import numpy as np
import pytest
import torch
import torch.nn as nn
from types import SimpleNamespace

from fastwam.datasets.lerobot.cloudedge_robot_video_dataset import sample_stale_index
from fastwam.models.wan22.leapbotce import (
    CloudPlanningCache,
    EdgeVisionEncoder,
    LeapBotCE,
    stale_loss_weight,
)


class TinyVision(nn.Module):
    embed_dim = 4

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(3, self.embed_dim)

    def forward(self, image):
        return self.proj(image.mean(dim=(-1, -2)))


def test_stale_loss_curriculum():
    assert stale_loss_weight(0, 100, 0.5) == 0.0
    assert stale_loss_weight(50, 100, 0.5) == pytest.approx(0.25)
    assert stale_loss_weight(200, 100, 0.5) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        stale_loss_weight(1, 10, 1.1)


def test_stale_index_respects_padding_and_boundary():
    pad = torch.tensor([True, True, False, False, False])
    stale, delay = sample_stale_index(4, 4, pad, np.random.RandomState(0))
    assert stale in {2, 3}
    assert delay == 4 - stale
    assert sample_stale_index(0, 20, torch.tensor([False]), np.random.RandomState(0)) == (0, 0)


def test_edge_encoder_freezes_backbone_and_trains_projector():
    backbone = TinyVision()
    encoder = EdgeVisionEncoder(
        text_dim=8, num_views=2, encoder=backbone, embed_dim=4, freeze=True
    )
    output = encoder(torch.rand(2, 2, 3, 32, 32))
    assert output.shape == (2, 1, 8)
    output.sum().backward()
    assert all(parameter.grad is None for parameter in backbone.parameters())
    assert encoder.projector[0].weight.grad is not None
    encoder.train()
    assert not backbone.training


class TinyInferenceScheduler:
    def build_inference_schedule(self, num_steps, device, dtype, shift_override=None):
        del shift_override
        return torch.ones(num_steps, device=device, dtype=dtype), torch.full(
            (num_steps,), -0.25, device=device, dtype=dtype
        )

    @staticmethod
    def step(prediction, delta, action):
        return action + prediction * delta


def test_sync_wrapper_matches_two_stage_edge_inference_with_fixed_seed():
    # Build a tiny inference-only shell around the real two-stage methods. This
    # validates the wrapper contract without loading the 5B frozen cloud expert.
    model = LeapBotCE.__new__(LeapBotCE)
    nn.Module.__init__(model)
    model.device = torch.device("cpu")
    model.torch_dtype = torch.float32
    model.edge_num_views = 2
    model.action_expert = SimpleNamespace(action_dim=2)
    model.infer_action_scheduler = TinyInferenceScheduler()

    context = torch.ones(1, 2, 4)
    context_mask = torch.ones(1, 2, dtype=torch.bool)
    cache = CloudPlanningCache([], [], context, context_mask, video_seq_len=1, video_tokens_per_frame=1)
    observed = {}

    def encode_cloud(input_image, prompt=None, context=None, context_mask=None, proprio=None):
        observed["input_shape"] = tuple(input_image.shape)
        observed["prompt"] = prompt
        observed["proprio"] = proprio
        return cache

    def edge_context(cache_context, cache_mask, current_views):
        assert current_views.shape == (1, 2, 3, 8, 4)
        return cache_context, cache_mask

    def predict_from_cache(cache_arg, noisy_action, timestep, edge_context_arg, edge_mask):
        assert cache_arg is cache
        assert timestep.shape == (1,)
        assert edge_context_arg is context
        assert edge_mask is context_mask
        return torch.full_like(noisy_action, 0.5)

    model.encode_cloud = encode_cloud
    model._edge_context = edge_context
    model._predict_from_cache = predict_from_cache
    image = torch.rand(1, 3, 8, 8)
    proprio = torch.rand(1, 8)

    two_stage = model.infer_action_edge(
        cache, model._split_views(image, 2), action_horizon=3, num_inference_steps=2, seed=7
    )
    synchronous = model.infer_action(
        "task", image, action_horizon=3, proprio=proprio, num_inference_steps=2, seed=7
    )

    assert observed["input_shape"] == (1, 3, 8, 8)
    assert observed["prompt"] == "task"
    assert observed["proprio"] is proprio
    assert two_stage["action"].dtype == torch.float32
    torch.testing.assert_close(synchronous["action"], two_stage["action"])
