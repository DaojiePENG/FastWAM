import numpy as np
import pytest
import torch
import torch.nn as nn
from types import SimpleNamespace

from fastwam.datasets.lerobot import cloudedge_robot_video_dataset as cloudedge_dataset
from fastwam.datasets.lerobot.cloudedge_robot_video_dataset import (
    CloudEdgeRobotVideoDataset,
    sample_stale_index,
)
from fastwam.models.wan22.leapbotce import (
    CloudPlanningCache,
    EdgeVisionEncoder,
    LeapBotCE,
    stale_loss_weight,
)
from fastwam.models.wan22.cloud_kv_adapter import CloudKVAdapter
from fastwam.models.wan22 import leapbotce_edge_train_factory
from fastwam.trainer import Wan22Trainer


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


def test_cloud_kv_adapter_is_identity_then_learns_residual():
    adapter = CloudKVAdapter(num_layers=2, cache_dim=6, rank=3)
    # The source cloud cache is treated as a frozen tensor.  Gradients should
    # still reach the lightweight interface adapter.
    keys = [torch.randn(2, 4, 6) for _ in range(2)]
    values = [torch.randn(2, 4, 6) for _ in range(2)]

    adapted_keys, adapted_values = adapter(keys, values)
    assert isinstance(adapted_keys, list)
    assert isinstance(adapted_values, list)
    for original, adapted in zip(keys, adapted_keys):
        torch.testing.assert_close(adapted, original)
        assert adapted.dtype == original.dtype
        assert adapted.shape == original.shape
    for original, adapted in zip(values, adapted_values):
        torch.testing.assert_close(adapted, original)

    # Zero-init up projections preserve the baseline, while still providing a
    # useful gradient to the adapter when the cache is frozen.
    loss = sum(item.square().mean() for item in adapted_keys + adapted_values)
    loss.backward()
    assert adapter.up_k.weight.grad is not None
    assert adapter.up_v.weight.grad is not None
    assert adapter.up_k.weight.grad.abs().sum() > 0
    assert adapter.up_v.weight.grad.abs().sum() > 0
    assert all(item.grad is None for item in keys + values)


def test_cloud_kv_adapter_validates_cache_contract():
    adapter = CloudKVAdapter(num_layers=2, cache_dim=4, rank=2)
    cache = [torch.zeros(1, 3, 4) for _ in range(2)]
    with pytest.raises(ValueError, match="layer count"):
        adapter(cache[:1], cache[:1])
    with pytest.raises(ValueError, match="shape mismatch"):
        adapter(cache, [torch.zeros(1, 2, 4) for _ in range(2)])
    with pytest.raises(ValueError, match="channel mismatch"):
        bad = [torch.zeros(1, 3, 5) for _ in range(2)]
        adapter(bad, bad)
    with pytest.raises(ValueError, match=r"shape \[batch, sequence, channels\]"):
        bad = [torch.zeros(1, 4) for _ in range(2)]
        adapter(bad, bad)


def test_cloud_kv_adapter_checkpoint_roundtrip(tmp_path):
    def make_shell():
        model = LeapBotCE.__new__(LeapBotCE)
        nn.Module.__init__(model)
        model.action_expert = nn.Linear(3, 3)
        model.edge_vision = SimpleNamespace(projector=nn.Linear(3, 3))
        model.proprio_encoder = None
        model.cloud_kv_adapter = CloudKVAdapter(num_layers=2, cache_dim=4, rank=2)
        model.edge_num_views = 2
        model.stale_loss_lambda_max = 0.5
        model.stale_loss_warmup_steps = 1
        model.torch_dtype = torch.float32
        return model

    source = make_shell()
    with torch.no_grad():
        source.cloud_kv_adapter.up_k.weight.fill_(0.125)
        source.cloud_kv_adapter.layer_scale_v.fill_(0.75)
    path = tmp_path / "leapbotce_delta.pt"
    source.save_checkpoint(path)

    restored = make_shell()
    restored.load_checkpoint(path)
    torch.testing.assert_close(
        restored.cloud_kv_adapter.up_k.weight,
        source.cloud_kv_adapter.up_k.weight,
    )
    torch.testing.assert_close(
        restored.cloud_kv_adapter.layer_scale_v,
        source.cloud_kv_adapter.layer_scale_v,
    )


def test_cloud_kv_adapter_is_in_dit_train_registry_without_unfreezing_cloud():
    class TinyTrainShell(nn.Module):
        def __init__(self):
            super().__init__()
            self.video_expert = nn.Linear(4, 4)
            self.cloud_kv_adapter = CloudKVAdapter(num_layers=2, cache_dim=4, rank=2)
            self.dit = nn.ModuleDict({"cloud_kv_adapter": self.cloud_kv_adapter})

    model = TinyTrainShell()
    Wan22Trainer._apply_dit_only_train_mode(model)
    assert all(not parameter.requires_grad for parameter in model.video_expert.parameters())
    assert all(parameter.requires_grad for parameter in model.cloud_kv_adapter.parameters())
    assert model.cloud_kv_adapter.training


def test_edge_factory_registers_optional_cloud_adapter(monkeypatch):
    class TinyLeapBotCE(nn.Module):
        def __init__(self):
            super().__init__()
            self.action_expert = nn.Linear(2, 2)
            self.edge_vision = SimpleNamespace(projector=nn.Linear(2, 2))
            self.cloud_kv_adapter = CloudKVAdapter(num_layers=1, cache_dim=4, rank=2)

    model = TinyLeapBotCE()
    monkeypatch.setattr(
        leapbotce_edge_train_factory,
        "_create_local_leapbotce",
        lambda **_kwargs: model,
    )
    output = leapbotce_edge_train_factory.create_leapbotce()
    assert output is model
    assert set(output.dit.keys()) == {
        "action_expert",
        "edge_vision_projector",
        "cloud_kv_adapter",
    }
    assert output.dit["cloud_kv_adapter"] is output.cloud_kv_adapter


def test_cloud_encoding_applies_kv_adapter_at_cache_boundary():
    class TinyVideoExpert:
        fuse_vae_embedding_in_latents = False

        @staticmethod
        def prepare(latents, timestep, context, context_mask, *_args):
            del timestep
            batch = latents.shape[0]
            tokens = torch.zeros(batch, 3, 4)
            t_mod = torch.zeros(batch, 1, 4)
            freqs = torch.zeros(3, 1, 2, dtype=torch.complex64)
            return (tokens, None, t_mod, context, context_mask, freqs, None, None, None, 1)

        @staticmethod
        def build_video_to_video_mask(seq_len, tokens_per_frame, device):
            del tokens_per_frame
            return torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)

    class TinyMoT:
        num_heads = 1
        attn_head_dim = 4
        num_layers = 2
        raw_keys = None
        raw_values = None

        @classmethod
        def prefill_video_cache_tensor(cls, *_args):
            key = torch.randn(1, 3, 4)
            value = torch.randn(1, 3, 4)
            cls.raw_keys = [key.clone(), key.clone()]
            cls.raw_values = [value.clone(), value.clone()]
            return cls.raw_keys, cls.raw_values

    model = LeapBotCE.__new__(LeapBotCE)
    nn.Module.__init__(model)
    model.device = torch.device("cpu")
    model.torch_dtype = torch.float32
    model.video_expert = TinyVideoExpert()
    model.mot = TinyMoT()
    model._encode_video_latents = lambda image: torch.zeros(image.shape[0], 1, 1, 1, 1)
    model.cloud_kv_adapter = CloudKVAdapter(num_layers=2, cache_dim=4, rank=2)
    with torch.no_grad():
        model.cloud_kv_adapter.up_k.weight.fill_(0.1)
        model.cloud_kv_adapter.up_v.weight.fill_(0.1)

    context = torch.zeros(1, 1, 4)
    context_mask = torch.ones(1, 1, dtype=torch.bool)
    cache = model._encode_cloud_tensor(
        torch.zeros(1, 3, 8, 8), context, context_mask
    )
    assert len(cache.video_cache_k) == 2
    assert cache.video_cache_k[0].shape == (1, 3, 4)
    assert cache.video_cache_k[0].dtype == torch.float32
    assert not torch.equal(cache.video_cache_k[0], TinyMoT.raw_keys[0])
    assert not torch.equal(cache.video_cache_v[0], TinyMoT.raw_values[0])


def test_stale_index_respects_padding_and_boundary():
    pad = torch.tensor([True, True, False, False, False])
    stale, delay = sample_stale_index(4, 4, pad, np.random.RandomState(0))
    assert stale in {2, 3}
    assert delay == 4 - stale
    assert sample_stale_index(0, 20, torch.tensor([False]), np.random.RandomState(0)) == (0, 0)


def test_dataset_keeps_cloud_image_and_proprio_time_aligned(monkeypatch):
    dataset = CloudEdgeRobotVideoDataset.__new__(CloudEdgeRobotVideoDataset)
    dataset.max_delay_steps = 2
    dataset.action_horizon = 2
    sample = {
        "video": torch.arange(5, dtype=torch.float32).view(1, 5, 1, 1).expand(3, 5, 1, 2),
        "image_is_pad": torch.zeros(5, dtype=torch.bool),
        "proprio": torch.arange(10, dtype=torch.float32).view(5, 2),
        "action": torch.zeros(5, 7),
        "action_is_pad": torch.zeros(5, dtype=torch.bool),
        "prompt": "task",
    }
    dataset.dataset = [sample]
    monkeypatch.setattr(cloudedge_dataset, "sample_stale_index", lambda *_: (0, 2))

    output = dataset[0]

    torch.testing.assert_close(output["cloud_stale_image"], sample["video"][:, 0])
    torch.testing.assert_close(output["cloud_stale_proprio"], sample["proprio"][0:1])
    torch.testing.assert_close(output["cloud_current_image"], sample["video"][:, 2])
    torch.testing.assert_close(output["cloud_current_proprio"], sample["proprio"][2:3])
    torch.testing.assert_close(output["edge_current_proprio"], sample["proprio"][2:3])


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


def test_cloud_cache_uses_cloud_state_and_retains_language_only_context():
    model = LeapBotCE.__new__(LeapBotCE)
    nn.Module.__init__(model)
    model.device = torch.device("cpu")
    model.torch_dtype = torch.float32
    model.proprio_dim = 2
    model.proprio_encoder = nn.Linear(2, 4, bias=False)
    with torch.no_grad():
        model.proprio_encoder.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, -1.0]]))

    language = torch.ones(1, 2, 4)
    language_mask = torch.ones(1, 2, dtype=torch.bool)
    captured = {}

    def encode_prompt(_):
        return language, language_mask

    def encode_tensor(image, cloud_context, cloud_mask, edge_context=None, edge_mask=None):
        captured["image"] = image
        captured["cloud_context"] = cloud_context
        captured["cloud_mask"] = cloud_mask
        return CloudPlanningCache([], [], edge_context, edge_mask, 1, 1)

    model.encode_prompt = encode_prompt
    model._encode_cloud_tensor = encode_tensor
    cloud_proprio = torch.tensor([[2.0, 3.0]])
    cache = model.encode_cloud(torch.zeros(1, 3, 8, 8), prompt="task", cloud_proprio=cloud_proprio)

    assert captured["cloud_context"].shape == (1, 3, 4)
    torch.testing.assert_close(
        captured["cloud_context"][:, -1], model.proprio_encoder(cloud_proprio)
    )
    assert captured["cloud_mask"].shape == (1, 3)
    torch.testing.assert_close(cache.context, language)
    torch.testing.assert_close(cache.context_mask, language_mask)


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
    model.proprio_dim = 8
    model.proprio_encoder = nn.Linear(8, 4, bias=False)

    context = torch.ones(1, 2, 4)
    context_mask = torch.ones(1, 2, dtype=torch.bool)
    cache = CloudPlanningCache([], [], context, context_mask, video_seq_len=1, video_tokens_per_frame=1)
    observed = {}

    def encode_cloud(input_image, prompt=None, context=None, context_mask=None,
                     cloud_proprio=None, proprio=None):
        observed["input_shape"] = tuple(input_image.shape)
        observed["prompt"] = prompt
        observed["cloud_proprio"] = cloud_proprio if cloud_proprio is not None else proprio
        return cache

    def edge_context(cache_context, cache_mask, current_views):
        assert current_views.shape == (1, 2, 3, 8, 4)
        assert cache_context.shape == (1, 3, 4)
        observed["edge_context"] = cache_context
        return cache_context, cache_mask

    def predict_from_cache(cache_arg, noisy_action, timestep, edge_context_arg, edge_mask):
        assert cache_arg is cache
        assert timestep.shape == (1,)
        assert edge_context_arg is observed["edge_context"]
        assert edge_mask.shape == (1, 3)
        return torch.full_like(noisy_action, 0.5)

    model.encode_cloud = encode_cloud
    model._edge_context = edge_context
    model._predict_from_cache = predict_from_cache
    image = torch.rand(1, 3, 8, 8)
    proprio = torch.rand(1, 8)

    two_stage = model.infer_action_edge(
        cache, model._split_views(image, 2), action_horizon=3, edge_proprio=proprio,
        num_inference_steps=2, seed=7
    )
    synchronous = model.infer_action(
        "task", image, action_horizon=3, proprio=proprio, num_inference_steps=2, seed=7
    )

    assert observed["input_shape"] == (1, 3, 8, 8)
    assert observed["prompt"] == "task"
    assert observed["cloud_proprio"] is proprio
    assert observed["edge_context"].shape == (1, 3, 4)
    assert two_stage["action"].dtype == torch.float32
    torch.testing.assert_close(synchronous["action"], two_stage["action"])
