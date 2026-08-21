import numpy as np
import pytest
import torch
import torch.nn as nn

from fastwam.datasets.lerobot.cloudedge_robot_video_dataset import sample_stale_index
from fastwam.models.wan22.leapbotce import EdgeVisionEncoder, stale_loss_weight


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
