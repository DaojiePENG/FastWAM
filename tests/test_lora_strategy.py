import math

import torch
from torch import nn
from fastwam.trainer import Wan22Trainer

from leapbot_va.lora import (
    LoRALinear,
    VideoLoRAConfig,
    inject_video_self_attention_lora,
    lora_parameters,
    merge_video_self_attention_lora,
)


class _Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q = nn.Linear(4, 4)
        self.k = nn.Linear(4, 4)
        self.v = nn.Linear(4, 4)
        self.o = nn.Linear(4, 4)


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _Attention()


class _Video(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([_Block(), _Block()])


def test_lora_injection_is_initially_exact_and_preserves_base_state_keys():
    torch.manual_seed(0)
    video = _Video()
    x = torch.randn(2, 3, 4)
    expected = video.blocks[0].self_attn.q(x)
    base_weight = video.blocks[0].self_attn.q.weight

    replaced = inject_video_self_attention_lora(
        video,
        VideoLoRAConfig(enabled=True, rank=2, alpha=2),
    )
    actual = video.blocks[0].self_attn.q(x)

    assert len(replaced) == 8
    assert isinstance(video.blocks[0].self_attn.q, LoRALinear)
    assert video.blocks[0].self_attn.q.weight is base_weight
    assert "blocks.0.self_attn.q.weight" in video.state_dict()
    assert "blocks.0.self_attn.q.lora_A" in video.state_dict()
    torch.testing.assert_close(actual, expected)


def test_only_lora_parameters_can_be_selected_from_frozen_video():
    video = _Video()
    inject_video_self_attention_lora(
        video,
        VideoLoRAConfig(enabled=True, rank=2, alpha=2),
    )
    video.requires_grad_(False)
    for parameter in lora_parameters(video):
        parameter.requires_grad_(True)

    trainable_names = {
        name for name, parameter in video.named_parameters() if parameter.requires_grad
    }
    assert trainable_names
    assert all(name.endswith(("lora_A", "lora_B")) for name in trainable_names)


def test_cosine_scheduler_preserves_optimizer_group_lr_ratio():
    first = nn.Parameter(torch.ones(()))
    second = nn.Parameter(torch.ones(()))
    optimizer = torch.optim.AdamW(
        [
            {"params": [first], "lr": 1e-5},
            {"params": [second], "lr": 1e-4},
        ]
    )
    trainer = Wan22Trainer.__new__(Wan22Trainer)
    trainer.optimizer = optimizer
    trainer.learning_rate = 1e-5
    scheduler = trainer._build_scheduler("cosine", total_train_steps=10, warmup_steps=0)

    for _ in range(10):
        optimizer.step()
        scheduler.step()
        assert math.isclose(
            optimizer.param_groups[1]["lr"] / optimizer.param_groups[0]["lr"],
            10.0,
        )


def test_lora_merge_restores_plain_linear_with_equivalent_output():
    torch.manual_seed(7)
    video = _Video()
    inject_video_self_attention_lora(
        video,
        VideoLoRAConfig(enabled=True, rank=2, alpha=2),
    )
    nn.init.normal_(video.blocks[0].self_attn.q.lora_B, std=0.01)
    video.eval()
    x = torch.randn(2, 3, 4)
    expected = video.blocks[0].self_attn.q(x)

    merged = merge_video_self_attention_lora(video)
    actual = video.blocks[0].self_attn.q(x)

    assert len(merged) == 8
    assert type(video.blocks[0].self_attn.q) is nn.Linear
    assert not any("lora_" in key for key in video.state_dict())
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
