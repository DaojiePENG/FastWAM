"""Minimal state-dict-compatible LoRA for LeapBot's frozen video expert."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class VideoLoRAConfig:
    enabled: bool = False
    rank: int = 16
    alpha: float = 16.0
    dropout: float = 0.0
    learning_rate_multiplier: float = 10.0

    def validate(self) -> None:
        if self.rank <= 0:
            raise ValueError("LoRA rank must be positive")
        if self.alpha <= 0:
            raise ValueError("LoRA alpha must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("LoRA dropout must be in [0,1)")
        if self.learning_rate_multiplier <= 0:
            raise ValueError("LoRA learning-rate multiplier must be positive")


class LoRALinear(nn.Linear):
    """An ``nn.Linear`` subclass that preserves base ``weight`` state keys."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        *,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(
            in_features,
            out_features,
            bias=bias,
            device=device,
            dtype=dtype,
        )
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.lora_dropout = nn.Dropout(float(dropout)) if dropout else nn.Identity()
        self.lora_A = nn.Parameter(
            torch.empty(self.rank, in_features, device=device, dtype=dtype)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(out_features, self.rank, device=device, dtype=dtype)
        )
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        *,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ) -> "LoRALinear":
        result = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
        )
        # Reuse the exact pretrained Parameter objects. This preserves keys
        # such as ``...q.weight`` and avoids allocating a second base matrix.
        result.weight = linear.weight
        result.bias = linear.bias
        return result

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        base = F.linear(input, self.weight, self.bias)
        update = F.linear(F.linear(self.lora_dropout(input), self.lora_A), self.lora_B)
        return base + update.to(base.dtype) * self.scaling


def inject_video_self_attention_lora(
    video_expert: nn.Module,
    config: VideoLoRAConfig,
) -> tuple[str, ...]:
    """Add LoRA to every video self-attention Q/K/V/O projection."""
    config.validate()
    replaced = []
    for layer_index, block in enumerate(video_expert.blocks):
        for projection_name in ("q", "k", "v", "o"):
            projection = getattr(block.self_attn, projection_name)
            if isinstance(projection, LoRALinear):
                continue
            if not isinstance(projection, nn.Linear):
                raise TypeError(
                    f"video block {layer_index} self_attn.{projection_name} "
                    f"must be nn.Linear, got {type(projection)}"
                )
            setattr(
                block.self_attn,
                projection_name,
                LoRALinear.from_linear(
                    projection,
                    rank=config.rank,
                    alpha=config.alpha,
                    dropout=config.dropout,
                ),
            )
            replaced.append(f"blocks.{layer_index}.self_attn.{projection_name}")
    if not replaced and not any(
        isinstance(module, LoRALinear) for module in video_expert.modules()
    ):
        raise RuntimeError("no video self-attention projections received LoRA")
    return tuple(replaced)


def lora_parameters(module: nn.Module) -> list[nn.Parameter]:
    result = []
    for child in module.modules():
        if isinstance(child, LoRALinear):
            result.extend((child.lora_A, child.lora_B))
    return result
