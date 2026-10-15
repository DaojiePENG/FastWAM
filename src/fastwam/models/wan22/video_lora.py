"""LoRA injection for the frozen video expert (a4 producer-side adaptation).

Wraps the attention projections (q/k/v/o) of the video expert's blocks with
low-rank residual adapters.  The frozen base Linear is held in a plain list so
it does NOT join the ``model.dit`` module tree — Wan22Trainer freezes the whole
model and then re-enables ``model.dit``, so anything registered under
``model.dit`` becomes trainable.  Keeping the base out of that registry is what
preserves the frozen base weights while the LoRA factors train.

Injection keeps the module functionally identical to the base network at step
zero (B is zero-initialized), so resuming from a checkpoint and adding LoRA is
a strict continuation.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """``y = base(x) + scaling * B(A(dropout(x)))`` with the base kept frozen.

    The base Linear is stored inside a list (``_base_holder``) instead of as a
    child module: lists are invisible to ``nn.Module`` registration, so the
    frozen weights never appear under ``model.dit`` and cannot be re-enabled
    by the trainer's freeze-then-enable pass.
    """

    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError(f"LoRALinear expects nn.Linear, got {type(base)!r}")
        self.in_features = base.in_features
        self.out_features = base.out_features
        if not (0 < rank <= min(self.in_features, self.out_features)):
            raise ValueError(f"LoRA rank {rank} out of range for {self.in_features}x{self.out_features}")
        self.rank = int(rank)
        self.scaling = float(alpha) / float(rank)
        # Base stays out of the module tree (see class docstring).
        self._base_holder = [base]
        self.lora_dropout = nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity()
        self.lora_a = nn.Parameter(torch.zeros(self.rank, self.in_features))
        self.lora_b = nn.Parameter(torch.zeros(self.out_features, self.rank))
        self.reset_parameters()

    def reset_parameters(self):
        # Kaiming-uniform on A, zeros on B: the adapter is an exact no-op at
        # injection time, so resumed training starts from the base behaviour.
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b)

    def to(self, *args, **kwargs):
        out = super().to(*args, **kwargs)
        base = self._base_holder[0]
        base.to(*args, **kwargs)
        return out

    def forward(self, x):
        base = self._base_holder[0]
        down = F.linear(self.lora_dropout(x), self.lora_a)
        up = F.linear(down, self.lora_b)
        return base(x) + up * self.scaling


def inject_video_lora(video_expert: nn.Module, rank: int = 16, alpha: float = 32.0,
                      dropout: float = 0.0, targets=("q", "k", "v", "o")):
    """Replace ``blocks[*].self_attn.<t>`` with LoRALinear for every block.

    Returns the list of injected wrappers (to be registered under
    ``model.dit`` so the trainer's freeze-then-enable pass trains exactly the
    LoRA factors and nothing else).  Raises if a target slot is missing or
    already wrapped, so double-injection cannot happen silently.
    """
    blocks = getattr(video_expert, "blocks", None)
    if blocks is None or len(blocks) == 0:
        raise RuntimeError("video expert exposes no .blocks to inject LoRA into")
    wrappers = []
    for i, block in enumerate(blocks):
        attn = getattr(block, "self_attn", None)
        if attn is None:
            raise RuntimeError(f"video block {i} has no self_attn")
        for t in targets:
            base = getattr(attn, t, None)
            if isinstance(base, LoRALinear):
                raise RuntimeError(f"blocks[{i}].self_attn.{t} already LoRA-wrapped")
            if not isinstance(base, nn.Linear):
                raise RuntimeError(f"blocks[{i}].self_attn.{t} is {type(base)!r}, expected nn.Linear")
            wrapper = LoRALinear(base, rank=rank, alpha=alpha, dropout=dropout)
            wrapper.to(device=base.weight.device, dtype=base.weight.dtype)
            setattr(attn, t, wrapper)
            wrappers.append(wrapper)
    return wrappers


def lora_parameter_count(wrappers) -> int:
    return sum(w.lora_a.numel() + w.lora_b.numel() for w in wrappers)
