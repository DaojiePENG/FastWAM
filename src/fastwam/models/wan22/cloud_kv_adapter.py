"""Lightweight adapters for the cloud-to-edge planning cache."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn


class CloudKVAdapter(nn.Module):
    """Shared low-rank residual adapter for per-layer video K/V caches.

    The adapter deliberately keeps the cache contract unchanged: each layer
    remains a ``[batch, sequence, channels]`` tensor.  The final projections
    are zero-initialized, so enabling the module starts as an exact identity
    and can be safely used for ablations or resumed training.
    """

    def __init__(
        self,
        num_layers: int,
        cache_dim: int,
        rank: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_layers = int(num_layers)
        self.cache_dim = int(cache_dim)
        self.rank = int(rank)
        self.dropout_p = float(dropout)
        if self.num_layers <= 0:
            raise ValueError(f"`num_layers` must be > 0, got {num_layers}")
        if self.cache_dim <= 0:
            raise ValueError(f"`cache_dim` must be > 0, got {cache_dim}")
        if self.rank <= 0 or self.rank > self.cache_dim:
            raise ValueError(
                f"`rank` must be in [1, cache_dim={self.cache_dim}], got {rank}"
            )
        if not 0.0 <= self.dropout_p < 1.0:
            raise ValueError(f"`dropout` must be in [0,1), got {dropout}")

        self.norm_k = nn.LayerNorm(self.cache_dim)
        self.norm_v = nn.LayerNorm(self.cache_dim)
        self.down_k = nn.Linear(self.cache_dim, self.rank, bias=False)
        self.down_v = nn.Linear(self.cache_dim, self.rank, bias=False)
        self.up_k = nn.Linear(self.rank, self.cache_dim, bias=False)
        self.up_v = nn.Linear(self.rank, self.cache_dim, bias=False)
        self.dropout = nn.Dropout(self.dropout_p)

        # A per-layer scale lets the same bottleneck specialize gently across
        # depth.  With zero up-projections, the initial module is still exact
        # identity while gradients to the up-projections are non-zero.
        self.layer_scale_k = nn.Parameter(torch.ones(self.num_layers))
        self.layer_scale_v = nn.Parameter(torch.ones(self.num_layers))
        nn.init.zeros_(self.up_k.weight)
        nn.init.zeros_(self.up_v.weight)

    def _adapt(
        self,
        cache: torch.Tensor,
        norm: nn.Module,
        down: nn.Module,
        up: nn.Module,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        if cache.ndim != 3:
            raise ValueError(
                "Cloud K/V cache tensors must have shape [batch, sequence, channels], "
                f"got {tuple(cache.shape)}"
            )
        if cache.shape[-1] != self.cache_dim:
            raise ValueError(
                f"Cloud K/V cache channel mismatch: expected {self.cache_dim}, "
                f"got {cache.shape[-1]}"
            )

        # Model parameters may be bf16 while a caller supplies fp32 cache
        # tensors (or vice versa).  Compute in the adapter parameter dtype and
        # return the original cache dtype so MoT can concatenate it unchanged.
        parameter_dtype = down.weight.dtype
        work = cache.to(device=down.weight.device, dtype=parameter_dtype)
        delta = up(self.dropout(torch.nn.functional.gelu(down(norm(work)))))
        delta = delta * scale.to(device=delta.device, dtype=delta.dtype)
        return cache + delta.to(device=cache.device, dtype=cache.dtype)

    def forward(
        self,
        cache_k: Sequence[torch.Tensor],
        cache_v: Sequence[torch.Tensor],
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Adapt layer-wise K/V lists without changing their cache contract."""
        if len(cache_k) != self.num_layers or len(cache_v) != self.num_layers:
            raise ValueError(
                "Cloud K/V cache layer count mismatch: "
                f"expected {self.num_layers}, got K={len(cache_k)}, V={len(cache_v)}"
            )

        adapted_k: list[torch.Tensor] = []
        adapted_v: list[torch.Tensor] = []
        for layer_idx, (key, value) in enumerate(zip(cache_k, cache_v)):
            if not isinstance(key, torch.Tensor) or not isinstance(value, torch.Tensor):
                raise TypeError("Cloud K/V cache entries must be torch.Tensor objects")
            if key.shape != value.shape:
                raise ValueError(
                    f"Cloud K/V cache[{layer_idx}] shape mismatch: "
                    f"K={tuple(key.shape)}, V={tuple(value.shape)}"
                )
            adapted_k.append(
                self._adapt(
                    key,
                    self.norm_k,
                    self.down_k,
                    self.up_k,
                    self.layer_scale_k[layer_idx],
                )
            )
            adapted_v.append(
                self._adapt(
                    value,
                    self.norm_v,
                    self.down_v,
                    self.up_v,
                    self.layer_scale_v[layer_idx],
                )
            )
        return adapted_k, adapted_v

    def config(self) -> dict[str, float | int]:
        """Return constructor metadata for lightweight checkpoints."""
        return {
            "num_layers": self.num_layers,
            "cache_dim": self.cache_dim,
            "rank": self.rank,
            "dropout": self.dropout_p,
        }
