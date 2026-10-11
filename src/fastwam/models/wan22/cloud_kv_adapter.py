"""Lightweight adapters for the cloud-to-edge planning cache."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class CloudPlanningCodec(nn.Module):
    """Compress per-layer video K/V caches into a compact planning token set.

    The 30-layer per-layer cache (``[layers, batch, tokens, channels]``) is
    reduced to ``M`` planning tokens: spatial average pooling of the value
    cache, a learnable softmax mixture across depth, and a bottleneck MLP.
    The resulting ``z`` is the entire cloud-to-edge transmission payload.
    With ``use_kv_decoder=False`` the same vector is reused as K and V at
    every action-expert layer (modulated by per-layer scalars); with the
    decoder enabled, low-rank projections decouple the addressing (K) from
    the content (V) while the transmitted payload stays ``z``.
    """

    def __init__(
        self,
        num_layers: int,
        cache_dim: int,
        pool_grid: Sequence[int],
        bottleneck_dim: int = 1024,
        use_kv_decoder: bool = False,
        decoder_rank: int = 256,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_layers = int(num_layers)
        self.cache_dim = int(cache_dim)
        self.pool_grid = (int(pool_grid[0]), int(pool_grid[1]))
        self.num_tokens = self.pool_grid[0] * self.pool_grid[1]
        self.bottleneck_dim = int(bottleneck_dim)
        self.use_kv_decoder = bool(use_kv_decoder)
        self.decoder_rank = int(decoder_rank)
        self.dropout_p = float(dropout)
        if self.num_layers <= 0:
            raise ValueError(f"`num_layers` must be > 0, got {num_layers}")
        if self.cache_dim <= 0:
            raise ValueError(f"`cache_dim` must be > 0, got {cache_dim}")
        if min(self.pool_grid) <= 0:
            raise ValueError(f"`pool_grid` entries must be > 0, got {self.pool_grid}")
        if not 0.0 <= self.dropout_p < 1.0:
            raise ValueError(f"`dropout` must be in [0,1), got {dropout}")

        # Uniform layer mixture at init; training decides which depths matter.
        self.layer_logits = nn.Parameter(torch.zeros(self.num_layers))
        self.norm = nn.LayerNorm(self.cache_dim)
        self.in_proj = nn.Linear(self.cache_dim, self.bottleneck_dim)
        self.out_proj = nn.Linear(self.bottleneck_dim, self.cache_dim)
        self.dropout = nn.Dropout(self.dropout_p)

        self.scale_k = nn.Parameter(torch.ones(self.num_layers))
        self.scale_v = nn.Parameter(torch.ones(self.num_layers))
        if self.use_kv_decoder:
            if not 0 < self.decoder_rank <= self.cache_dim:
                raise ValueError(
                    f"`decoder_rank` must be in (0, {self.cache_dim}], got {decoder_rank}"
                )
            self.dec_down_k = nn.Linear(self.cache_dim, self.decoder_rank, bias=False)
            self.dec_up_k = nn.Linear(self.decoder_rank, self.cache_dim, bias=False)
            self.dec_down_v = nn.Linear(self.cache_dim, self.decoder_rank, bias=False)
            self.dec_up_v = nn.Linear(self.decoder_rank, self.cache_dim, bias=False)

    def _spatial_pool(self, tokens: torch.Tensor, token_grid: tuple[int, int]) -> torch.Tensor:
        """Average-pool ``[layers, batch, tokens, channels]`` over the 2D token grid."""
        layers, batch, seq_len, channels = tokens.shape
        if token_grid[0] * token_grid[1] != seq_len:
            raise ValueError(
                f"Token grid {token_grid} does not match sequence length {seq_len}"
            )
        grid = tokens.view(
            layers * batch, token_grid[0], token_grid[1], channels
        ).permute(0, 3, 1, 2)
        grid = F.adaptive_avg_pool2d(grid, self.pool_grid)
        return grid.permute(0, 2, 3, 1).reshape(layers, batch, -1, channels)

    def forward(
        self,
        cache_k: Sequence[torch.Tensor],
        cache_v: Sequence[torch.Tensor],
        token_grid: tuple[int, int],
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Compress layer-wise K/V lists into ``num_tokens`` planning tokens."""
        if len(cache_k) != self.num_layers or len(cache_v) != self.num_layers:
            raise ValueError(
                "Cloud K/V cache layer count mismatch: "
                f"expected {self.num_layers}, got K={len(cache_k)}, V={len(cache_v)}"
            )
        first = cache_v[0]
        for key, value in zip(cache_k, cache_v):
            if key.shape != value.shape or value.shape != first.shape:
                raise ValueError("Cloud K/V cache tensors must share one shape.")
        if first.shape[-1] != self.cache_dim:
            raise ValueError(
                f"Cloud K/V cache channel mismatch: expected {self.cache_dim}, "
                f"got {first.shape[-1]}"
            )

        parameter = self.in_proj.weight
        parameter_dtype = parameter.dtype
        value_stack = torch.stack(list(cache_v), dim=0).to(
            device=parameter.device, dtype=parameter_dtype
        )
        pooled = self._spatial_pool(value_stack, token_grid)
        weights = torch.softmax(self.layer_logits.to(parameter_dtype), dim=0)
        mixed = torch.einsum("l,lbmc->bmc", weights, pooled)
        z = self.out_proj(self.dropout(F.gelu(self.in_proj(self.norm(mixed)))))

        if self.use_kv_decoder:
            keys = self.dec_up_k(self.dec_down_k(z))
            values = self.dec_up_v(self.dec_down_v(z))
        else:
            keys = values = z

        out_dtype = first.dtype
        adapted_k = [(keys * self.scale_k[i].to(keys.dtype)).to(out_dtype)
                     for i in range(self.num_layers)]
        adapted_v = [(values * self.scale_v[i].to(values.dtype)).to(out_dtype)
                     for i in range(self.num_layers)]
        return adapted_k, adapted_v

    def config(self) -> dict[str, float | int | bool | tuple[int, int]]:
        """Return constructor metadata for lightweight checkpoints."""
        return {
            "num_layers": self.num_layers,
            "cache_dim": self.cache_dim,
            "pool_grid": self.pool_grid,
            "bottleneck_dim": self.bottleneck_dim,
            "use_kv_decoder": self.use_kv_decoder,
            "decoder_rank": self.decoder_rank,
            "dropout": self.dropout_p,
        }
