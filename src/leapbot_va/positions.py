"""Checkpoint-compatible hierarchical temporal positions for LeapBot-VA.

FastWAM's pretrained RoPE coordinates are local to the sequence passed to each
expert.  Replacing those coordinates with episode-global indices changes the
pretrained attention phases, even before any history is used.  This module
keeps the native local RoPE coordinates and represents episode progress with a
separate, initially no-op embedding::

    video_pre = video_expert.pre_dit(
        ...,
        frame_position_ids=positions.local_video_rope_ids(num_frames, device=device),
    )
    video_pre = positions.apply_video_pre_dit(video_pre, absolute_block_ids)

    action_pre = action_expert.pre_dit(
        ...,
        position_ids=positions.local_action_rope_ids(num_actions, device=device),
    )
    action_pre = positions.apply_action_pre_dit(
        action_pre,
        absolute_control_ids,
        absolute_block_ids,
    )

The episode embedding uses analytic sinusoidal features rather than a finite
lookup table, so it supports arbitrarily long episodes.  All three output
projections are exactly zero-initialized: adding this module to a FastWAM
checkpoint is therefore an exact identity until the new projections train.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn


_INTEGER_DTYPES = {
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
}


def sinusoidal_episode_features(
    position_ids: torch.Tensor,
    feature_dim: int,
    *,
    max_period: float = 10_000.0,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return table-free sinusoidal features for non-negative episode positions.

    Args:
        position_ids: Integer tensor of any shape.
        feature_dim: Even feature dimension.
        max_period: Longest approximate sinusoidal period.
        dtype: Output floating-point dtype.  Trigonometry is evaluated in
            float64 first, which keeps large absolute positions well behaved.
    """

    if not isinstance(position_ids, torch.Tensor):
        raise TypeError("position_ids must be a torch.Tensor")
    if position_ids.dtype not in _INTEGER_DTYPES:
        raise TypeError("position_ids must have an integer dtype")
    if feature_dim < 2 or feature_dim % 2:
        raise ValueError("feature_dim must be a positive even integer")
    if not math.isfinite(max_period) or max_period <= 1.0:
        raise ValueError("max_period must be finite and greater than 1")
    if not dtype.is_floating_point:
        raise TypeError("feature dtype must be floating point")
    if bool((position_ids < 0).any().item()):
        raise ValueError("episode position ids must be non-negative")

    half_dim = feature_dim // 2
    frequency_index = torch.arange(
        half_dim,
        device=position_ids.device,
        dtype=torch.float64,
    )
    inverse_frequencies = torch.exp(
        -math.log(max_period) * frequency_index / float(half_dim)
    )
    phase = position_ids.to(torch.float64).unsqueeze(-1) * inverse_frequencies
    features = torch.cat((phase.sin(), phase.cos()), dim=-1)
    return features.to(dtype=dtype)


class HierarchicalTemporalPositionEmbedding(nn.Module):
    """Add episode-global progress without changing FastWAM's local RoPE.

    ``video_dim`` and ``action_dim`` are the hidden dimensions of the video
    and action pre-DiT token tensors, not the raw observation/action sizes.
    Video positions are absolute replan-block indices.  Action positions are
    hierarchical: every token receives both its coarse replan-block index and
    its fine absolute executed-control-step index.  The shared coarse clock
    lets a block's observation and actions identify the same episode segment,
    while the fine clock preserves actual controller progress.
    """

    def __init__(
        self,
        video_dim: int,
        action_dim: int,
        *,
        feature_dim: int = 128,
        max_period: float = 10_000.0,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if video_dim <= 0:
            raise ValueError("video_dim must be positive")
        if action_dim <= 0:
            raise ValueError("action_dim must be positive")
        if feature_dim < 2 or feature_dim % 2:
            raise ValueError("feature_dim must be a positive even integer")
        if not math.isfinite(max_period) or max_period <= 1.0:
            raise ValueError("max_period must be finite and greater than 1")

        self.video_dim = int(video_dim)
        self.action_dim = int(action_dim)
        self.feature_dim = int(feature_dim)
        self.max_period = float(max_period)
        self.video_projection = nn.Linear(
            self.feature_dim,
            self.video_dim,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.action_block_projection = nn.Linear(
            self.feature_dim,
            self.action_dim,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.action_control_projection = nn.Linear(
            self.feature_dim,
            self.action_dim,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Restore the exact FastWAM-compatible no-op initialization."""

        nn.init.zeros_(self.video_projection.weight)
        nn.init.zeros_(self.action_block_projection.weight)
        nn.init.zeros_(self.action_control_projection.weight)

    @staticmethod
    def _local_rope_ids(
        length: int,
        *,
        batch_size: int | None,
        device: torch.device | str | None,
    ) -> torch.Tensor:
        if length < 0:
            raise ValueError("local RoPE length must be non-negative")
        if batch_size is not None and batch_size <= 0:
            raise ValueError("batch_size must be positive when provided")
        ids = torch.arange(length, dtype=torch.long, device=device)
        if batch_size is None:
            return ids
        return ids.unsqueeze(0).expand(batch_size, -1)

    @classmethod
    def local_video_rope_ids(
        cls,
        num_frames: int,
        *,
        batch_size: int | None = None,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        """Return native per-call video RoPE ids ``0 .. num_frames-1``."""

        return cls._local_rope_ids(
            num_frames,
            batch_size=batch_size,
            device=device,
        )

    @classmethod
    def local_action_rope_ids(
        cls,
        num_actions: int,
        *,
        batch_size: int | None = None,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        """Return native per-block action RoPE ids ``0 .. num_actions-1``."""

        return cls._local_rope_ids(
            num_actions,
            batch_size=batch_size,
            device=device,
        )

    @staticmethod
    def _validate_tokens(tokens: torch.Tensor, expected_dim: int, name: str) -> None:
        if not isinstance(tokens, torch.Tensor):
            raise TypeError(f"{name} tokens must be a torch.Tensor")
        if tokens.ndim != 3:
            raise ValueError(f"{name} tokens must have shape [B,S,D]")
        if tokens.shape[-1] != expected_dim:
            raise ValueError(
                f"{name} token dimension must be {expected_dim}, "
                f"got {tokens.shape[-1]}"
            )
        if not tokens.dtype.is_floating_point:
            raise TypeError(f"{name} tokens must have a floating-point dtype")

    @staticmethod
    def _normalize_position_ids(
        position_ids: torch.Tensor,
        *,
        batch_size: int,
        sequence_length: int,
        device: torch.device,
        name: str,
    ) -> torch.Tensor:
        if not isinstance(position_ids, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if position_ids.dtype not in _INTEGER_DTYPES:
            raise TypeError(f"{name} must have an integer dtype")
        if position_ids.ndim == 1:
            if position_ids.shape[0] != sequence_length:
                raise ValueError(
                    f"{name} length must be {sequence_length}, "
                    f"got {position_ids.shape[0]}"
                )
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        elif position_ids.ndim == 2:
            if position_ids.shape != (batch_size, sequence_length):
                raise ValueError(
                    f"{name} shape must be [{batch_size},{sequence_length}], "
                    f"got {list(position_ids.shape)}"
                )
        else:
            raise ValueError(f"{name} must have shape [S] or [B,S]")
        if bool((position_ids < 0).any().item()):
            raise ValueError(f"{name} must be non-negative")
        return position_ids.to(device=device, dtype=torch.long)

    def _project(
        self,
        position_ids: torch.Tensor,
        projection: nn.Linear,
        *,
        output_dtype: torch.dtype,
        output_device: torch.device,
    ) -> torch.Tensor:
        if projection.weight.device != output_device:
            raise ValueError(
                "position module and token tensors must be on the same device; "
                "move the module with .to(tokens.device)"
            )
        features = sinusoidal_episode_features(
            position_ids,
            self.feature_dim,
            max_period=self.max_period,
            dtype=projection.weight.dtype,
        )
        return projection(features).to(dtype=output_dtype)

    def video_offsets(
        self,
        tokens: torch.Tensor,
        absolute_block_ids: torch.Tensor,
        *,
        tokens_per_frame: int,
    ) -> torch.Tensor:
        """Return per-token video offsets with shape ``[B,S,video_dim]``."""

        self._validate_tokens(tokens, self.video_dim, "video")
        if tokens_per_frame <= 0:
            raise ValueError("tokens_per_frame must be positive")
        batch_size, sequence_length, _ = tokens.shape
        if sequence_length % tokens_per_frame:
            raise ValueError("video token count must be divisible by tokens_per_frame")
        num_frames = sequence_length // tokens_per_frame
        block_ids = self._normalize_position_ids(
            absolute_block_ids,
            batch_size=batch_size,
            sequence_length=num_frames,
            device=tokens.device,
            name="absolute_block_ids",
        )
        frame_offsets = self._project(
            block_ids,
            self.video_projection,
            output_dtype=tokens.dtype,
            output_device=tokens.device,
        )
        return frame_offsets.repeat_interleave(tokens_per_frame, dim=1)

    def action_offsets(
        self,
        tokens: torch.Tensor,
        absolute_control_ids: torch.Tensor,
        absolute_block_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Return summed coarse-block and fine-control action offsets."""

        self._validate_tokens(tokens, self.action_dim, "action")
        batch_size, sequence_length, _ = tokens.shape
        control_ids = self._normalize_position_ids(
            absolute_control_ids,
            batch_size=batch_size,
            sequence_length=sequence_length,
            device=tokens.device,
            name="absolute_control_ids",
        )
        block_ids = self._normalize_position_ids(
            absolute_block_ids,
            batch_size=batch_size,
            sequence_length=sequence_length,
            device=tokens.device,
            name="absolute_block_ids",
        )
        control_offsets = self._project(
            control_ids,
            self.action_control_projection,
            output_dtype=tokens.dtype,
            output_device=tokens.device,
        )
        block_offsets = self._project(
            block_ids,
            self.action_block_projection,
            output_dtype=tokens.dtype,
            output_device=tokens.device,
        )
        return block_offsets + control_offsets

    def add_video(
        self,
        tokens: torch.Tensor,
        absolute_block_ids: torch.Tensor,
        *,
        tokens_per_frame: int,
    ) -> torch.Tensor:
        """Add episode block embeddings to pre-DiT video tokens."""

        return tokens + self.video_offsets(
            tokens,
            absolute_block_ids,
            tokens_per_frame=tokens_per_frame,
        )

    def add_action(
        self,
        tokens: torch.Tensor,
        absolute_control_ids: torch.Tensor,
        absolute_block_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Add episode block and control-step embeddings to action tokens."""

        return tokens + self.action_offsets(
            tokens,
            absolute_control_ids,
            absolute_block_ids,
        )

    def apply_video_pre_dit(
        self,
        pre_state: Mapping[str, Any],
        absolute_block_ids: torch.Tensor,
    ) -> dict[str, Any]:
        """Return a copied video ``pre_dit`` state with positioned tokens."""

        try:
            tokens = pre_state["tokens"]
            tokens_per_frame = int(pre_state["meta"]["tokens_per_frame"])
        except (KeyError, TypeError) as exc:
            raise ValueError(
                "video pre_state must contain tokens and meta.tokens_per_frame"
            ) from exc
        result = dict(pre_state)
        result["tokens"] = self.add_video(
            tokens,
            absolute_block_ids,
            tokens_per_frame=tokens_per_frame,
        )
        result["meta"] = dict(pre_state["meta"])
        result["meta"]["absolute_block_ids"] = absolute_block_ids
        return result

    def apply_action_pre_dit(
        self,
        pre_state: Mapping[str, Any],
        absolute_control_ids: torch.Tensor,
        absolute_block_ids: torch.Tensor,
    ) -> dict[str, Any]:
        """Return a copied action ``pre_dit`` state with positioned tokens."""

        try:
            tokens = pre_state["tokens"]
        except (KeyError, TypeError) as exc:
            raise ValueError("action pre_state must contain tokens") from exc
        result = dict(pre_state)
        result["tokens"] = self.add_action(
            tokens,
            absolute_control_ids,
            absolute_block_ids,
        )
        result["meta"] = dict(pre_state.get("meta", {}))
        result["meta"]["absolute_control_ids"] = absolute_control_ids
        result["meta"]["absolute_block_ids"] = absolute_block_ids
        return result
