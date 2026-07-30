"""Explicit, per-episode causal KV memory for LeapBot-VA.

The runtime deliberately keeps memory outside the model module.  This avoids
global mutable caches, makes concurrent evaluation environments safe, and
makes the observation/action commit protocol testable without loading a 6B
parameter checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Literal, Sequence

import torch


CausalMode = Literal["interleaved", "vision_causal", "action_aggregator"]
Modality = Literal["video", "action"]

VALID_CAUSAL_MODES: tuple[str, ...] = (
    "interleaved",
    "vision_causal",
    "action_aggregator",
)
VALID_EXIT_DEPTHS: tuple[int, ...] = (8, 16, 24, 30)


class MemoryPhase(str, Enum):
    EXPECT_OBSERVATION = "expect_observation"
    EXPECT_ACTION_COMMIT = "expect_action_commit"


@dataclass(frozen=True)
class LeapMemoryConfig:
    """Immutable settings that must remain fixed for one episode."""

    exit_depth: int = 30
    causal_mode: CausalMode = "interleaved"
    max_history_blocks: int = 70
    retained_history_blocks: int | None = None
    action_horizon: int = 32
    replan_steps: int = 10

    def __post_init__(self) -> None:
        if self.exit_depth not in VALID_EXIT_DEPTHS:
            raise ValueError(
                f"exit_depth must be one of {VALID_EXIT_DEPTHS}, got {self.exit_depth}"
            )
        if self.causal_mode not in VALID_CAUSAL_MODES:
            raise ValueError(
                f"causal_mode must be one of {VALID_CAUSAL_MODES}, got {self.causal_mode}"
            )
        if self.max_history_blocks <= 0:
            raise ValueError("max_history_blocks must be positive")
        if self.retained_history_blocks is not None:
            if isinstance(self.retained_history_blocks, bool) or not isinstance(
                self.retained_history_blocks, int
            ):
                raise ValueError("retained_history_blocks must be an integer or None")
            if self.retained_history_blocks < 0:
                raise ValueError("retained_history_blocks must be non-negative or None")
            if self.retained_history_blocks > self.max_history_blocks:
                raise ValueError(
                    "retained_history_blocks cannot exceed max_history_blocks"
                )
        if self.action_horizon <= 0:
            raise ValueError("action_horizon must be positive")
        if self.replan_steps <= 0 or self.replan_steps > self.action_horizon:
            raise ValueError("replan_steps must be in [1, action_horizon]")


@dataclass
class KVSegment:
    """K/V produced by one real observation or one executed action slice."""

    modality: Modality
    block_index: int
    positions: torch.Tensor
    keys: list[torch.Tensor]
    values: list[torch.Tensor]

    def __post_init__(self) -> None:
        if self.modality not in ("video", "action"):
            raise ValueError(f"unsupported modality: {self.modality}")
        if len(self.keys) != len(self.values):
            raise ValueError("keys and values must contain the same number of layers")
        if not self.keys:
            raise ValueError("a KV segment must contain at least one layer")
        seq_len = int(self.keys[0].shape[1])
        if self.positions.numel() != seq_len:
            raise ValueError(
                f"position count {self.positions.numel()} does not match KV length {seq_len}"
            )
        for layer, (key, value) in enumerate(zip(self.keys, self.values)):
            if key.shape != value.shape:
                raise ValueError(f"layer {layer}: key/value shape mismatch")
            if key.ndim != 3:
                raise ValueError(f"layer {layer}: KV tensors must be [B,S,D]")
            if key.shape[1] != seq_len:
                raise ValueError(f"layer {layer}: inconsistent sequence length")

    @property
    def num_layers(self) -> int:
        return len(self.keys)

    @property
    def num_tokens(self) -> int:
        return int(self.positions.numel())

    @property
    def nbytes(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in (*self.keys, *self.values)
        )

    def detached(self) -> "KVSegment":
        return KVSegment(
            modality=self.modality,
            block_index=self.block_index,
            positions=self.positions.detach(),
            keys=[tensor.detach() for tensor in self.keys],
            values=[tensor.detach() for tensor in self.values],
        )


@dataclass
class MemorySnapshot:
    # Keep references to the immutable/detached segments rather than only a
    # tail length.  A completed commit may evict segments from the front, and
    # rollback must be able to restore that transaction without copying the KV
    # tensors themselves.
    segments: tuple[KVSegment, ...]
    phase: MemoryPhase
    completed_blocks: int
    next_action_position: int
    prompt_fingerprint: str | None
    pending_context: torch.Tensor | None
    pending_context_mask: torch.Tensor | None


@dataclass
class LeapMemoryState:
    """Mutable episode state with transactional observation commits."""

    config: LeapMemoryConfig
    segments: list[KVSegment] = field(default_factory=list)
    phase: MemoryPhase = MemoryPhase.EXPECT_OBSERVATION
    completed_blocks: int = 0
    next_action_position: int = 0
    prompt_fingerprint: str | None = None
    pending_context: torch.Tensor | None = field(default=None, repr=False)
    pending_context_mask: torch.Tensor | None = field(default=None, repr=False)

    def snapshot(self) -> MemorySnapshot:
        return MemorySnapshot(
            segments=tuple(self.segments),
            phase=self.phase,
            completed_blocks=self.completed_blocks,
            next_action_position=self.next_action_position,
            prompt_fingerprint=self.prompt_fingerprint,
            pending_context=self.pending_context,
            pending_context_mask=self.pending_context_mask,
        )

    def rollback(self, snapshot: MemorySnapshot) -> None:
        self.segments[:] = snapshot.segments
        self.phase = snapshot.phase
        self.completed_blocks = snapshot.completed_blocks
        self.next_action_position = snapshot.next_action_position
        self.prompt_fingerprint = snapshot.prompt_fingerprint
        self.pending_context = snapshot.pending_context
        self.pending_context_mask = snapshot.pending_context_mask

    def bind_prompt(self, fingerprint: str) -> None:
        if self.prompt_fingerprint is None:
            self.prompt_fingerprint = fingerprint
        elif self.prompt_fingerprint != fingerprint:
            raise ValueError("prompt/context changed inside an active episode; reset memory first")

    def begin_observation(self) -> int:
        if self.phase is not MemoryPhase.EXPECT_OBSERVATION:
            raise RuntimeError("executed actions must be committed before the next observation")
        if (
            self.segments
            and self.segments[-1].modality == "action"
            and self.segments[-1].num_tokens != self.config.replan_steps
        ):
            raise RuntimeError(
                "a partial action commit is terminal for the current training contract; "
                "reset memory before starting another observation"
            )
        if self.next_action_position != self.completed_blocks * self.config.replan_steps:
            raise RuntimeError(
                "episode action positions diverged from fixed replanning boundaries; "
                "reset memory before continuing"
            )
        if self.completed_blocks >= self.config.max_history_blocks:
            raise RuntimeError(
                "episode KV capacity exceeded: "
                f"{self.completed_blocks}/{self.config.max_history_blocks} blocks"
            )
        return self.completed_blocks

    def append_observation(
        self,
        segment: KVSegment,
        *,
        context: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> None:
        if segment.modality != "video":
            raise ValueError("append_observation requires a video segment")
        expected_block = self.begin_observation()
        if segment.block_index != expected_block:
            raise ValueError(
                f"observation block index must be {expected_block}, got {segment.block_index}"
            )
        self._validate_depth(segment)
        self.segments.append(segment.detached())
        self.pending_context = context.detach()
        self.pending_context_mask = context_mask.detach()
        self.phase = MemoryPhase.EXPECT_ACTION_COMMIT

    def append_actions(self, segment: KVSegment) -> None:
        if self.phase is not MemoryPhase.EXPECT_ACTION_COMMIT:
            raise RuntimeError("an observation must be appended before executed actions")
        if segment.modality != "action":
            raise ValueError("append_actions requires an action segment")
        if segment.block_index != self.completed_blocks:
            raise ValueError(
                f"action block index must be {self.completed_blocks}, got {segment.block_index}"
            )
        if segment.num_tokens <= 0 or segment.num_tokens > self.config.replan_steps:
            raise ValueError(
                f"executed action count must be in [1,{self.config.replan_steps}], "
                f"got {segment.num_tokens}"
            )
        self._validate_depth(segment)
        expected_positions = torch.arange(
            self.next_action_position,
            self.next_action_position + segment.num_tokens,
            device=segment.positions.device,
            dtype=segment.positions.dtype,
        )
        if not torch.equal(segment.positions, expected_positions):
            raise ValueError("executed action positions are not contiguous")
        self.segments.append(segment.detached())
        self.next_action_position += segment.num_tokens
        self.completed_blocks += 1
        self.pending_context = None
        self.pending_context_mask = None
        self.phase = MemoryPhase.EXPECT_OBSERVATION
        if segment.num_tokens == self.config.replan_steps:
            self._evict_completed_history()

    def _evict_completed_history(self) -> None:
        """Apply the inference-only retention window at a safe block boundary.

        ``completed_blocks`` and ``next_action_position`` are absolute episode
        clocks and are deliberately untouched.  This method is called only
        after a full action block has been appended, so every segment older
        than the cutoff belongs to a completed video/action block.  In
        particular, an observation awaiting its action commit is never
        eligible for eviction.
        """

        retained = self.config.retained_history_blocks
        if retained is None:
            return
        first_retained_block = self.completed_blocks - retained
        self.segments[:] = [
            segment
            for segment in self.segments
            if segment.block_index >= first_retained_block
        ]

    def _validate_depth(self, segment: KVSegment) -> None:
        if segment.num_layers != self.config.exit_depth:
            raise ValueError(
                f"segment has {segment.num_layers} layers, expected {self.config.exit_depth}"
            )

    def selected_segments_for_video(self) -> Sequence[KVSegment]:
        if self.config.causal_mode == "interleaved":
            return self.segments
        if self.config.causal_mode == "vision_causal":
            return [segment for segment in self.segments if segment.modality == "video"]
        return []

    def selected_segments_for_action(self) -> Sequence[KVSegment]:
        return self.segments

    def materialize(
        self,
        segments: Iterable[KVSegment],
    ) -> list[dict[str, torch.Tensor]] | None:
        selected = list(segments)
        if not selected:
            return None
        result: list[dict[str, torch.Tensor]] = []
        for layer in range(self.config.exit_depth):
            result.append(
                {
                    "k": torch.cat([segment.keys[layer] for segment in selected], dim=1),
                    "v": torch.cat([segment.values[layer] for segment in selected], dim=1),
                }
            )
        return result

    @property
    def cache_nbytes(self) -> int:
        return sum(segment.nbytes for segment in self.segments)

    @property
    def token_counts(self) -> dict[str, int]:
        return {
            modality: sum(
                segment.num_tokens for segment in self.segments if segment.modality == modality
            )
            for modality in ("video", "action")
        }

    @property
    def retained_completed_blocks(self) -> int:
        """Number of completed action blocks still represented in the cache.

        ``completed_blocks`` is the absolute episode clock and therefore keeps
        increasing under a rolling retention ablation.  Action segments are
        the unambiguous completed-block marker: the current observation may
        already be present while it is still awaiting its action commit.
        """

        return len(
            {
                segment.block_index
                for segment in self.segments
                if segment.modality == "action"
            }
        )

    def reset(self) -> None:
        self.segments.clear()
        self.phase = MemoryPhase.EXPECT_OBSERVATION
        self.completed_blocks = 0
        self.next_action_position = 0
        self.prompt_fingerprint = None
        self.pending_context = None
        self.pending_context_mask = None


def build_block_causal_mask(
    query_modalities: Sequence[Modality],
    query_blocks: Sequence[int],
    key_modalities: Sequence[Modality],
    key_blocks: Sequence[int],
    mode: CausalMode,
    *,
    query_is_future_video: Sequence[bool] | None = None,
    key_is_future_video: Sequence[bool] | None = None,
) -> torch.Tensor:
    """Build the coarse block mask without confusing real and future video.

    Runtime incremental attention filters immutable history before the call, so
    it can use an all-true rectangular mask.  The production packed training
    path additionally tracks individual video frames; the optional future-video
    flags make this public reference helper preserve the same hard isolation.
    """

    if mode not in VALID_CAUSAL_MODES:
        raise ValueError(f"unsupported causal mode: {mode}")
    if len(query_modalities) != len(query_blocks):
        raise ValueError("query modalities/blocks length mismatch")
    if len(key_modalities) != len(key_blocks):
        raise ValueError("key modalities/blocks length mismatch")
    query_future = (
        [False] * len(query_blocks)
        if query_is_future_video is None
        else list(query_is_future_video)
    )
    key_future = (
        [False] * len(key_blocks)
        if key_is_future_video is None
        else list(key_is_future_video)
    )
    if len(query_future) != len(query_blocks):
        raise ValueError("query future-video flags length mismatch")
    if len(key_future) != len(key_blocks):
        raise ValueError("key future-video flags length mismatch")
    if any(
        flag and modality != "video"
        for flag, modality in zip(query_future, query_modalities)
    ):
        raise ValueError("only video queries can be marked as future video")
    if any(
        flag and modality != "video"
        for flag, modality in zip(key_future, key_modalities)
    ):
        raise ValueError("only video keys can be marked as future video")

    mask = torch.zeros((len(query_blocks), len(key_blocks)), dtype=torch.bool)
    for q_idx, (q_modality, q_block, q_future) in enumerate(
        zip(query_modalities, query_blocks, query_future)
    ):
        for k_idx, (k_modality, k_block, k_future) in enumerate(
            zip(key_modalities, key_blocks, key_future)
        ):
            if k_block > q_block:
                continue
            # Future-video supervision is transient.  It is visible only to
            # future-video queries in the same block and can never become
            # historical information for a real observation or ActionDiT.
            if k_future and not (
                q_modality == "video" and q_future and k_block == q_block
            ):
                continue
            if q_modality == "action":
                mask[q_idx, k_idx] = True
            elif k_block == q_block:
                # The real observation is encoded before this block's action
                # is predicted/executed. Future-video queries may jointly read
                # all video supervision tokens, while a real query cannot.
                mask[q_idx, k_idx] = k_modality == "video"
            elif mode == "interleaved":
                mask[q_idx, k_idx] = True
            elif mode == "vision_causal":
                mask[q_idx, k_idx] = k_modality == "video"
            # action_aggregator has no cross-block visual attention.
    return mask
