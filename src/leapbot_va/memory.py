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

from leapbot_va.episode_memory import EpisodeMemoryConfig


CausalMode = Literal["interleaved", "vision_causal", "action_aggregator"]
Modality = Literal["video", "action"]
HistoryStorageMode = Literal["incremental_kv", "strict_replay", "packed_replay"]

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
    history_storage_mode: HistoryStorageMode = "incremental_kv"
    history_window_blocks: int = 8
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
        if self.history_storage_mode not in (
            "incremental_kv",
            "strict_replay",
            "packed_replay",
        ):
            raise ValueError(
                "history_storage_mode must be incremental_kv, strict_replay, or packed_replay"
            )
        if (
            isinstance(self.history_window_blocks, bool)
            or not isinstance(self.history_window_blocks, int)
            or self.history_window_blocks <= 0
            or (
                self.history_storage_mode in ("strict_replay", "packed_replay")
                and self.history_window_blocks > self.max_history_blocks
            )
        ):
            raise ValueError(
                "history_window_blocks must be a positive integer no greater "
                "than max_history_blocks"
            )
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


@dataclass(frozen=True)
class ReplayBlock:
    """Real inputs required to reconstruct one completed causal block."""

    block_index: int
    observation_latents: torch.Tensor
    context: torch.Tensor
    context_mask: torch.Tensor
    executed_actions: torch.Tensor

    def __post_init__(self) -> None:
        if self.block_index < 0:
            raise ValueError("replay block_index must be non-negative")
        if (
            self.observation_latents.ndim != 5
            or self.observation_latents.shape[0] != 1
            or self.observation_latents.shape[2] != 1
        ):
            raise ValueError(
                "replay observation_latents must be [1,C,1,H,W]"
            )
        if self.context.ndim != 3 or self.context.shape[0] != 1:
            raise ValueError("replay context must be [1,L,D]")
        if self.context_mask.shape != self.context.shape[:2]:
            raise ValueError("replay context_mask must match context [1,L]")
        if self.context_mask.dtype != torch.bool:
            raise ValueError("replay context_mask must be boolean")
        if self.executed_actions.ndim != 3 or self.executed_actions.shape[0] != 1:
            raise ValueError("replay executed_actions must be [1,T,D]")
        if self.executed_actions.shape[1] <= 0:
            raise ValueError("replay executed_actions must be non-empty")

    @property
    def nbytes(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in (
                self.observation_latents,
                self.context,
                self.context_mask,
                self.executed_actions,
            )
        )

    def detached(self) -> "ReplayBlock":
        return ReplayBlock(
            block_index=self.block_index,
            observation_latents=self.observation_latents.detach(),
            context=self.context.detach(),
            context_mask=self.context_mask.detach(),
            executed_actions=self.executed_actions.detach(),
        )


@dataclass(frozen=True)
class ClosedReplayBlock:
    """One real observation/executed-action/next-observation transition."""

    start: ReplayBlock
    next_observation_latents: torch.Tensor
    next_context: torch.Tensor
    next_context_mask: torch.Tensor

    def __post_init__(self) -> None:
        if (
            self.next_observation_latents.ndim != 5
            or self.next_observation_latents.shape[0] != 1
            or self.next_observation_latents.shape[2] != 1
        ):
            raise ValueError("next observation latents must be [1,C,1,H,W]")
        if self.next_context.ndim != 3 or self.next_context.shape[0] != 1:
            raise ValueError("next context must be [1,L,D]")
        if self.next_context_mask.shape != self.next_context.shape[:2]:
            raise ValueError("next context mask must match next context")
        if self.next_context_mask.dtype != torch.bool:
            raise ValueError("next context mask must be boolean")

    @property
    def block_index(self) -> int:
        return self.start.block_index

    @property
    def nbytes(self) -> int:
        return self.start.nbytes + sum(
            tensor.numel() * tensor.element_size()
            for tensor in (
                self.next_observation_latents,
                self.next_context,
                self.next_context_mask,
            )
        )

    def detached(self) -> "ClosedReplayBlock":
        return ClosedReplayBlock(
            start=self.start.detached(),
            next_observation_latents=self.next_observation_latents.detach(),
            next_context=self.next_context.detach(),
            next_context_mask=self.next_context_mask.detach(),
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
    replay_blocks: tuple[ReplayBlock, ...]
    episode_anchor: ReplayBlock | None
    pending_observation_latents: torch.Tensor | None
    pending_replay_context: torch.Tensor | None
    pending_replay_context_mask: torch.Tensor | None
    episode_state: torch.Tensor | None
    pch_closed_blocks: tuple[ClosedReplayBlock, ...]
    handoff_blocks: tuple[ClosedReplayBlock, ...]


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
    replay_blocks: list[ReplayBlock] = field(default_factory=list, repr=False)
    episode_anchor: ReplayBlock | None = field(default=None, repr=False)
    pending_observation_latents: torch.Tensor | None = field(
        default=None, repr=False
    )
    pending_replay_context: torch.Tensor | None = field(default=None, repr=False)
    pending_replay_context_mask: torch.Tensor | None = field(
        default=None, repr=False
    )
    episode_memory_config: EpisodeMemoryConfig = field(
        default_factory=EpisodeMemoryConfig
    )
    episode_state: torch.Tensor | None = field(default=None, repr=False)
    initial_episode_state: torch.Tensor | None = field(default=None, repr=False)
    pch_closed_blocks: list[ClosedReplayBlock] = field(default_factory=list, repr=False)
    handoff_blocks: list[ClosedReplayBlock] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        episode = self.episode_memory_config
        if episode.enabled:
            if self.config.history_storage_mode != "packed_replay":
                raise ValueError("episode memory requires packed_replay history storage")
            if self.config.history_window_blocks != episode.window_blocks:
                raise ValueError("PCH window and episode-memory window must match")
            if self.episode_state is None or self.initial_episode_state is None:
                raise ValueError("enabled episode memory requires initial and current H")
            expected = (1, episode.num_slots, episode.state_dim)
            if tuple(self.episode_state.shape) != expected:
                raise ValueError(f"episode state must have shape {expected}")
            if tuple(self.initial_episode_state.shape) != expected:
                raise ValueError(f"initial episode state must have shape {expected}")

    def snapshot(self) -> MemorySnapshot:
        return MemorySnapshot(
            segments=tuple(self.segments),
            phase=self.phase,
            completed_blocks=self.completed_blocks,
            next_action_position=self.next_action_position,
            prompt_fingerprint=self.prompt_fingerprint,
            pending_context=self.pending_context,
            pending_context_mask=self.pending_context_mask,
            replay_blocks=tuple(self.replay_blocks),
            episode_anchor=self.episode_anchor,
            pending_observation_latents=self.pending_observation_latents,
            pending_replay_context=self.pending_replay_context,
            pending_replay_context_mask=self.pending_replay_context_mask,
            episode_state=self.episode_state,
            pch_closed_blocks=tuple(self.pch_closed_blocks),
            handoff_blocks=tuple(self.handoff_blocks),
        )

    def rollback(self, snapshot: MemorySnapshot) -> None:
        self.segments[:] = snapshot.segments
        self.phase = snapshot.phase
        self.completed_blocks = snapshot.completed_blocks
        self.next_action_position = snapshot.next_action_position
        self.prompt_fingerprint = snapshot.prompt_fingerprint
        self.pending_context = snapshot.pending_context
        self.pending_context_mask = snapshot.pending_context_mask
        self.replay_blocks[:] = snapshot.replay_blocks
        self.episode_anchor = snapshot.episode_anchor
        self.pending_observation_latents = snapshot.pending_observation_latents
        self.pending_replay_context = snapshot.pending_replay_context
        self.pending_replay_context_mask = snapshot.pending_replay_context_mask
        self.episode_state = snapshot.episode_state
        self.pch_closed_blocks[:] = snapshot.pch_closed_blocks
        self.handoff_blocks[:] = snapshot.handoff_blocks

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
        if (
            not self.episode_memory_config.enabled
            and self.completed_blocks >= self.config.max_history_blocks
        ):
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

    def stage_replay_observation(
        self,
        *,
        observation_latents: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> None:
        if self.config.history_storage_mode not in ("strict_replay", "packed_replay"):
            return
        if self.phase is not MemoryPhase.EXPECT_ACTION_COMMIT:
            raise RuntimeError(
                "replay observation can only be staged after observation K/V"
            )
        if self.pending_observation_latents is not None:
            raise RuntimeError("a replay observation is already pending")
        if (
            observation_latents.ndim != 5
            or observation_latents.shape[0] != 1
            or observation_latents.shape[2] != 1
        ):
            raise ValueError("observation_latents must be [1,C,1,H,W]")
        self.pending_observation_latents = observation_latents.detach()
        self.pending_replay_context = context.detach()
        self.pending_replay_context_mask = context_mask.detach()

    def replay_prefix(self) -> tuple[ReplayBlock, ...]:
        """Return one optional V0 anchor plus the recent completed window."""

        if self.config.history_storage_mode not in ("strict_replay", "packed_replay"):
            return ()
        recent = tuple(self.replay_blocks[-self.config.history_window_blocks :])
        if self.episode_memory_config.enabled:
            return (
                *(closed.start for closed in self.handoff_blocks),
                *recent,
            )
        if self.episode_anchor is not None and recent and recent[0].block_index > 0:
            return (self.episode_anchor, *recent)
        return recent

    def strict_replay_prefix(self) -> tuple[ReplayBlock, ...]:
        if self.config.history_storage_mode != "strict_replay":
            return ()
        return self.replay_prefix()

    def begin_strict_replay_rebuild(self) -> tuple[ReplayBlock, ...]:
        if self.config.history_storage_mode != "strict_replay":
            return ()
        if self.phase is not MemoryPhase.EXPECT_OBSERVATION:
            raise RuntimeError("strict replay rebuild requires a committed block")
        self.segments.clear()
        return self.replay_prefix()

    def begin_packed_replay_rebuild(self) -> tuple[ReplayBlock, ...]:
        if self.config.history_storage_mode != "packed_replay":
            return ()
        if self.phase is not MemoryPhase.EXPECT_OBSERVATION:
            raise RuntimeError("packed replay rebuild requires a committed block")
        self.segments.clear()
        return self.replay_prefix()

    def commit_replay_block(self, executed_actions: torch.Tensor) -> None:
        if self.config.history_storage_mode not in ("strict_replay", "packed_replay"):
            return
        if self.phase is not MemoryPhase.EXPECT_OBSERVATION:
            raise RuntimeError("replay block can only finalize after action commit")
        if (
            self.pending_observation_latents is None
            or self.pending_replay_context is None
            or self.pending_replay_context_mask is None
        ):
            raise RuntimeError("no staged real observation is available for replay")
        block_index = self.completed_blocks - 1
        block = ReplayBlock(
            block_index=block_index,
            observation_latents=self.pending_observation_latents,
            context=self.pending_replay_context,
            context_mask=self.pending_replay_context_mask,
            executed_actions=executed_actions.detach(),
        ).detached()
        if (
            not self.episode_memory_config.enabled
            and block_index == 0
            and self.episode_anchor is None
        ):
            self.episode_anchor = block
        self.replay_blocks.append(block)
        if not self.episode_memory_config.enabled:
            del self.replay_blocks[: -self.config.history_window_blocks]
        self.pending_observation_latents = None
        self.pending_replay_context = None
        self.pending_replay_context_mask = None
    def close_previous_transition(
        self,
        *,
        next_observation_latents: torch.Tensor,
        next_context: torch.Tensor,
        next_context_mask: torch.Tensor,
    ) -> tuple[ClosedReplayBlock, ...] | None:
        """Close the previous real transition and move only exited PCH into Q."""

        if not self.episode_memory_config.enabled or self.completed_blocks == 0:
            return None
        if self.phase is not MemoryPhase.EXPECT_OBSERVATION:
            raise RuntimeError("transition closure requires a committed action block")
        previous_index = self.completed_blocks - 1
        if not self.replay_blocks or self.replay_blocks[-1].block_index != previous_index:
            raise RuntimeError("the previous executed block is unavailable for closure")
        if self.pch_closed_blocks and self.pch_closed_blocks[-1].block_index >= previous_index:
            raise RuntimeError("the previous transition was already closed")
        closed = ClosedReplayBlock(
            start=self.replay_blocks[-1],
            next_observation_latents=next_observation_latents,
            next_context=next_context,
            next_context_mask=next_context_mask,
        ).detached()
        self.pch_closed_blocks.append(closed)

        window = self.episode_memory_config.window_blocks
        if len(self.replay_blocks) > window:
            exiting = self.replay_blocks.pop(0)
            if (
                not self.pch_closed_blocks
                or self.pch_closed_blocks[0].block_index != exiting.block_index
            ):
                raise RuntimeError("PCH closure order diverged from replay order")
            self.handoff_blocks.append(self.pch_closed_blocks.pop(0))
        if len(self.replay_blocks) > window:
            raise RuntimeError("PCH exceeded its fixed window")
        chunk = self.episode_memory_config.chunk_blocks
        if len(self.handoff_blocks) > chunk:
            raise RuntimeError("handoff buffer exceeded one atomic chunk")
        return tuple(self.handoff_blocks) if len(self.handoff_blocks) == chunk else None

    def commit_handoff(self, new_episode_state: torch.Tensor) -> None:
        if not self.episode_memory_config.enabled:
            raise RuntimeError("commit_handoff requires enabled episode memory")
        if len(self.handoff_blocks) != self.episode_memory_config.chunk_blocks:
            raise RuntimeError("handoff can commit only one complete chunk")
        if self.episode_state is None or new_episode_state.shape != self.episode_state.shape:
            raise ValueError("new episode state has incompatible shape")
        first = self.handoff_blocks[0].block_index
        expected = list(range(first, first + self.episode_memory_config.chunk_blocks))
        if [block.block_index for block in self.handoff_blocks] != expected:
            raise RuntimeError("handoff chunk is not contiguous")
        self.episode_state = new_episode_state.detach()
        self.handoff_blocks.clear()

    @property
    def episode_partition(self) -> dict[str, tuple[int, int]]:
        """Return half-open H/Q/PCH block ranges at the current decision."""

        t = self.completed_blocks
        w = self.episode_memory_config.window_blocks
        c = self.episode_memory_config.chunk_blocks
        e = max(0, t - w)
        q = c * (e // c)
        return {"H": (0, q), "Q": (q, e), "PCH": (e, t)}

    def commit_packed_replay_actions(self, executed_actions: torch.Tensor) -> None:
        """Commit real actions without an ActionDiT prefill in packed replay mode."""

        if self.config.history_storage_mode != "packed_replay":
            raise RuntimeError("commit_packed_replay_actions requires packed_replay")
        if self.phase is not MemoryPhase.EXPECT_ACTION_COMMIT:
            raise RuntimeError("an observation must be appended before executed actions")
        if executed_actions.ndim != 3 or int(executed_actions.shape[0]) != 1:
            raise ValueError("executed_actions must be [1,T,D]")
        count = int(executed_actions.shape[1])
        if count <= 0 or count > self.config.replan_steps:
            raise ValueError(
                f"executed action count must be in [1,{self.config.replan_steps}], got {count}"
            )
        self.next_action_position += count
        self.completed_blocks += 1
        self.pending_context = None
        self.pending_context_mask = None
        self.phase = MemoryPhase.EXPECT_OBSERVATION
        if count == self.config.replan_steps:
            self._evict_completed_history()
        self.commit_replay_block(executed_actions)
        # PCH K/V is an atomic rebuild product. Never mix it with the next block.
        self.segments.clear()

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

        retained = (
            self.config.history_window_blocks
            if self.config.history_storage_mode in ("strict_replay", "packed_replay")
            else self.config.retained_history_blocks
        )
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

    def selected_segments_for_future_video(self) -> Sequence[KVSegment]:
        """Return the causal real prefix for a same-block imagined video.

        This differs from ``selected_segments_for_video`` only for the
        action-aggregator ablation: ordinary observations are independently
        encoded, while the future-video query must still read its current real
        observation.
        """

        if not self.segments or self.segments[-1].modality != "video":
            raise RuntimeError(
                "future-video conditioning requires a pending real observation"
            )
        if self.config.causal_mode == "interleaved":
            return self.segments
        if self.config.causal_mode == "vision_causal":
            return [segment for segment in self.segments if segment.modality == "video"]
        return [self.segments[-1]]

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
    def replay_nbytes(self) -> int:
        unique = {id(block): block for block in self.replay_blocks}
        if self.episode_anchor is not None:
            unique[id(self.episode_anchor)] = self.episode_anchor
        replay = sum(block.nbytes for block in unique.values())
        handoff = sum(block.nbytes for block in self.handoff_blocks)
        state = (
            0
            if self.episode_state is None
            else self.episode_state.numel() * self.episode_state.element_size()
        )
        return replay + handoff + state

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
        self.replay_blocks.clear()
        self.episode_anchor = None
        self.pending_observation_latents = None
        self.pending_replay_context = None
        self.pending_replay_context_mask = None
        self.pch_closed_blocks.clear()
        self.handoff_blocks.clear()
        self.episode_state = (
            None
            if self.initial_episode_state is None
            else self.initial_episode_state.detach().clone()
        )


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
            # Same-block future video is a transient world-model condition.
            # It is visible to its own video queries and to ActionDiT, but
            # never to real observations and never after the replanning call.
            if k_future and not (
                k_block == q_block
                and (
                    (q_modality == "video" and q_future)
                    or q_modality == "action"
                )
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
