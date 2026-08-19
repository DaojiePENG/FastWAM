"""Fixed-capacity, scan-compatible episode world state.

The update operator is generated from one closed interaction chunk without
reading the incoming state.  It is therefore safe to compose with an
associative prefix scan while remaining exactly equivalent to a grouped
prediction--observation correction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


@dataclass(frozen=True)
class EpisodeMemoryConfig:
    enabled: bool = False
    window_blocks: int = 8
    chunk_blocks: int = 4
    num_slots: int = 32
    state_dim: int = 1024
    group_dim: int = 16
    updater_dim: int = 256
    updater_heads: int = 8
    reader_rank: int = 64
    first_frame_memory: bool = True

    def __post_init__(self) -> None:
        positive = {
            "window_blocks": self.window_blocks,
            "chunk_blocks": self.chunk_blocks,
            "num_slots": self.num_slots,
            "state_dim": self.state_dim,
            "group_dim": self.group_dim,
            "updater_dim": self.updater_dim,
            "updater_heads": self.updater_heads,
            "reader_rank": self.reader_rank,
        }
        if any(isinstance(value, bool) or int(value) <= 0 for value in positive.values()):
            raise ValueError("episode-memory dimensions must be positive integers")
        if self.chunk_blocks >= self.window_blocks:
            raise ValueError("episode-memory chunk_blocks must be smaller than window_blocks")
        if self.state_dim % self.group_dim:
            raise ValueError("state_dim must be divisible by group_dim")
        if self.updater_dim % self.updater_heads:
            raise ValueError("updater_dim must be divisible by updater_heads")

    @property
    def num_groups(self) -> int:
        return self.state_dim // self.group_dim


@dataclass
class EpisodeAffineUpdate:
    """Grouped affine map ``h -> G h + B``.

    ``matrix`` is ``[..., slots, groups, d, d]`` and ``bias`` is
    ``[..., slots, groups, d]``.  Keeping the 16-D groups explicit makes
    composition closed without materialising a state_dim-square matrix.
    """

    matrix: torch.Tensor
    bias: torch.Tensor

    def __post_init__(self) -> None:
        if self.matrix.ndim < 5 or self.bias.ndim < 4:
            raise ValueError("affine update tensors have too few dimensions")
        if self.matrix.shape[:-1] != self.bias.shape:
            raise ValueError("affine matrix/bias geometry mismatch")
        if self.matrix.shape[-2] != self.matrix.shape[-1]:
            raise ValueError("affine group matrices must be square")

    def apply(self, state: torch.Tensor) -> torch.Tensor:
        grouped = _group_state(state, self.matrix.shape[-3], self.matrix.shape[-1])
        result = torch.matmul(self.matrix, grouped.unsqueeze(-1)).squeeze(-1) + self.bias
        return result.flatten(-2)

    def compose_after(self, earlier: "EpisodeAffineUpdate") -> "EpisodeAffineUpdate":
        """Return ``self(earlier(h))``; temporal order is not commuted."""

        if self.matrix.shape != earlier.matrix.shape or self.bias.shape != earlier.bias.shape:
            raise ValueError("cannot compose affine updates with different shapes")
        matrix = torch.matmul(self.matrix, earlier.matrix)
        bias = (
            torch.matmul(self.matrix, earlier.bias.unsqueeze(-1)).squeeze(-1)
            + self.bias
        )
        return EpisodeAffineUpdate(matrix=matrix, bias=bias)

    def to(self, *args, **kwargs) -> "EpisodeAffineUpdate":
        return EpisodeAffineUpdate(
            matrix=self.matrix.to(*args, **kwargs),
            bias=self.bias.to(*args, **kwargs),
        )


def _group_state(state: torch.Tensor, groups: int, group_dim: int) -> torch.Tensor:
    if state.ndim < 3:
        raise ValueError("episode state must be [...,slots,state_dim]")
    if state.shape[-1] != groups * group_dim:
        raise ValueError("episode state dimension does not match affine groups")
    return state.reshape(*state.shape[:-1], groups, group_dim)


def identity_update(
    batch_shape: Sequence[int],
    *,
    num_slots: int,
    num_groups: int,
    group_dim: int,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
) -> EpisodeAffineUpdate:
    eye = torch.eye(group_dim, device=device, dtype=dtype)
    matrix = eye.expand(*batch_shape, num_slots, num_groups, group_dim, group_dim).clone()
    bias = torch.zeros(
        *batch_shape, num_slots, num_groups, group_dim, device=device, dtype=dtype
    )
    return EpisodeAffineUpdate(matrix=matrix, bias=bias)


def compose_updates(
    later: EpisodeAffineUpdate, earlier: EpisodeAffineUpdate
) -> EpisodeAffineUpdate:
    return later.compose_after(earlier)


def associative_affine_scan(
    updates: EpisodeAffineUpdate,
    *,
    checkpoint_stages: bool = False,
) -> EpisodeAffineUpdate:
    """Inclusive Hillis--Steele scan over dimension 1 in logarithmic depth.

    Input geometry is ``[batch,chunks,...]``.  The implementation is fully
    differentiable, supports non-power-of-two chunk counts, and changes only
    the parenthesisation of the chronological affine composition.
    """

    if updates.matrix.ndim != 6 or updates.bias.ndim != 5:
        raise ValueError("scan expects matrix [B,J,N,G,D,D] and bias [B,J,N,G,D]")
    chunks = int(updates.matrix.shape[1])
    if chunks <= 0:
        raise ValueError("scan requires at least one chunk")
    matrix = updates.matrix.float()
    bias = updates.bias.float()
    offset = 1
    while offset < chunks:
        later_m = matrix[:, offset:]
        earlier_m = matrix[:, :-offset]
        later_b = bias[:, offset:]
        earlier_b = bias[:, :-offset]
        def compose_stage(lm, em, lb, eb):
            composed_matrix = torch.matmul(lm, em)
            composed_bias = (
                torch.matmul(lm, eb.unsqueeze(-1)).squeeze(-1) + lb
            )
            return composed_matrix, composed_bias

        if checkpoint_stages and torch.is_grad_enabled():
            composed_m, composed_b = checkpoint(
                compose_stage,
                later_m,
                earlier_m,
                later_b,
                earlier_b,
                use_reentrant=False,
            )
        else:
            composed_m, composed_b = compose_stage(
                later_m, earlier_m, later_b, earlier_b
            )
        matrix = torch.cat([matrix[:, :offset], composed_m], dim=1)
        bias = torch.cat([bias[:, :offset], composed_b], dim=1)
        offset *= 2
    return EpisodeAffineUpdate(matrix=matrix, bias=bias)


def apply_prefix_updates(
    initial_state: torch.Tensor, prefixes: EpisodeAffineUpdate
) -> torch.Tensor:
    if prefixes.matrix.ndim != 6:
        raise ValueError("prefix updates must include batch and chunk dimensions")
    state = initial_state[:, None].expand(-1, prefixes.matrix.shape[1], -1, -1)
    return prefixes.apply(state)


@dataclass
class PredictionCorrectionDiagnostics:
    predicted_observation: torch.Tensor
    observed_evidence: torch.Tensor
    corrected_observation: torch.Tensor
    gain: torch.Tensor
    prediction_diagonal: torch.Tensor
    prediction_bias: torch.Tensor
    observation_direction: torch.Tensor
    transition_matrix: torch.Tensor
    transition_bias: torch.Tensor


class EpisodeChunkUpdater(nn.Module):
    """Generate and compose one state-independent operator per transition."""

    def __init__(
        self,
        config: EpisodeMemoryConfig,
        *,
        video_dim: int,
        action_dim: int,
    ) -> None:
        super().__init__()
        self.config = config
        hidden = config.updater_dim
        self.video_adapter = nn.Sequential(
            nn.LayerNorm(video_dim), nn.Linear(video_dim, hidden), nn.SiLU()
        )
        self.action_adapter = nn.Sequential(
            nn.LayerNorm(action_dim), nn.Linear(action_dim, hidden), nn.SiLU()
        )
        # Start observation, executed action and real successor have distinct
        # causal roles. No within-chunk position embedding is used: the same
        # closed transition must induce the same operator regardless of where
        # an arbitrary C-block handoff boundary places it.
        self.role_embedding = nn.Parameter(torch.empty(3, hidden))
        self.slot_queries = nn.Parameter(torch.empty(config.num_slots, hidden))
        self.predict_attention = nn.MultiheadAttention(
            hidden, config.updater_heads, batch_first=True
        )
        self.observe_attention = nn.MultiheadAttention(
            hidden, config.updater_heads, batch_first=True
        )
        self.predict_norm = nn.LayerNorm(hidden)
        self.observe_norm = nn.LayerNorm(hidden)
        group_output = config.num_groups * config.group_dim
        self.a_head = nn.Linear(hidden, group_output)
        self.b_head = nn.Linear(hidden, group_output)
        self.c_head = nn.Linear(hidden, group_output)
        self.y_head = nn.Linear(hidden, config.num_groups)
        self.k_head = nn.Linear(hidden, config.num_groups)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.role_embedding, std=0.02)
        nn.init.normal_(self.slot_queries, std=0.02)
        nn.init.zeros_(self.a_head.weight)
        nn.init.zeros_(self.a_head.bias)
        nn.init.zeros_(self.b_head.weight)
        nn.init.zeros_(self.b_head.bias)
        nn.init.normal_(self.c_head.weight, std=0.01)
        nn.init.zeros_(self.c_head.bias)
        nn.init.normal_(self.y_head.weight, std=0.01)
        nn.init.zeros_(self.y_head.bias)
        nn.init.zeros_(self.k_head.weight)
        nn.init.constant_(self.k_head.bias, -2.0)

    @staticmethod
    def _pool_tokens(tokens: torch.Tensor, expected_steps: int, name: str) -> torch.Tensor:
        if tokens.ndim == 3:
            tokens = tokens.unsqueeze(2)
        if tokens.ndim != 4 or int(tokens.shape[1]) != expected_steps:
            raise ValueError(
                f"{name} must be [B,{expected_steps},S,D] or [B,{expected_steps},D]"
            )
        return tokens.mean(dim=2)

    def forward(
        self,
        video_tokens: torch.Tensor,
        action_tokens: torch.Tensor,
    ) -> tuple[EpisodeAffineUpdate, PredictionCorrectionDiagnostics]:
        c = self.config.chunk_blocks
        video = self.video_adapter(
            self._pool_tokens(video_tokens, c + 1, "video_tokens")
        )
        action = self.action_adapter(
            self._pool_tokens(action_tokens, c, "action_tokens")
        )
        if video.shape[0] != action.shape[0]:
            raise ValueError("video/action chunk batches must match")

        # Every transition is encoded independently. Prediction sees only
        # (o_i, a_i_exec), while correction sees only the real successor
        # o_{i+1}. Flattening B and C makes all transition operators parallel.
        batch = int(video.shape[0])
        predict_seq = torch.stack(
            (
                video[:, :-1] + self.role_embedding[0],
                action + self.role_embedding[1],
            ),
            dim=2,
        ).flatten(0, 1)
        observe_seq = (
            video[:, 1:] + self.role_embedding[2]
        ).flatten(0, 1).unsqueeze(1)
        queries = self.slot_queries[None].expand(batch * c, -1, -1)
        pred_slots = self.predict_norm(
            queries
            + self.predict_attention(
                queries, predict_seq, predict_seq, need_weights=False
            )[0]
        )
        obs_slots = self.observe_norm(
            queries
            + self.observe_attention(
                queries, observe_seq, observe_seq, need_weights=False
            )[0]
        )

        n, g, d = self.config.num_slots, self.config.num_groups, self.config.group_dim
        a = 1.0 + 0.01 * torch.tanh(self.a_head(pred_slots)).reshape(
            batch, c, n, g, d
        )
        b = 0.01 * torch.tanh(self.b_head(pred_slots)).reshape(
            batch, c, n, g, d
        )
        direction = F.normalize(
            self.c_head(pred_slots).reshape(batch, c, n, g, d),
            dim=-1,
            eps=1e-6,
        )
        observed = self.y_head(obs_slots).reshape(batch, c, n, g)
        gain = 0.25 * torch.sigmoid(self.k_head(obs_slots)).reshape(
            batch, c, n, g
        )

        eye = torch.eye(d, device=video.device, dtype=video.dtype)
        erase = eye - gain[..., None, None] * (
            direction[..., :, None] * direction[..., None, :]
        )
        prediction = torch.diag_embed(a)
        transition_matrix = torch.matmul(erase, prediction)
        transition_bias = (
            torch.matmul(erase, b.unsqueeze(-1)).squeeze(-1)
            + gain[..., None] * direction * observed[..., None]
        )

        # Chronological affine composition is associative. The last inclusive
        # prefix is exactly T_{C-1} o ... o T_0 and is applied to H only once.
        transition_prefixes = associative_affine_scan(
            EpisodeAffineUpdate(
                matrix=transition_matrix,
                bias=transition_bias,
            )
        )
        matrix = transition_prefixes.matrix[:, -1]
        bias = transition_prefixes.bias[:, -1]
        predicted_observation = (direction * b).sum(dim=-1)
        corrected_observation = (direction * transition_bias).sum(dim=-1)
        return (
            EpisodeAffineUpdate(matrix=matrix, bias=bias),
            PredictionCorrectionDiagnostics(
                predicted_observation=predicted_observation,
                observed_evidence=observed,
                corrected_observation=corrected_observation,
                gain=gain,
                prediction_diagonal=a,
                prediction_bias=b,
                observation_direction=direction,
                transition_matrix=transition_matrix,
                transition_bias=transition_bias,
            ),
        )

def prediction_correction_loss(
    diagnostics: PredictionCorrectionDiagnostics,
    *,
    input_state: torch.Tensor | None = None,
    valid_mask: torch.Tensor | None = None,
    prediction_weight: float = 0.1,
    correction_weight: float = 0.1,
) -> torch.Tensor:
    predicted_observation = diagnostics.predicted_observation
    corrected_observation = diagnostics.corrected_observation
    if input_state is not None:
        groups = int(diagnostics.prediction_diagonal.shape[-2])
        group_dim = int(diagnostics.prediction_diagonal.shape[-1])
        if input_state.ndim != 3:
            raise ValueError("input_state must be [chunks,slots,state_dim]")
        transition_prefixes = associative_affine_scan(
            EpisodeAffineUpdate(
                matrix=diagnostics.transition_matrix,
                bias=diagnostics.transition_bias,
            )
        )
        post_states = apply_prefix_updates(input_state.float(), transition_prefixes)
        transition_inputs = torch.cat(
            (input_state[:, None].float(), post_states[:, :-1]), dim=1
        )
        grouped = _group_state(transition_inputs, groups, group_dim)
        predicted_group = (
            diagnostics.prediction_diagonal * grouped
            + diagnostics.prediction_bias
        )
        predicted_observation = (
            diagnostics.observation_direction * predicted_group
        ).sum(dim=-1)
        corrected_group = predicted_group + (
            diagnostics.gain[..., None]
            * diagnostics.observation_direction
            * (diagnostics.observed_evidence - predicted_observation)[..., None]
        )
        corrected_observation = (
            diagnostics.observation_direction * corrected_group
        ).sum(dim=-1)
    def masked_mse(value: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        error = (value - target).square()
        if valid_mask is None:
            return error.mean()
        if valid_mask.numel() != error.shape[0]:
            raise ValueError("valid_mask must contain one value per chunk")
        mask = valid_mask.to(device=error.device, dtype=error.dtype).reshape(
            error.shape[0], *([1] * (error.ndim - 1))
        )
        denominator = mask.sum() * error[0].numel()
        return (error * mask).sum() / denominator.clamp_min(1.0)

    prediction = masked_mse(predicted_observation, diagnostics.observed_evidence)
    correction = masked_mse(corrected_observation, diagnostics.observed_evidence)
    return float(prediction_weight) * prediction + float(correction_weight) * correction


class EpisodeMemoryReader(nn.Module):
    """Shared H-to-K/V projection with per-layer low-rank adapters."""

    def __init__(
        self,
        config: EpisodeMemoryConfig,
        *,
        video_dim: int,
        action_dim: int,
        attention_dim: int,
        num_heads: int,
        num_layers: int,
    ) -> None:
        super().__init__()
        if attention_dim % num_heads:
            raise ValueError("attention_dim must be divisible by num_heads")
        self.config = config
        self.attention_dim = int(attention_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.attention_dim // self.num_heads
        self.num_layers = int(num_layers)
        self.state_norm = nn.LayerNorm(config.state_dim)
        self.key = nn.Linear(config.state_dim, attention_dim, bias=False)
        self.value = nn.Linear(config.state_dim, attention_dim, bias=False)
        self.query = nn.ModuleDict(
            {
                "video": nn.Linear(video_dim, attention_dim, bias=False),
                "action": nn.Linear(action_dim, attention_dim, bias=False),
            }
        )
        self.adapter_down = nn.ModuleList(
            [nn.Linear(config.state_dim, config.reader_rank, bias=False) for _ in range(num_layers)]
        )
        self.adapter_up = nn.ModuleList(
            [nn.Linear(config.reader_rank, config.state_dim, bias=False) for _ in range(num_layers)]
        )
        self.output = nn.ModuleDict(
            {
                "video": nn.Linear(attention_dim, video_dim, bias=False),
                "action": nn.Linear(attention_dim, action_dim, bias=False),
            }
        )
        self.gates = nn.ParameterDict(
            {
                "video": nn.Parameter(torch.zeros(num_layers)),
                "action": nn.Parameter(torch.zeros(num_layers)),
            }
        )
        for up in self.adapter_up:
            nn.init.zeros_(up.weight)

    def forward(
        self,
        modality: str,
        layer: int,
        query_tokens: torch.Tensor,
        state: torch.Tensor,
    ) -> torch.Tensor:
        if modality not in self.query:
            raise ValueError(f"unsupported memory-reader modality: {modality}")
        if not 0 <= int(layer) < self.num_layers:
            raise ValueError("memory-reader layer is out of range")
        if state.ndim != 3 or state.shape[-2:] != (
            self.config.num_slots,
            self.config.state_dim,
        ):
            raise ValueError("episode state must be [B,num_slots,state_dim]")
        if state.shape[0] != query_tokens.shape[0]:
            if state.shape[0] == 1:
                state = state.expand(query_tokens.shape[0], -1, -1)
            else:
                raise ValueError("episode state/query batch mismatch")
        normalized = self.state_norm(state)
        adapted = normalized + self.adapter_up[layer](F.silu(self.adapter_down[layer](normalized)))
        q = self.query[modality](query_tokens)
        k = self.key(adapted)
        v = self.value(adapted)
        batch, queries = q.shape[:2]
        q = q.view(batch, queries, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, self.config.num_slots, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, self.config.num_slots, self.num_heads, self.head_dim).transpose(1, 2)
        attended = F.scaled_dot_product_attention(q, k, v)
        attended = attended.transpose(1, 2).reshape(batch, queries, self.attention_dim)
        gate = torch.tanh(self.gates[modality][layer]).to(dtype=attended.dtype)
        return gate * self.output[modality](attended)


class EpisodeMemoryModule(nn.Module):
    """Owned parameters for H initialisation, chunk writing and layer reads."""

    def __init__(
        self,
        config: EpisodeMemoryConfig,
        *,
        video_dim: int,
        action_dim: int,
        attention_dim: int,
        num_heads: int,
        num_layers: int,
    ) -> None:
        super().__init__()
        self.config = config
        self.empty_state = nn.Parameter(torch.zeros(config.num_slots, config.state_dim))
        nn.init.normal_(self.empty_state, std=0.02)
        self.updater = EpisodeChunkUpdater(
            config, video_dim=video_dim, action_dim=action_dim
        )
        self.reader = EpisodeMemoryReader(
            config,
            video_dim=video_dim,
            action_dim=action_dim,
            attention_dim=attention_dim,
            num_heads=num_heads,
            num_layers=num_layers,
        )

    def initial_state(
        self, batch_size: int, *, device=None, dtype=None
    ) -> torch.Tensor:
        state = self.empty_state
        if device is not None or dtype is not None:
            state = state.to(device=device, dtype=dtype)
        return state.unsqueeze(0).expand(int(batch_size), -1, -1).clone()


def build_episode_prefix_states(
    updater: EpisodeChunkUpdater,
    initial_state: torch.Tensor,
    video_tokens: torch.Tensor,
    action_tokens: torch.Tensor,
    *,
    chunk_valid_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, EpisodeAffineUpdate, PredictionCorrectionDiagnostics]:
    """Generate all chunk operators in parallel and scan their H prefixes.

    video_tokens has shape [B,J*C+1,Sv,Dv] and action_tokens has shape
    [B,J*C,Sa,Da]. Returned states are inclusive chunk-boundary states;
    callers use initial_state for the zero-chunk prefix.
    """

    if video_tokens.ndim not in (4, 5) or action_tokens.ndim not in (4, 5):
        raise ValueError("episode clean tokens must include batch and block axes")
    batch = int(video_tokens.shape[0])
    blocks = int(action_tokens.shape[1])
    c = updater.config.chunk_blocks
    if blocks <= 0 or blocks % c:
        raise ValueError("action block count must be a positive multiple of C")
    chunks = blocks // c
    if int(video_tokens.shape[1]) != blocks + 1:
        raise ValueError("closed transitions require exactly one extra observation")
    if chunk_valid_mask is None:
        chunk_valid_mask = torch.ones(
            (batch, chunks), dtype=torch.bool, device=video_tokens.device
        )
    elif tuple(chunk_valid_mask.shape) != (batch, chunks):
        raise ValueError("chunk_valid_mask must be [batch,chunks]")
    chunk_valid_mask = chunk_valid_mask.to(device=video_tokens.device, dtype=torch.bool)
    video_chunks = torch.stack(
        [video_tokens[:, index * c : (index + 1) * c + 1] for index in range(chunks)],
        dim=1,
    )
    action_chunks = action_tokens.reshape(
        batch, chunks, c, *action_tokens.shape[2:]
    )
    flat_video = video_chunks.flatten(0, 1)
    flat_action = action_chunks.flatten(0, 1)
    updates, diagnostics = updater(flat_video, flat_action)
    matrix = updates.matrix.reshape(
        batch, chunks, *updates.matrix.shape[1:]
    )
    bias = updates.bias.reshape(batch, chunks, *updates.bias.shape[1:])
    valid_matrix = chunk_valid_mask[:, :, None, None, None, None]
    identity = torch.eye(
        updater.config.group_dim,
        device=matrix.device,
        dtype=matrix.dtype,
    ).reshape(1, 1, 1, 1, updater.config.group_dim, updater.config.group_dim)
    matrix = torch.where(valid_matrix, matrix, identity)
    bias = torch.where(
        chunk_valid_mask[:, :, None, None, None],
        bias,
        torch.zeros((), device=bias.device, dtype=bias.dtype),
    )
    prefixes = associative_affine_scan(
        EpisodeAffineUpdate(matrix=matrix, bias=bias),
        checkpoint_stages=True,
    )
    states = apply_prefix_updates(initial_state.float(), prefixes)
    return states, prefixes, diagnostics
