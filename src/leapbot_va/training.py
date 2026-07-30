"""Runtime-isomorphic causal-history training for LeapBot-VA."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from leapbot_va.memory import VALID_CAUSAL_MODES

if TYPE_CHECKING:
    from leapbot_va.models.leapbot import LeapBotVA


DEFAULT_HISTORY_VAE_BATCH_CHUNK_SIZE = 1


def _encode_single_frame_video_batch(
    model: "LeapBotVA",
    videos: torch.Tensor,
    *,
    tiled: bool,
) -> torch.Tensor:
    """Encode a batch whose members are independent one-frame videos.

    ``WanVideoVAE.encode`` deliberately iterates over its leading dimension and
    invokes ``single_encode`` once per member.  Its underlying ``VideoVAE_.encode``
    is batch-separable, however: temporal cache tensors retain the batch axis and
    no layer mixes different batch members.  Calling ``single_encode`` with a
    small batch therefore runs exactly the same T=1 temporal program while
    avoiding one heavyweight encoder launch per observation.

    Non-Wan or tiled encoders retain the public model path.  In particular, this
    helper never silently bypasses a custom encoder's tiling semantics.
    """

    if videos.ndim != 5 or int(videos.shape[2]) != 1:
        raise ValueError(
            "independent history observations must be [N,C,1,H,W]; "
            f"got {tuple(videos.shape)}"
        )
    if not tiled:
        from fastwam.models.wan22.wan_video_vae import WanVideoVAE

        if isinstance(model.vae, WanVideoVAE):
            return model.vae.single_encode(videos, device=model.device)
    return model._encode_video_latents(videos, tiled=tiled)


@torch.no_grad()
def encode_independent_history_video_latents(
    model: "LeapBotVA",
    history_video: torch.Tensor,
    history_valid_blocks: torch.Tensor,
    *,
    empty_latent_reference: torch.Tensor,
    tiled: bool = False,
    chunk_size: int = DEFAULT_HISTORY_VAE_BATCH_CHUNK_SIZE,
) -> torch.Tensor:
    """Encode valid ``[B,C,H,Y,X]`` history frames as independent T=1 videos.

    Valid ``(batch, history)`` pairs are visited in row-major order, encoded in
    bounded chunks along a synthetic batch dimension, and scattered back to
    ``[B,C_latent,H,Y_latent,X_latent]``.  History is never placed on the VAE
    temporal axis, so causal VAE state cannot leak across replanning blocks.
    Invalid/padded blocks remain zero and H=0 performs no encoder call.
    """

    if history_video.ndim != 5:
        raise ValueError(
            "history_video must be [B,C,H,Y,X], "
            f"got {tuple(history_video.shape)}"
        )
    batch, _, history_blocks, _, _ = history_video.shape
    if history_valid_blocks.dtype != torch.bool or history_valid_blocks.shape != (
        batch,
        history_blocks,
    ):
        raise ValueError(
            "history_valid_blocks must be bool [B,H] matching history_video"
        )
    if history_valid_blocks.device != history_video.device:
        raise ValueError("history_video and history_valid_blocks must share a device")
    if empty_latent_reference.ndim != 5 or int(empty_latent_reference.shape[0]) != batch:
        raise ValueError("empty_latent_reference must be [B,C,T,Y,X] with matching B")
    chunk_size = int(chunk_size)
    if chunk_size <= 0:
        raise ValueError("history VAE batch chunk_size must be positive")

    latent_channels = int(empty_latent_reference.shape[1])
    latent_height = int(empty_latent_reference.shape[3])
    latent_width = int(empty_latent_reference.shape[4])
    if history_blocks == 0:
        return empty_latent_reference[:, :, :0]

    valid_pairs = torch.nonzero(history_valid_blocks, as_tuple=False)
    if valid_pairs.numel() == 0:
        return empty_latent_reference.new_zeros(
            (batch, latent_channels, history_blocks, latent_height, latent_width)
        )

    flat_latents = None
    for start in range(0, int(valid_pairs.shape[0]), chunk_size):
        pairs = valid_pairs[start : start + chunk_size]
        # Advanced indexing gathers [N,C,Y,X].  The explicit singleton below is
        # the only temporal dimension ever presented to the VAE.
        video_chunk = history_video[pairs[:, 0], :, pairs[:, 1]].unsqueeze(2)
        latent_chunk = _encode_single_frame_video_batch(
            model,
            video_chunk,
            tiled=tiled,
        )
        if not isinstance(latent_chunk, torch.Tensor) or latent_chunk.ndim != 5:
            raise ValueError("history VAE encoder must return a [N,C,T,Y,X] tensor")
        expected = (
            int(pairs.shape[0]),
            latent_channels,
            1,
            latent_height,
            latent_width,
        )
        if tuple(latent_chunk.shape) != expected:
            raise ValueError(
                "each independent T=1 history observation must encode to one latent "
                f"frame with shape {expected}, got {tuple(latent_chunk.shape)}"
            )
        if flat_latents is None:
            flat_latents = latent_chunk.new_zeros(
                (
                    batch * history_blocks,
                    latent_channels,
                    1,
                    latent_height,
                    latent_width,
                )
            )
        flat_indices = pairs[:, 0] * history_blocks + pairs[:, 1]
        flat_latents.index_copy_(0, flat_indices, latent_chunk)

    if flat_latents is None:  # Kept explicit for static analyzers.
        raise RuntimeError("valid history observations produced no latent chunks")
    return (
        flat_latents.reshape(
            batch,
            history_blocks,
            latent_channels,
            latent_height,
            latent_width,
        )
        .permute(0, 2, 1, 3, 4)
        .contiguous()
    )


def history_window_indices(
    *,
    current_episode_step: int,
    history_blocks: int,
    replan_steps: int,
    current_window_offset: int,
) -> tuple[list[int], slice, list[int]]:
    """Return within-window observations/actions and absolute block positions."""

    if replan_steps <= 0 or history_blocks < 0:
        raise ValueError("invalid replan/history size")
    if current_episode_step % replan_steps:
        raise ValueError("current sample is not on a replanning boundary")
    if current_episode_step < history_blocks * replan_steps:
        raise ValueError("history would cross the episode boundary")
    start = current_window_offset - history_blocks * replan_steps
    observations = [start + block * replan_steps for block in range(history_blocks)]
    positions = list(
        range(
            current_episode_step // replan_steps - history_blocks,
            current_episode_step // replan_steps,
        )
    )
    return observations, slice(start, current_window_offset), positions


def build_packed_history_attention_mask(
    history_valid_blocks: torch.Tensor,
    *,
    video_tokens_per_frame: int,
    current_video_frames: int,
    replan_steps: int,
    action_horizon: int,
    causal_mode: str,
) -> torch.Tensor:
    """Build [B,S,S] mask with no action-to-future-video leakage."""

    if causal_mode not in VALID_CAUSAL_MODES:
        raise ValueError(f"unsupported causal mode: {causal_mode}")
    if history_valid_blocks.ndim != 2 or history_valid_blocks.dtype != torch.bool:
        raise ValueError("history_valid_blocks must be bool [B,H]")
    if min(video_tokens_per_frame, current_video_frames, replan_steps, action_horizon) <= 0:
        raise ValueError("token/frame/action counts must be positive")

    batch, max_history = history_valid_blocks.shape
    video_frames = max_history + current_video_frames
    video_len = video_frames * video_tokens_per_frame
    history_action_len = max_history * replan_steps
    action_len = history_action_len + action_horizon
    total = video_len + action_len
    device = history_valid_blocks.device

    video_block = torch.cat(
        [
            torch.arange(max_history).repeat_interleave(video_tokens_per_frame),
            torch.full((current_video_frames * video_tokens_per_frame,), max_history),
        ]
    ).to(device)
    video_frame = torch.arange(video_frames, device=device).repeat_interleave(
        video_tokens_per_frame
    )
    action_block = torch.cat(
        [
            torch.arange(max_history).repeat_interleave(replan_steps),
            torch.full((action_horizon,), max_history),
        ]
    ).to(device)
    block = torch.cat([video_block, action_block])
    modality = torch.cat(
        [
            torch.zeros(video_len, dtype=torch.long),
            torch.ones(action_len, dtype=torch.long),
        ]
    ).to(device)
    frame = torch.cat(
        [video_frame, torch.full((action_len,), -1, dtype=torch.long, device=device)]
    )

    q_block = block[:, None]
    k_block = block[None, :]
    q_action = modality[:, None].bool()
    k_action = modality[None, :].bool()
    q_frame = frame[:, None]
    k_frame = frame[None, :]
    earlier = k_block < q_block
    same_block = k_block == q_block

    # Historical action blocks read their same-block real observation. Current
    # action reads only current real-frame keys (frame=max_history), excluding
    # every future-video supervision frame in the same block.
    same_action_block = same_block & (
        k_action | (~k_action & ((q_block < max_history) | (k_frame == max_history)))
    )
    action_allowed = earlier | same_action_block

    current_future_query = (q_block == max_history) & (q_frame > max_history)
    same_video_block = same_block & ~k_action & (
        current_future_query | (k_frame == q_frame)
    )
    if causal_mode == "interleaved":
        earlier_video_allowed = earlier
    elif causal_mode == "vision_causal":
        earlier_video_allowed = earlier & ~k_action
    else:
        earlier_video_allowed = torch.zeros_like(earlier)
    video_allowed = same_video_block | earlier_video_allowed
    base_allowed = torch.where(q_action, action_allowed, video_allowed)

    valid_blocks = torch.cat(
        [history_valid_blocks, torch.ones((batch, 1), dtype=torch.bool, device=device)],
        dim=1,
    )
    token_valid = valid_blocks[:, block]
    mask = (
        base_allowed.unsqueeze(0)
        & token_valid.unsqueeze(2)
        & token_valid.unsqueeze(1)
    )
    # Keep padded query rows numerically safe while ensuring those tokens can
    # never become keys for a valid query.
    invalid_query_self = (~token_valid).unsqueeze(2) & torch.eye(
        total, dtype=torch.bool, device=device
    ).unsqueeze(0)
    return mask | invalid_query_self


def build_query_context_masks(
    base_context_mask: torch.Tensor,
    history_valid_blocks: torch.Tensor,
    *,
    video_tokens_per_frame: int,
    current_video_frames: int,
    replan_steps: int,
    action_horizon: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Language + exactly one corresponding proprio token per query."""

    batch, text_len = base_context_mask.shape
    max_history = history_valid_blocks.shape[1]
    total_blocks = max_history + 1
    video_block = torch.cat(
        [
            torch.arange(max_history).repeat_interleave(video_tokens_per_frame),
            torch.full((current_video_frames * video_tokens_per_frame,), max_history),
        ]
    ).to(base_context_mask.device)
    action_block = torch.cat(
        [
            torch.arange(max_history).repeat_interleave(replan_steps),
            torch.full((action_horizon,), max_history),
        ]
    ).to(base_context_mask.device)
    valid_blocks = torch.cat(
        [history_valid_blocks, torch.ones((batch, 1), dtype=torch.bool, device=base_context_mask.device)],
        dim=1,
    )

    def make(block_ids: torch.Tensor) -> torch.Tensor:
        result = torch.zeros(
            (batch, block_ids.numel(), text_len + total_blocks),
            dtype=torch.bool,
            device=base_context_mask.device,
        )
        for block_index in range(total_blocks):
            rows = block_ids == block_index
            query_valid = valid_blocks[:, block_index].view(batch, 1, 1)
            result[:, rows, :text_len] = base_context_mask[:, None, :] & query_valid
            result[:, rows, text_len + block_index] = valid_blocks[:, block_index].view(batch, 1)
        return result

    return make(video_block), make(action_block)


def validate_packed_history_metadata(
    history_valid_blocks: torch.Tensor,
    history_block_positions: torch.Tensor,
    current_block_positions: torch.Tensor,
    episode_steps: torch.Tensor,
    *,
    replan_steps: int,
    full_episode_history: bool,
) -> torch.Tensor:
    """Validate that packed history describes one causal episode prefix.

    Returns the number of valid blocks per sample.  Full-episode samples must
    contain the exact prefix ``0..current_block-1``; short-window ablations may
    start later, but must still be left-aligned and strictly precede current.
    """

    if history_valid_blocks.ndim != 2 or history_valid_blocks.dtype != torch.bool:
        raise ValueError("history_valid_blocks must be bool [B,H]")
    if history_block_positions.shape != history_valid_blocks.shape:
        raise ValueError("history_block_positions must match history_valid_blocks")
    batch = int(history_valid_blocks.shape[0])
    if current_block_positions.shape != (batch,) or episode_steps.shape != (batch,):
        raise ValueError("current_block_positions and episode_steps must be [B]")
    if replan_steps <= 0:
        raise ValueError("replan_steps must be positive")

    history_counts = history_valid_blocks.sum(dim=1)
    slots = torch.arange(
        history_valid_blocks.shape[1], device=history_valid_blocks.device
    )[None, :]
    expected_valid = slots < history_counts[:, None]
    if not torch.equal(history_valid_blocks, expected_valid):
        raise ValueError("history_valid_blocks must be left-aligned without internal gaps")

    valid_positions = history_block_positions[history_valid_blocks]
    if valid_positions.numel() and bool((valid_positions < 0).any().item()):
        raise ValueError("valid history block positions must be non-negative")
    if bool((episode_steps != current_block_positions * replan_steps).any().item()):
        raise ValueError("sample must lie on its declared replanning boundary")
    if history_valid_blocks.shape[1] > 0:
        current_grid = current_block_positions[:, None].expand_as(history_block_positions)
        if bool(
            (
                history_valid_blocks
                & (history_block_positions >= current_grid)
            ).any().item()
        ):
            raise ValueError("history positions must strictly precede the current block")

    if full_episode_history:
        if not torch.equal(current_block_positions, history_counts):
            raise ValueError(
                "full-episode training must expose every preceding observation/action block"
            )
        absolute_slots = slots.expand_as(history_valid_blocks)
        if not torch.equal(
            history_block_positions[history_valid_blocks],
            absolute_slots[history_valid_blocks],
        ):
            raise ValueError(
                "full-episode history positions must be the complete prefix 0..current_block-1"
            )
    return history_counts


def resolve_full_episode_history_batch(
    full_history_flag: torch.Tensor | bool | None,
    *,
    batch_size: int,
    device: torch.device,
) -> bool:
    """Resolve a homogeneous per-sample history contract for one batch.

    The packed validator has different invariants for complete prefixes and
    short-window ablations.  Collapsing a mixed boolean batch with ``all()``
    would incorrectly apply the weaker short-window rules to rows that claim
    to contain a complete episode prefix, so mixed batches are rejected.
    """

    if full_history_flag is None:
        raise ValueError("causal history samples must declare full_episode_history")
    flags = torch.as_tensor(full_history_flag, device=device, dtype=torch.bool).reshape(-1)
    if flags.numel() == 1 and batch_size > 1:
        flags = flags.expand(batch_size)
    if flags.numel() != batch_size:
        raise ValueError(
            "full_episode_history must provide one flag per sample: "
            f"expected={batch_size} got={flags.numel()}"
        )
    if bool((flags != flags[0]).any().item()):
        raise ValueError(
            "full_episode_history must be homogeneous within a packed batch"
        )
    return bool(flags[0].item())


def _materialize_attached_segments(
    segments: Iterable[dict[str, Any]],
    *,
    modalities: set[str] | None = None,
) -> list[dict[str, torch.Tensor]] | None:
    """Concatenate chronological K/V without severing the autograd graph."""

    selected = [
        segment
        for segment in segments
        if modalities is None or str(segment["modality"]) in modalities
    ]
    if not selected:
        return None
    num_layers = len(selected[0]["kv"])
    if any(len(segment["kv"]) != num_layers for segment in selected):
        raise ValueError("incremental history segments have inconsistent depth")
    return [
        {
            "k": torch.cat(
                [segment["kv"][layer]["k"] for segment in selected], dim=1
            ),
            "v": torch.cat(
                [segment["kv"][layer]["v"] for segment in selected], dim=1
            ),
        }
        for layer in range(num_layers)
    ]


def _select_attached_segments(
    segments: Iterable[dict[str, Any]],
    *,
    modalities: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Return chronological segment references without materializing their K/V."""

    return [
        segment
        for segment in segments
        if modalities is None or str(segment["modality"]) in modalities
    ]


def _flatten_segment_kv_inputs(
    segments: list[dict[str, Any]],
    *,
    num_layers: int,
) -> tuple[torch.Tensor, ...]:
    """Flatten raw segment K/V into direct activation-checkpoint inputs."""

    tensors: list[torch.Tensor] = []
    for segment in segments:
        segment_kv = segment["kv"]
        if len(segment_kv) < num_layers:
            raise ValueError(
                "incremental history segment has insufficient depth: "
                f"got={len(segment_kv)} expected={num_layers}"
            )
        for layer in range(num_layers):
            tensors.extend((segment_kv[layer]["k"], segment_kv[layer]["v"]))
    return tuple(tensors)


def _materialize_flat_segment_kv(
    flat_kv: tuple[torch.Tensor, ...],
    *,
    num_segments: int,
    num_layers: int,
) -> list[dict[str, torch.Tensor]] | None:
    """Temporarily concatenate direct checkpoint inputs inside its closure."""

    if num_segments == 0:
        if flat_kv:
            raise ValueError("zero history segments cannot provide K/V tensors")
        return None
    expected = num_segments * num_layers * 2
    if len(flat_kv) != expected:
        raise ValueError(
            f"flattened history has {len(flat_kv)} tensors, expected {expected}"
        )

    def tensor_at(segment: int, layer: int, value_offset: int) -> torch.Tensor:
        return flat_kv[(segment * num_layers + layer) * 2 + value_offset]

    return [
        {
            "k": torch.cat(
                [tensor_at(segment, layer, 0) for segment in range(num_segments)],
                dim=1,
            ),
            "v": torch.cat(
                [tensor_at(segment, layer, 1) for segment in range(num_segments)],
                dim=1,
            ),
        }
        for layer in range(num_layers)
    ]


def _use_segment_activation_checkpointing(model: "LeapBotVA") -> bool:
    """Resolve the training-only checkpoint switch without changing inference."""

    configured = getattr(
        model,
        "history_segment_activation_checkpointing",
        bool(getattr(model.mot, "mot_checkpoint_mixed_attn", False)),
    )
    # FastWAM deliberately keeps the root model in eval mode while putting only
    # MoT and trainable auxiliaries in train mode.  The executable DiT state,
    # rather than ``model.training``, therefore determines whether this is a
    # training graph.
    return bool(configured) and model.mot.training and torch.is_grad_enabled()


def _prefill_with_segment_history(
    model: "LeapBotVA",
    *,
    expert_name: str,
    tokens: torch.Tensor,
    freqs: torch.Tensor,
    t_mod: torch.Tensor,
    context_payload: dict[str, torch.Tensor],
    history_segments: list[dict[str, Any]],
    max_layers: int,
    exit_depths: tuple[int, ...] | None = None,
    segment_valid_mask: torch.Tensor | None = None,
    return_segment_kv: bool,
) -> tuple[
    torch.Tensor | dict[int, torch.Tensor] | None,
    list[dict[str, torch.Tensor]] | None,
]:
    """Run an unchanged MoT prefill with optional segment-level recomputation.

    Raw chronological segment K/V tensors are direct checkpoint inputs.  The
    only full-prefix concatenations are built inside ``run`` and can therefore
    be discarded after the forward pass and recreated during backward.  No
    segment is detached, summarized, compressed, or reordered.
    """

    context = context_payload["context"]
    context_mask = context_payload["mask"]
    if not isinstance(context, torch.Tensor) or not isinstance(
        context_mask, torch.Tensor
    ):
        raise ValueError("incremental training requires tensor context and mask")
    requested_exits = None if exit_depths is None else tuple(exit_depths)

    if not _use_segment_activation_checkpointing(model):
        return model.mot.prefill_expert_segment(
            expert_name=expert_name,
            tokens=tokens,
            freqs=freqs,
            t_mod=t_mod,
            context_payload={"context": context, "mask": context_mask},
            history_kv=_materialize_attached_segments(history_segments),
            max_layers=max_layers,
            segment_valid_mask=segment_valid_mask,
            exit_depths=requested_exits,
        )

    raw_kv = _flatten_segment_kv_inputs(
        history_segments,
        num_layers=max_layers,
    )
    num_segments = len(history_segments)
    num_exit_outputs = 0 if requested_exits is None else len(requested_exits)
    valid_mask_input = (
        torch.empty((0,), dtype=torch.bool, device=tokens.device)
        if segment_valid_mask is None
        else segment_valid_mask
    )

    def run(
        inner_tokens: torch.Tensor,
        inner_freqs: torch.Tensor,
        inner_t_mod: torch.Tensor,
        inner_context: torch.Tensor,
        inner_context_mask: torch.Tensor,
        inner_valid_mask: torch.Tensor,
        *inner_raw_kv: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        history_kv = _materialize_flat_segment_kv(
            tuple(inner_raw_kv),
            num_segments=num_segments,
            num_layers=max_layers,
        )
        hidden, segment_kv = model.mot.prefill_expert_segment(
            expert_name=expert_name,
            tokens=inner_tokens,
            freqs=inner_freqs,
            t_mod=inner_t_mod,
            context_payload={
                "context": inner_context,
                "mask": inner_context_mask,
            },
            history_kv=history_kv,
            max_layers=max_layers,
            segment_valid_mask=(
                None if inner_valid_mask.numel() == 0 else inner_valid_mask
            ),
            exit_depths=requested_exits,
            checkpoint_internal=False,
        )
        outputs: list[torch.Tensor] = []
        if requested_exits is not None:
            if not isinstance(hidden, dict):
                raise RuntimeError("multi-exit prefill did not return depth outputs")
            outputs.extend(hidden[depth] for depth in requested_exits)
        if return_segment_kv:
            for layer_kv in segment_kv:
                outputs.extend((layer_kv["k"], layer_kv["v"]))
        if not outputs:
            raise ValueError("checkpointed prefill must return hidden states or K/V")
        return tuple(outputs)

    checkpoint_outputs = checkpoint(
        run,
        tokens,
        freqs,
        t_mod,
        context,
        context_mask,
        valid_mask_input,
        *raw_kv,
        use_reentrant=False,
        preserve_rng_state=True,
    )
    if isinstance(checkpoint_outputs, torch.Tensor):
        flat_outputs = (checkpoint_outputs,)
    else:
        flat_outputs = tuple(checkpoint_outputs)

    hidden_outputs: torch.Tensor | dict[int, torch.Tensor] | None = None
    if requested_exits is not None:
        hidden_outputs = {
            depth: flat_outputs[index]
            for index, depth in enumerate(requested_exits)
        }
    segment_outputs: list[dict[str, torch.Tensor]] | None = None
    if return_segment_kv:
        kv_outputs = flat_outputs[num_exit_outputs:]
        if len(kv_outputs) != max_layers * 2:
            raise RuntimeError(
                "checkpointed prefill returned an invalid K/V tensor count"
            )
        segment_outputs = [
            {"k": kv_outputs[layer * 2], "v": kv_outputs[layer * 2 + 1]}
            for layer in range(max_layers)
        ]
    return hidden_outputs, segment_outputs


def _forward_action_with_segment_history(
    model: "LeapBotVA",
    *,
    action_tokens: torch.Tensor,
    action_freqs: torch.Tensor,
    action_t_mod: torch.Tensor,
    action_context_payload: dict[str, torch.Tensor],
    history_segments: list[dict[str, Any]],
    max_layers: int,
    exit_depths: tuple[int, ...],
    action_valid_mask: torch.Tensor | None = None,
) -> dict[int, torch.Tensor]:
    """Run transient ActionDiT while rematerializing only its K/V prefix."""

    if not history_segments:
        raise RuntimeError("current action has no real observation prefix")
    context = action_context_payload["context"]
    context_mask = action_context_payload["mask"]
    requested_exits = tuple(exit_depths)
    if not _use_segment_activation_checkpointing(model):
        hidden = model.mot.forward_action_with_history(
            action_tokens=action_tokens,
            action_freqs=action_freqs,
            action_t_mod=action_t_mod,
            action_context_payload={"context": context, "mask": context_mask},
            history_kv=_materialize_attached_segments(history_segments),
            max_layers=max_layers,
            exit_depths=requested_exits,
            action_valid_mask=action_valid_mask,
        )
        if not isinstance(hidden, dict):
            raise RuntimeError("multi-exit ActionDiT did not return depth outputs")
        return hidden

    raw_kv = _flatten_segment_kv_inputs(
        history_segments,
        num_layers=max_layers,
    )
    num_segments = len(history_segments)
    valid_mask_input = (
        torch.empty((0,), dtype=torch.bool, device=action_tokens.device)
        if action_valid_mask is None
        else action_valid_mask
    )

    def run(
        inner_tokens: torch.Tensor,
        inner_freqs: torch.Tensor,
        inner_t_mod: torch.Tensor,
        inner_context: torch.Tensor,
        inner_context_mask: torch.Tensor,
        inner_valid_mask: torch.Tensor,
        *inner_raw_kv: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        history_kv = _materialize_flat_segment_kv(
            tuple(inner_raw_kv),
            num_segments=num_segments,
            num_layers=max_layers,
        )
        hidden = model.mot.forward_action_with_history(
            action_tokens=inner_tokens,
            action_freqs=inner_freqs,
            action_t_mod=inner_t_mod,
            action_context_payload={
                "context": inner_context,
                "mask": inner_context_mask,
            },
            history_kv=history_kv,
            max_layers=max_layers,
            exit_depths=requested_exits,
            action_valid_mask=(
                None if inner_valid_mask.numel() == 0 else inner_valid_mask
            ),
            checkpoint_internal=False,
        )
        if not isinstance(hidden, dict):
            raise RuntimeError("multi-exit ActionDiT did not return depth outputs")
        return tuple(hidden[depth] for depth in requested_exits)

    checkpoint_outputs = checkpoint(
        run,
        action_tokens,
        action_freqs,
        action_t_mod,
        context,
        context_mask,
        valid_mask_input,
        *raw_kv,
        use_reentrant=False,
        preserve_rng_state=True,
    )
    if isinstance(checkpoint_outputs, torch.Tensor):
        flat_outputs = (checkpoint_outputs,)
    else:
        flat_outputs = tuple(checkpoint_outputs)
    return {
        depth: flat_outputs[index]
        for index, depth in enumerate(requested_exits)
    }


def _video_history_for_mode(
    segments: list[dict[str, Any]], causal_mode: str
) -> list[dict[str, torch.Tensor]] | None:
    if causal_mode == "interleaved":
        return _materialize_attached_segments(segments)
    if causal_mode == "vision_causal":
        return _materialize_attached_segments(segments, modalities={"video"})
    if causal_mode == "action_aggregator":
        return None
    raise ValueError(f"unsupported causal mode: {causal_mode}")


def _video_history_segments_for_mode(
    segments: list[dict[str, Any]], causal_mode: str
) -> list[dict[str, Any]]:
    """Select the runtime video prefix while preserving raw segment tensors."""

    if causal_mode == "interleaved":
        return _select_attached_segments(segments)
    if causal_mode == "vision_causal":
        return _select_attached_segments(segments, modalities={"video"})
    if causal_mode == "action_aggregator":
        return []
    raise ValueError(f"unsupported causal mode: {causal_mode}")


def _future_video_history_for_mode(
    completed_segments: list[dict[str, Any]],
    current_real_segment: dict[str, Any],
    causal_mode: str,
) -> list[dict[str, torch.Tensor]]:
    """Select causal past plus the current real frame for future-video queries."""

    if causal_mode == "interleaved":
        selected = [*completed_segments, current_real_segment]
    elif causal_mode == "vision_causal":
        selected = [
            segment
            for segment in completed_segments
            if segment["modality"] == "video"
        ]
        selected.append(current_real_segment)
    elif causal_mode == "action_aggregator":
        selected = [current_real_segment]
    else:
        raise ValueError(f"unsupported causal mode: {causal_mode}")
    materialized = _materialize_attached_segments(selected)
    if materialized is None:
        raise RuntimeError("current real observation was not added to video history")
    return materialized


def _future_video_history_segments_for_mode(
    completed_segments: list[dict[str, Any]],
    current_real_segment: dict[str, Any],
    causal_mode: str,
) -> list[dict[str, Any]]:
    """Select future-video prefix segments without concatenating their K/V."""

    if causal_mode == "interleaved":
        return [*completed_segments, current_real_segment]
    if causal_mode == "vision_causal":
        return [
            *_select_attached_segments(
                completed_segments,
                modalities={"video"},
            ),
            current_real_segment,
        ]
    if causal_mode == "action_aggregator":
        return [current_real_segment]
    raise ValueError(f"unsupported causal mode: {causal_mode}")


def _context_for_proprio(
    model: "LeapBotVA",
    context: torch.Tensor,
    context_mask: torch.Tensor,
    proprio: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return model._append_proprio_to_context(
        context=context,
        context_mask=context_mask,
        proprio=proprio.to(device=model.device, dtype=model.torch_dtype),
    )


def current_video_segment_attention_mask(
    *, tokens_per_frame: int, num_frames: int, device: torch.device
) -> torch.Tensor:
    """FastWAM first-frame-causal mask for one current video block."""

    if tokens_per_frame <= 0 or num_frames <= 0:
        raise ValueError("tokens_per_frame and num_frames must be positive")
    total = tokens_per_frame * num_frames
    mask = torch.ones((total, total), dtype=torch.bool, device=device)
    mask[:tokens_per_frame, tokens_per_frame:] = False
    return mask


def future_video_token_valid_mask(
    image_is_pad: torch.Tensor | None,
    *,
    temporal_downsample_factor: int,
    num_future_latent_frames: int,
    tokens_per_frame: int,
    device: torch.device,
) -> torch.Tensor | None:
    """Return a token mask only when an episode has a fully padded latent tail."""

    if image_is_pad is None:
        return None
    if image_is_pad.dtype != torch.bool or image_is_pad.ndim != 2:
        raise ValueError("image_is_pad must be bool [B,T]")
    if (
        min(
            temporal_downsample_factor,
            num_future_latent_frames,
            tokens_per_frame,
        )
        <= 0
    ):
        raise ValueError("future-video mask dimensions must be positive")
    tail = image_is_pad[:, 1:].to(device=device)
    expected_tail = num_future_latent_frames * temporal_downsample_factor
    if int(tail.shape[1]) != expected_tail:
        raise ValueError(
            "image padding cannot be aligned with future video latents: "
            f"tail={tail.shape[1]} expected={expected_tail}"
        )
    latent_valid = ~tail.reshape(
        tail.shape[0], num_future_latent_frames, temporal_downsample_factor
    ).all(dim=2)
    token_valid = latent_valid.repeat_interleave(tokens_per_frame, dim=1)
    if bool(token_valid.all().item()):
        return None
    return token_valid


def _runtime_single_observation_latents(
    model: "LeapBotVA", image: torch.Tensor, *, tiled: bool
) -> torch.Tensor:
    """Run the exact batch-one, T=1 VAE program used by memory inference."""

    if image.ndim != 4 or tuple(image.shape[:2]) != (1, 3):
        raise ValueError(
            "runtime observation image must have shape [1,3,H,W], "
            f"got {tuple(image.shape)}"
        )
    latents = model._encode_input_image_latents_tensor(
        image.to(device=model.device, dtype=model.torch_dtype),
        tiled=tiled,
    )
    if not isinstance(latents, torch.Tensor) or latents.ndim != 5:
        raise ValueError("runtime observation VAE must return [1,C,T,H,W]")
    if int(latents.shape[0]) != 1 or int(latents.shape[2]) != 1:
        raise ValueError(
            "one real observation must encode to exactly one latent frame, "
            f"got {tuple(latents.shape)}"
        )
    return latents


def _video_prediction_at_depth(
    model: "LeapBotVA",
    *,
    depth: int,
    hidden: torch.Tensor,
    pre_state: dict[str, Any],
) -> torch.Tensor:
    if depth == model.mot.num_layers:
        return model.video_expert.post_dit(hidden, pre_state)
    pred_tokens = model.video_exit_heads[str(depth)](hidden, pre_state["t"])
    return model.video_expert.unpatchify(
        pred_tokens, pre_state["meta"]["grid_size"]
    )


def _action_prediction_at_depth(
    model: "LeapBotVA",
    *,
    depth: int,
    hidden: torch.Tensor,
    pre_state: dict[str, Any],
) -> torch.Tensor:
    if depth == model.mot.num_layers:
        return model.action_expert.post_dit(hidden, pre_state)
    return model.action_exit_heads[str(depth)](hidden)


def _packed_causal_history_reference_loss(
    model: "LeapBotVA", sample, tiled: bool = False
):
    """Retained one-shot reference used only by numerical audit tests."""

    inputs = model.build_inputs(sample, tiled=tiled, append_proprio=False)
    current_latents = inputs["input_latents"]
    action = inputs["action"]
    batch = current_latents.shape[0]
    configured_history_valid = sample["history_valid_blocks"].to(
        model.device, dtype=torch.bool
    )
    full_episode_history = resolve_full_episode_history_batch(
        sample.get("full_episode_history"),
        batch_size=batch,
        device=model.device,
    )
    replan_steps = int(sample["history_action"].shape[2])
    action_horizon = int(action.shape[1])
    model.validate_temporal_contract(
        replan_steps=replan_steps,
        action_horizon=action_horizon,
    )
    history_counts = validate_packed_history_metadata(
        configured_history_valid,
        sample["history_block_positions"].to(model.device),
        sample["current_block_position"].to(model.device),
        sample["episode_step"].to(model.device),
        replan_steps=replan_steps,
        full_episode_history=full_episode_history,
    )
    max_history = int(history_counts.max().item())
    # Capacity is a runtime upper bound, not a reason to execute padded tokens.
    # Cropping only a suffix that is invalid for every sample preserves every
    # real episode-prefix block while making full BPTT practical on H800s.
    history_valid = configured_history_valid[:, :max_history]

    history_video = sample["history_video"][:, :, :max_history].to(
        model.device, dtype=model.torch_dtype
    )
    history_latents = encode_independent_history_video_latents(
        model,
        history_video,
        history_valid,
        empty_latent_reference=current_latents,
        tiled=tiled,
        chunk_size=int(
            getattr(
                model,
                "history_vae_batch_chunk_size",
                DEFAULT_HISTORY_VAE_BATCH_CHUNK_SIZE,
            )
        ),
    )

    noise_video = torch.randn_like(current_latents)
    timestep_video = model.train_video_scheduler.sample_training_t(
        batch, model.device, current_latents.dtype
    )
    noisy_current = model.train_video_scheduler.add_noise(
        current_latents, noise_video, timestep_video
    )
    target_video = model.train_video_scheduler.training_target(
        current_latents, noise_video, timestep_video
    )
    noisy_current[:, :, 0:1] = current_latents[:, :, 0:1]
    packed_video = torch.cat([history_latents, noisy_current], dim=2)

    history_action = sample["history_action"][:, :max_history].to(
        model.device, dtype=model.torch_dtype
    )
    history_action = history_action.reshape(
        batch, max_history * replan_steps, int(action.shape[-1])
    )
    noise_action = torch.randn_like(action)
    timestep_action = model.train_action_scheduler.sample_training_t(
        batch, model.device, action.dtype
    )
    noisy_action = model.train_action_scheduler.add_noise(action, noise_action, timestep_action)
    target_action = model.train_action_scheduler.training_target(action, noise_action, timestep_action)
    packed_action = torch.cat([history_action, noisy_action], dim=1)

    context = inputs["context"]
    context_mask = inputs["context_mask"]
    if model.proprio_encoder is None:
        raise ValueError("causal history training requires proprio_encoder")
    history_proprio = sample["history_proprio"][:, :max_history].to(
        model.device, dtype=model.torch_dtype
    )
    current_proprio = sample["proprio"][:, 0:1].to(model.device, dtype=model.torch_dtype)
    proprio_tokens = model.proprio_encoder(torch.cat([history_proprio, current_proprio], dim=1))
    packed_context = torch.cat([context, proprio_tokens.to(context.dtype)], dim=1)
    packed_context_mask = torch.cat(
        [
            context_mask,
            torch.ones((batch, max_history + 1), dtype=torch.bool, device=model.device),
        ],
        dim=1,
    )

    history_positions = (
        sample["history_block_positions"][:, :max_history]
        .to(model.device)
        .clamp_min(0)
    )
    current_block = sample["current_block_position"].to(model.device).view(batch, 1)
    current_video_frames = int(current_latents.shape[2])
    # Keep FastWAM's pretrained RoPE coordinates local to each block.  The
    # episode-global block index is carried by the separate temporal embedding
    # below, so adding history does not phase-shift the native current block.
    frame_positions = torch.cat(
        [
            torch.zeros_like(history_positions),
            torch.arange(
                current_video_frames,
                device=model.device,
                dtype=history_positions.dtype,
            )
            .view(1, -1)
            .expand(batch, -1),
        ],
        dim=1,
    )
    absolute_video_blocks = torch.cat(
        [
            history_positions,
            current_block.expand(-1, current_video_frames),
        ],
        dim=1,
    )
    frame_timesteps = torch.cat(
        [
            torch.zeros((batch, max_history), device=model.device, dtype=timestep_video.dtype),
            timestep_video[:, None].expand(-1, current_video_frames).clone(),
        ],
        dim=1,
    )
    frame_timesteps[:, max_history] = 0
    video_pre = model.video_expert.pre_dit(
        x=packed_video,
        timestep=timestep_video,
        context=packed_context,
        context_mask=packed_context_mask,
        action=None,
        fuse_vae_embedding_in_latents=inputs["fuse_vae_embedding_in_latents"],
        frame_position_ids=frame_positions,
        frame_timesteps=frame_timesteps,
    )
    video_pre = model.temporal_positions.apply_video_pre_dit(
        video_pre,
        absolute_video_blocks,
    )

    history_action_positions = (
        sample["history_block_positions"][:, :max_history]
        .to(model.device)
        .unsqueeze(-1)
        * replan_steps
        + torch.arange(replan_steps, device=model.device).view(1, 1, -1)
    ).clamp_min(0).reshape(batch, max_history * replan_steps)
    episode_step = sample["episode_step"].to(model.device).view(batch, 1)
    current_action_positions = episode_step + torch.arange(action_horizon, device=model.device)
    local_history_action_positions = (
        torch.arange(replan_steps, device=model.device)
        .view(1, 1, -1)
        .expand(batch, max_history, -1)
        .reshape(batch, max_history * replan_steps)
    )
    local_current_action_positions = (
        torch.arange(action_horizon, device=model.device)
        .view(1, -1)
        .expand(batch, -1)
    )
    action_positions = torch.cat(
        [local_history_action_positions, local_current_action_positions], dim=1
    )
    absolute_action_positions = torch.cat(
        [history_action_positions, current_action_positions], dim=1
    )
    history_action_blocks = (
        history_positions.unsqueeze(-1)
        .expand(-1, -1, replan_steps)
        .reshape(batch, max_history * replan_steps)
    )
    current_action_blocks = current_block.expand(-1, action_horizon)
    absolute_action_blocks = torch.cat(
        [history_action_blocks, current_action_blocks], dim=1
    )
    action_token_timesteps = torch.cat(
        [
            torch.zeros_like(history_action_positions, dtype=timestep_action.dtype),
            timestep_action[:, None].expand(-1, action_horizon),
        ],
        dim=1,
    )
    action_pre = model.action_expert.pre_dit(
        action_tokens=packed_action,
        timestep=timestep_action,
        context=packed_context,
        context_mask=packed_context_mask,
        position_ids=action_positions,
        token_timesteps=action_token_timesteps,
    )
    action_pre = model.temporal_positions.apply_action_pre_dit(
        action_pre,
        absolute_action_positions,
        absolute_action_blocks,
    )

    tokens_per_frame = int(video_pre["meta"]["tokens_per_frame"])
    attention_mask = build_packed_history_attention_mask(
        history_valid,
        video_tokens_per_frame=tokens_per_frame,
        current_video_frames=current_video_frames,
        replan_steps=replan_steps,
        action_horizon=action_horizon,
        causal_mode=model.causal_mode,
    )
    video_context_mask, action_context_mask = build_query_context_masks(
        context_mask,
        history_valid,
        video_tokens_per_frame=tokens_per_frame,
        current_video_frames=current_video_frames,
        replan_steps=replan_steps,
        action_horizon=action_horizon,
    )
    outputs = model.mot(
        embeds_all={"video": video_pre["tokens"], "action": action_pre["tokens"]},
        attention_mask=attention_mask,
        freqs_all={"video": video_pre["freqs"], "action": action_pre["freqs"]},
        context_all={
            "video": {"context": video_pre["context"], "mask": video_context_mask},
            "action": {"context": action_pre["context"], "mask": action_context_mask},
        },
        t_mod_all={"video": video_pre["t_mod"], "action": action_pre["t_mod"]},
        exit_depths=model.training_exit_depths,
    )

    losses = {}
    action_is_pad = inputs["action_is_pad"]
    for depth in model.training_exit_depths:
        hidden = outputs[depth]
        if depth == model.mot.num_layers:
            pred_video = model.video_expert.post_dit(hidden["video"], video_pre)
            pred_action = model.action_expert.post_dit(hidden["action"], action_pre)
        else:
            pred_video_tokens = model.video_exit_heads[str(depth)](
                hidden["video"], video_pre["t"]
            )
            pred_video = model.video_expert.unpatchify(
                pred_video_tokens, video_pre["meta"]["grid_size"]
            )
            pred_action = model.action_exit_heads[str(depth)](hidden["action"])

        pred_video = pred_video[:, :, max_history + 1 :]
        target_video_loss = target_video[:, :, 1:]
        video_per_sample = model._compute_video_loss_per_sample(
            pred_video,
            target_video_loss,
            inputs["image_is_pad"],
            include_initial_video_step=False,
        )
        video_weight = model.train_video_scheduler.training_weight(timestep_video).to(
            video_per_sample.device, dtype=video_per_sample.dtype
        )
        loss_video = (video_per_sample * video_weight).mean()

        pred_action = pred_action[:, -action_horizon:]
        action_token_loss = F.mse_loss(
            pred_action.float(), target_action.float(), reduction="none"
        ).mean(dim=2)
        if action_is_pad is not None:
            valid = (~action_is_pad).to(action_token_loss.dtype)
            action_per_sample = (action_token_loss * valid).sum(1) / valid.sum(1).clamp_min(1)
        else:
            action_per_sample = action_token_loss.mean(1)
        action_weight = model.train_action_scheduler.training_weight(timestep_action).to(
            action_per_sample.device, dtype=action_per_sample.dtype
        )
        loss_action = (action_per_sample * action_weight).mean()
        losses[depth] = (
            model.loss_lambda_video * loss_video + model.loss_lambda_action * loss_action,
            loss_video,
            loss_action,
        )

    final_depth = model.mot.num_layers
    if tuple(model.training_exit_depths) == (final_depth,):
        total = losses[final_depth][0]
    else:
        shallow = [losses[depth][0] for depth in model.training_exit_depths if depth != final_depth]
        total = losses[final_depth][0] + torch.stack(shallow).mean()
    metrics = {}
    for depth, (_, video_loss, action_loss) in losses.items():
        metrics[f"loss_video_d{depth}"] = model.loss_lambda_video * float(video_loss.detach())
        metrics[f"loss_action_d{depth}"] = model.loss_lambda_action * float(action_loss.detach())
    metrics["history_blocks_mean"] = float(history_counts.float().mean().detach())
    metrics["history_blocks_max"] = float(history_counts.max().detach())
    return total, metrics


def causal_history_training_loss(model: "LeapBotVA", sample, tiled: bool = False):
    """Full-gradient history training with the exact rollout attention decomposition.

    Completed blocks are encoded chronologically as clean ``V -> A`` segments.
    The current real observation is then encoded exactly once with the runtime
    batch-one/T=1 VAE and prefill path.  Current noisy actions read only those
    persistent real-data K/V tensors.  Noisy future-video tokens execute later
    in a separate transient branch and can therefore never affect ActionDiT.
    """

    if int(model.video_expert.patch_size[0]) != 1:
        raise ValueError(
            "incremental full-BPTT requires a temporal video patch size of 1"
        )
    if bool(getattr(model.video_expert, "action_conditioned", False)):
        raise ValueError(
            "future-only video decomposition requires action_conditioned=false"
        )

    inputs = model.build_inputs(sample, tiled=tiled, append_proprio=False)
    current_latents = inputs["input_latents"]
    action = inputs["action"]
    batch = int(current_latents.shape[0])
    action_horizon = int(action.shape[1])
    replan_steps = int(sample["history_action"].shape[2])
    model.validate_temporal_contract(
        replan_steps=replan_steps,
        action_horizon=action_horizon,
    )
    full_episode_history = resolve_full_episode_history_batch(
        sample.get("full_episode_history"),
        batch_size=batch,
        device=model.device,
    )
    history_valid = sample["history_valid_blocks"].to(
        model.device, dtype=torch.bool
    )
    history_positions = sample["history_block_positions"].to(
        model.device, dtype=torch.long
    )
    current_block_positions = sample["current_block_position"].to(
        model.device, dtype=torch.long
    )
    episode_steps = sample["episode_step"].to(model.device, dtype=torch.long)
    history_counts = validate_packed_history_metadata(
        history_valid,
        history_positions,
        current_block_positions,
        episode_steps,
        replan_steps=replan_steps,
        full_episode_history=full_episode_history,
    )
    if model.proprio_encoder is None:
        raise ValueError("causal history training requires proprio_encoder")

    history_action = sample["history_action"].to(
        model.device, dtype=model.torch_dtype, non_blocking=True
    )
    history_proprio = sample["history_proprio"].to(
        model.device, dtype=model.torch_dtype, non_blocking=True
    )
    current_proprio = sample["proprio"][:, 0].to(
        model.device, dtype=model.torch_dtype, non_blocking=True
    )

    noise_video = torch.randn_like(current_latents)
    timestep_video = model.train_video_scheduler.sample_training_t(
        batch, model.device, current_latents.dtype
    )
    noisy_video = model.train_video_scheduler.add_noise(
        current_latents, noise_video, timestep_video
    )
    target_video = model.train_video_scheduler.training_target(
        current_latents, noise_video, timestep_video
    )

    noise_action = torch.randn_like(action)
    timestep_action = model.train_action_scheduler.sample_training_t(
        batch, model.device, action.dtype
    )
    noisy_action = model.train_action_scheduler.add_noise(
        action, noise_action, timestep_action
    )
    target_action = model.train_action_scheduler.training_target(
        action, noise_action, timestep_action
    )

    training_depths = tuple(int(depth) for depth in model.training_exit_depths)
    final_depth = int(model.mot.num_layers)
    if not training_depths or training_depths[-1] != final_depth:
        raise ValueError("training exits must include the final MoT depth")
    video_losses: dict[int, list[torch.Tensor]] = {
        depth: [] for depth in training_depths
    }
    action_losses: dict[int, list[torch.Tensor]] = {
        depth: [] for depth in training_depths
    }

    for sample_index in range(batch):
        base_context = inputs["context"][sample_index : sample_index + 1]
        base_context_mask = inputs["context_mask"][sample_index : sample_index + 1]
        completed_segments: list[dict[str, Any]] = []
        history_count = int(history_counts[sample_index].item())

        for history_index in range(history_count):
            block_position = history_positions[
                sample_index, history_index
            ].reshape(1)
            block_context, block_context_mask = _context_for_proprio(
                model,
                base_context,
                base_context_mask,
                history_proprio[
                    sample_index : sample_index + 1, history_index
                ],
            )

            block_image = sample["history_video"][
                sample_index : sample_index + 1,
                :,
                history_index,
            ]
            block_latents = _runtime_single_observation_latents(
                model, block_image, tiled=tiled
            )
            video_pre = model._prepare_real_observation_pre_dit(
                latents=block_latents,
                context=block_context,
                context_mask=block_context_mask,
                block_index=int(block_position.item()),
            )
            _, video_kv = _prefill_with_segment_history(
                model,
                expert_name="video",
                tokens=video_pre["tokens"],
                freqs=video_pre["freqs"],
                t_mod=video_pre["t_mod"],
                context_payload={
                    "context": video_pre["context"],
                    "mask": video_pre["context_mask"],
                },
                history_segments=_video_history_segments_for_mode(
                    completed_segments, model.causal_mode
                ),
                max_layers=final_depth,
                return_segment_kv=True,
            )
            if video_kv is None:
                raise RuntimeError("history video prefill did not return K/V")
            completed_segments.append(
                {
                    "modality": "video",
                    "block_index": int(block_position.item()),
                    "kv": video_kv,
                }
            )

            executed_actions = history_action[
                sample_index : sample_index + 1, history_index
            ]
            clean_action_timestep = torch.zeros(
                (1,), device=model.device, dtype=executed_actions.dtype
            )
            action_pre = model._prepare_action_segment_pre_dit(
                actions=executed_actions,
                timestep=clean_action_timestep,
                context=block_context,
                context_mask=block_context_mask,
                absolute_start=int(block_position.item()) * replan_steps,
                block_index=int(block_position.item()),
            )
            if not completed_segments:
                raise RuntimeError("history action has no same-block real observation")
            _, action_kv = _prefill_with_segment_history(
                model,
                expert_name="action",
                tokens=action_pre["tokens"],
                freqs=action_pre["freqs"],
                t_mod=action_pre["t_mod"],
                context_payload={
                    "context": action_pre["context"],
                    "mask": action_pre["context_mask"],
                },
                history_segments=completed_segments,
                max_layers=final_depth,
                return_segment_kv=True,
            )
            if action_kv is None:
                raise RuntimeError("history action prefill did not return K/V")
            completed_segments.append(
                {
                    "modality": "action",
                    "block_index": int(block_position.item()),
                    "kv": action_kv,
                }
            )

        current_block = current_block_positions[sample_index].reshape(1)
        current_context, current_context_mask = _context_for_proprio(
            model,
            base_context,
            base_context_mask,
            current_proprio[sample_index : sample_index + 1],
        )
        current_image = sample["video"][
            sample_index : sample_index + 1, :, 0
        ]
        current_real_latents = _runtime_single_observation_latents(
            model, current_image, tiled=tiled
        )
        current_real_pre = model._prepare_real_observation_pre_dit(
            latents=current_real_latents,
            context=current_context,
            context_mask=current_context_mask,
            block_index=int(current_block.item()),
        )
        _, current_real_kv = _prefill_with_segment_history(
            model,
            expert_name="video",
            tokens=current_real_pre["tokens"],
            freqs=current_real_pre["freqs"],
            t_mod=current_real_pre["t_mod"],
            context_payload={
                "context": current_real_pre["context"],
                "mask": current_real_pre["context_mask"],
            },
            history_segments=_video_history_segments_for_mode(
                completed_segments, model.causal_mode
            ),
            max_layers=final_depth,
            return_segment_kv=True,
        )
        if current_real_kv is None:
            raise RuntimeError("current real observation prefill did not return K/V")
        current_real_segment = {
            "modality": "video",
            "block_index": int(current_block.item()),
            "kv": current_real_kv,
        }

        # This prefix is finalized before any future-video token is built.
        action_prefix_segments = [*completed_segments, current_real_segment]
        current_action_pre = model._prepare_action_segment_pre_dit(
            actions=noisy_action[sample_index : sample_index + 1],
            timestep=timestep_action[sample_index : sample_index + 1],
            context=current_context,
            context_mask=current_context_mask,
            absolute_start=int(episode_steps[sample_index].item()),
            block_index=int(current_block.item()),
        )
        current_action_valid_mask = None
        if inputs["action_is_pad"] is not None:
            candidate_action_valid = ~inputs["action_is_pad"][
                sample_index : sample_index + 1
            ]
            if not bool(candidate_action_valid.all().item()):
                current_action_valid_mask = candidate_action_valid
        action_hidden_by_depth = _forward_action_with_segment_history(
            model,
            action_tokens=current_action_pre["tokens"],
            action_freqs=current_action_pre["freqs"],
            action_t_mod=current_action_pre["t_mod"],
            action_context_payload={
                "context": current_action_pre["context"],
                "mask": current_action_pre["context_mask"],
            },
            history_segments=action_prefix_segments,
            max_layers=final_depth,
            exit_depths=training_depths,
            action_valid_mask=current_action_valid_mask,
        )

        # The video loss is intentionally later and transient.  It sees the
        # current real frame through attached K/V but can never enter action_prefix.
        future_latents = noisy_video[
            sample_index : sample_index + 1, :, 1:
        ]
        num_future_frames = int(future_latents.shape[2])
        if num_future_frames <= 0:
            raise ValueError("causal video training requires future latent frames")
        future_video_timestep = timestep_video[
            sample_index : sample_index + 1
        ]
        future_frame_timesteps = future_video_timestep[:, None].expand(
            -1, num_future_frames
        )
        future_video_pre = model.video_expert.pre_dit(
            x=future_latents,
            timestep=future_video_timestep,
            context=current_context,
            context_mask=current_context_mask,
            action=None,
            fuse_vae_embedding_in_latents=inputs[
                "fuse_vae_embedding_in_latents"
            ],
            frame_position_ids=torch.arange(
                1,
                num_future_frames + 1,
                device=model.device,
                dtype=torch.long,
            ),
            frame_timesteps=future_frame_timesteps,
        )
        future_video_pre = model.temporal_positions.apply_video_pre_dit(
            future_video_pre,
            current_block[:, None].expand(-1, num_future_frames),
        )
        future_video_valid_mask = future_video_token_valid_mask(
            None
            if inputs["image_is_pad"] is None
            else inputs["image_is_pad"][sample_index : sample_index + 1],
            temporal_downsample_factor=int(model.vae.temporal_downsample_factor),
            num_future_latent_frames=num_future_frames,
            tokens_per_frame=int(future_video_pre["meta"]["tokens_per_frame"]),
            device=model.device,
        )
        video_hidden_by_depth, _ = _prefill_with_segment_history(
            model,
            expert_name="video",
            tokens=future_video_pre["tokens"],
            freqs=future_video_pre["freqs"],
            t_mod=future_video_pre["t_mod"],
            context_payload={
                "context": future_video_pre["context"],
                "mask": future_video_pre["context_mask"],
            },
            history_segments=_future_video_history_segments_for_mode(
                completed_segments,
                current_real_segment,
                model.causal_mode,
            ),
            max_layers=final_depth,
            exit_depths=training_depths,
            segment_valid_mask=future_video_valid_mask,
            return_segment_kv=False,
        )
        if not isinstance(video_hidden_by_depth, dict):
            raise RuntimeError("multi-exit video DiT did not return depth outputs")

        for depth in training_depths:
            pred_action = _action_prediction_at_depth(
                model,
                depth=depth,
                hidden=action_hidden_by_depth[depth],
                pre_state=current_action_pre,
            )
            action_token_loss = F.mse_loss(
                pred_action.float(),
                target_action[sample_index : sample_index + 1].float(),
                reduction="none",
            ).mean(dim=2)
            if inputs["action_is_pad"] is not None:
                valid_action = (
                    ~inputs["action_is_pad"][sample_index : sample_index + 1]
                ).to(action_token_loss.dtype)
                action_per_sample = (
                    (action_token_loss * valid_action).sum(1)
                    / valid_action.sum(1).clamp_min(1)
                )
            else:
                action_per_sample = action_token_loss.mean(1)
            action_per_sample = action_per_sample * (
                model.train_action_scheduler.training_weight(
                    timestep_action[sample_index : sample_index + 1]
                ).to(
                    action_per_sample.device,
                    dtype=action_per_sample.dtype,
                )
            )
            action_losses[depth].append(action_per_sample.squeeze(0))

            pred_future_video = _video_prediction_at_depth(
                model,
                depth=depth,
                hidden=video_hidden_by_depth[depth],
                pre_state=future_video_pre,
            )
            video_per_sample = model._compute_video_loss_per_sample(
                pred_future_video,
                target_video[sample_index : sample_index + 1, :, 1:],
                None
                if inputs["image_is_pad"] is None
                else inputs["image_is_pad"][sample_index : sample_index + 1],
                include_initial_video_step=False,
            )
            video_per_sample = video_per_sample * (
                model.train_video_scheduler.training_weight(
                    future_video_timestep
                ).to(
                    video_per_sample.device,
                    dtype=video_per_sample.dtype,
                )
            )
            video_losses[depth].append(video_per_sample.squeeze(0))

    losses: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    for depth in training_depths:
        loss_video = torch.stack(video_losses[depth]).mean()
        loss_action = torch.stack(action_losses[depth]).mean()
        losses[depth] = (
            model.loss_lambda_video * loss_video
            + model.loss_lambda_action * loss_action,
            loss_video,
            loss_action,
        )

    if training_depths == (final_depth,):
        total = losses[final_depth][0]
    else:
        shallow = [
            losses[depth][0]
            for depth in training_depths
            if depth != final_depth
        ]
        total = losses[final_depth][0] + torch.stack(shallow).mean()
    metrics: dict[str, float] = {}
    for depth, (_, video_loss, action_loss) in losses.items():
        metrics[f"loss_video_d{depth}"] = model.loss_lambda_video * float(
            video_loss.detach()
        )
        metrics[f"loss_action_d{depth}"] = model.loss_lambda_action * float(
            action_loss.detach()
        )
    metrics["history_blocks_mean"] = float(
        history_counts.float().mean().detach()
    )
    metrics["history_blocks_max"] = float(history_counts.max().detach())
    return total, metrics
