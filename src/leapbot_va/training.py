"""Packed causal-history attention and multi-exit training for LeapBot-VA."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable

import torch
import torch.nn.functional as F

from leapbot_va.memory import VALID_CAUSAL_MODES

if TYPE_CHECKING:
    from leapbot_va.models.leapbot import LeapBotVA


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


def causal_history_training_loss(model: "LeapBotVA", sample, tiled: bool = False):
    """Original video/action flow matching over a clean real-history prefix."""

    inputs = model.build_inputs(sample, tiled=tiled, append_proprio=False)
    current_latents = inputs["input_latents"]
    action = inputs["action"]
    batch = current_latents.shape[0]
    configured_history_valid = sample["history_valid_blocks"].to(
        model.device, dtype=torch.bool
    )
    full_history_flag = sample.get("full_episode_history")
    full_episode_history = full_history_flag is not None and bool(
        torch.as_tensor(full_history_flag, device=model.device).all().item()
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
    history_latents = []
    for block_index in range(max_history):
        history_latents.append(
            model._encode_video_latents(history_video[:, :, block_index : block_index + 1], tiled=tiled)
        )
    history_latents = (
        torch.cat(history_latents, dim=2)
        if history_latents
        else current_latents[:, :, :0]
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


def _materialize_incremental_segments(
    segments: Iterable[dict[str, Any]],
    *,
    modalities: set[str] | None = None,
) -> list[dict[str, torch.Tensor]] | None:
    selected = [
        segment
        for segment in segments
        if modalities is None or str(segment["modality"]) in modalities
    ]
    if not selected:
        return None
    num_layers = len(selected[0]["kv"])
    if any(len(segment["kv"]) != num_layers for segment in selected):
        raise ValueError("incremental prefix segments have inconsistent depth")
    return [
        {
            "k": torch.cat([segment["kv"][layer]["k"] for segment in selected], dim=1),
            "v": torch.cat([segment["kv"][layer]["v"] for segment in selected], dim=1),
        }
        for layer in range(num_layers)
    ]


def _materialize_incremental_batch(
    sample_segments: list[list[dict[str, Any]]],
    sample_indices: torch.Tensor,
    *,
    modalities: set[str] | None = None,
) -> list[dict[str, torch.Tensor]] | None:
    """Stack equal-length per-sample prefixes into one attention batch."""

    individual = [
        _materialize_incremental_segments(
            sample_segments[int(index)], modalities=modalities
        )
        for index in sample_indices.tolist()
    ]
    if all(prefix is None for prefix in individual):
        return None
    if any(prefix is None for prefix in individual):
        raise ValueError("batched prefixes must be uniformly empty or populated")
    populated = [prefix for prefix in individual if prefix is not None]
    num_layers = len(populated[0])
    if any(len(prefix) != num_layers for prefix in populated):
        raise ValueError("batched prefixes have inconsistent depth")
    result = []
    for layer in range(num_layers):
        sequence_lengths = {int(prefix[layer]["k"].shape[1]) for prefix in populated}
        if len(sequence_lengths) != 1:
            raise ValueError(
                "samples grouped for incremental attention have unequal prefix lengths"
            )
        result.append(
            {
                "k": torch.cat([prefix[layer]["k"] for prefix in populated], dim=0),
                "v": torch.cat([prefix[layer]["v"] for prefix in populated], dim=0),
            }
        )
    return result


def _video_prefix_batch_for_mode(
    sample_segments: list[list[dict[str, Any]]],
    sample_indices: torch.Tensor,
    causal_mode: str,
) -> list[dict[str, torch.Tensor]] | None:
    if causal_mode == "interleaved":
        return _materialize_incremental_batch(sample_segments, sample_indices)
    if causal_mode == "vision_causal":
        return _materialize_incremental_batch(
            sample_segments, sample_indices, modalities={"video"}
        )
    if causal_mode == "action_aggregator":
        return None
    raise ValueError(f"unsupported causal mode: {causal_mode}")


def _video_prefix_for_mode(
    segments: list[dict[str, Any]], causal_mode: str
) -> list[dict[str, torch.Tensor]] | None:
    if causal_mode == "interleaved":
        return _materialize_incremental_segments(segments)
    if causal_mode == "vision_causal":
        return _materialize_incremental_segments(segments, modalities={"video"})
    if causal_mode == "action_aggregator":
        return None
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
    """Inference-equivalent current-video mask used only for training.

    The first real frame cannot read noisy future-video supervision.  Future
    tokens may attend bidirectionally inside the current video block.  Only the
    first-frame K/V is later exposed to the action branch.
    """

    if tokens_per_frame <= 0 or num_frames <= 0:
        raise ValueError("tokens_per_frame and num_frames must be positive")
    total = tokens_per_frame * num_frames
    mask = torch.ones((total, total), dtype=torch.bool, device=device)
    mask[:tokens_per_frame, tokens_per_frame:] = False
    return mask


def incremental_detached_prefix_training_loss(
    model: "LeapBotVA", sample, tiled: bool = False
):
    """Train the current block against the complete real episode prefix.

    Historical observation/action K/V is built sequentially with the exact
    rollout primitives and remains fully visible to attention.  It is encoded
    under ``no_grad`` and detached solely to bound activation memory; no
    historical block is removed.  The current real observation retains
    gradients and is the only current-video K/V exposed to ActionDiT.
    """

    final_depth = int(model.mot.num_layers)
    if tuple(model.training_exit_depths) != (final_depth,):
        raise NotImplementedError(
            "incremental detached-prefix training currently supports the final exit only; "
            "train shallow exits after the full-prefix recipe is selected"
        )
    full_history_flag = sample.get("full_episode_history")
    if full_history_flag is None or not bool(torch.as_tensor(full_history_flag).all().item()):
        raise ValueError(
            "incremental_detached_prefix requires data.train.full_episode_history=true"
        )

    inputs = model.build_inputs(sample, tiled=tiled, append_proprio=False)
    current_latents = inputs["input_latents"]
    action = inputs["action"]
    batch = int(current_latents.shape[0])
    action_horizon = int(action.shape[1])
    history_valid = sample["history_valid_blocks"].to(model.device, dtype=torch.bool)
    replan_steps = int(sample["history_action"].shape[2])
    model.validate_temporal_contract(
        replan_steps=replan_steps,
        action_horizon=action_horizon,
    )
    history_counts = validate_packed_history_metadata(
        history_valid,
        sample["history_block_positions"].to(model.device),
        sample["current_block_position"].to(model.device),
        sample["episode_step"].to(model.device),
        replan_steps=replan_steps,
        full_episode_history=True,
    )

    history_video = sample["history_video"].to(
        model.device, dtype=model.torch_dtype, non_blocking=True
    )
    history_action = sample["history_action"].to(
        model.device, dtype=model.torch_dtype, non_blocking=True
    )
    history_proprio = sample["history_proprio"].to(
        model.device, dtype=model.torch_dtype, non_blocking=True
    )
    history_positions = sample["history_block_positions"].to(model.device)
    current_block_positions = sample["current_block_position"].to(model.device)
    episode_steps = sample["episode_step"].to(model.device)

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

    sample_segments: list[list[dict[str, Any]]] = [[] for _ in range(batch)]

    # Recompute old K/V using current weights on every batch, but vectorize all
    # episodes that are active at the same prefix depth.  Each active sample has
    # exactly the same prefix token length, so this is numerically identical to
    # per-sample rollout prefill while using the H800 efficiently.
    with torch.no_grad():
        max_history = int(history_counts.max().item())
        for history_index in range(max_history):
            active = torch.nonzero(
                history_counts > history_index, as_tuple=False
            ).flatten()
            base_context = inputs["context"].index_select(0, active)
            base_context_mask = inputs["context_mask"].index_select(0, active)
            block_context, block_context_mask = _context_for_proprio(
                model,
                base_context,
                base_context_mask,
                history_proprio.index_select(0, active)[:, history_index],
            )
            block_image = history_video.index_select(0, active)[
                :, :, history_index : history_index + 1
            ]
            block_latent = model._encode_video_latents(block_image, tiled=tiled)
            clean_timestep = torch.zeros(
                (active.numel(),), device=model.device, dtype=block_latent.dtype
            )
            block_positions = history_positions.index_select(0, active)[
                :, history_index
            ]
            video_pre = model.video_expert.pre_dit(
                x=block_latent,
                timestep=clean_timestep,
                context=block_context,
                context_mask=block_context_mask,
                action=None,
                fuse_vae_embedding_in_latents=inputs["fuse_vae_embedding_in_latents"],
                frame_position_ids=torch.zeros(
                    (active.numel(), 1),
                    dtype=torch.long,
                    device=model.device,
                ),
            )
            video_pre = model.temporal_positions.apply_video_pre_dit(
                video_pre,
                block_positions[:, None],
            )
            _, video_kv = model.mot.prefill_expert_segment(
                expert_name="video",
                tokens=video_pre["tokens"],
                freqs=video_pre["freqs"],
                t_mod=video_pre["t_mod"],
                context_payload={
                    "context": video_pre["context"],
                    "mask": video_pre["context_mask"],
                },
                history_kv=_video_prefix_batch_for_mode(
                    sample_segments, active, model.causal_mode
                ),
                max_layers=final_depth,
            )
            for local_index, sample_index in enumerate(active.tolist()):
                sample_segments[sample_index].append(
                    {
                        "modality": "video",
                        "kv": [
                            {
                                "k": layer["k"][local_index : local_index + 1],
                                "v": layer["v"][local_index : local_index + 1],
                            }
                            for layer in video_kv
                        ],
                    }
                )

            executed_actions = history_action.index_select(0, active)[
                :, history_index
            ]
            absolute_executed_positions = (
                block_positions[:, None] * replan_steps
                + torch.arange(replan_steps, device=model.device)[None, :]
            )
            local_executed_positions = (
                torch.arange(replan_steps, device=model.device)
                .view(1, -1)
                .expand(active.numel(), -1)
            )
            action_pre = model.action_expert.pre_dit(
                action_tokens=executed_actions,
                timestep=torch.zeros(
                    (active.numel(),),
                    device=model.device,
                    dtype=executed_actions.dtype,
                ),
                context=block_context,
                context_mask=block_context_mask,
                position_ids=local_executed_positions,
            )
            action_pre = model.temporal_positions.apply_action_pre_dit(
                action_pre,
                absolute_executed_positions,
                block_positions[:, None].expand(-1, replan_steps),
            )
            _, action_kv = model.mot.prefill_expert_segment(
                expert_name="action",
                tokens=action_pre["tokens"],
                freqs=action_pre["freqs"],
                t_mod=action_pre["t_mod"],
                context_payload={
                    "context": action_pre["context"],
                    "mask": action_pre["context_mask"],
                },
                history_kv=_materialize_incremental_batch(sample_segments, active),
                max_layers=final_depth,
            )
            for local_index, sample_index in enumerate(active.tolist()):
                sample_segments[sample_index].append(
                    {
                        "modality": "action",
                        "kv": [
                            {
                                "k": layer["k"][local_index : local_index + 1],
                                "v": layer["v"][local_index : local_index + 1],
                            }
                            for layer in action_kv
                        ],
                    }
                )

    total_per_sample: list[torch.Tensor] = []
    video_per_sample_all: list[torch.Tensor] = []
    action_per_sample_all: list[torch.Tensor] = []
    for history_count in torch.unique(history_counts, sorted=True).tolist():
        group = torch.nonzero(
            history_counts == int(history_count), as_tuple=False
        ).flatten()
        base_context = inputs["context"].index_select(0, group)
        base_context_mask = inputs["context_mask"].index_select(0, group)
        current_context, current_context_mask = _context_for_proprio(
            model,
            base_context,
            base_context_mask,
            sample["proprio"].to(model.device).index_select(0, group)[:, 0],
        )
        current_latent = noisy_current.index_select(0, group)
        current_video_frames = int(current_latent.shape[2])
        current_video_timestep = timestep_video.index_select(0, group)
        frame_timesteps = current_video_timestep[:, None].expand(
            -1, current_video_frames
        ).clone()
        frame_timesteps[:, 0] = 0
        group_blocks = current_block_positions.index_select(0, group)
        video_pre = model.video_expert.pre_dit(
            x=current_latent,
            timestep=current_video_timestep,
            context=current_context,
            context_mask=current_context_mask,
            action=None,
            fuse_vae_embedding_in_latents=inputs["fuse_vae_embedding_in_latents"],
            frame_position_ids=(
                torch.arange(current_video_frames, device=model.device)
                .view(1, -1)
                .expand(group.numel(), -1)
            ),
            frame_timesteps=frame_timesteps,
        )
        video_pre = model.temporal_positions.apply_video_pre_dit(
            video_pre,
            group_blocks[:, None].expand(-1, current_video_frames),
        )
        tokens_per_frame = int(video_pre["meta"]["tokens_per_frame"])
        video_hidden, current_video_kv = model.mot.prefill_expert_segment(
            expert_name="video",
            tokens=video_pre["tokens"],
            freqs=video_pre["freqs"],
            t_mod=video_pre["t_mod"],
            context_payload={
                "context": video_pre["context"],
                "mask": video_pre["context_mask"],
            },
            history_kv=_video_prefix_batch_for_mode(
                sample_segments, group, model.causal_mode
            ),
            max_layers=final_depth,
            segment_attention_mask=current_video_segment_attention_mask(
                tokens_per_frame=tokens_per_frame,
                num_frames=current_video_frames,
                device=model.device,
            ),
        )
        pred_video = model.video_expert.post_dit(video_hidden, video_pre)
        group_video_loss = model._compute_video_loss_per_sample(
            pred_video[:, :, 1:],
            target_video.index_select(0, group)[:, :, 1:],
            None
            if inputs["image_is_pad"] is None
            else inputs["image_is_pad"].index_select(0, group),
            include_initial_video_step=False,
        )
        group_video_loss = group_video_loss * model.train_video_scheduler.training_weight(
            current_video_timestep
        ).to(group_video_loss.device, dtype=group_video_loss.dtype)

        historical_action_prefix = _materialize_incremental_batch(
            sample_segments, group
        )
        current_real_kv = [
            {
                "k": layer["k"][:, :tokens_per_frame],
                "v": layer["v"][:, :tokens_per_frame],
            }
            for layer in current_video_kv
        ]
        if historical_action_prefix is None:
            action_prefix = current_real_kv
        else:
            action_prefix = [
                {
                    "k": torch.cat(
                        [historical_action_prefix[layer]["k"], current_real_kv[layer]["k"]],
                        dim=1,
                    ),
                    "v": torch.cat(
                        [historical_action_prefix[layer]["v"], current_real_kv[layer]["v"]],
                        dim=1,
                    ),
                }
                for layer in range(final_depth)
            ]

        group_episode_steps = episode_steps.index_select(0, group)
        current_action_positions = group_episode_steps[:, None] + torch.arange(
            action_horizon, device=model.device
        )[None, :]
        local_current_action_positions = (
            torch.arange(action_horizon, device=model.device)
            .view(1, -1)
            .expand(group.numel(), -1)
        )
        action_pre = model.action_expert.pre_dit(
            action_tokens=noisy_action.index_select(0, group),
            timestep=timestep_action.index_select(0, group),
            context=current_context,
            context_mask=current_context_mask,
            position_ids=local_current_action_positions,
        )
        action_pre = model.temporal_positions.apply_action_pre_dit(
            action_pre,
            current_action_positions,
            group_blocks[:, None].expand(-1, action_horizon),
        )
        action_hidden = model.mot.forward_action_with_history(
            action_tokens=action_pre["tokens"],
            action_freqs=action_pre["freqs"],
            action_t_mod=action_pre["t_mod"],
            action_context_payload={
                "context": action_pre["context"],
                "mask": action_pre["context_mask"],
            },
            history_kv=action_prefix,
            max_layers=final_depth,
        )
        pred_action = model.action_expert.post_dit(action_hidden, action_pre)
        action_token_loss = F.mse_loss(
            pred_action.float(),
            target_action.index_select(0, group).float(),
            reduction="none",
        ).mean(dim=2)
        action_is_pad = inputs["action_is_pad"]
        if action_is_pad is not None:
            valid = (~action_is_pad.index_select(0, group)).to(action_token_loss.dtype)
            group_action_loss = (action_token_loss * valid).sum(1) / valid.sum(1).clamp_min(1)
        else:
            group_action_loss = action_token_loss.mean(1)
        group_action_loss = group_action_loss * model.train_action_scheduler.training_weight(
            timestep_action.index_select(0, group)
        ).to(group_action_loss.device, dtype=group_action_loss.dtype)

        total_per_sample.append(
            model.loss_lambda_video * group_video_loss
            + model.loss_lambda_action * group_action_loss
        )
        video_per_sample_all.append(group_video_loss)
        action_per_sample_all.append(group_action_loss)

    total = torch.cat(total_per_sample).mean()
    loss_video_mean = torch.cat(video_per_sample_all).mean()
    loss_action_mean = torch.cat(action_per_sample_all).mean()
    return total, {
        f"loss_video_d{final_depth}": model.loss_lambda_video
        * float(loss_video_mean.detach()),
        f"loss_action_d{final_depth}": model.loss_lambda_action
        * float(loss_action_mean.detach()),
        "history_blocks_mean": float(history_counts.float().mean().detach()),
        "history_blocks_max": float(history_counts.max().detach()),
        "prefix_detached": 1.0,
    }
