"""Packed causal-history attention and multi-exit training for LeapBot-VA."""

from __future__ import annotations

from typing import TYPE_CHECKING

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


def causal_history_training_loss(model: "LeapBotVA", sample, tiled: bool = False):
    """Original video/action flow matching over a clean real-history prefix."""

    inputs = model.build_inputs(sample, tiled=tiled, append_proprio=False)
    current_latents = inputs["input_latents"]
    action = inputs["action"]
    batch = current_latents.shape[0]
    history_valid = sample["history_valid_blocks"].to(model.device, dtype=torch.bool)
    max_history = int(history_valid.shape[1])
    replan_steps = int(sample["history_action"].shape[2])
    action_horizon = int(action.shape[1])

    history_video = sample["history_video"].to(model.device, dtype=model.torch_dtype)
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

    history_action = sample["history_action"].to(model.device, dtype=model.torch_dtype)
    history_action = history_action.reshape(batch, max_history * replan_steps, -1)
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
    history_proprio = sample["history_proprio"].to(model.device, dtype=model.torch_dtype)
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

    history_positions = sample["history_block_positions"].to(model.device).clamp_min(0)
    current_block = sample["current_block_position"].to(model.device).view(batch, 1)
    current_video_frames = int(current_latents.shape[2])
    frame_positions = torch.cat(
        [history_positions, current_block.expand(-1, current_video_frames)], dim=1
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

    history_action_positions = (
        sample["history_block_positions"].to(model.device).unsqueeze(-1) * replan_steps
        + torch.arange(replan_steps, device=model.device).view(1, 1, -1)
    ).clamp_min(0).reshape(batch, -1)
    episode_step = sample["episode_step"].to(model.device).view(batch, 1)
    current_action_positions = episode_step + torch.arange(action_horizon, device=model.device)
    action_positions = torch.cat([history_action_positions, current_action_positions], dim=1)
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
    return total, metrics
