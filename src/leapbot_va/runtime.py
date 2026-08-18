"""Hydra construction helpers for LeapBot-VA."""

from __future__ import annotations

import torch
from omegaconf import DictConfig, OmegaConf


def _as_dict(value, *, default=None):
    if isinstance(value, DictConfig):
        value = OmegaConf.to_container(value, resolve=True)
    if value is None:
        value = {} if default is None else default
    if not isinstance(value, dict):
        raise ValueError(f"expected dict-like config, got {type(value)}")
    return value


def create_leapbot(
    model_id: str,
    tokenizer_model_id: str,
    video_dit_config,
    tokenizer_max_len: int = 512,
    load_text_encoder: bool = True,
    proprio_dim: int | None = None,
    action_dit_config=None,
    action_dit_pretrained_path: str | None = None,
    skip_dit_load_from_pretrain: bool = False,
    video_scheduler=None,
    action_scheduler=None,
    loss=None,
    mot_checkpoint_mixed_attn: bool = True,
    redirect_common_files: bool = True,
    exit_depths=(8, 16, 24, 30),
    causal_mode: str = "interleaved",
    training_exit_depths=(30,),
    history_training_mode: str = "incremental_full_bptt",
    packed_history_attention_backend: str = "dense",
    history_window_blocks: int = 8,
    episode_memory=None,
    history_vae_batch_chunk_size: int = 1,
    replan_steps: int = 10,
    action_horizon: int = 32,
    num_video_frames: int = 9,
    future_video_conditioning: str = "lingbot_teacher_forced_v1",
    future_video_condition_noise_probability: float = 0.5,
    future_video_condition_min_u: float = 0.5,
    future_video_condition_max_u: float = 1.0,
    future_video_condition_clean_warmup_steps: int = 0,
    future_video_condition_noise_ramp_steps: int = 0,
    future_video_denoise_steps: int = -1,
    training_strategy: str = "full_dit",
    video_lora=None,
    model_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
):
    from .models.leapbot import LeapBotVA

    video_dit_config = _as_dict(video_dit_config)
    action_dit_config = _as_dict(action_dit_config)
    video_scheduler = _as_dict(video_scheduler)
    action_scheduler = _as_dict(action_scheduler)
    loss = _as_dict(loss)
    video_lora = _as_dict(video_lora)
    episode_memory = _as_dict(episode_memory)
    required = {"train_shift", "infer_shift", "num_train_timesteps"}
    missing = required - set(action_scheduler)
    if missing:
        raise ValueError(f"action_scheduler missing keys: {sorted(missing)}")

    model = LeapBotVA.from_wan22_pretrained(
        device=device,
        torch_dtype=model_dtype,
        model_id=model_id,
        tokenizer_model_id=tokenizer_model_id,
        tokenizer_max_len=int(tokenizer_max_len),
        load_text_encoder=bool(load_text_encoder),
        proprio_dim=None if proprio_dim is None else int(proprio_dim),
        redirect_common_files=bool(redirect_common_files),
        video_dit_config=video_dit_config,
        action_dit_config=action_dit_config,
        action_dit_pretrained_path=action_dit_pretrained_path,
        skip_dit_load_from_pretrain=bool(skip_dit_load_from_pretrain),
        mot_checkpoint_mixed_attn=bool(mot_checkpoint_mixed_attn),
        video_train_shift=float(video_scheduler.get("train_shift", 5.0)),
        video_infer_shift=float(video_scheduler.get("infer_shift", 5.0)),
        video_num_train_timesteps=int(video_scheduler.get("num_train_timesteps", 1000)),
        action_train_shift=float(action_scheduler["train_shift"]),
        action_infer_shift=float(action_scheduler["infer_shift"]),
        action_num_train_timesteps=int(action_scheduler["num_train_timesteps"]),
        loss_lambda_video=float(loss.get("lambda_video", 1.0)),
        loss_lambda_action=float(loss.get("lambda_action", 1.0)),
    )
    requested_depths = tuple(int(depth) for depth in exit_depths)
    if requested_depths != model.exit_depths:
        raise ValueError(
            "exit_depths are fixed during construction; "
            f"requested {requested_depths}, model created {model.exit_depths}"
        )
    from .lora import VideoLoRAConfig

    model.configure_finetuning(
        training_strategy=str(training_strategy),
        video_lora_config=VideoLoRAConfig(
            enabled=bool(video_lora.get("enabled", False)),
            rank=int(video_lora.get("rank", 16)),
            alpha=float(video_lora.get("alpha", 16.0)),
            dropout=float(video_lora.get("dropout", 0.0)),
            learning_rate_multiplier=float(
                video_lora.get("learning_rate_multiplier", 10.0)
            ),
        ),
    )
    from .episode_memory import EpisodeMemoryConfig

    resolved_episode_memory = EpisodeMemoryConfig(
        enabled=bool(episode_memory.get("enabled", False)),
        window_blocks=int(episode_memory.get("window_blocks", 8)),
        chunk_blocks=int(episode_memory.get("chunk_blocks", 4)),
        num_slots=int(episode_memory.get("num_slots", 32)),
        state_dim=int(episode_memory.get("state_dim", 1024)),
        group_dim=int(episode_memory.get("group_dim", 16)),
        updater_dim=int(episode_memory.get("updater_dim", 256)),
        updater_heads=int(episode_memory.get("updater_heads", 8)),
        reader_rank=int(episode_memory.get("reader_rank", 64)),
        video_reads=episode_memory.get("video_reads"),
        action_reads=bool(episode_memory.get("action_reads", True)),
    )
    model.configure_causal_training(
        causal_mode=str(causal_mode),
        training_exit_depths=tuple(int(depth) for depth in training_exit_depths),
        history_training_mode=str(history_training_mode),
        packed_history_attention_backend=str(packed_history_attention_backend),
        history_window_blocks=int(history_window_blocks),
        replan_steps=int(replan_steps),
        action_horizon=int(action_horizon),
        num_video_frames=int(num_video_frames),
        future_video_condition_noise_probability=float(
            future_video_condition_noise_probability
        ),
        future_video_condition_min_u=float(future_video_condition_min_u),
        future_video_condition_max_u=float(future_video_condition_max_u),
        future_video_condition_clean_warmup_steps=int(
            future_video_condition_clean_warmup_steps
        ),
        future_video_condition_noise_ramp_steps=int(
            future_video_condition_noise_ramp_steps
        ),
        future_video_denoise_steps=int(future_video_denoise_steps),
        episode_memory_config=resolved_episode_memory,
    )
    if str(future_video_conditioning) != model.future_video_conditioning:
        raise ValueError(
            "unsupported future-video conditioning contract: "
            f"{future_video_conditioning}"
        )
    resolved_history_vae_chunk = int(history_vae_batch_chunk_size)
    if resolved_history_vae_chunk <= 0:
        raise ValueError("history_vae_batch_chunk_size must be positive")
    model.history_vae_batch_chunk_size = resolved_history_vae_chunk
    return model
