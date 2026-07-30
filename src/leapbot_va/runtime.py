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
    model_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
):
    from .models.leapbot import LeapBotVA

    video_dit_config = _as_dict(video_dit_config)
    action_dit_config = _as_dict(action_dit_config)
    video_scheduler = _as_dict(video_scheduler)
    action_scheduler = _as_dict(action_scheduler)
    loss = _as_dict(loss)
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
    model.configure_causal_training(
        causal_mode=str(causal_mode),
        training_exit_depths=tuple(int(depth) for depth in training_exit_depths),
    )
    return model
