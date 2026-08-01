"""Validate the real 6B incremental full-BPTT path on one LIBERO episode."""

from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path
from typing import Any

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data._utils.collate import default_collate

from fastwam.runtime import _mixed_precision_to_model_dtype, _normalize_mixed_precision
from fastwam.trainer import Wan22Trainer
from fastwam.utils import misc
from fastwam.utils.config_resolvers import register_default_resolvers
from fastwam.utils.logging_config import get_logger, setup_logging


register_default_resolvers()
logger = get_logger(__name__)


RUNTIME_OBSERVATION_VAE_CONTRACT = (
    "batch1_t1_via__encode_input_image_latents_tensor"
)
FUTURE_VIDEO_VAE_CONTRACT = (
    "full_clip_encode_shared_by_video_target_and_teacher_forced_action_condition"
)
FORMAL_REAL_HISTORY_BLOCKS = 50
FORMAL_CAPACITY_HISTORY_BLOCKS = 70
FORMAL_REPLAN_STEPS = 10
FORMAL_ACTION_HORIZON = 32


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def _validate_smoke_contract(cfg: DictConfig) -> dict[str, Any]:
    """Resolve a smoke configuration without touching a checkpoint or CUDA."""

    smoke = cfg.get("smoke")
    if smoke is None:
        raise ValueError("missing +smoke configuration")
    batch_size = int(cfg.get("batch_size", -1))
    if batch_size <= 0:
        raise ValueError("incremental_full_bptt smoke requires a positive batch size")
    batch_repeats = int(smoke.get("batch_repeats", 1))
    if batch_repeats != 1:
        raise ValueError(
            "smoke.batch_repeats=1 is required; batch capacity must use distinct "
            "real episode prefixes through smoke.history_blocks_per_sample"
        )
    history_blocks_per_sample_raw = smoke.get("history_blocks_per_sample")
    if history_blocks_per_sample_raw is None:
        history_blocks_per_sample = None
        if batch_size != 1:
            raise ValueError(
                "cfg.batch_size=1 is required unless "
                "smoke.history_blocks_per_sample is provided"
            )
    else:
        raw_history_values = list(history_blocks_per_sample_raw)
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in raw_history_values
        ):
            raise ValueError(
                "smoke.history_blocks_per_sample must contain only integers"
            )
        history_blocks_per_sample = [int(value) for value in raw_history_values]
        if len(history_blocks_per_sample) != batch_size:
            raise ValueError(
                "smoke.history_blocks_per_sample must contain exactly one history "
                f"length per batch row: batch={batch_size} "
                f"lengths={len(history_blocks_per_sample)}"
            )
        if history_blocks_per_sample != sorted(set(history_blocks_per_sample)):
            raise ValueError(
                "smoke.history_blocks_per_sample must be strictly increasing and unique"
            )
        if not history_blocks_per_sample or history_blocks_per_sample[0] < 0:
            raise ValueError("smoke history lengths must be non-negative")
        if history_blocks_per_sample[-1] > FORMAL_REAL_HISTORY_BLOCKS:
            raise ValueError(
                "optimizer topology smoke may use only real released prefixes H<=50"
            )

    model_cfg = cfg.get("model")
    if model_cfg is None:
        raise ValueError("missing model configuration")
    history_training_mode = str(model_cfg.get("history_training_mode", ""))
    if history_training_mode != "incremental_full_bptt":
        raise ValueError(
            "smoke requires model.history_training_mode=incremental_full_bptt"
        )
    history_vae_chunk = int(model_cfg.get("history_vae_batch_chunk_size", -1))
    if history_vae_chunk != 1:
        raise ValueError(
            "runtime-equivalent observation encoding requires "
            "model.history_vae_batch_chunk_size=1"
        )
    training_strategy = str(model_cfg.get("training_strategy", ""))
    video_lora_cfg = model_cfg.get("video_lora")
    video_lora_enabled = bool(
        video_lora_cfg is not None and video_lora_cfg.get("enabled", False)
    )
    replan_steps = int(model_cfg.get("replan_steps", -1))
    action_horizon = int(model_cfg.get("action_horizon", -1))
    num_video_frames = int(model_cfg.get("num_video_frames", -1))
    if (replan_steps, action_horizon) != (
        FORMAL_REPLAN_STEPS,
        FORMAL_ACTION_HORIZON,
    ):
        raise ValueError(
            "formal LIBERO-Long smoke requires replan_steps/action_horizon=10/32, "
            f"got {replan_steps}/{action_horizon}"
        )
    conditioning_contract = str(
        model_cfg.get("future_video_conditioning", "lingbot_teacher_forced_v1")
    )
    condition_noise_probability = float(
        model_cfg.get("future_video_condition_noise_probability", -1.0)
    )
    condition_min_u = float(
        model_cfg.get("future_video_condition_min_u", -1.0)
    )
    condition_max_u = float(
        model_cfg.get("future_video_condition_max_u", -1.0)
    )
    if (
        conditioning_contract != "lingbot_teacher_forced_v1"
        or num_video_frames != 9
        or condition_noise_probability != 0.5
        or condition_min_u != 0.5
        or condition_max_u != 1.0
    ):
        raise ValueError(
            "formal smoke requires LingBot teacher forcing, 9 video frames, "
            "noise probability 0.5, and u=[0.5,1.0]"
        )

    data_cfg = cfg.get("data")
    train_data_cfg = None if data_cfg is None else data_cfg.get("train")
    if train_data_cfg is None:
        raise ValueError("missing data.train configuration")
    if not bool(train_data_cfg.get("full_episode_history", False)):
        raise ValueError(
            "full-prefix smoke requires data.train.full_episode_history=true"
        )
    data_replan_steps = int(train_data_cfg.get("replan_steps", -1))
    if data_replan_steps != replan_steps:
        raise ValueError(
            "dataset/model replan step mismatch: "
            f"data={data_replan_steps} model={replan_steps}"
        )
    capacity = int(train_data_cfg.get("max_history_blocks", -1))

    target_history_blocks = int(smoke.get("history_blocks", FORMAL_REAL_HISTORY_BLOCKS))
    if target_history_blocks < 0:
        raise ValueError("smoke.history_blocks must be non-negative")
    synthetic_source_raw = smoke.get("synthetic_source_history_blocks")
    synthetic_source = (
        None if synthetic_source_raw is None else int(synthetic_source_raw)
    )
    if history_blocks_per_sample is not None:
        if synthetic_source is not None:
            raise ValueError(
                "optimizer topology smoke cannot synthesize or extend history"
            )
        optimizer_updates = int(smoke.get("optimizer_updates", -1))
        microbatches_per_update = int(
            smoke.get("microbatches_per_update", -1)
        )
        if optimizer_updates != 2:
            raise ValueError(
                "optimizer topology smoke requires exactly two optimizer updates"
            )
        if microbatches_per_update < 2:
            raise ValueError(
                "optimizer topology smoke requires at least two microbatches per update"
            )
        if training_strategy != "video_lora_action_full" or not video_lora_enabled:
            raise ValueError(
                "optimizer topology smoke requires the formal "
                "video_lora_action_full strategy with video LoRA enabled"
            )
        target_history_blocks = max(history_blocks_per_sample)
        selection_history_blocks: int | list[int] = history_blocks_per_sample
        history_provenance = "distinct_real_episode_prefixes"
        smoke_profile = f"real_optimizer_topology_b{batch_size}"
        measurement_scope = "optimizer_memory_topology_only_not_loss_or_quality"
    elif synthetic_source is None:
        optimizer_updates = 0
        microbatches_per_update = 1
        if target_history_blocks > FORMAL_REAL_HISTORY_BLOCKS:
            raise ValueError(
                "released LIBERO data provides at most a real H50 prefix; "
                "H70 must set synthetic_source_history_blocks=50 and is capacity-only"
            )
        selection_history_blocks = target_history_blocks
        history_provenance = "real_episode_prefix"
        smoke_profile = (
            "real_h50"
            if target_history_blocks == FORMAL_REAL_HISTORY_BLOCKS
            else f"diagnostic_real_h{target_history_blocks}"
        )
        measurement_scope = "training_path_smoke"
    else:
        optimizer_updates = 0
        microbatches_per_update = 1
        if (
            synthetic_source != FORMAL_REAL_HISTORY_BLOCKS
            or target_history_blocks != FORMAL_CAPACITY_HISTORY_BLOCKS
        ):
            raise ValueError(
                "the only supported synthetic smoke is real H50 extended to H70"
            )
        selection_history_blocks = synthetic_source
        history_provenance = "synthetic_capacity_extension_from_real_h50"
        smoke_profile = "synthetic_h70_capacity"
        measurement_scope = "capacity_oom_only_not_loss_or_quality"
    if capacity < target_history_blocks:
        raise ValueError(
            "data.train.max_history_blocks is smaller than the requested smoke: "
            f"{capacity}<{target_history_blocks}"
        )

    precision = _normalize_mixed_precision(str(cfg.get("mixed_precision", "")))
    if precision != "bf16":
        raise ValueError(
            f"formal H800 smoke requires mixed_precision=bf16, got {precision}"
        )
    device = str(smoke.get("device", "cuda:0"))
    try:
        parsed_device = torch.device(device)
    except (TypeError, RuntimeError) as error:
        raise ValueError(f"invalid smoke.device: {device}") from error
    if parsed_device.type != "cuda":
        raise ValueError("the real 6B smoke requires a CUDA device")
    if bool(smoke.get("tiled", False)):
        raise ValueError(
            "formal rollout-equivalent observation encoding requires smoke.tiled=false"
        )

    return {
        "smoke_profile": smoke_profile,
        "measurement_scope": measurement_scope,
        "batch_size": batch_size,
        "history_blocks": target_history_blocks,
        "selection_history_blocks": selection_history_blocks,
        "history_blocks_per_sample": history_blocks_per_sample,
        "real_history_blocks": selection_history_blocks,
        "history_provenance": history_provenance,
        "is_synthetic_capacity_smoke": synthetic_source is not None,
        "is_optimizer_topology_smoke": history_blocks_per_sample is not None,
        "optimizer_updates": optimizer_updates,
        "microbatches_per_update": microbatches_per_update,
        "synthetic_source_history_blocks": synthetic_source,
        "configured_max_history_blocks": capacity,
        "history_training_mode": history_training_mode,
        "history_vae_batch_chunk_size": history_vae_chunk,
        "training_strategy": training_strategy,
        "video_lora_enabled": video_lora_enabled,
        "runtime_observation_vae_contract": RUNTIME_OBSERVATION_VAE_CONTRACT,
        "future_video_vae_contract": FUTURE_VIDEO_VAE_CONTRACT,
        "future_video_conditioning": conditioning_contract,
        "future_video_condition_noise_probability": condition_noise_probability,
        "future_video_condition_min_u": condition_min_u,
        "future_video_condition_max_u": condition_max_u,
        "num_video_frames": num_video_frames,
        "replan_steps": replan_steps,
        "action_horizon": action_horizon,
        "mixed_precision": precision,
        "device": str(parsed_device),
    }


def _extend_history_for_capacity_smoke(
    sample: dict[str, torch.Tensor],
    *,
    source_history_blocks: int,
    target_history_blocks: int,
    replan_steps: int,
) -> dict[str, int]:
    """Extend a real prefix only for worst-case capacity/memory validation.

    LIBERO's longest released episode has 50 replanning blocks, while LeapBot
    reserves 70.  A 70-block smoke therefore cannot be a real loss benchmark.
    This helper repeats the final real history block into the unused fixed
    slots and advances only temporal metadata.  It is deliberately limited to
    forward/backward, mask, shape, and OOM validation; callers record the
    synthetic provenance in the result JSON.
    """

    if source_history_blocks <= 0:
        raise ValueError("synthetic capacity smoke requires a non-empty real prefix")
    if target_history_blocks <= source_history_blocks:
        raise ValueError("target synthetic history must exceed the real source history")
    if replan_steps <= 0:
        raise ValueError("replan_steps must be positive")
    source_counts = sample["history_valid_blocks"].sum(dim=1)
    if not bool((source_counts == source_history_blocks).all().item()):
        raise ValueError(
            "source history mismatch: "
            f"expected {source_history_blocks}, got {source_counts.tolist()}"
        )
    capacity = int(sample["history_valid_blocks"].shape[1])
    if target_history_blocks > capacity:
        raise ValueError(
            f"synthetic target exceeds configured capacity: {target_history_blocks}>{capacity}"
        )

    video = sample["history_video"].clone()
    video[:, :, source_history_blocks:target_history_blocks] = video[
        :, :, source_history_blocks - 1 : source_history_blocks
    ].expand(-1, -1, target_history_blocks - source_history_blocks, -1, -1)
    sample["history_video"] = video

    for key in ("history_action", "history_proprio"):
        value = sample[key].clone()
        value[:, source_history_blocks:target_history_blocks] = value[
            :, source_history_blocks - 1 : source_history_blocks
        ].expand(-1, target_history_blocks - source_history_blocks, *value.shape[2:])
        sample[key] = value

    history_valid = torch.zeros_like(sample["history_valid_blocks"])
    history_valid[:, :target_history_blocks] = True
    sample["history_valid_blocks"] = history_valid
    history_positions = torch.full_like(sample["history_block_positions"], -1)
    history_positions[:, :target_history_blocks] = torch.arange(
        target_history_blocks,
        dtype=history_positions.dtype,
        device=history_positions.device,
    )
    sample["history_block_positions"] = history_positions
    sample["current_block_position"] = torch.full_like(
        sample["current_block_position"], target_history_blocks
    )
    sample["episode_step"] = torch.full_like(
        sample["episode_step"], target_history_blocks * replan_steps
    )
    return {
        "source_history_blocks": source_history_blocks,
        "target_history_blocks": target_history_blocks,
        "repeated_source_block": source_history_blocks - 1,
    }


def _finite_scalar_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    resolved: dict[str, float] = {}
    for name, value in metrics.items():
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise AssertionError(f"metric {name} is not scalar: {tuple(value.shape)}")
            scalar = float(value.detach().float().item())
        else:
            scalar = float(value)
        if not math.isfinite(scalar):
            raise AssertionError(f"metric {name} is non-finite: {scalar}")
        resolved[str(name)] = scalar
    return resolved


def _select_distinct_real_history_indices(
    dataset: Any,
    history_blocks_per_sample: list[int],
) -> list[int]:
    """Select one genuine, distinct dataset row for every requested prefix."""

    remaining = set(int(value) for value in history_blocks_per_sample)
    selected: dict[int, int] = {}
    for dataset_index, frame_index in enumerate(dataset._valid_replan_indices):
        history_blocks = int(
            dataset._episode_step[frame_index] // dataset.replan_steps
        )
        if history_blocks in remaining:
            selected[history_blocks] = int(dataset_index)
            remaining.remove(history_blocks)
            if not remaining:
                break
    if remaining:
        missing = sorted(remaining)
        raise ValueError(f"dataset contains no real samples for history lengths {missing}")
    result = [selected[int(value)] for value in history_blocks_per_sample]
    if len(result) != len(set(result)):
        raise AssertionError("distinct history requests resolved to duplicate dataset rows")
    return result


def _formal_adamw(model: torch.nn.Module, cfg: DictConfig) -> torch.optim.AdamW:
    """Build the same local AdamW parameter groups used by formal training."""

    learning_rate = float(cfg.learning_rate)
    weight_decay = float(cfg.weight_decay)
    group_getter = getattr(model, "optimizer_parameter_groups", None)
    if group_getter is None:
        trainable_parameters: Any = [
            parameter for parameter in model.parameters() if parameter.requires_grad
        ]
    else:
        trainable_parameters = group_getter(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )
    return torch.optim.AdamW(
        trainable_parameters,
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.95),
    )


def _gradient_statistics(
    trainable: list[torch.nn.Parameter],
) -> dict[str, int | float | bool]:
    gradients = [parameter.grad for parameter in trainable if parameter.grad is not None]
    if not gradients:
        raise AssertionError("no trainable parameter received a gradient")
    finite_gradients = True
    grad_norm_squared = 0.0
    max_abs_gradient = 0.0
    for gradient in gradients:
        finite_gradients = finite_gradients and bool(
            torch.isfinite(gradient).all().item()
        )
        gradient_float = gradient.detach().float()
        grad_norm_squared += float(gradient_float.square().sum().item())
        max_abs_gradient = max(
            max_abs_gradient,
            float(gradient_float.abs().max().item()),
        )
    if not finite_gradients:
        raise AssertionError("non-finite gradient in full-prefix smoke test")
    gradient_l2_norm = math.sqrt(grad_norm_squared)
    if not math.isfinite(gradient_l2_norm) or not math.isfinite(
        max_abs_gradient
    ):
        raise AssertionError("non-finite aggregate gradient statistics")
    return {
        "parameters_with_grad": int(
            sum(parameter.numel() for parameter in trainable if parameter.grad is not None)
        ),
        "gradient_tensors": len(gradients),
        "finite_gradients": finite_gradients,
        "gradient_l2_norm": gradient_l2_norm,
        "max_abs_gradient": max_abs_gradient,
    }


def _cuda_memory_snapshot(device: torch.device) -> dict[str, float]:
    return {
        "allocated_gib": float(torch.cuda.memory_allocated(device) / 2**30),
        "reserved_gib": float(torch.cuda.memory_reserved(device) / 2**30),
        "peak_allocated_gib": float(
            torch.cuda.max_memory_allocated(device) / 2**30
        ),
        "peak_reserved_gib": float(torch.cuda.max_memory_reserved(device) / 2**30),
    }


def run_smoke(cfg: DictConfig) -> dict:
    contract = _validate_smoke_contract(cfg)
    smoke = cfg.smoke
    output_dir = Path(str(cfg.output_dir)).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output_dir / "config.yaml", resolve=True)
    output_path = output_dir / "full_prefix_smoke.json"

    if bool(smoke.get("validate_only", False)):
        result = {
            "kind": "incremental_full_bptt_smoke_config",
            "status": "validated_without_dataset_checkpoint_or_cuda",
            "contract": contract,
        }
        _write_json(output_path, result)
        logger.info("Full-prefix smoke configuration is valid: %s", output_path)
        return result

    checkpoint_raw = smoke.get("checkpoint")
    if checkpoint_raw is None:
        raise ValueError("execution requires +smoke.checkpoint=/path/to/checkpoint.pt")
    checkpoint = Path(str(checkpoint_raw)).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use smoke.validate_only=true for a dry run")

    cuda_device = torch.device(contract["device"])
    torch.cuda.set_device(cuda_device)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError(f"BF16 is not supported by {cuda_device}")
    misc.register_work_dir(str(output_dir))

    target_history_blocks = int(contract["history_blocks"])
    selection_history_raw = contract["selection_history_blocks"]
    selection_history_blocks = (
        [int(value) for value in selection_history_raw]
        if isinstance(selection_history_raw, list)
        else [int(selection_history_raw)]
    )
    dataset = instantiate(cfg.data.train)
    if int(dataset.replan_steps) != int(contract["replan_steps"]):
        raise AssertionError(
            "instantiated dataset replan_steps differs from the validated config"
        )
    selected_indices = _select_distinct_real_history_indices(
        dataset,
        selection_history_blocks,
    )
    # The production loss executes each row through an independent batch-one
    # episode-prefix graph.  Collation only retains those graphs until their
    # mean loss is backpropagated together.
    sample = default_collate([dataset[index] for index in selected_indices])
    source_counts = sample["history_valid_blocks"].sum(dim=1)
    if source_counts.tolist() != selection_history_blocks:
        raise AssertionError(
            "selected dataset sample does not contain its complete real prefix: "
            f"expected H={selection_history_blocks}, got {source_counts.tolist()}"
        )
    if sample["current_block_position"].tolist() != selection_history_blocks:
        raise AssertionError("current block is not contiguous with the real prefix")
    expected_episode_steps = [
        value * int(dataset.replan_steps) for value in selection_history_blocks
    ]
    if sample["episode_step"].tolist() != expected_episode_steps:
        raise AssertionError("episode_step is inconsistent with the real prefix")
    full_episode_history = sample.get("full_episode_history")
    if full_episode_history is None or not bool(full_episode_history.all().item()):
        raise AssertionError("dataset sample is not marked as full_episode_history")

    synthetic_history_extension = None
    synthetic_source = contract["synthetic_source_history_blocks"]
    if synthetic_source is not None:
        if len(selected_indices) != 1:
            raise AssertionError("synthetic capacity extension must remain batch one")
        synthetic_history_extension = _extend_history_for_capacity_smoke(
            sample,
            source_history_blocks=int(synthetic_source),
            target_history_blocks=target_history_blocks,
            replan_steps=int(dataset.replan_steps),
        )
    actual_history_counts = sample["history_valid_blocks"].sum(dim=1)
    expected_actual_histories = (
        [target_history_blocks]
        if synthetic_source is not None
        else selection_history_blocks
    )
    if actual_history_counts.tolist() != expected_actual_histories:
        raise AssertionError(
            (actual_history_counts.tolist(), expected_actual_histories)
        )

    precision = str(contract["mixed_precision"])
    dtype = _mixed_precision_to_model_dtype(precision)
    model = instantiate(cfg.model, model_dtype=dtype, device=str(cuda_device))
    model.load_checkpoint(str(checkpoint), optimizer=None)
    Wan22Trainer._apply_dit_only_train_mode(model)
    if model.history_training_mode != contract["history_training_mode"]:
        raise AssertionError("instantiated model history_training_mode mismatch")
    if int(model.history_vae_batch_chunk_size) != 1:
        raise AssertionError("instantiated model does not use batch-one observation VAE")
    if model.future_video_conditioning != contract["future_video_conditioning"]:
        raise AssertionError("instantiated model future-video contract mismatch")
    if int(model.training_num_video_frames) != int(contract["num_video_frames"]):
        raise AssertionError("instantiated model video-frame contract mismatch")
    model.validate_temporal_contract(
        replan_steps=int(contract["replan_steps"]),
        action_horizon=int(contract["action_horizon"]),
    )

    runtime_observation_shapes: list[tuple[int, ...]] = []
    runtime_observation_dtypes: list[str] = []
    runtime_observation_devices: list[str] = []
    future_video_shapes: list[tuple[int, ...]] = []
    original_observation_encoder = model._encode_input_image_latents_tensor
    original_video_encoder = model._encode_video_latents

    def recording_observation_encoder(*args, **kwargs):
        image = args[0] if args else kwargs.get("input_image")
        if not isinstance(image, torch.Tensor):
            raise AssertionError("runtime observation encoder did not receive a tensor")
        runtime_observation_shapes.append(tuple(int(size) for size in image.shape))
        runtime_observation_dtypes.append(str(image.dtype))
        runtime_observation_devices.append(str(image.device))
        return original_observation_encoder(*args, **kwargs)

    def recording_video_encoder(*args, **kwargs):
        video = args[0] if args else kwargs.get("video_tensor")
        if not isinstance(video, torch.Tensor):
            raise AssertionError("future-video encoder did not receive a tensor")
        future_video_shapes.append(tuple(int(size) for size in video.shape))
        return original_video_encoder(*args, **kwargs)

    seed = int(cfg.seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    is_optimizer_topology_smoke = bool(contract["is_optimizer_topology_smoke"])
    optimizer = _formal_adamw(model, cfg) if is_optimizer_topology_smoke else None
    optimizer_updates = int(contract["optimizer_updates"])
    microbatches_per_update = int(contract["microbatches_per_update"])
    execution_updates = optimizer_updates if is_optimizer_topology_smoke else 1
    execution_microbatches = (
        microbatches_per_update if is_optimizer_topology_smoke else 1
    )
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    torch.cuda.empty_cache()
    baseline_memory = {
        "allocated_gib": float(torch.cuda.memory_allocated(cuda_device) / 2**30),
        "reserved_gib": float(torch.cuda.memory_reserved(cuda_device) / 2**30),
    }
    torch.cuda.synchronize(cuda_device)

    phase = "forward"
    update_records: list[dict[str, Any]] = []
    loss_values: list[float] = []
    metric_values: list[dict[str, float]] = []
    forward_s = 0.0
    backward_s = 0.0
    gradient_statistics: dict[str, int | float | bool] | None = None
    try:
        model._encode_input_image_latents_tensor = recording_observation_encoder
        model._encode_video_latents = recording_video_encoder
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        for update_index in range(execution_updates):
            torch.cuda.reset_peak_memory_stats(cuda_device)
            update_forward_s = 0.0
            update_backward_s = 0.0
            update_loss_values: list[float] = []
            clip_grad_norm = None
            for microbatch_index in range(execution_microbatches):
                phase = f"forward_update_{update_index + 1}_microbatch_{microbatch_index + 1}"
                forward_start = time.perf_counter()
                with torch.autocast(device_type="cuda", dtype=dtype, enabled=True):
                    loss, raw_metrics = model.training_loss(sample, tiled=False)
                torch.cuda.synchronize(cuda_device)
                elapsed_forward = time.perf_counter() - forward_start
                forward_s += elapsed_forward
                update_forward_s += elapsed_forward

                if not isinstance(loss, torch.Tensor) or loss.numel() != 1:
                    raise AssertionError("training loss must be a scalar tensor")
                loss_value = float(loss.detach().float().item())
                if not math.isfinite(loss_value):
                    raise AssertionError(f"non-finite training loss: {loss_value}")
                metrics = _finite_scalar_metrics(raw_metrics)
                loss_values.append(loss_value)
                update_loss_values.append(loss_value)
                metric_values.append(metrics)

                phase = f"backward_update_{update_index + 1}_microbatch_{microbatch_index + 1}"
                backward_start = time.perf_counter()
                scaled_loss = loss / float(execution_microbatches)
                scaled_loss.backward()
                torch.cuda.synchronize(cuda_device)
                elapsed_backward = time.perf_counter() - backward_start
                backward_s += elapsed_backward
                update_backward_s += elapsed_backward

            if update_index == execution_updates - 1:
                gradient_statistics = _gradient_statistics(trainable)
            if optimizer is not None:
                phase = f"optimizer_update_{update_index + 1}"
                clip_result = torch.nn.utils.clip_grad_norm_(
                    trainable,
                    float(cfg.max_grad_norm),
                    error_if_nonfinite=True,
                )
                clip_grad_norm = float(clip_result.detach().float().item())
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.synchronize(cuda_device)
            update_records.append(
                {
                    "update": update_index + 1,
                    "microbatches": execution_microbatches,
                    "loss_mean": sum(update_loss_values) / len(update_loss_values),
                    "forward_s": update_forward_s,
                    "backward_s": update_backward_s,
                    "clip_grad_norm": clip_grad_norm,
                    "gpu_memory": _cuda_memory_snapshot(cuda_device),
                }
            )
    except torch.cuda.OutOfMemoryError as error:
        oom_result = {
            "kind": "incremental_full_bptt_training_smoke",
            "status": "oom",
            "failed_phase": phase,
            "error": str(error),
            "checkpoint": str(checkpoint),
            "contract": contract,
            "baseline_gpu_memory": baseline_memory,
            "gpu_memory_at_failure": _cuda_memory_snapshot(cuda_device),
        }
        if synthetic_history_extension is not None:
            oom_result["synthetic_history_extension"] = synthetic_history_extension
        _write_json(output_path, oom_result)
        logger.exception("Full-prefix smoke OOM during %s; report: %s", phase, output_path)
        raise
    finally:
        model._encode_input_image_latents_tensor = original_observation_encoder
        model._encode_video_latents = original_video_encoder

    if gradient_statistics is None:
        gradient_statistics = _gradient_statistics(trainable)
    loss_value = sum(loss_values) / len(loss_values)
    metric_names = set(metric_values[0])
    if any(set(values) != metric_names for values in metric_values):
        raise AssertionError("training metric keys changed during topology smoke")
    metrics = {
        name: sum(values[name] for values in metric_values) / len(metric_values)
        for name in sorted(metric_names)
    }
    timed_memory = dict(update_records[-1]["gpu_memory"])
    timed_memory["peak_allocated_gib"] = max(
        float(record["gpu_memory"]["peak_allocated_gib"])
        for record in update_records
    )
    timed_memory["peak_reserved_gib"] = max(
        float(record["gpu_memory"]["peak_reserved_gib"])
        for record in update_records
    )
    optimizer_groups = []
    if optimizer is not None:
        optimizer_groups = [
            {
                "name": str(group.get("group_name", f"group_{index}")),
                "learning_rate": float(group["lr"]),
                "weight_decay": float(group["weight_decay"]),
                "parameters": int(
                    sum(parameter.numel() for parameter in group["params"])
                ),
            }
            for index, group in enumerate(optimizer.param_groups)
        ]

    expected_observation_calls_per_microbatch = sum(
        history_blocks + 1 for history_blocks in expected_actual_histories
    )
    expected_observation_calls = (
        expected_observation_calls_per_microbatch
        * execution_microbatches
        * execution_updates
    )
    expected_image_shape = (
        1,
        3,
        int(sample["video"].shape[-2]),
        int(sample["video"].shape[-1]),
    )
    if len(runtime_observation_shapes) != expected_observation_calls:
        raise AssertionError(
            "runtime observation VAE call count mismatch: "
            f"expected {expected_observation_calls}, got {len(runtime_observation_shapes)}"
        )
    if any(shape != expected_image_shape for shape in runtime_observation_shapes):
        raise AssertionError(
            "every real observation must use the online [1,3,H,W] encoder call; "
            f"expected {expected_image_shape}, got {sorted(set(runtime_observation_shapes))}"
        )
    expected_observation_dtype = str(model.torch_dtype)
    expected_observation_device = str(torch.device(model.device))
    if set(runtime_observation_dtypes) != {expected_observation_dtype}:
        raise AssertionError(
            "runtime observation VAE dtype mismatch: "
            f"expected {expected_observation_dtype}, got {sorted(set(runtime_observation_dtypes))}"
        )
    if set(runtime_observation_devices) != {expected_observation_device}:
        raise AssertionError(
            "runtime observation VAE device mismatch: "
            f"expected {expected_observation_device}, "
            f"got {sorted(set(runtime_observation_devices))}"
        )
    expected_future_video_shape = tuple(int(size) for size in sample["video"].shape)
    expected_future_video_calls = execution_updates * execution_microbatches
    if future_video_shapes != [expected_future_video_shape] * expected_future_video_calls:
        raise AssertionError(
            "future-video target/condition must share exactly one full-clip VAE "
            "call per microbatch: "
            f"expected {expected_future_video_calls} calls with shape "
            f"{expected_future_video_shape}, got {future_video_shapes}"
        )

    result = {
        "kind": "incremental_full_bptt_training_smoke",
        "status": "passed",
        "checkpoint": str(checkpoint),
        "dataset_indices": selected_indices,
        "dataset_index": selected_indices[0] if len(selected_indices) == 1 else None,
        "batch_size": int(contract["batch_size"]),
        "history_blocks": target_history_blocks,
        "history_blocks_per_sample": expected_actual_histories,
        "history_provenance": contract["history_provenance"],
        "measurement_scope": contract["measurement_scope"],
        "contract": contract,
        "causal_mode": model.causal_mode,
        "training_strategy": model.training_strategy,
        "video_lora_config": model.video_lora_config.__dict__,
        "training_exit_depths": [int(depth) for depth in model.training_exit_depths],
        "history_training_mode": model.history_training_mode,
        "loss": loss_value,
        "metrics": metrics,
        "timing_s": {
            "forward": forward_s,
            "backward": backward_s,
            "forward_backward": forward_s + backward_s,
        },
        "optimizer_topology": {
            "enabled": optimizer is not None,
            "optimizer": "adamw_beta0.9_0.95",
            "learning_rate": float(cfg.learning_rate),
            "weight_decay": float(cfg.weight_decay),
            "max_grad_norm": float(cfg.max_grad_norm),
            "updates": execution_updates if optimizer is not None else 0,
            "microbatches_per_update": execution_microbatches,
            "parameter_groups": optimizer_groups,
            "update_records": update_records,
        },
        # Preserve the old flat timing/memory keys for existing report readers.
        "forward_s": forward_s,
        "backward_s": backward_s,
        "baseline_gpu_memory": baseline_memory,
        "timed_gpu_memory": timed_memory,
        "peak_gpu_allocated_gib": timed_memory["peak_allocated_gib"],
        "peak_gpu_reserved_gib": timed_memory["peak_reserved_gib"],
        "runtime_observation_vae": {
            "contract": RUNTIME_OBSERVATION_VAE_CONTRACT,
            "calls": len(runtime_observation_shapes),
            "expected_calls": expected_observation_calls,
            "input_shape": list(expected_image_shape),
            "input_dtype": expected_observation_dtype,
            "input_device": expected_observation_device,
            "all_inputs_match_contract": True,
        },
        "future_video_vae": {
            "contract": FUTURE_VIDEO_VAE_CONTRACT,
            "calls": len(future_video_shapes),
            "expected_calls": expected_future_video_calls,
            "input_shape": list(expected_future_video_shape),
            "compared_to_runtime_observation_latents": False,
        },
        "trainable_parameters": int(sum(parameter.numel() for parameter in trainable)),
        **gradient_statistics,
    }
    if synthetic_history_extension is not None:
        result["synthetic_history_extension"] = synthetic_history_extension
    _write_json(output_path, result)
    logger.info("Full-prefix smoke complete: %s", output_path)
    return result


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig) -> None:
    setup_logging(log_level=logging.INFO, is_main_process=True)
    run_smoke(cfg)


if __name__ == "__main__":
    main()
