"""Run a real 6B full-prefix forward/backward smoke test on one LIBERO sample."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data._utils.collate import default_collate

from fastwam.runtime import _mixed_precision_to_model_dtype, _normalize_mixed_precision
from fastwam.models.wan22.fastwam import FastWAM
from fastwam.utils import misc
from fastwam.utils.config_resolvers import register_default_resolvers
from fastwam.utils.logging_config import get_logger, setup_logging


register_default_resolvers()
logger = get_logger(__name__)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


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


def run_smoke(cfg: DictConfig) -> dict:
    smoke = cfg.get("smoke")
    if smoke is None:
        raise ValueError("missing +smoke configuration")
    checkpoint = Path(str(smoke.checkpoint)).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    output_dir = Path(str(cfg.output_dir)).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    misc.register_work_dir(str(output_dir))
    OmegaConf.save(cfg, output_dir / "config.yaml", resolve=True)

    precision = _normalize_mixed_precision(str(cfg.mixed_precision))
    dtype = _mixed_precision_to_model_dtype(precision)
    device = str(smoke.get("device", "cuda:0"))
    target_history_blocks = int(smoke.get("history_blocks", 50))
    synthetic_source = smoke.get("synthetic_source_history_blocks")
    selection_history_blocks = (
        target_history_blocks if synthetic_source is None else int(synthetic_source)
    )
    batch_repeats = int(smoke.get("batch_repeats", 1))
    if batch_repeats <= 0:
        raise ValueError("smoke.batch_repeats must be positive")

    dataset = instantiate(cfg.data.train)
    selected_index = None
    for dataset_index, frame_index in enumerate(dataset._valid_replan_indices):
        block = dataset._episode_step[frame_index] // dataset.replan_steps
        if block == selection_history_blocks:
            selected_index = dataset_index
            break
    if selected_index is None:
        raise ValueError(f"dataset contains no sample with H={selection_history_blocks}")
    sample = default_collate([dataset[selected_index] for _ in range(batch_repeats)])
    synthetic_history_extension = None
    if synthetic_source is not None:
        synthetic_history_extension = _extend_history_for_capacity_smoke(
            sample,
            source_history_blocks=selection_history_blocks,
            target_history_blocks=target_history_blocks,
            replan_steps=int(dataset.replan_steps),
        )
    actual_history_counts = sample["history_valid_blocks"].sum(dim=1)
    if not bool((actual_history_counts == target_history_blocks).all().item()):
        raise AssertionError((actual_history_counts.tolist(), target_history_blocks))
    actual_history_blocks = target_history_blocks

    model = instantiate(cfg.model, model_dtype=dtype, device=device)
    model.load_checkpoint(str(checkpoint), optimizer=None)
    model.eval()
    model.requires_grad_(False)
    model.configure_trainable_parameters()
    if model.proprio_encoder is not None:
        model.proprio_encoder.train()
        model.proprio_encoder.requires_grad_(True)

    vae_first_frame_equivalence = None
    if bool(smoke.get("check_vae_first_frame", True)):
        video = sample["video"].to(device=model.device, dtype=model.torch_dtype)
        with torch.no_grad(), torch.autocast(
            device_type="cuda", dtype=dtype, enabled=precision != "no"
        ):
            full_clip_latents = model._encode_video_latents(video, tiled=False)
            single_frame_latents = torch.cat(
                [
                    model._encode_input_image_latents_tensor(
                        video[index, :, 0], tiled=False
                    )
                    for index in range(batch_repeats)
                ],
                dim=0,
            )
        difference = full_clip_latents[:, :, :1].float() - single_frame_latents.float()
        vae_first_frame_equivalence = {
            "max_abs_delta": float(difference.abs().max().item()),
            "rmse": float(difference.square().mean().sqrt().item()),
        }
        vae_atol = float(smoke.get("vae_first_frame_atol", 1.0e-5))
        if vae_first_frame_equivalence["max_abs_delta"] > vae_atol:
            raise AssertionError(
                "full-clip and single-image VAE first latents differ: "
                f"{vae_first_frame_equivalence} atol={vae_atol}"
            )
        del video, full_clip_latents, single_frame_latents, difference
        torch.cuda.empty_cache()

    seed = int(cfg.seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cpu_rng_state = torch.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state(torch.device(device))
    native_reference = None
    if bool(smoke.get("compare_native", smoke.get("compare_native_h0", True))):
        with torch.no_grad(), torch.autocast(
            device_type="cuda", dtype=dtype, enabled=precision != "no"
        ):
            native_loss, native_metrics = FastWAM.training_loss(
                model, sample, tiled=False
            )
        native_reference = {
            "loss": float(native_loss.detach().float().item()),
            "loss_video": float(native_metrics["loss_video"]),
            "loss_action": float(native_metrics["loss_action"]),
        }
        del native_loss
        torch.set_rng_state(cpu_rng_state)
        torch.cuda.set_rng_state(cuda_rng_state, torch.device(device))
        torch.cuda.empty_cache()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(torch.device(device))
    start = time.perf_counter()
    with torch.autocast(
        device_type="cuda", dtype=dtype, enabled=precision != "no"
    ):
        loss, metrics = model.training_loss(sample)
    torch.cuda.synchronize(torch.device(device))
    forward_s = time.perf_counter() - start

    backward_start = time.perf_counter()
    loss.backward()
    torch.cuda.synchronize(torch.device(device))
    backward_s = time.perf_counter() - backward_start
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    gradients = [parameter.grad for parameter in trainable if parameter.grad is not None]
    if not gradients:
        raise AssertionError("no trainable parameter received a gradient")
    finite_gradients = all(bool(torch.isfinite(gradient).all().item()) for gradient in gradients)
    if not finite_gradients:
        raise AssertionError("non-finite gradient in full-prefix smoke test")

    result = {
        "kind": "full_prefix_training_smoke",
        "checkpoint": str(checkpoint),
        "dataset_index": selected_index,
        "batch_size": batch_repeats,
        "history_blocks": actual_history_blocks,
        "causal_mode": model.causal_mode,
        "history_training_mode": model.history_training_mode,
        "loss": float(loss.detach().float().item()),
        "metrics": metrics,
        "forward_s": forward_s,
        "backward_s": backward_s,
        "peak_gpu_allocated_gib": float(
            torch.cuda.max_memory_allocated(torch.device(device)) / 2**30
        ),
        "trainable_parameters": int(sum(parameter.numel() for parameter in trainable)),
        "parameters_with_grad": int(sum(parameter.numel() for parameter in trainable if parameter.grad is not None)),
        "finite_gradients": finite_gradients,
    }
    if vae_first_frame_equivalence is not None:
        result["vae_first_frame_equivalence"] = vae_first_frame_equivalence
    if synthetic_history_extension is not None:
        result["synthetic_history_extension"] = synthetic_history_extension
    if native_reference is not None:
        incremental_values = {
            "loss": result["loss"],
            "loss_video": float(metrics["loss_video_d30"]),
            "loss_action": float(metrics["loss_action_d30"]),
        }
        deltas = {
            key: abs(incremental_values[key] - native_reference[key])
            for key in incremental_values
        }
        result["native_reference"] = native_reference
        result["native_absolute_delta"] = deltas
        h0_atol = float(smoke.get("h0_native_atol", 1.0e-5))
        if target_history_blocks == 0 and max(deltas.values()) > h0_atol:
            raise AssertionError(
                "H0 incremental/native loss mismatch exceeds BF16 tolerance: "
                f"incremental={incremental_values} native={native_reference} "
                f"delta={deltas} atol={h0_atol}"
            )
        if (
            model.causal_mode == "action_aggregator"
            and deltas["loss_video"] > float(
                smoke.get("action_aggregator_video_atol", 1.0e-5)
            )
        ):
            raise AssertionError(
                "action_aggregator video path must remain native-equivalent because "
                "historical K/V is consumed only by ActionDiT: "
                f"incremental={incremental_values} native={native_reference} "
                f"delta={deltas}"
            )
    output_path = output_dir / "full_prefix_smoke.json"
    _write_json(output_path, result)
    logger.info("Full-prefix smoke complete: %s", output_path)
    return result


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig) -> None:
    setup_logging(log_level=logging.INFO, is_main_process=True)
    run_smoke(cfg)


if __name__ == "__main__":
    main()
