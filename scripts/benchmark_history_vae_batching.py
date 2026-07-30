"""Benchmark real Wan VAE per-observation encoding against LeapBot batching."""

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
from fastwam.utils import misc
from fastwam.utils.config_resolvers import register_default_resolvers
from fastwam.utils.logging_config import get_logger, setup_logging
from leapbot_va.training import encode_independent_history_video_latents


register_default_resolvers()
logger = get_logger(__name__)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def _timed_cuda_call(device: torch.device, function):
    torch.cuda.empty_cache()
    baseline = int(torch.cuda.memory_allocated(device))
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    value = function()
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    peak = int(torch.cuda.max_memory_allocated(device))
    return value, {
        "elapsed_s": float(elapsed),
        "baseline_allocated_gib": float(baseline / 2**30),
        "peak_allocated_gib": float(peak / 2**30),
        "operation_peak_delta_gib": float((peak - baseline) / 2**30),
    }


def run_benchmark(cfg: DictConfig) -> dict:
    benchmark = cfg.get("vae_benchmark")
    if benchmark is None:
        raise ValueError("missing +vae_benchmark configuration")
    checkpoint = Path(str(benchmark.checkpoint)).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    output_dir = Path(str(cfg.output_dir)).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    misc.register_work_dir(str(output_dir))
    OmegaConf.save(cfg, output_dir / "config.yaml", resolve=True)

    precision = _normalize_mixed_precision(str(cfg.mixed_precision))
    dtype = _mixed_precision_to_model_dtype(precision)
    device = torch.device(str(benchmark.get("device", "cuda:0")))
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("real Wan VAE benchmark requires CUDA")
    history_blocks = int(benchmark.get("history_blocks", 8))
    batch_size = int(benchmark.get("batch_size", 2))
    chunk_size = int(benchmark.get("chunk_size", 4))
    if min(history_blocks, batch_size, chunk_size) <= 0:
        raise ValueError("history_blocks, batch_size, and chunk_size must be positive")

    dataset = instantiate(cfg.data.train)
    selected = []
    for dataset_index, frame_index in enumerate(dataset._valid_replan_indices):
        history = int(dataset._episode_step[frame_index] // dataset.replan_steps)
        if history == history_blocks:
            selected.append(dataset_index)
            if len(selected) == batch_size:
                break
    if len(selected) != batch_size:
        raise ValueError(
            f"dataset has only {len(selected)} selected samples with H={history_blocks}"
        )
    sample = default_collate([dataset[index] for index in selected])

    model = instantiate(cfg.model, model_dtype=dtype, device=str(device))
    model.load_checkpoint(str(checkpoint), optimizer=None)
    model.eval().requires_grad_(False)
    history_video = sample["history_video"][:, :, :history_blocks].to(
        device=device, dtype=dtype
    )
    history_valid = sample["history_valid_blocks"][:, :history_blocks].to(
        device=device, dtype=torch.bool
    )
    if not bool(history_valid.all().item()):
        raise AssertionError("selected exact-H samples must have every sliced block valid")

    autocast = lambda: torch.autocast(
        device_type="cuda", dtype=dtype, enabled=precision != "no"
    )
    with torch.no_grad(), autocast():
        warmup = model._encode_video_latents(history_video[:1, :, :1], tiled=False)
    del warmup

    def reference_call():
        with torch.no_grad(), autocast():
            return torch.cat(
                [
                    model._encode_video_latents(
                        history_video[:, :, block : block + 1], tiled=False
                    )
                    for block in range(history_blocks)
                ],
                dim=2,
            )

    reference, reference_timing = _timed_cuda_call(device, reference_call)
    reference_cpu = reference.float().cpu()
    empty_reference = reference[:, :, :1]

    def batched_call():
        with torch.no_grad(), autocast():
            return encode_independent_history_video_latents(
                model,
                history_video,
                history_valid,
                empty_latent_reference=empty_reference,
                tiled=False,
                chunk_size=chunk_size,
            )

    batched, batched_timing = _timed_cuda_call(device, batched_call)
    batched_cpu = batched.float().cpu()
    delta = batched_cpu - reference_cpu
    max_abs = float(delta.abs().max().item())
    rmse = float(delta.square().mean().sqrt().item())
    reference_abs_max = float(reference_cpu.abs().max().item())
    relative_max = max_abs / max(reference_abs_max, 1.0e-12)
    max_abs_atol = float(benchmark.get("max_abs_atol", 1.0e-2))
    rmse_atol = float(benchmark.get("rmse_atol", 1.0e-3))

    loss_equivalence = None
    if bool(benchmark.get("check_loss_equivalence", True)):
        loss_seed = int(benchmark.get("loss_seed", 271828))

        def loss_call(active_chunk_size: int):
            model.history_vae_batch_chunk_size = active_chunk_size
            torch.manual_seed(loss_seed)
            torch.cuda.manual_seed_all(loss_seed)
            with torch.no_grad(), autocast():
                loss, metrics = model.training_loss(sample)
            return {
                "loss": float(loss.detach().float().item()),
                "loss_video": float(metrics[f"loss_video_d{model.mot.num_layers}"]),
                "loss_action": float(metrics[f"loss_action_d{model.mot.num_layers}"]),
            }

        per_observation_loss, per_observation_timing = _timed_cuda_call(
            device, lambda: loss_call(1)
        )
        batched_loss, batched_loss_timing = _timed_cuda_call(
            device, lambda: loss_call(chunk_size)
        )
        loss_deltas = {
            key: abs(float(batched_loss[key]) - float(per_observation_loss[key]))
            for key in per_observation_loss
        }
        loss_atol = float(benchmark.get("loss_atol", 1.0e-3))
        loss_equivalence = {
            "seed": loss_seed,
            "per_observation_chunk1": per_observation_loss,
            "selected_chunk": batched_loss,
            "absolute_deltas": loss_deltas,
            "loss_atol": loss_atol,
            "passed": max(loss_deltas.values()) <= loss_atol,
            "per_observation_timing": per_observation_timing,
            "selected_chunk_timing": batched_loss_timing,
        }
        model.history_vae_batch_chunk_size = chunk_size

    result = {
        "kind": "real_wan_history_vae_batching_equivalence",
        "checkpoint": str(checkpoint),
        "precision": precision,
        "dtype": str(dtype),
        "device": str(device),
        "dataset_indices": selected,
        "batch_size": batch_size,
        "history_blocks": history_blocks,
        "valid_observations": int(history_valid.sum().item()),
        "chunk_size": chunk_size,
        "reference_single_encode_calls": batch_size * history_blocks,
        "batched_single_encode_calls": (
            int(history_valid.sum().item()) + chunk_size - 1
        )
        // chunk_size,
        "max_abs_delta": max_abs,
        "rmse": rmse,
        "reference_abs_max": reference_abs_max,
        "relative_max_delta": relative_max,
        "max_abs_atol": max_abs_atol,
        "rmse_atol": rmse_atol,
        "latent_tolerance_passed": max_abs <= max_abs_atol and rmse <= rmse_atol,
        "reference": reference_timing,
        "batched": batched_timing,
        "speedup": float(
            reference_timing["elapsed_s"] / batched_timing["elapsed_s"]
        ),
        "loss_equivalence": loss_equivalence,
    }
    output_path = output_dir / "history_vae_batching.json"
    _write_json(output_path, result)
    if not result["latent_tolerance_passed"]:
        raise AssertionError(
            "batched real Wan VAE encoding exceeds tolerance: "
            f"max_abs={max_abs} (atol={max_abs_atol}) "
            f"rmse={rmse} (atol={rmse_atol}); details={output_path}"
        )
    if loss_equivalence is not None and not loss_equivalence["passed"]:
        raise AssertionError(
            "batched history VAE changes complete training loss beyond tolerance: "
            f"deltas={loss_equivalence['absolute_deltas']} "
            f"atol={loss_equivalence['loss_atol']}; details={output_path}"
        )
    logger.info("Real Wan history VAE benchmark complete: %s", output_path)
    return result


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig) -> None:
    setup_logging(log_level=logging.INFO, is_main_process=True)
    run_benchmark(cfg)


if __name__ == "__main__":
    main()
