"""Measure stochastic FastWAM training losses without updating model weights.

The reference uses the regular model ``training_loss`` implementation so the
video/action objectives match training, but runs under ``torch.inference_mode``
with every parameter frozen.  No optimizer or scheduler is constructed.
"""

from __future__ import annotations

import gc
import json
import logging
import math
import time
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from fastwam.runtime import (
    _mixed_precision_to_model_dtype,
    _normalize_mixed_precision,
    build_datasets,
)
from fastwam.utils import misc
from fastwam.utils.config_resolvers import register_default_resolvers
from fastwam.utils.logging_config import get_logger, setup_logging
from fastwam.utils.pytorch_utils import set_global_seed
from fastwam.utils.samplers import ResumableEpochSampler


register_default_resolvers()
logger = get_logger(__name__)


def _summary(values: list[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("cannot summarize an empty metric")
    array = np.asarray(values, dtype=np.float64)
    std = float(array.std(ddof=1)) if array.size > 1 else 0.0
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": std,
        "sem": float(std / math.sqrt(array.size)),
        "min": float(array.min()),
        "p50": float(np.quantile(array, 0.50)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(array.max()),
    }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def _create_wandb_run(cfg: DictConfig, output_dir: Path):
    if not bool(cfg.wandb.enabled):
        return None
    import wandb

    run = wandb.init(
        entity=cfg.wandb.workspace,
        project=cfg.wandb.project,
        name=cfg.wandb.name,
        group=None if cfg.wandb.group in (None, "", "null") else str(cfg.wandb.group),
        mode=str(cfg.wandb.mode),
        dir=str(output_dir),
        config=OmegaConf.to_container(cfg, resolve=True),
        job_type="frozen-loss-reference",
    )
    run.define_metric("reference/batch")
    run.define_metric("reference/*", step_metric="reference/batch")
    return run


def run_frozen_reference(cfg: DictConfig) -> dict:
    reference = cfg.get("reference")
    if reference is None:
        raise ValueError("missing `reference` config")
    checkpoint = Path(str(reference.checkpoint)).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"reference checkpoint not found: {checkpoint}")

    output_dir = Path(str(cfg.output_dir)).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    misc.register_work_dir(str(output_dir))
    OmegaConf.save(cfg, output_dir / "config.yaml", resolve=True)

    seed = int(reference.get("seed", cfg.seed))
    noise_seed = int(reference.get("noise_seed", seed + 1_000_000))
    batch_size = int(reference.get("batch_size", cfg.batch_size))
    max_batches = int(reference.max_batches)
    num_workers = int(reference.get("num_workers", cfg.num_workers))
    if batch_size <= 0 or max_batches <= 0 or num_workers < 0:
        raise ValueError("batch_size/max_batches must be positive and num_workers non-negative")

    precision = _normalize_mixed_precision(str(cfg.mixed_precision))
    model_dtype = _mixed_precision_to_model_dtype(precision)
    device = str(reference.get("device", "cuda:0" if torch.cuda.is_available() else "cpu"))
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA reference requested but CUDA is unavailable")

    worker_init_fn = set_global_seed(seed, get_worker_init_fn=True)
    logger.info("Instantiating frozen reference model on %s (%s)", device, model_dtype)
    model = instantiate(cfg.model, model_dtype=model_dtype, device=device)
    payload = model.load_checkpoint(str(checkpoint), optimizer=None)
    checkpoint_step = payload.get("step") if isinstance(payload, dict) else None
    del payload
    gc.collect()

    model.eval()
    model.requires_grad_(False)
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if trainable_parameters != 0:
        raise AssertionError(f"frozen reference has {trainable_parameters} trainable parameters")
    parameter_versions_before = {
        name: int(parameter._version) for name, parameter in model.named_parameters()
    }

    train_dataset, _ = build_datasets(cfg.data)
    sampler = ResumableEpochSampler(
        dataset=train_dataset,
        seed=seed,
        batch_size=batch_size,
        num_processes=1,
    )
    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=device.startswith("cuda"),
        worker_init_fn=worker_init_fn,
    )

    wandb_run = _create_wandb_run(cfg, output_dir)
    metric_values: dict[str, list[float]] = {
        "loss_total": [],
        "loss_video": [],
        "loss_action": [],
    }
    batch_records: list[dict] = []
    sample_count = 0
    start = time.perf_counter()
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(torch.device(device))

    try:
        for batch_index, sample in enumerate(loader):
            if batch_index >= max_batches:
                break
            # The same batch index receives the same diffusion noise/timesteps
            # across checkpoints, making paired loss comparisons meaningful.
            batch_seed = noise_seed + batch_index
            torch.manual_seed(batch_seed)
            if device.startswith("cuda"):
                torch.cuda.manual_seed_all(batch_seed)

            batch_start = time.perf_counter()
            with torch.inference_mode():
                with torch.autocast(
                    device_type="cuda",
                    dtype=model_dtype,
                    enabled=device.startswith("cuda") and precision != "no",
                ):
                    loss, loss_dict = model.training_loss(sample)
            if device.startswith("cuda"):
                torch.cuda.synchronize(torch.device(device))

            record = {
                "batch": int(batch_index),
                "noise_seed": int(batch_seed),
                "samples": int(sample["video"].shape[0]),
                "loss_total": float(loss.detach().float().item()),
                "loss_video": float(loss_dict["loss_video"]),
                "loss_action": float(loss_dict["loss_action"]),
                "elapsed_s": float(time.perf_counter() - batch_start),
            }
            for key in metric_values:
                metric_values[key].append(record[key])
            batch_records.append(record)
            sample_count += record["samples"]
            logger.info(
                "reference batch=%d/%d samples=%d loss=%.6f action=%.6f video=%.6f elapsed=%.3fs",
                batch_index + 1,
                max_batches,
                sample_count,
                record["loss_total"],
                record["loss_action"],
                record["loss_video"],
                record["elapsed_s"],
            )
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "reference/batch": batch_index,
                        "reference/loss_total": record["loss_total"],
                        "reference/loss_action": record["loss_action"],
                        "reference/loss_video": record["loss_video"],
                        "reference/batch_elapsed_s": record["elapsed_s"],
                        "reference/samples": sample_count,
                    }
                )
    finally:
        elapsed_s = time.perf_counter() - start

    if not batch_records:
        raise RuntimeError("reference loader produced no batches")

    changed_parameters = [
        name
        for name, parameter in model.named_parameters()
        if int(parameter._version) != parameter_versions_before[name]
    ]
    parameters_with_grad = [
        name for name, parameter in model.named_parameters() if parameter.grad is not None
    ]
    peak_gpu_gib = (
        float(torch.cuda.max_memory_allocated(torch.device(device)) / 2**30)
        if device.startswith("cuda")
        else 0.0
    )
    summary = {key: _summary(values) for key, values in metric_values.items()}
    result = {
        "kind": "frozen_training_loss_reference",
        "checkpoint": str(checkpoint),
        "checkpoint_step": checkpoint_step,
        "model_target": str(cfg.model.get("_target_", "")),
        "dataset_target": str(cfg.data.train.get("_target_", "")),
        "seed": seed,
        "noise_seed": noise_seed,
        "batch_size": batch_size,
        "num_batches": len(batch_records),
        "num_samples": sample_count,
        "device": device,
        "mixed_precision": precision,
        "elapsed_s": float(elapsed_s),
        "samples_per_s": float(sample_count / max(elapsed_s, 1e-9)),
        "peak_gpu_allocated_gib": peak_gpu_gib,
        "summary": summary,
        "invariants": {
            "optimizer_created": False,
            "backward_called": False,
            "inference_mode": True,
            "trainable_parameters": trainable_parameters,
            "parameters_with_grad": parameters_with_grad,
            "changed_parameter_versions": changed_parameters,
            "weights_unchanged": not changed_parameters and not parameters_with_grad,
        },
        "batches": batch_records,
    }
    output_path = output_dir / str(reference.get("output_name", "frozen_loss_reference.json"))
    _write_json(output_path, result)

    if wandb_run is not None:
        for metric_name, stats in summary.items():
            for stat_name, value in stats.items():
                wandb_run.summary[f"reference/{metric_name}_{stat_name}"] = value
        wandb_run.summary["reference/num_samples"] = sample_count
        wandb_run.summary["reference/peak_gpu_allocated_gib"] = peak_gpu_gib
        wandb_run.summary["reference/weights_unchanged"] = result["invariants"]["weights_unchanged"]
        wandb_run.summary["reference/output_path"] = str(output_path)
        logger.info("W&B frozen reference: %s", wandb_run.url)
        wandb_run.finish()

    if changed_parameters or parameters_with_grad:
        raise AssertionError(
            "frozen reference mutated model state: "
            f"changed_versions={changed_parameters[:5]} gradients={parameters_with_grad[:5]}"
        )
    logger.info("Frozen loss reference complete: %s", output_path)
    return result


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig) -> None:
    setup_logging(log_level=logging.INFO, is_main_process=True)
    run_frozen_reference(cfg)


if __name__ == "__main__":
    main()
