"""Paired fixed-noise loss audit across exact causal-history lengths.

This is a checkpoint-selection tool, not a training loop.  Every checkpoint is
evaluated on the same LeRobot samples with the same flow-matching noise and
timesteps.  Reporting exact history lengths makes it possible to distinguish a
healthy H=0 FastWAM-compatible path from degradation introduced by longer
causal prefixes.
"""

from __future__ import annotations

import gc
import json
import logging
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data._utils.collate import default_collate

from fastwam.models.wan22.fastwam import FastWAM
from fastwam.runtime import _mixed_precision_to_model_dtype, _normalize_mixed_precision
from fastwam.utils import misc
from fastwam.utils.config_resolvers import register_default_resolvers
from fastwam.utils.logging_config import get_logger, setup_logging


register_default_resolvers()
logger = get_logger(__name__)


def select_history_samples(
    dataset,
    *,
    history_lengths: list[int],
    samples_per_history: int,
    seed: int,
) -> list[dict[str, int]]:
    """Select deterministic, paired dataset indices for exact history lengths."""

    if samples_per_history <= 0:
        raise ValueError("samples_per_history must be positive")
    if not history_lengths:
        raise ValueError("history_lengths must not be empty")
    if any(value < 0 for value in history_lengths):
        raise ValueError("history lengths must be non-negative")

    requested = set(int(value) for value in history_lengths)
    candidates: dict[int, list[int]] = defaultdict(list)
    for dataset_index, frame_index in enumerate(dataset._valid_replan_indices):
        history = int(dataset._episode_step[frame_index] // dataset.replan_steps)
        if history in requested:
            candidates[history].append(dataset_index)

    selected: list[dict[str, int]] = []
    for history in history_lengths:
        available = candidates.get(int(history), [])
        if not available:
            raise ValueError(f"dataset contains no sample with H={history}")
        count = min(samples_per_history, len(available))
        rng = np.random.default_rng(seed + int(history) * 1_000_003)
        positions = np.sort(rng.choice(len(available), size=count, replace=False))
        for replica, position in enumerate(positions.tolist()):
            selected.append(
                {
                    "history_blocks": int(history),
                    "replica": int(replica),
                    "dataset_index": int(available[position]),
                }
            )
    return selected


def _summary(values: list[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("cannot summarize empty values")
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


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[int(record["history_blocks"])].append(record)

    def metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            key: _summary([float(row[key]) for row in rows])
            for key in ("loss", "loss_video", "loss_action")
        }

    return {
        "overall": metrics(records),
        "by_history": {
            str(history): metrics(grouped[history]) for history in sorted(grouped)
        },
    }


def summarize_paired_variant_delta(
    candidate: list[dict[str, Any]],
    reference: list[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize candidate-reference loss deltas for identical sample/noise pairs."""

    reference_by_key = {
        (int(row["dataset_index"]), int(row["noise_seed"])): row
        for row in reference
    }
    grouped: dict[int, list[dict[str, float]]] = defaultdict(list)
    all_rows: list[dict[str, float]] = []
    for row in candidate:
        key = (int(row["dataset_index"]), int(row["noise_seed"]))
        baseline = reference_by_key[key]
        delta = {
            "loss": float(row["loss"] - baseline["loss"]),
            "loss_video": float(row["loss_video"] - baseline["loss_video"]),
            "loss_action": float(row["loss_action"] - baseline["loss_action"]),
        }
        all_rows.append(delta)
        grouped[int(row["history_blocks"])].append(delta)

    def means(rows: list[dict[str, float]]) -> dict[str, float | int]:
        return {
            "count": len(rows),
            **{
                f"{key}_delta_mean": float(np.mean([row[key] for row in rows]))
                for key in ("loss", "loss_video", "loss_action")
            },
        }

    return {
        "overall": means(all_rows),
        "by_history": {
            str(history): means(rows) for history, rows in sorted(grouped.items())
        },
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def _checkpoint_label(path: Path, index: int) -> str:
    if path.parent.name == "weights":
        return f"{path.parent.parent.parent.name}:{path.stem}"
    return f"checkpoint_{index}:{path.stem}"


def _evaluate_checkpoint(
    cfg: DictConfig,
    *,
    checkpoint: Path,
    label: str,
    dataset,
    selected: list[dict[str, int]],
    dtype: torch.dtype,
    precision: str,
    device: str,
    noise_seed: int,
    include_native: bool,
) -> dict[str, Any]:
    logger.info("Loading checkpoint %s from %s", label, checkpoint)
    model = instantiate(cfg.model, model_dtype=dtype, device=device)
    payload = model.load_checkpoint(str(checkpoint), optimizer=None)
    checkpoint_step = payload.get("step") if isinstance(payload, dict) else None
    del payload
    model.eval()
    model.requires_grad_(False)

    records: list[dict[str, Any]] = []
    absolute_no_history_records: list[dict[str, Any]] = []
    native_records: list[dict[str, Any]] = []
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(torch.device(device))
    started = time.perf_counter()

    for ordinal, selection in enumerate(selected):
        sample = default_collate([dataset[selection["dataset_index"]]])
        actual_history = int(sample["history_valid_blocks"].sum().item())
        if actual_history != int(selection["history_blocks"]):
            raise AssertionError(
                f"history mismatch dataset_index={selection['dataset_index']} "
                f"expected={selection['history_blocks']} actual={actual_history}"
            )
        sample_seed = int(noise_seed + ordinal)

        def evaluate(loss_fn) -> tuple[float, dict[str, float], float]:
            torch.manual_seed(sample_seed)
            torch.cuda.manual_seed_all(sample_seed)
            before = time.perf_counter()
            with torch.inference_mode(), torch.autocast(
                device_type="cuda", dtype=dtype, enabled=precision != "no"
            ):
                loss, raw_metrics = loss_fn(model, sample)
            torch.cuda.synchronize(torch.device(device))
            metrics = {
                str(key): float(value) for key, value in raw_metrics.items()
            }
            return float(loss.detach().float().item()), metrics, time.perf_counter() - before

        incremental_loss, incremental_metrics, elapsed_s = evaluate(
            lambda active_model, active_sample: active_model.training_loss(active_sample)
        )
        record = {
            **selection,
            "noise_seed": sample_seed,
            "loss": incremental_loss,
            "loss_video": float(incremental_metrics[f"loss_video_d{model.mot.num_layers}"]),
            "loss_action": float(incremental_metrics[f"loss_action_d{model.mot.num_layers}"]),
            "elapsed_s": float(elapsed_s),
        }
        records.append(record)

        # Retain the current observation/action absolute positions while
        # removing every historical K/V segment.  Comparing this with native
        # FastWAM isolates temporal-RoPE extrapolation; comparing full history
        # with this variant isolates the effect of adding historical content.
        absolute_no_history_sample = dict(sample)
        absolute_no_history_sample["history_valid_blocks"] = torch.zeros_like(
            sample["history_valid_blocks"]
        )
        absolute_loss, absolute_metrics, absolute_elapsed_s = evaluate(
            lambda active_model, _active_sample: active_model.training_loss(
                absolute_no_history_sample
            )
        )
        absolute_no_history_records.append(
            {
                **selection,
                "noise_seed": sample_seed,
                "loss": absolute_loss,
                "loss_video": float(
                    absolute_metrics[f"loss_video_d{model.mot.num_layers}"]
                ),
                "loss_action": float(
                    absolute_metrics[f"loss_action_d{model.mot.num_layers}"]
                ),
                "elapsed_s": float(absolute_elapsed_s),
            }
        )

        if include_native:
            native_loss, native_metrics, native_elapsed_s = evaluate(FastWAM.training_loss)
            native_records.append(
                {
                    **selection,
                    "noise_seed": sample_seed,
                    "loss": native_loss,
                    "loss_video": float(native_metrics["loss_video"]),
                    "loss_action": float(native_metrics["loss_action"]),
                    "elapsed_s": float(native_elapsed_s),
                }
            )
        logger.info(
            "%s H=%d replica=%d action=%.6f video=%.6f elapsed=%.2fs",
            label,
            selection["history_blocks"],
            selection["replica"],
            record["loss_action"],
            record["loss_video"],
            elapsed_s,
        )

    result: dict[str, Any] = {
        "label": label,
        "checkpoint": str(checkpoint),
        "checkpoint_step": checkpoint_step,
        "causal_mode": str(model.causal_mode),
        "history_training_mode": str(model.history_training_mode),
        "elapsed_s": float(time.perf_counter() - started),
        "peak_gpu_allocated_gib": float(
            torch.cuda.max_memory_allocated(torch.device(device)) / 2**30
        ),
        "records": records,
        "summary": summarize_records(records),
        "absolute_no_history_records": absolute_no_history_records,
        "absolute_no_history_summary": summarize_records(
            absolute_no_history_records
        ),
        "full_minus_absolute_no_history": summarize_paired_variant_delta(
            records, absolute_no_history_records
        ),
    }
    if native_records:
        result["native_records"] = native_records
        result["native_summary"] = summarize_records(native_records)
        result["absolute_no_history_minus_native"] = summarize_paired_variant_delta(
            absolute_no_history_records, native_records
        )

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _paired_deltas(checkpoints: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(checkpoints) < 2:
        return []
    reference = checkpoints[0]
    reference_rows = {
        (int(row["dataset_index"]), int(row["noise_seed"])): row
        for row in reference["records"]
    }
    results = []
    for candidate in checkpoints[1:]:
        rows = []
        for row in candidate["records"]:
            key = (int(row["dataset_index"]), int(row["noise_seed"]))
            baseline = reference_rows[key]
            rows.append(
                {
                    "history_blocks": int(row["history_blocks"]),
                    "dataset_index": key[0],
                    "noise_seed": key[1],
                    "loss_action_delta": float(row["loss_action"] - baseline["loss_action"]),
                    "loss_video_delta": float(row["loss_video"] - baseline["loss_video"]),
                }
            )
        by_history: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_history[int(row["history_blocks"])].append(row)
        results.append(
            {
                "reference": reference["label"],
                "candidate": candidate["label"],
                "action_delta_mean": float(
                    np.mean([row["loss_action_delta"] for row in rows])
                ),
                "video_delta_mean": float(
                    np.mean([row["loss_video_delta"] for row in rows])
                ),
                "by_history": {
                    str(history): {
                        "count": len(group),
                        "action_delta_mean": float(
                            np.mean([row["loss_action_delta"] for row in group])
                        ),
                        "video_delta_mean": float(
                            np.mean([row["loss_video_delta"] for row in group])
                        ),
                    }
                    for history, group in sorted(by_history.items())
                },
            }
        )
    return results


def run_audit(cfg: DictConfig) -> dict[str, Any]:
    audit = cfg.get("stratified")
    if audit is None:
        raise ValueError("missing +stratified configuration")
    checkpoints = [Path(str(value)).expanduser().resolve() for value in audit.checkpoints]
    if not checkpoints:
        raise ValueError("stratified.checkpoints must not be empty")
    missing = [str(path) for path in checkpoints if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing checkpoints: {missing}")

    output_dir = Path(str(cfg.output_dir)).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    misc.register_work_dir(str(output_dir))
    OmegaConf.save(cfg, output_dir / "config.yaml", resolve=True)

    precision = _normalize_mixed_precision(str(cfg.mixed_precision))
    dtype = _mixed_precision_to_model_dtype(precision)
    device = str(audit.get("device", "cuda:0"))
    if not device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("history-stratified 6B audit requires CUDA")

    dataset = instantiate(cfg.data.train)
    if not bool(getattr(dataset, "full_episode_history", False)):
        raise ValueError("stratified audit requires full_episode_history=true")
    history_lengths = [int(value) for value in audit.history_lengths]
    selected = select_history_samples(
        dataset,
        history_lengths=history_lengths,
        samples_per_history=int(audit.get("samples_per_history", 2)),
        seed=int(audit.get("selection_seed", cfg.seed)),
    )

    checkpoint_results = []
    for index, checkpoint in enumerate(checkpoints):
        checkpoint_results.append(
            _evaluate_checkpoint(
                cfg,
                checkpoint=checkpoint,
                label=_checkpoint_label(checkpoint, index),
                dataset=dataset,
                selected=selected,
                dtype=dtype,
                precision=precision,
                device=device,
                noise_seed=int(audit.get("noise_seed", int(cfg.seed) + 2_000_000)),
                include_native=bool(audit.get("include_native", index == 0)),
            )
        )

    result = {
        "kind": "paired_history_stratified_loss_audit",
        "history_lengths": history_lengths,
        "samples_per_history_requested": int(audit.get("samples_per_history", 2)),
        "selected_samples": selected,
        "precision": precision,
        "device": device,
        "checkpoints": checkpoint_results,
        "paired_deltas_from_first": _paired_deltas(checkpoint_results),
    }
    output_path = output_dir / str(audit.get("output_name", "history_stratified_loss.json"))
    _write_json(output_path, result)
    logger.info("History-stratified loss audit complete: %s", output_path)
    return result


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig) -> None:
    setup_logging(log_level=logging.INFO, is_main_process=True)
    run_audit(cfg)


if __name__ == "__main__":
    main()
