"""Measure the formal 8-GPU ZeRO-2 training peak on real long prefixes.

This is a hardware acceptance probe, not a training entry point.  It executes
two real optimizer updates with the same model, loss, optimizer and DeepSpeed
preparation as formal training, but deliberately never writes a model or
optimizer checkpoint.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset

from fastwam.runtime import (
    _mixed_precision_to_model_dtype,
    _normalize_mixed_precision,
    _resolve_train_device,
    _seed_model_initialization,
)
from fastwam.trainer import Wan22Trainer
from fastwam.utils import misc
from fastwam.utils.config_resolvers import register_default_resolvers
from fastwam.utils.logging_config import get_logger, setup_logging


register_default_resolvers()
logger = get_logger(__name__)

FORMAL_WORLD_SIZE = 8
FORMAL_OPTIMIZER_UPDATES = 2
FORMAL_HISTORY_MIN = 41
FORMAL_HISTORY_MAX = 50
FORMAL_REPLAN_STEPS = 10
FORMAL_ACTION_HORIZON = 32
SUPPORTED_BATCH_SIZES = (16, 18, 20, 22)
SUPPORTED_CAUSAL_MODES = (
    "action_aggregator",
    "interleaved",
    "vision_causal",
)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def find_checkpoint_files(output_dir: Path) -> list[str]:
    checkpoint_root = output_dir / "checkpoints"
    return sorted(
        str(path.relative_to(output_dir))
        for path in checkpoint_root.rglob("*")
        if path.is_file()
    )


def read_formal_optimizer_parameter_groups(optimizer: Any) -> dict[str, Any]:
    """Verify the two formal AdamW groups after DeepSpeed preparation."""

    resolved: dict[str, Any] = {}
    for group in getattr(optimizer, "param_groups", []):
        name = str(group.get("group_name", ""))
        if name in resolved:
            raise RuntimeError(f"duplicate optimizer parameter group: {name!r}")
        if name not in {"action_and_aux", "video_lora"}:
            raise RuntimeError(f"unexpected optimizer parameter group: {name!r}")
        resolved[name] = {
            "weight_decay": float(group["weight_decay"]),
            "parameters": int(
                sum(parameter.numel() for parameter in group["params"])
            ),
        }
    expected_weight_decay = {"action_and_aux": 1.0e-2, "video_lora": 0.0}
    if set(resolved) != set(expected_weight_decay):
        raise RuntimeError(
            "formal optimizer groups are incomplete after DeepSpeed preparation: "
            f"{sorted(resolved)}"
        )
    for name, expected in expected_weight_decay.items():
        if resolved[name]["weight_decay"] != expected:
            raise RuntimeError(
                f"optimizer group {name} weight_decay mismatch: "
                f"expected={expected} actual={resolved[name]['weight_decay']}"
            )
    return resolved


def read_deepspeed_engine_step_state(engine: Any) -> dict[str, int | bool]:
    """Read the concrete DeepSpeedEngine counters used by DS 0.18.5."""

    values: dict[str, int | bool] = {}
    for name in ("global_steps", "skipped_steps"):
        value = getattr(engine, name, None)
        if isinstance(value, bool) or not isinstance(value, int):
            raise RuntimeError(
                f"prepared DeepSpeed engine is missing integer {name}: {value!r}"
            )
        values[name] = int(value)
    step_applied = getattr(engine, "_step_applied", None)
    if not isinstance(step_applied, bool):
        raise RuntimeError(
            "prepared DeepSpeed engine is missing boolean _step_applied evidence"
        )
    values["_step_applied"] = step_applied
    return values


def validate_deepspeed_engine_step_transition(
    before: dict[str, int | bool],
    after: dict[str, int | bool],
    *,
    optimizer_step_was_skipped: bool,
) -> dict[str, int | bool]:
    """Require one applied, non-skipped DeepSpeed update on every probe step."""

    global_step_delta = int(after["global_steps"]) - int(before["global_steps"])
    skipped_step_delta = int(after["skipped_steps"]) - int(before["skipped_steps"])
    step_applied = bool(after["_step_applied"])
    if global_step_delta != 1:
        raise RuntimeError(
            "DeepSpeed engine global_steps did not advance exactly once: "
            f"before={before['global_steps']} after={after['global_steps']}"
        )
    if optimizer_step_was_skipped or skipped_step_delta != 0 or not step_applied:
        raise RuntimeError(
            "capacity-probe optimizer update was skipped or not applied: "
            f"accelerator_skipped={optimizer_step_was_skipped} "
            f"skipped_step_delta={skipped_step_delta} "
            f"engine_step_applied={step_applied}"
        )
    return {
        "engine_global_steps_before": int(before["global_steps"]),
        "engine_global_steps_after": int(after["global_steps"]),
        "engine_global_steps_delta": global_step_delta,
        "engine_skipped_steps_before": int(before["skipped_steps"]),
        "engine_skipped_steps_after": int(after["skipped_steps"]),
        "engine_skipped_steps_delta": skipped_step_delta,
        "engine_step_applied": step_applied,
        "accelerator_optimizer_step_was_skipped": bool(
            optimizer_step_was_skipped
        ),
    }


def validate_capacity_probe_contract(cfg: DictConfig) -> dict[str, Any]:
    """Fail closed before model, dataset, CUDA or distributed initialization."""

    probe = cfg.get("capacity_probe")
    if probe is None:
        raise ValueError("missing +capacity_probe configuration")
    batch_size = int(cfg.get("batch_size", -1))
    if batch_size not in SUPPORTED_BATCH_SIZES:
        raise ValueError(
            "capacity probe batch_size must be one of "
            f"{SUPPORTED_BATCH_SIZES}, got {batch_size}"
        )
    optimizer_updates = int(probe.get("optimizer_updates", -1))
    if optimizer_updates != FORMAL_OPTIMIZER_UPDATES:
        raise ValueError("capacity probe requires exactly two optimizer updates")
    history_min = int(probe.get("history_min", -1))
    history_max = int(probe.get("history_max", -1))
    if (history_min, history_max) != (FORMAL_HISTORY_MIN, FORMAL_HISTORY_MAX):
        raise ValueError("capacity probe requires real H41-H50 prefixes")
    if int(cfg.get("gradient_accumulation_steps", -1)) != 1:
        raise ValueError("capacity probe requires gradient_accumulation_steps=1")
    if _normalize_mixed_precision(str(cfg.get("mixed_precision", ""))) != "bf16":
        raise ValueError("capacity probe requires BF16")
    if bool(cfg.get("wandb", {}).get("enabled", True)):
        raise ValueError("capacity probe forbids W&B")
    if int(cfg.get("save_every", -1)) != 0 or int(cfg.get("eval_every", -1)) != 0:
        raise ValueError("capacity probe forbids checkpoint/evaluation callbacks")
    if int(cfg.get("max_steps", -1)) != optimizer_updates:
        raise ValueError("capacity probe max_steps must equal optimizer_updates")
    if str(cfg.get("lr_scheduler_type", "")) != "constant":
        raise ValueError("capacity probe requires a constant learning rate")
    if float(cfg.get("weight_decay", -1.0)) != 1.0e-2:
        raise ValueError("capacity probe requires weight_decay=1e-2")
    if float(cfg.get("max_grad_norm", -1.0)) != 1.0:
        raise ValueError("capacity probe requires max_grad_norm=1.0")

    model = cfg.get("model")
    if model is None:
        raise ValueError("capacity probe is missing model configuration")
    causal_mode = str(model.get("causal_mode", ""))
    if causal_mode not in SUPPORTED_CAUSAL_MODES:
        raise ValueError(
            "capacity probe causal_mode must be one of "
            f"{SUPPORTED_CAUSAL_MODES}, got {causal_mode!r}"
        )
    if str(model.get("history_training_mode", "")) != "incremental_full_bptt":
        raise ValueError("capacity probe requires incremental_full_bptt")
    if int(model.get("history_vae_batch_chunk_size", -1)) != 1:
        raise ValueError("capacity probe requires history VAE chunk1")
    if str(model.get("training_strategy", "")) != "video_lora_action_full":
        raise ValueError("capacity probe requires video_lora_action_full")
    video_lora = model.get("video_lora")
    if video_lora is None or not bool(video_lora.get("enabled", False)):
        raise ValueError("capacity probe requires enabled VideoDiT LoRA")
    if int(video_lora.get("rank", -1)) != 16:
        raise ValueError("capacity probe requires VideoDiT LoRA rank 16")
    if (
        float(video_lora.get("alpha", -1.0)) != 16.0
        or float(video_lora.get("dropout", -1.0)) != 0.0
        or float(video_lora.get("learning_rate_multiplier", -1.0)) != 1.0
    ):
        raise ValueError(
            "capacity probe requires VideoDiT LoRA alpha16/dropout0/LR multiplier1"
        )
    if list(model.get("training_exit_depths", [])) != [30]:
        raise ValueError("capacity probe requires the D30 exit only")
    if not bool(model.get("mot_checkpoint_mixed_attn", False)):
        raise ValueError("capacity probe requires formal activation checkpointing")
    if (int(model.get("replan_steps", -1)), int(model.get("action_horizon", -1))) != (
        FORMAL_REPLAN_STEPS,
        FORMAL_ACTION_HORIZON,
    ):
        raise ValueError("capacity probe requires replan_steps=10/action_horizon=32")
    if (
        str(model.get("future_video_conditioning", ""))
        != "lingbot_teacher_forced_v1"
        or int(model.get("num_video_frames", -1)) != 9
        or float(model.get("future_video_condition_noise_probability", -1.0)) != 0.5
        or float(model.get("future_video_condition_min_u", -1.0)) != 0.5
        or float(model.get("future_video_condition_max_u", -1.0)) != 1.0
    ):
        raise ValueError(
            "capacity probe requires LingBot future-video frames=9, "
            "probability=0.5, u=[0.5,1.0]"
        )

    data = cfg.get("data", {}).get("train")
    if data is None:
        raise ValueError("capacity probe is missing training data configuration")
    if not bool(data.get("full_episode_history", False)):
        raise ValueError("capacity probe requires full_episode_history=true")
    if int(data.get("max_history_blocks", -1)) != 70:
        raise ValueError("capacity probe requires max_history_blocks=70")
    if int(data.get("min_history_blocks", -1)) != 0:
        raise ValueError("capacity probe requires min_history_blocks=0")
    if int(data.get("replan_steps", -1)) != FORMAL_REPLAN_STEPS:
        raise ValueError("capacity probe data requires replan_steps=10")

    return {
        "batch_size_per_rank": batch_size,
        "world_size": FORMAL_WORLD_SIZE,
        "gradient_accumulation_steps": 1,
        "global_batch_size": FORMAL_WORLD_SIZE * batch_size,
        "optimizer_updates": optimizer_updates,
        "history_min": history_min,
        "history_max": history_max,
        "mixed_precision": "bf16",
        "zero_stage": 2,
        "causal_mode": causal_mode,
        "history_training_mode": "incremental_full_bptt",
        "history_vae_batch_chunk_size": 1,
        "world_model_conditioning": "lingbot_teacher_forced_v1",
        "num_video_frames": 9,
        "future_video_condition_noise_probability": 0.5,
        "future_video_condition_min_u": 0.5,
        "future_video_condition_max_u": 1.0,
        "training_strategy": "video_lora_action_full",
        "video_lora": {
            "rank": 16,
            "alpha": 16.0,
            "dropout": 0.0,
            "learning_rate_multiplier": 1.0,
        },
        "training_exit_depths": [30],
        "optimizer": "adamw_beta0.9_0.95_clip1.0",
        "optimizer_parameter_groups": {
            "action_and_aux": {"weight_decay": 1.0e-2},
            "video_lora": {"weight_decay": 0.0},
        },
        "checkpoints_forbidden": True,
        "wandb_forbidden": True,
    }


@dataclass(frozen=True)
class RealPrefixSelection:
    dataset_index: int
    frame_index: int
    episode_step: int
    history_blocks: int


def select_real_high_history_prefixes(
    dataset: Any,
    *,
    batch_size: int,
    history_min: int = FORMAL_HISTORY_MIN,
    history_max: int = FORMAL_HISTORY_MAX,
) -> list[RealPrefixSelection]:
    """Choose distinct, genuine dataset rows with the largest real prefixes."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    candidates: list[RealPrefixSelection] = []
    for dataset_index, frame_index_raw in enumerate(dataset._valid_replan_indices):
        frame_index = int(frame_index_raw)
        episode_step = int(dataset._episode_step[frame_index])
        if episode_step % int(dataset.replan_steps) != 0:
            raise ValueError("dataset contains a non-replanning-boundary sample")
        history_blocks = episode_step // int(dataset.replan_steps)
        if history_min <= history_blocks <= history_max:
            candidates.append(
                RealPrefixSelection(
                    dataset_index=int(dataset_index),
                    frame_index=frame_index,
                    episode_step=episode_step,
                    history_blocks=history_blocks,
                )
            )
    candidates.sort(key=lambda item: (-item.history_blocks, item.dataset_index))
    selected = candidates[:batch_size]
    if len(selected) != batch_size:
        raise ValueError(
            "dataset has too few distinct real H41-H50 samples: "
            f"required={batch_size} available={len(candidates)}"
        )
    indices = [item.dataset_index for item in selected]
    if len(indices) != len(set(indices)):
        raise AssertionError("capacity selection contains duplicate dataset rows")
    return selected


class FixedRealPrefixProbeDataset(Dataset):
    """Repeat a distinct real local batch across ranks and two updates.

    Repetition changes only which genuine dataset row is scheduled on a rank;
    it never synthesizes, extends, truncates or copies tensors inside a prefix.
    Accelerate shards consecutive DataLoader batches, so every rank receives
    the same list of distinct real rows for each update.
    """

    def __init__(
        self,
        dataset: Dataset,
        selections: list[RealPrefixSelection],
        *,
        world_size: int,
        optimizer_updates: int,
    ) -> None:
        if not selections:
            raise ValueError("capacity probe selections cannot be empty")
        if world_size <= 0 or optimizer_updates <= 0:
            raise ValueError("world_size/optimizer_updates must be positive")
        self.dataset = dataset
        self.selections = tuple(selections)
        self.world_size = int(world_size)
        self.optimizer_updates = int(optimizer_updates)
        one_batch = [item.dataset_index for item in self.selections]
        self._selection_by_index = {
            item.dataset_index: item for item in self.selections
        }
        self._indices = one_batch * (self.world_size * self.optimizer_updates)

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, index: int):
        dataset_index = self._indices[int(index)]
        sample = dict(self.dataset[dataset_index])
        selection = self._selection_by_index[dataset_index]
        sample["capacity_probe_source_dataset_index"] = torch.tensor(
            selection.dataset_index, dtype=torch.int64
        )
        sample["capacity_probe_expected_history_blocks"] = torch.tensor(
            selection.history_blocks, dtype=torch.int64
        )
        return sample


class CapacityProbeTrainer(Wan22Trainer):
    """Formal Trainer preparation with a checkpoint-free two-update loop."""

    def _build_loader(self, dataset, worker_init_fn=None):
        # Sequential local-sized batches are intentional. Accelerator assigns
        # one consecutive batch to every rank, preserving the fixed B-row real
        # stress batch constructed above.
        self.train_sampler = None
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=worker_init_fn,
            drop_last=True,
        )

    def save_checkpoint(self):  # pragma: no cover - guard against future refactors
        raise RuntimeError("capacity probe must never write a checkpoint or state shard")

    def _write_rank_oom(self, *, update: int, phase: str, error: BaseException) -> None:
        device = self.accelerator.device
        try:
            engine_step_state: dict[str, int | bool] | None = (
                read_deepspeed_engine_step_state(self.model)
            )
        except RuntimeError:
            engine_step_state = None
        payload = {
            "kind": "zero2_real_high_history_capacity_probe_rank_failure",
            "status": "oom",
            "rank": int(self.accelerator.process_index),
            "update": int(update),
            "phase": phase,
            "error_type": type(error).__name__,
            "error": str(error),
            "allocated_gib": float(torch.cuda.memory_allocated(device) / 2**30),
            "reserved_gib": float(torch.cuda.memory_reserved(device) / 2**30),
            "peak_allocated_gib": float(
                torch.cuda.max_memory_allocated(device) / 2**30
            ),
            "peak_reserved_gib": float(
                torch.cuda.max_memory_reserved(device) / 2**30
            ),
            "training_asset_manifest_sha256": os.environ.get(
                "LEAPBOT_TRAINING_ASSET_MANIFEST_SHA256"
            ),
            "engine_step_state": engine_step_state,
        }
        _atomic_json(
            Path(self.output_dir)
            / f"capacity_probe.rank_{self.accelerator.process_index:03d}.oom.json",
            payload,
        )

    def run_capacity_probe(
        self,
        *,
        selections: list[RealPrefixSelection],
        contract: dict[str, Any],
    ) -> dict[str, Any] | None:
        if self.distributed_training_topology is None:
            raise RuntimeError("capacity probe must run under DeepSpeed")
        topology = dict(self.distributed_training_topology)
        if topology.get("zero_stage") != 2:
            raise RuntimeError(f"capacity probe requires ZeRO-2, got {topology}")
        if int(self.accelerator.num_processes) != FORMAL_WORLD_SIZE:
            raise RuntimeError(
                "capacity probe requires exactly eight ranks, got "
                f"{self.accelerator.num_processes}"
            )
        deepspeed_engine = self.model
        initial_engine_step_state = read_deepspeed_engine_step_state(
            deepspeed_engine
        )
        optimizer_parameter_groups = read_formal_optimizer_parameter_groups(
            self.optimizer
        )

        self._set_dit_only_train_mode()
        model = (
            self.model
            if hasattr(self.model, "training_loss")
            else self.accelerator.unwrap_model(self.model)
        )
        data_iter = iter(self.train_loader)
        device = self.accelerator.device
        if device.type != "cuda":
            raise RuntimeError("capacity probe requires CUDA")
        torch.cuda.empty_cache()
        self.accelerator.wait_for_everyone()

        update_records: list[dict[str, Any]] = []
        all_losses_finite = True
        all_gradients_finite = True
        for update in range(1, FORMAL_OPTIMIZER_UPDATES + 1):
            phase = "load_batch"
            try:
                sample = next(data_iter)
                histories = sample["history_valid_blocks"].sum(dim=1).to(
                    device=device, dtype=torch.int64
                )
                source_indices = sample[
                    "capacity_probe_source_dataset_index"
                ].to(device=device, dtype=torch.int64)
                expected_histories = sample[
                    "capacity_probe_expected_history_blocks"
                ].to(device=device, dtype=torch.int64)
                if histories.numel() != self.batch_size:
                    raise RuntimeError(
                        "prepared capacity loader changed the per-rank batch size"
                    )
                if not bool(
                    ((histories >= FORMAL_HISTORY_MIN) & (histories <= FORMAL_HISTORY_MAX))
                    .all()
                    .item()
                ):
                    raise RuntimeError(
                        f"capacity loader emitted history outside H41-H50: {histories.tolist()}"
                    )
                if not torch.equal(histories, expected_histories):
                    raise RuntimeError(
                        "real prefix history differs from its dataset metadata: "
                        f"actual={histories.tolist()} expected={expected_histories.tolist()}"
                    )
                if int(torch.unique(source_indices).numel()) != self.batch_size:
                    raise RuntimeError(
                        "one capacity micro-batch must contain distinct real dataset rows"
                    )
                full_history = sample.get("full_episode_history")
                if full_history is None or not bool(full_history.all().item()):
                    raise RuntimeError("capacity sample is not marked full_episode_history")

                torch.cuda.reset_peak_memory_stats(device)
                baseline = torch.tensor(
                    [
                        torch.cuda.memory_allocated(device),
                        torch.cuda.memory_reserved(device),
                    ],
                    device=device,
                    dtype=torch.float64,
                )
                start = time.perf_counter()
                engine_step_before = read_deepspeed_engine_step_state(
                    deepspeed_engine
                )
                with self.accelerator.accumulate(self.model):
                    phase = "forward"
                    with self.accelerator.autocast():
                        loss, metrics = model.training_loss(sample)
                    local_loss_finite = bool(
                        torch.isfinite(loss.detach()).all().item()
                    )
                    if not local_loss_finite:
                        raise FloatingPointError("non-finite capacity-probe loss")
                    phase = "backward"
                    self.accelerator.backward(loss)
                    if not self.accelerator.sync_gradients:
                        raise RuntimeError(
                            "GA1 capacity probe unexpectedly disabled gradient sync"
                        )
                    phase = "gradient_clip_and_optimizer_step"
                    grad_norm = self._optimizer_step_with_validated_gradients()
                    if not math.isfinite(grad_norm):
                        raise FloatingPointError(
                            "non-finite capacity-probe gradient norm"
                        )
                    optimizer_step_was_skipped = bool(
                        self.accelerator.optimizer_step_was_skipped
                    )
                    if not optimizer_step_was_skipped:
                        self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                engine_step_after = read_deepspeed_engine_step_state(
                    deepspeed_engine
                )
                engine_step_evidence = validate_deepspeed_engine_step_transition(
                    engine_step_before,
                    engine_step_after,
                    optimizer_step_was_skipped=optimizer_step_was_skipped,
                )
                torch.cuda.synchronize(device)
                elapsed = time.perf_counter() - start

                local_memory = torch.stack(
                    [
                        baseline[0],
                        baseline[1],
                        torch.tensor(
                            torch.cuda.memory_allocated(device),
                            device=device,
                            dtype=torch.float64,
                        ),
                        torch.tensor(
                            torch.cuda.memory_reserved(device),
                            device=device,
                            dtype=torch.float64,
                        ),
                        torch.tensor(
                            torch.cuda.max_memory_allocated(device),
                            device=device,
                            dtype=torch.float64,
                        ),
                        torch.tensor(
                            torch.cuda.max_memory_reserved(device),
                            device=device,
                            dtype=torch.float64,
                        ),
                    ]
                ).reshape(1, 6)
                gathered_memory = self.accelerator.gather(local_memory).reshape(-1, 6)
                gathered_histories = self.accelerator.gather(histories).reshape(
                    self.accelerator.num_processes, self.batch_size
                )
                gathered_source_indices = self.accelerator.gather(
                    source_indices
                ).reshape(self.accelerator.num_processes, self.batch_size)
                gathered_losses = self.accelerator.gather(
                    loss.detach().float().reshape(1)
                ).reshape(-1)
                gathered_elapsed = self.accelerator.gather(
                    torch.tensor([elapsed], device=device, dtype=torch.float64)
                ).reshape(-1)
                local_engine_evidence = torch.tensor(
                    [
                        engine_step_evidence["engine_global_steps_before"],
                        engine_step_evidence["engine_global_steps_after"],
                        engine_step_evidence["engine_global_steps_delta"],
                        engine_step_evidence["engine_skipped_steps_before"],
                        engine_step_evidence["engine_skipped_steps_after"],
                        engine_step_evidence["engine_skipped_steps_delta"],
                        int(engine_step_evidence["engine_step_applied"]),
                        int(
                            engine_step_evidence[
                                "accelerator_optimizer_step_was_skipped"
                            ]
                        ),
                    ],
                    device=device,
                    dtype=torch.int64,
                ).reshape(1, 8)
                gathered_engine_evidence = self.accelerator.gather(
                    local_engine_evidence
                ).reshape(self.accelerator.num_processes, 8)
                if not bool(
                    (gathered_engine_evidence == gathered_engine_evidence[0]).all().item()
                ):
                    raise RuntimeError(
                        "DeepSpeed engine step evidence differs across ranks: "
                        f"{gathered_engine_evidence.cpu().tolist()}"
                    )
                all_losses_finite = all_losses_finite and bool(
                    torch.isfinite(gathered_losses).all().item()
                )
                all_gradients_finite = all_gradients_finite and math.isfinite(grad_norm)

                if self.accelerator.is_main_process:
                    per_rank = []
                    for rank, memory in enumerate(gathered_memory.cpu().tolist()):
                        per_rank.append(
                            {
                                "rank": rank,
                                "source_dataset_indices": gathered_source_indices[
                                    rank
                                ].cpu().tolist(),
                                "history_blocks": gathered_histories[rank].cpu().tolist(),
                                "loss": float(gathered_losses[rank].item()),
                                "elapsed_s": float(gathered_elapsed[rank].item()),
                                "baseline_allocated_gib": float(memory[0] / 2**30),
                                "baseline_reserved_gib": float(memory[1] / 2**30),
                                "end_allocated_gib": float(memory[2] / 2**30),
                                "end_reserved_gib": float(memory[3] / 2**30),
                                "peak_allocated_gib": float(memory[4] / 2**30),
                                "peak_reserved_gib": float(memory[5] / 2**30),
                            }
                        )
                    update_records.append(
                        {
                            "update": update,
                            "loss_mean": float(gathered_losses.mean().item()),
                            "grad_norm_mean_across_ranks": float(grad_norm),
                            "rank0_loss_metrics": {
                                key: float(value) for key, value in metrics.items()
                            },
                            "engine_step_evidence": engine_step_evidence,
                            "global_peak_allocated_gib": max(
                                row["peak_allocated_gib"] for row in per_rank
                            ),
                            "global_peak_reserved_gib": max(
                                row["peak_reserved_gib"] for row in per_rank
                            ),
                            "per_rank": per_rank,
                        }
                    )
            except torch.cuda.OutOfMemoryError as error:
                self._write_rank_oom(update=update, phase=phase, error=error)
                raise

        final_engine_step_state = read_deepspeed_engine_step_state(
            deepspeed_engine
        )
        engine_global_steps_delta = int(
            final_engine_step_state["global_steps"]
        ) - int(initial_engine_step_state["global_steps"])
        engine_skipped_steps_delta = int(
            final_engine_step_state["skipped_steps"]
        ) - int(initial_engine_step_state["skipped_steps"])
        if engine_global_steps_delta != FORMAL_OPTIMIZER_UPDATES:
            raise RuntimeError(
                "capacity probe did not execute exactly two DeepSpeed global steps: "
                f"start={initial_engine_step_state['global_steps']} "
                f"end={final_engine_step_state['global_steps']}"
            )
        if engine_skipped_steps_delta != 0:
            raise RuntimeError(
                "DeepSpeed recorded skipped optimizer steps during capacity probe"
            )

        if not self.accelerator.is_main_process:
            return None
        checkpoint_files = find_checkpoint_files(Path(self.output_dir))
        if checkpoint_files:
            raise RuntimeError(
                "capacity probe unexpectedly wrote checkpoint files: "
                f"{checkpoint_files}"
            )
        training_asset_manifest_sha256 = os.environ.get(
            "LEAPBOT_TRAINING_ASSET_MANIFEST_SHA256", ""
        )
        if re.fullmatch(r"[0-9a-f]{64}", training_asset_manifest_sha256) is None:
            raise RuntimeError(
                "capacity probe is missing a valid training asset manifest identity"
            )
        timeout_seconds_raw = os.environ.get(
            "LEAPBOT_CAPACITY_PROBE_TIMEOUT_SECONDS", ""
        )
        if not timeout_seconds_raw.isdigit() or int(timeout_seconds_raw) <= 0:
            raise RuntimeError("capacity probe is missing its wall-clock timeout contract")
        result = {
            "kind": "zero2_real_high_history_capacity_probe",
            "status": "passed",
            "code_commit": os.environ.get("LEAPBOT_CODE_COMMIT"),
            "contract": contract,
            "source_checkpoint": str(self.resume),
            "selection": [item.__dict__ for item in selections],
            "selection_provenance": (
                "distinct_real_dataset_rows_repeated_only_across_ranks_and_updates"
            ),
            "history_tensors_synthesized_or_extended": False,
            "engine_topology": topology,
            "engine_global_steps": {
                "start": int(initial_engine_step_state["global_steps"]),
                "end": int(final_engine_step_state["global_steps"]),
                "delta": engine_global_steps_delta,
            },
            "engine_skipped_steps": {
                "start": int(initial_engine_step_state["skipped_steps"]),
                "end": int(final_engine_step_state["skipped_steps"]),
                "delta": engine_skipped_steps_delta,
            },
            "trainable_parameters": int(
                sum(
                    parameter.numel()
                    for parameter in self.accelerator.unwrap_model(self.model).parameters()
                    if parameter.requires_grad
                )
            ),
            "optimizer": {
                "name": "AdamW",
                "betas": [0.9, 0.95],
                "max_grad_norm": 1.0,
                "parameter_groups": optimizer_parameter_groups,
            },
            "updates": update_records,
            "all_losses_finite": all_losses_finite,
            "all_gradients_finite": all_gradients_finite,
            "global_peak_allocated_gib": max(
                record["global_peak_allocated_gib"] for record in update_records
            ),
            "global_peak_reserved_gib": max(
                record["global_peak_reserved_gib"] for record in update_records
            ),
            "wandb_used": False,
            "checkpoint_files_written": [],
            "training_asset_manifest_sha256": training_asset_manifest_sha256,
            "timeout_seconds": int(timeout_seconds_raw),
        }
        _atomic_json(Path(self.output_dir) / "capacity_probe.json", result)
        return result


def run_probe(cfg: DictConfig) -> dict[str, Any] | None:
    contract = validate_capacity_probe_contract(cfg)
    output_dir = Path(str(cfg.output_dir)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    misc.register_work_dir(str(output_dir))
    # Accelerator/DeepSpeed is constructed below.  Before that point all eight
    # Python processes may have an uninitialized torch.distributed backend, so
    # the launcher-provided global rank is the race-free writer identity.
    is_main = int(os.environ.get("RANK", "0")) == 0
    setup_logging(log_level=logging.INFO, is_main_process=is_main)
    if is_main:
        OmegaConf.save(
            OmegaConf.create(OmegaConf.to_container(cfg, resolve=True)),
            output_dir / "config.yaml",
        )

    model_device = _resolve_train_device()
    model_dtype = _mixed_precision_to_model_dtype(str(cfg.mixed_precision))
    _seed_model_initialization(int(cfg.seed))
    model = instantiate(cfg.model, model_dtype=model_dtype, device=model_device)
    base_dataset = instantiate(cfg.data.train)
    selections = select_real_high_history_prefixes(
        base_dataset,
        batch_size=int(cfg.batch_size),
        history_min=int(contract["history_min"]),
        history_max=int(contract["history_max"]),
    )
    probe_dataset = FixedRealPrefixProbeDataset(
        base_dataset,
        selections,
        world_size=FORMAL_WORLD_SIZE,
        optimizer_updates=FORMAL_OPTIMIZER_UPDATES,
    )
    trainer = CapacityProbeTrainer(
        cfg=cfg,
        model=model,
        train_dataset=probe_dataset,
        val_dataset=None,
    )
    return trainer.run_capacity_probe(selections=selections, contract=contract)


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig) -> None:
    run_probe(cfg)


if __name__ == "__main__":
    main()
