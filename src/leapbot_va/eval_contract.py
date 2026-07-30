"""Resolve LeapBot LIBERO evaluation behavior into canonical contracts."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any, Mapping

from omegaconf import DictConfig, OmegaConf

from leapbot_va.eval_fingerprint import normalize_json_value, sha256_file


def _to_resolved_container(value: Any) -> Any:
    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=True)
    return normalize_json_value(value)


def resolve_dataset_stats_path(cfg: DictConfig) -> Path:
    """Use exactly the dataset-stat search order used by evaluation."""
    explicit = cfg.EVALUATION.get("dataset_stats_path")
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(Path(os.path.expanduser(os.path.expandvars(str(explicit)))))

    checkpoint = Path(os.path.expanduser(os.path.expandvars(str(cfg.ckpt))))
    for parent in list(checkpoint.parents)[:4]:
        candidates.append(parent / "dataset_stats.json")

    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.is_file():
            return resolved
    raise FileNotFoundError(
        "Failed to locate dataset_stats.json. Tried explicit "
        "EVALUATION.dataset_stats_path and checkpoint parent directories. "
        "Please pass EVALUATION.dataset_stats_path=/path/to/dataset_stats.json."
    )


def resolve_libero_task_and_initial_states(cfg: DictConfig):
    """Resolve the task and trusted official initial-state file."""
    from libero.libero import benchmark, get_libero_path

    suite_name = str(cfg.EVALUATION.task_suite_name)
    task_id = int(cfg.EVALUATION.task_id)
    benchmark_dict = benchmark.get_benchmark_dict()
    if suite_name not in benchmark_dict:
        raise ValueError(f"Unknown LIBERO benchmark suite: {suite_name}")
    task_suite = benchmark_dict[suite_name]()
    task = task_suite.get_task(task_id)
    initial_states_path = (
        Path(get_libero_path("init_states"))
        / task.problem_folder
        / task.init_states_file
    ).resolve()
    if not initial_states_path.is_file():
        raise FileNotFoundError(
            f"LIBERO initial-state file does not exist: {initial_states_path}"
        )
    return task_suite, task, initial_states_path


def _git_source_identity(source_root: str | Path) -> dict[str, Any]:
    root = Path(source_root).resolve()

    def _git(*args: str) -> str:
        completed = subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return completed.stdout.strip()

    revision = _git("rev-parse", "HEAD")
    dirty_output = _git("status", "--porcelain", "--untracked-files=normal")
    return {"revision": revision, "dirty": bool(dirty_output)}


def _task_choice_from_mapping(hydra_choices: Mapping[str, Any] | None) -> str:
    choice = None if hydra_choices is None else hydra_choices.get("task")
    if choice is None or not str(choice).strip():
        raise ValueError("Hydra runtime choice 'task' is required by evaluation contract")
    return str(choice)


def build_runtime_contract(
    cfg: DictConfig,
    *,
    config_name: str,
    hydra_choices: Mapping[str, Any] | None,
    dataset_stats_path: str | Path,
    source_root: str | Path,
    configured_causal_mode: str | None = None,
) -> dict[str, Any]:
    """Expand every inference-relevant setting into a stable JSON contract."""
    evaluation = cfg.EVALUATION
    memory_raw = evaluation.get("memory", {})
    memory_cfg = _to_resolved_container(memory_raw or {})
    memory_enabled = bool(memory_cfg.get("enabled", False))
    episode_capacity = (
        int(memory_cfg.get("max_history_blocks", 0)) if memory_enabled else 0
    )
    retained = memory_cfg.get("retained_history_blocks", None) if memory_enabled else 0
    if retained is not None:
        retained = int(retained)
    if episode_capacity < 0 or (retained is not None and retained < 0):
        raise ValueError("memory capacity/retention must be non-negative")
    if retained is not None and retained > episode_capacity:
        raise ValueError("retained_history_blocks cannot exceed max_history_blocks")

    action_horizon_raw = evaluation.get("action_horizon", None)
    action_horizon = (
        int(cfg.data.train.num_frames) - 1
        if action_horizon_raw is None
        else int(action_horizon_raw)
    )
    inference_steps_raw = evaluation.get("num_inference_steps", None)
    inference_steps = (
        int(cfg.get("eval_num_inference_steps", 20))
        if inference_steps_raw is None
        else int(inference_steps_raw)
    )
    video_size = list(cfg.data.train.get("video_size", [224, 224]))
    if len(video_size) != 2:
        raise ValueError(f"data.train.video_size must be [H, W], got {video_size}")
    if action_horizon <= 0 or inference_steps <= 0:
        raise ValueError("action_horizon and num_inference_steps must be positive")

    mode = configured_causal_mode
    if mode is None:
        mode = cfg.model.get("causal_mode", None)
    memory_mode = memory_cfg.get("causal_mode", mode) if memory_enabled else None
    processor_config = _to_resolved_container(cfg.data.train.processor)
    model_config = _to_resolved_container(cfg.model)
    dataset_stats = Path(dataset_stats_path).resolve()

    contract = {
        "config": {
            "name": str(config_name),
            "task_choice": _task_choice_from_mapping(hydra_choices),
        },
        "source": _git_source_identity(source_root),
        "seed": None if cfg.get("seed") is None else int(cfg.seed),
        "inference": {
            "num_inference_steps": inference_steps,
            "replan_steps": int(evaluation.get("replan_steps", 5)),
            "action_horizon": action_horizon,
            "num_steps_wait": int(evaluation.get("num_steps_wait", 5)),
            "sigma_shift": (
                None
                if evaluation.get("sigma_shift", None) is None
                else float(evaluation.sigma_shift)
            ),
            "text_cfg_scale": float(evaluation.get("text_cfg_scale", 1.0)),
            "negative_prompt": str(evaluation.get("negative_prompt", "")),
            "rand_device": str(evaluation.get("rand_device", "cpu")),
            "tiled": bool(evaluation.get("tiled", False)),
            "visualize_future_video": bool(
                evaluation.get("visualize_future_video", False)
            ),
        },
        "action_execution": {
            "binarize_gripper": bool(evaluation.get("binarize_gripper", False)),
            "invert_gripper_action": True,
            "use_action_ensembler": bool(
                evaluation.get("use_action_ensembler", False)
            ),
        },
        "normalization": {
            "dataset_stats_sha256": sha256_file(dataset_stats),
        },
        "precision_and_adapters": {
            "mixed_precision": str(cfg.get("mixed_precision", "bf16")).lower(),
            "merge_video_lora": bool(evaluation.get("merge_video_lora", False)),
        },
        "memory": {
            "enabled": memory_enabled,
            "causal_mode": str(memory_mode) if memory_mode is not None else None,
            "exit_depth": int(memory_cfg.get("exit_depth", 0)) if memory_enabled else 0,
            "episode_capacity": episode_capacity,
            "retained_history_blocks": retained,
            "effective_history_cap": (
                episode_capacity if retained is None else int(retained)
            ),
        },
        "input": {
            "height": int(video_size[0]),
            "width": int(video_size[1]),
            "concat_multi_camera": cfg.data.train.get(
                "concat_multi_camera", None
            ),
            "num_frames": int(cfg.data.train.num_frames),
            "action_video_freq_ratio": int(
                cfg.data.train.action_video_freq_ratio
            ),
            "processor": processor_config,
        },
        # The complete resolved model subtree catches future behavior settings
        # that predate an explicit field in this contract builder.
        "model": model_config,
    }
    return normalize_json_value(contract)


def build_result_contract(
    cfg: DictConfig,
    *,
    task: Any,
    initial_states_path: str | Path,
) -> dict[str, Any]:
    trials = int(cfg.EVALUATION.num_trials)
    if trials <= 0:
        raise ValueError("EVALUATION.num_trials must be positive")
    initial_states = Path(initial_states_path).resolve()
    contract = {
        "suite": str(cfg.EVALUATION.task_suite_name),
        "task": {
            "id": int(cfg.EVALUATION.task_id),
            "problem_folder": str(task.problem_folder),
            "init_states_file": str(task.init_states_file),
        },
        "trials": trials,
        "initial_states_sha256": sha256_file(initial_states),
    }
    return normalize_json_value(contract)
