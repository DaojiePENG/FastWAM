#!/usr/bin/env python3
"""Aggregate LIBERO task JSON files and select LeapBot Pareto configurations."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

from leapbot_va.eval_contract import (
    KV_RETENTION_SEMANTICS,
    STRICT_REPLAY_SEMANTICS,
)
from leapbot_va.eval_fingerprint import normalize_evaluation_fingerprint


def percentile(values: list[float], q: float) -> float:
    if not values:
        return math.inf
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - position) + ordered[high] * (position - low)


def optional_percentile(values: list[float], q: float) -> float | None:
    return percentile(values, q) if values else None


def _validated_kv_retention_contract(
    memory: dict,
    *,
    context: str,
) -> tuple[str, int]:
    """Validate that a run describes physical KV retention, not a history window."""

    if not isinstance(memory, dict):
        raise TypeError(f"{context}: runtime memory contract must be an object")
    storage_mode = memory.get("history_storage_mode", "incremental_kv")
    expected_semantics = (
        STRICT_REPLAY_SEMANTICS
        if storage_mode == "strict_replay"
        else KV_RETENTION_SEMANTICS
    )
    semantics = memory.get("retention_semantics")
    if semantics != expected_semantics:
        raise ValueError(
            f"{context}: memory.retention_semantics must be "
            f"{expected_semantics!r}, got {semantics!r}"
        )

    integer_fields = (
        "episode_capacity",
        "effective_kv_retention_cap",
        "effective_history_cap",
    )
    values = {}
    for field in integer_fields:
        value = memory.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(
                f"{context}: memory.{field} must be a non-negative integer"
            )
        values[field] = value

    retained = memory.get("retained_history_blocks")
    if retained is not None and (
        isinstance(retained, bool) or not isinstance(retained, int) or retained < 0
    ):
        raise ValueError(
            f"{context}: memory.retained_history_blocks must be a non-negative "
            "integer or null"
        )
    episode_capacity = values["episode_capacity"]
    if retained is not None and retained > episode_capacity:
        raise ValueError(
            f"{context}: memory.retained_history_blocks cannot exceed "
            "memory.episode_capacity"
        )
    if storage_mode == "strict_replay":
        history_window = memory.get("history_window_blocks")
        if (
            isinstance(history_window, bool)
            or not isinstance(history_window, int)
            or history_window <= 0
            or history_window > episode_capacity
        ):
            raise ValueError(
                f"{context}: strict replay history_window_blocks must be in "
                "[1,episode_capacity]"
            )
        expected_cap = history_window
    else:
        expected_cap = episode_capacity if retained is None else retained
    kv_cap = values["effective_kv_retention_cap"]
    if kv_cap != expected_cap:
        raise ValueError(
            f"{context}: memory.effective_kv_retention_cap={kv_cap} does not "
            f"match the configured physical KV cap {expected_cap}"
        )
    if values["effective_history_cap"] != kv_cap:
        raise ValueError(
            f"{context}: compatibility alias memory.effective_history_cap="
            f"{values['effective_history_cap']} does not match "
            f"memory.effective_kv_retention_cap={kv_cap}"
        )
    return semantics, kv_cap


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return center - radius, center + radius


def config_key(payload: dict, path: Path) -> str:
    """Return a readable identity that cannot merge distinct run contracts."""

    try:
        fingerprint = normalize_evaluation_fingerprint(
            payload["evaluation_fingerprint"]
        )
    except (KeyError, TypeError, ValueError):
        # ``validate_inputs`` rejects this legacy identity before formal
        # aggregation.  The fallback keeps standalone diagnostic helpers able
        # to explain which old file was encountered.
        return f"legacy/{path.stem}"
    runtime = fingerprint["runtime_contract"]
    memory = runtime["memory"]
    _, kv_retention_cap = _validated_kv_retention_contract(
        memory,
        context=str(path),
    )
    runtime_tag = fingerprint["runtime_contract_sha256"][:12]
    checkpoint_tag = fingerprint["checkpoint_sha256"][:12]
    if memory["enabled"]:
        return "/".join(
            [
                str(memory["causal_mode"]),
                f"d{memory['exit_depth']}",
                f"kvret{kv_retention_cap}",
                f"cap{memory['episode_capacity']}",
                f"rt-{runtime_tag}",
                f"ckpt-{checkpoint_tag}",
            ]
        )
    config_name = str(runtime["config"]["name"])
    family = "fastwam_release" if config_name == "sim_libero" else "leapbot_no_memory"
    return f"{family}/rt-{runtime_tag}/ckpt-{checkpoint_tag}"


def model_family(payload: dict) -> str:
    """Return the explicit model family used by formal model selection."""

    fingerprint = normalize_evaluation_fingerprint(
        payload["evaluation_fingerprint"]
    )
    runtime = fingerprint["runtime_contract"]
    if runtime["memory"]["enabled"]:
        return "leapbot_memory"
    if str(runtime["config"]["name"]) == "sim_libero":
        return "fastwam_release"
    return "leapbot_no_memory"


def _new_group() -> dict:
    return {
        "model_family": None,
        "memory_enabled": None,
        "retention_semantics": None,
        "effective_kv_retention_cap": None,
        "successes": 0,
        "episodes": 0,
        "completion_steps": [],
        "latency": [],
        "input_preprocess": [],
        "model_inference": [],
        "action_postprocess": [],
        "conditioning": [],
        "observation_prefill": [],
        "future_video_setup": [],
        "future_video_denoise": [],
        "future_video_cache": [],
        "action_setup": [],
        "action_denoise": [],
        "causal_model": [],
        "action_commit": [],
        "cache": [],
        "transient_future_video_cache": [],
        "gpu": [],
    }


def aggregate(paths: list[Path]) -> list[dict]:
    groups = defaultdict(_new_group)
    for path in paths:
        payload = json.loads(path.read_text())
        key = config_key(payload, path)
        group = groups[key]
        family = model_family(payload)
        memory_enabled = family == "leapbot_memory"
        fingerprint = normalize_evaluation_fingerprint(
            payload["evaluation_fingerprint"]
        )
        retention_semantics, kv_retention_cap = _validated_kv_retention_contract(
            fingerprint["runtime_contract"]["memory"],
            context=str(path),
        )
        if group["model_family"] not in (None, family):
            raise ValueError(f"{key}: mixed model families in one aggregate group")
        if group["memory_enabled"] not in (None, memory_enabled):
            raise ValueError(f"{key}: mixed memory contracts in one aggregate group")
        if group["retention_semantics"] not in (None, retention_semantics):
            raise ValueError(f"{key}: mixed KV retention semantics in one group")
        if group["effective_kv_retention_cap"] not in (None, kv_retention_cap):
            raise ValueError(f"{key}: mixed physical KV retention caps in one group")
        group["model_family"] = family
        group["memory_enabled"] = memory_enabled
        group["retention_semantics"] = retention_semantics
        group["effective_kv_retention_cap"] = kv_retention_cap
        group["successes"] += int(payload.get("successes", 0))
        group["episodes"] += int(payload.get("total_episodes", 0))
        group["completion_steps"].extend(
            float(step) for step in payload.get("completion_steps", [])
        )
        for episode in payload.get("memory_metrics", []):
            group["cache"].append(float(episode.get("peak_cache_bytes", 0)))
            if "peak_transient_future_video_cache_bytes" in episode:
                group["transient_future_video_cache"].append(
                    float(episode["peak_transient_future_video_cache_bytes"])
                )
            if "peak_gpu_bytes" in episode:
                group["gpu"].append(float(episode["peak_gpu_bytes"]))
            for replan in episode.get("replans", []):
                timing = replan.get("timing", {})
                for field, target in (
                    ("input_preprocess_s", "input_preprocess"),
                    ("model_inference_s", "model_inference"),
                    ("action_postprocess_s", "action_postprocess"),
                    ("conditioning_s", "conditioning"),
                    ("future_video_setup_s", "future_video_setup"),
                    ("future_video_denoise_s", "future_video_denoise"),
                    ("future_video_cache_s", "future_video_cache"),
                    ("action_setup_s", "action_setup"),
                    ("causal_model_s", "causal_model"),
                ):
                    if field in timing:
                        group[target].append(float(timing[field]))
                observation = float(timing.get("observation_prefill_s", 0))
                denoise = float(timing.get("action_denoise_s", 0))
                commit = float(replan.get("commit", {}).get("commit_s", 0))
                if "observation_prefill_s" in timing:
                    group["observation_prefill"].append(observation)
                if "action_denoise_s" in timing:
                    group["action_denoise"].append(denoise)
                if "commit_s" in replan.get("commit", {}):
                    group["action_commit"].append(commit)
                future_video = sum(
                    float(timing.get(field, 0))
                    for field in (
                        "future_video_setup_s",
                        "future_video_denoise_s",
                        "future_video_cache_s",
                    )
                )
                total = float(
                    timing.get(
                        "total_inference_s", observation + future_video + denoise
                    )
                ) + commit
                group["latency"].append(total)

    rows = []
    for key, values in groups.items():
        rate = values["successes"] / values["episodes"] if values["episodes"] else 0.0
        ci_low, ci_high = wilson(values["successes"], values["episodes"])
        rows.append(
            {
                "config": key,
                "model_family": values["model_family"],
                "memory_enabled": values["memory_enabled"],
                "retention_semantics": values["retention_semantics"],
                "effective_kv_retention_cap": values[
                    "effective_kv_retention_cap"
                ],
                "successes": values["successes"],
                "episodes": values["episodes"],
                "success_rate": rate,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "mean_completion_steps": (
                    sum(values["completion_steps"]) / len(values["completion_steps"])
                    if values["completion_steps"]
                    else math.inf
                ),
                "p50_completion_steps": percentile(values["completion_steps"], 0.50),
                "p95_completion_steps": percentile(values["completion_steps"], 0.95),
                "p50_latency_s": percentile(values["latency"], 0.50),
                "p95_latency_s": percentile(values["latency"], 0.95),
                "p50_input_preprocess_s": optional_percentile(values["input_preprocess"], 0.50),
                "p95_input_preprocess_s": optional_percentile(values["input_preprocess"], 0.95),
                "p50_model_inference_s": optional_percentile(values["model_inference"], 0.50),
                "p95_model_inference_s": optional_percentile(values["model_inference"], 0.95),
                "p50_action_postprocess_s": optional_percentile(values["action_postprocess"], 0.50),
                "p95_action_postprocess_s": optional_percentile(values["action_postprocess"], 0.95),
                "p50_conditioning_s": optional_percentile(values["conditioning"], 0.50),
                "p95_conditioning_s": optional_percentile(values["conditioning"], 0.95),
                "p50_observation_prefill_s": optional_percentile(values["observation_prefill"], 0.50),
                "p95_observation_prefill_s": optional_percentile(values["observation_prefill"], 0.95),
                "p50_future_video_setup_s": optional_percentile(values["future_video_setup"], 0.50),
                "p95_future_video_setup_s": optional_percentile(values["future_video_setup"], 0.95),
                "p50_future_video_denoise_s": optional_percentile(values["future_video_denoise"], 0.50),
                "p95_future_video_denoise_s": optional_percentile(values["future_video_denoise"], 0.95),
                "p50_future_video_cache_s": optional_percentile(values["future_video_cache"], 0.50),
                "p95_future_video_cache_s": optional_percentile(values["future_video_cache"], 0.95),
                "p50_action_setup_s": optional_percentile(values["action_setup"], 0.50),
                "p95_action_setup_s": optional_percentile(values["action_setup"], 0.95),
                "p50_action_denoise_s": optional_percentile(values["action_denoise"], 0.50),
                "p95_action_denoise_s": optional_percentile(values["action_denoise"], 0.95),
                "p50_causal_model_s": optional_percentile(values["causal_model"], 0.50),
                "p95_causal_model_s": optional_percentile(values["causal_model"], 0.95),
                "p50_action_commit_s": optional_percentile(values["action_commit"], 0.50),
                "p95_action_commit_s": optional_percentile(values["action_commit"], 0.95),
                "peak_cache_gib": max(values["cache"], default=0) / 2**30,
                "peak_transient_future_video_cache_gib": max(
                    values["transient_future_video_cache"], default=0
                )
                / 2**30,
                "peak_gpu_gib": max(values["gpu"], default=0) / 2**30,
            }
        )
    return sorted(rows, key=lambda row: row["config"])


def aggregate_per_task(paths: list[Path]) -> list[dict]:
    groups = defaultdict(_new_group)
    descriptions = {}
    for path in paths:
        payload = json.loads(path.read_text())
        key = config_key(payload, path)
        task_id = int(payload.get("task_id", -1))
        group_key = (key, task_id)
        group = groups[group_key]
        group["successes"] += int(payload.get("successes", 0))
        group["episodes"] += int(payload.get("total_episodes", 0))
        group["completion_steps"].extend(
            float(step) for step in payload.get("completion_steps", [])
        )
        descriptions[group_key] = str(payload.get("task_description", ""))

    rows = []
    for (key, task_id), values in groups.items():
        rate = values["successes"] / values["episodes"] if values["episodes"] else 0.0
        ci_low, ci_high = wilson(values["successes"], values["episodes"])
        rows.append(
            {
                "config": key,
                "task_id": task_id,
                "task_description": descriptions[(key, task_id)],
                "successes": values["successes"],
                "episodes": values["episodes"],
                "success_rate": rate,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "mean_completion_steps": (
                    sum(values["completion_steps"]) / len(values["completion_steps"])
                    if values["completion_steps"]
                    else math.inf
                ),
                "p50_completion_steps": percentile(values["completion_steps"], 0.50),
                "p95_completion_steps": percentile(values["completion_steps"], 0.95),
            }
        )
    return sorted(rows, key=lambda row: (row["config"], row["task_id"]))


def aggregate_by_kv_retention(paths: list[Path]) -> list[dict]:
    """Aggregate against physically retained KV blocks, not an information window."""
    groups = defaultdict(
        lambda: {
            "samples": 0,
            "cache_after_observation": [],
            "cache_after_commit": [],
            "transient_future_video_cache": [],
            "input_preprocess": [],
            "model_inference": [],
            "action_postprocess": [],
            "conditioning": [],
            "observation_prefill": [],
            "future_video_setup": [],
            "future_video_denoise": [],
            "future_video_cache": [],
            "action_setup": [],
            "action_denoise": [],
            "causal_model": [],
            "action_commit": [],
            "total_replan": [],
            "episode_blocks": [],
        }
    )
    for path in paths:
        payload = json.loads(path.read_text())
        key = config_key(payload, path)
        fingerprint = normalize_evaluation_fingerprint(
            payload["evaluation_fingerprint"]
        )
        runtime_memory = fingerprint["runtime_contract"]["memory"]
        retention_semantics, retention_cap = _validated_kv_retention_contract(
            runtime_memory,
            context=str(path),
        )
        for episode in payload.get("memory_metrics", []):
            for replan in episode.get("replans", []):
                memory = replan.get("memory", {})
                if "completed_blocks" not in memory:
                    continue
                episode_blocks = int(memory["completed_blocks"])
                retained_kv_blocks = int(
                    memory.get(
                        "retained_history_blocks",
                        min(episode_blocks, retention_cap),
                    )
                )
                expected_retained = min(episode_blocks, retention_cap)
                if retained_kv_blocks != expected_retained:
                    raise ValueError(
                        f"{path}: runtime retained KV blocks={retained_kv_blocks} "
                        "do not "
                        f"match min(episode_blocks={episode_blocks}, "
                        f"kv_retention_cap={retention_cap})={expected_retained}"
                    )
                group = groups[
                    (
                        key,
                        retention_semantics,
                        retention_cap,
                        retained_kv_blocks,
                    )
                ]
                group["samples"] += 1
                group["episode_blocks"].append(float(episode_blocks))
                if "cache_bytes" in memory:
                    group["cache_after_observation"].append(float(memory["cache_bytes"]))
                if "transient_future_video_cache_bytes" in memory:
                    group["transient_future_video_cache"].append(
                        float(memory["transient_future_video_cache_bytes"])
                    )

                timing = replan.get("timing", {})
                commit = replan.get("commit", {})
                for field, target in (
                    ("input_preprocess_s", "input_preprocess"),
                    ("model_inference_s", "model_inference"),
                    ("action_postprocess_s", "action_postprocess"),
                    ("conditioning_s", "conditioning"),
                    ("future_video_setup_s", "future_video_setup"),
                    ("future_video_denoise_s", "future_video_denoise"),
                    ("future_video_cache_s", "future_video_cache"),
                    ("action_setup_s", "action_setup"),
                    ("causal_model_s", "causal_model"),
                ):
                    if field in timing:
                        group[target].append(float(timing[field]))
                if "cache_bytes" in commit:
                    group["cache_after_commit"].append(float(commit["cache_bytes"]))
                if "observation_prefill_s" in timing:
                    group["observation_prefill"].append(
                        float(timing["observation_prefill_s"])
                    )
                if "action_denoise_s" in timing:
                    group["action_denoise"].append(float(timing["action_denoise_s"]))
                if "commit_s" in commit:
                    group["action_commit"].append(float(commit["commit_s"]))
                total = float(
                    timing.get(
                        "total_inference_s",
                        float(timing.get("observation_prefill_s", 0))
                        + float(timing.get("future_video_setup_s", 0))
                        + float(timing.get("future_video_denoise_s", 0))
                        + float(timing.get("future_video_cache_s", 0))
                        + float(timing.get("action_denoise_s", 0)),
                    )
                ) + float(commit.get("commit_s", 0))
                group["total_replan"].append(total)

    def cache_percentile(values: list[float], q: float) -> float | None:
        value = optional_percentile(values, q)
        return None if value is None else value / 2**30

    rows = []
    for (
        key,
        retention_semantics,
        retention_cap,
        retained_kv_blocks,
    ), values in groups.items():
        rows.append(
            {
                "config": key,
                "retention_semantics": retention_semantics,
                "effective_kv_retention_cap": retention_cap,
                "kv_retained_blocks_before_replan": retained_kv_blocks,
                # Compatibility alias for existing plotting/report consumers.
                "history_blocks_before_replan": retained_kv_blocks,
                "p50_episode_blocks_before_replan": optional_percentile(
                    values["episode_blocks"], 0.50
                ),
                "p95_episode_blocks_before_replan": optional_percentile(
                    values["episode_blocks"], 0.95
                ),
                "samples": values["samples"],
                "p50_cache_after_observation_gib": cache_percentile(
                    values["cache_after_observation"], 0.50
                ),
                "p95_cache_after_observation_gib": cache_percentile(
                    values["cache_after_observation"], 0.95
                ),
                "p50_cache_after_commit_gib": cache_percentile(
                    values["cache_after_commit"], 0.50
                ),
                "p95_cache_after_commit_gib": cache_percentile(
                    values["cache_after_commit"], 0.95
                ),
                "p50_transient_future_video_cache_gib": cache_percentile(
                    values["transient_future_video_cache"], 0.50
                ),
                "p95_transient_future_video_cache_gib": cache_percentile(
                    values["transient_future_video_cache"], 0.95
                ),
                "p50_input_preprocess_s": optional_percentile(
                    values["input_preprocess"], 0.50
                ),
                "p95_input_preprocess_s": optional_percentile(
                    values["input_preprocess"], 0.95
                ),
                "p50_model_inference_s": optional_percentile(
                    values["model_inference"], 0.50
                ),
                "p95_model_inference_s": optional_percentile(
                    values["model_inference"], 0.95
                ),
                "p50_action_postprocess_s": optional_percentile(
                    values["action_postprocess"], 0.50
                ),
                "p95_action_postprocess_s": optional_percentile(
                    values["action_postprocess"], 0.95
                ),
                "p50_conditioning_s": optional_percentile(
                    values["conditioning"], 0.50
                ),
                "p95_conditioning_s": optional_percentile(
                    values["conditioning"], 0.95
                ),
                "p50_observation_prefill_s": optional_percentile(
                    values["observation_prefill"], 0.50
                ),
                "p95_observation_prefill_s": optional_percentile(
                    values["observation_prefill"], 0.95
                ),
                "p50_future_video_setup_s": optional_percentile(
                    values["future_video_setup"], 0.50
                ),
                "p95_future_video_setup_s": optional_percentile(
                    values["future_video_setup"], 0.95
                ),
                "p50_future_video_denoise_s": optional_percentile(
                    values["future_video_denoise"], 0.50
                ),
                "p95_future_video_denoise_s": optional_percentile(
                    values["future_video_denoise"], 0.95
                ),
                "p50_future_video_cache_s": optional_percentile(
                    values["future_video_cache"], 0.50
                ),
                "p95_future_video_cache_s": optional_percentile(
                    values["future_video_cache"], 0.95
                ),
                "p50_action_setup_s": optional_percentile(
                    values["action_setup"], 0.50
                ),
                "p95_action_setup_s": optional_percentile(
                    values["action_setup"], 0.95
                ),
                "p50_action_denoise_s": optional_percentile(
                    values["action_denoise"], 0.50
                ),
                "p95_action_denoise_s": optional_percentile(
                    values["action_denoise"], 0.95
                ),
                "p50_causal_model_s": optional_percentile(
                    values["causal_model"], 0.50
                ),
                "p95_causal_model_s": optional_percentile(
                    values["causal_model"], 0.95
                ),
                "p50_action_commit_s": optional_percentile(
                    values["action_commit"], 0.50
                ),
                "p95_action_commit_s": optional_percentile(
                    values["action_commit"], 0.95
                ),
                "p50_total_replan_s": optional_percentile(
                    values["total_replan"], 0.50
                ),
                "p95_total_replan_s": optional_percentile(
                    values["total_replan"], 0.95
                ),
            }
        )
    return sorted(
        rows,
        key=lambda row: (row["config"], row["kv_retained_blocks_before_replan"]),
    )


def aggregate_by_history(paths: list[Path]) -> list[dict]:
    """Compatibility alias for :func:`aggregate_by_kv_retention`."""

    return aggregate_by_kv_retention(paths)


def validate_inputs(
    paths: list[Path],
    expected_tasks: int | None = None,
    expected_trials_per_task: int | None = None,
    require_profiled: bool = False,
) -> None:
    """Reject incomplete/duplicated evaluation sets before model selection."""
    groups: dict[str, dict[int, Path]] = defaultdict(dict)
    errors: list[str] = []
    for path in paths:
        payload = json.loads(path.read_text())
        try:
            fingerprint = normalize_evaluation_fingerprint(
                payload["evaluation_fingerprint"]
            )
            key = config_key(payload, path)
        except (KeyError, TypeError, ValueError) as error:
            key = f"invalid/{path.stem}"
            fingerprint = None
            errors.append(f"{path}: invalid or legacy evaluation fingerprint: {error}")
        task_id = int(payload.get("task_id", -1))
        if task_id in groups[key]:
            errors.append(
                f"{key}: duplicate task {task_id}: {groups[key][task_id]} and {path}"
            )
        groups[key][task_id] = path

        episodes = int(payload.get("total_episodes", 0))
        if fingerprint is not None:
            result_contract = fingerprint["result_contract"]
            runtime_memory = fingerprint["runtime_contract"]["memory"]
            if str(payload.get("task_suite")) != str(result_contract["suite"]):
                errors.append(
                    f"{key}/task{task_id}: payload task_suite does not match fingerprint"
                )
            if task_id != int(result_contract["task"]["id"]):
                errors.append(
                    f"{key}/task{task_id}: payload task_id does not match fingerprint"
                )
            if episodes != int(result_contract["trials"]):
                errors.append(
                    f"{key}/task{task_id}: total_episodes does not match fingerprint trials"
                )

            payload_memory = payload.get("memory_config") or {}
            payload_enabled = bool(payload_memory.get("enabled", False))
            if payload_enabled != bool(runtime_memory["enabled"]):
                errors.append(
                    f"{key}/task{task_id}: memory enabled flag does not match fingerprint"
                )
            if payload_enabled:
                expected_memory = {
                    "causal_mode": str(runtime_memory["causal_mode"]),
                    "exit_depth": int(runtime_memory["exit_depth"]),
                    "max_history_blocks": int(runtime_memory["episode_capacity"]),
                    "retained_history_blocks": runtime_memory[
                        "retained_history_blocks"
                    ],
                }
                actual_memory = {
                    "causal_mode": str(payload_memory.get("causal_mode")),
                    "exit_depth": int(payload_memory.get("exit_depth", -1)),
                    "max_history_blocks": int(
                        payload_memory.get("max_history_blocks", -1)
                    ),
                    "retained_history_blocks": payload_memory.get(
                        "retained_history_blocks", None
                    ),
                }
                if actual_memory != expected_memory:
                    errors.append(
                        f"{key}/task{task_id}: memory_config does not match fingerprint: "
                        f"actual={actual_memory} expected={expected_memory}"
                    )
        if expected_trials_per_task is not None and episodes != expected_trials_per_task:
            errors.append(
                f"{key}/task{task_id}: expected {expected_trials_per_task} episodes, got {episodes}"
            )
        if len(payload.get("completion_steps", [])) != episodes:
            errors.append(
                f"{key}/task{task_id}: completion_steps length does not match episodes"
            )
        metrics = payload.get("memory_metrics", [])
        if len(metrics) != episodes:
            errors.append(
                f"{key}/task{task_id}: memory_metrics length does not match episodes"
            )
        if require_profiled:
            memory_enabled = bool((payload.get("memory_config") or {}).get("enabled"))
            for episode_index, episode in enumerate(metrics):
                replans = episode.get("replans", [])
                if not replans:
                    errors.append(
                        f"{key}/task{task_id}/episode{episode_index}: no replanning metrics"
                    )
                    continue
                missing = sum(
                    "total_inference_s" not in replan.get("timing", {})
                    for replan in replans
                )
                if missing:
                    errors.append(
                        f"{key}/task{task_id}/episode{episode_index}: "
                        f"{missing}/{len(replans)} replans lack total_inference_s"
                    )
                for timing_field in (
                    "input_preprocess_s",
                    "model_inference_s",
                    "action_postprocess_s",
                    "latency_residual_s",
                ):
                    missing_stage = sum(
                        timing_field not in replan.get("timing", {})
                        for replan in replans
                    )
                    if missing_stage:
                        errors.append(
                            f"{key}/task{task_id}/episode{episode_index}: "
                            f"{missing_stage}/{len(replans)} replans lack {timing_field}"
                        )
                for replan_index, replan in enumerate(replans):
                    timing = replan.get("timing", {})
                    outer_fields = (
                        "total_inference_s",
                        "input_preprocess_s",
                        "model_inference_s",
                        "action_postprocess_s",
                        "latency_residual_s",
                    )
                    if all(field in timing for field in outer_fields):
                        values = {field: float(timing[field]) for field in outer_fields}
                        if any(not math.isfinite(value) or value < 0 for value in values.values()):
                            errors.append(
                                f"{key}/task{task_id}/episode{episode_index}/"
                                f"replan{replan_index}: invalid outer latency value"
                            )
                        else:
                            stage_sum = sum(
                                values[field]
                                for field in outer_fields
                                if field != "total_inference_s"
                            )
                            tolerance = max(1e-6, values["total_inference_s"] * 1e-5)
                            if abs(values["total_inference_s"] - stage_sum) > tolerance:
                                errors.append(
                                    f"{key}/task{task_id}/episode{episode_index}/"
                                    f"replan{replan_index}: outer latency stages do not close"
                                )
                if memory_enabled:
                    missing_prefill = sum(
                        "observation_prefill_s" not in replan.get("timing", {})
                        for replan in replans
                    )
                    missing_denoise = sum(
                        "action_denoise_s" not in replan.get("timing", {})
                        for replan in replans
                    )
                    missing_commit = sum(
                        "commit_s" not in replan.get("commit", {})
                        for replan in replans
                    )
                    if missing_prefill:
                        errors.append(
                            f"{key}/task{task_id}/episode{episode_index}: "
                            f"{missing_prefill}/{len(replans)} replans lack observation_prefill_s"
                        )
                    if missing_denoise:
                        errors.append(
                            f"{key}/task{task_id}/episode{episode_index}: "
                            f"{missing_denoise}/{len(replans)} replans lack action_denoise_s"
                        )
                    for timing_field in (
                        "conditioning_s",
                        "future_video_setup_s",
                        "future_video_denoise_s",
                        "future_video_cache_s",
                        "action_setup_s",
                        "causal_model_s",
                        "causal_model_residual_s",
                    ):
                        missing_stage = sum(
                            timing_field not in replan.get("timing", {})
                            for replan in replans
                        )
                        if missing_stage:
                            errors.append(
                                f"{key}/task{task_id}/episode{episode_index}: "
                                f"{missing_stage}/{len(replans)} replans lack {timing_field}"
                            )
                    for replan_index, replan in enumerate(replans):
                        timing = replan.get("timing", {})
                        causal_fields = (
                            "causal_model_s",
                            "conditioning_s",
                            "observation_prefill_s",
                            "future_video_setup_s",
                            "future_video_denoise_s",
                            "future_video_cache_s",
                            "action_setup_s",
                            "action_denoise_s",
                            "causal_model_residual_s",
                        )
                        if all(field in timing for field in causal_fields):
                            values = {
                                field: float(timing[field]) for field in causal_fields
                            }
                            if any(
                                not math.isfinite(value) or value < 0
                                for value in values.values()
                            ):
                                errors.append(
                                    f"{key}/task{task_id}/episode{episode_index}/"
                                    f"replan{replan_index}: invalid causal latency value"
                                )
                            else:
                                stage_sum = sum(
                                    values[field]
                                    for field in causal_fields
                                    if field != "causal_model_s"
                                )
                                tolerance = max(
                                    1e-6, values["causal_model_s"] * 1e-5
                                )
                                if abs(values["causal_model_s"] - stage_sum) > tolerance:
                                    errors.append(
                                        f"{key}/task{task_id}/episode{episode_index}/"
                                        f"replan{replan_index}: causal latency stages do not close"
                                    )
                    if missing_commit:
                        errors.append(
                            f"{key}/task{task_id}/episode{episode_index}: "
                            f"{missing_commit}/{len(replans)} replans lack action commit timing"
                        )
                if "peak_gpu_bytes" not in episode:
                    errors.append(
                        f"{key}/task{task_id}/episode{episode_index}: peak GPU metric missing"
                    )
                if "peak_cache_bytes" not in episode:
                    errors.append(
                        f"{key}/task{task_id}/episode{episode_index}: peak cache metric missing"
                    )
                if "peak_transient_future_video_cache_bytes" not in episode:
                    errors.append(
                        f"{key}/task{task_id}/episode{episode_index}: "
                        "peak transient future-video cache metric missing"
                    )

    if expected_tasks is not None:
        expected_ids = set(range(expected_tasks))
        for key, task_paths in groups.items():
            actual_ids = set(task_paths)
            if actual_ids != expected_ids:
                errors.append(
                    f"{key}: task ids mismatch; missing={sorted(expected_ids - actual_ids)} "
                    f"unexpected={sorted(actual_ids - expected_ids)}"
                )
    if errors:
        preview = "\n".join(f"- {error}" for error in errors[:50])
        suffix = "" if len(errors) <= 50 else f"\n- ... {len(errors) - 50} more"
        raise ValueError(f"evaluation completeness validation failed:\n{preview}{suffix}")


def non_dominated(rows: list[dict]) -> list[dict]:
    result = []
    for candidate in rows:
        dominated = any(
            other is not candidate
            and other["success_rate"] >= candidate["success_rate"]
            and other["p50_latency_s"] <= candidate["p50_latency_s"]
            and other["peak_gpu_gib"] <= candidate["peak_gpu_gib"]
            and (
                other["success_rate"] > candidate["success_rate"]
                or other["p50_latency_s"] < candidate["p50_latency_s"]
                or other["peak_gpu_gib"] < candidate["peak_gpu_gib"]
            )
            for other in rows
        )
        if not dominated:
            result.append(candidate)
    return result


def choose_default(rows: list[dict]) -> dict | None:
    if not rows:
        return None
    best = max(rows, key=lambda row: row["success_rate"])
    eligible = [
        row
        for row in rows
        if row["success_rate"] >= best["success_rate"] - 0.01
        and row["ci_low"] <= best["ci_high"]
        and best["ci_low"] <= row["ci_high"]
    ]
    return min(eligible, key=lambda row: row["p50_latency_s"])


def choose_leapbot_default(rows: list[dict]) -> dict | None:
    """Choose the default only from memory-enabled LeapBot configurations."""

    leapbot_rows = [
        row
        for row in rows
        if row.get("model_family") == "leapbot_memory"
        and row.get("memory_enabled") is True
    ]
    return choose_default(leapbot_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("evaluate_results/leapbot/pareto"))
    parser.add_argument("--expected-tasks", type=int)
    parser.add_argument("--expected-trials-per-task", type=int)
    parser.add_argument("--require-profiled", action="store_true")
    args = parser.parse_args()
    paths = []
    for item in args.inputs:
        paths.extend(sorted(item.rglob("*_results.json")) if item.is_dir() else [item])
    validate_inputs(
        paths,
        expected_tasks=args.expected_tasks,
        expected_trials_per_task=args.expected_trials_per_task,
        require_profiled=args.require_profiled,
    )
    rows = aggregate(paths)
    per_task_rows = aggregate_per_task(paths)
    kv_retention_rows = aggregate_by_kv_retention(paths)
    frontier = non_dominated(rows)
    leapbot_default = choose_leapbot_default(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ["config"])
        writer.writeheader()
        writer.writerows(rows)
    with (args.output_dir / "per_task.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(per_task_rows[0]) if per_task_rows else ["config", "task_id"],
        )
        writer.writeheader()
        writer.writerows(per_task_rows)
    kv_retention_fields = (
        list(kv_retention_rows[0])
        if kv_retention_rows
        else ["config", "kv_retained_blocks_before_replan"]
    )
    for filename in ("kv_retention_profile.csv", "history_profile.csv"):
        with (args.output_dir / filename).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=kv_retention_fields)
            writer.writeheader()
            writer.writerows(kv_retention_rows)
    (args.output_dir / "pareto.json").write_text(
        json.dumps(
            {
                "retention_semantics": KV_RETENTION_SEMANTICS,
                "leapbot_default": leapbot_default,
                "overall_frontier": frontier,
                # Compatibility aliases for existing report consumers.
                "default": leapbot_default,
                "frontier": frontier,
                "all": rows,
            },
            indent=2,
        )
    )
    print(
        "LeapBot default:",
        None if leapbot_default is None else leapbot_default["config"],
    )
    print("overall non-dominated:", len(frontier), "of", len(rows))


if __name__ == "__main__":
    main()
