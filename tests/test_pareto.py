import importlib.util
import hashlib
import json
from pathlib import Path

from leapbot_va.eval_fingerprint import build_evaluation_fingerprint


_PATH = Path(__file__).parents[1] / "experiments" / "leapbot" / "pareto.py"
_SPEC = importlib.util.spec_from_file_location("leapbot_pareto", _PATH)
pareto = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(pareto)
_PLOT_PATH = Path(__file__).parents[1] / "experiments" / "leapbot" / "plot_pareto.py"
_PLOT_SPEC = importlib.util.spec_from_file_location("leapbot_plot_pareto", _PLOT_PATH)
plot_pareto = importlib.util.module_from_spec(_PLOT_SPEC)
_PLOT_SPEC.loader.exec_module(plot_pareto)


def _attach_fingerprint(payload, *, checkpoint_bytes=b"checkpoint", config_name=None):
    memory = payload.get("memory_config") or {}
    enabled = bool(memory.get("enabled", False))
    episode_capacity = int(memory.get("max_history_blocks", 0)) if enabled else 0
    retained = memory.get("retained_history_blocks", None) if enabled else 0
    runtime = {
        "config": {
            "name": config_name
            or ("sim_leapbot_libero" if enabled else "sim_libero")
        },
        "memory": {
            "enabled": enabled,
            "causal_mode": memory.get("causal_mode") if enabled else None,
            "exit_depth": int(memory.get("exit_depth", 0)) if enabled else 0,
            "episode_capacity": episode_capacity,
            "retained_history_blocks": retained,
            "effective_history_cap": (
                episode_capacity if retained is None else int(retained)
            ),
        },
    }
    result = {
        "suite": payload.setdefault("task_suite", "libero_10"),
        "task": {"id": int(payload.get("task_id", 0))},
        "trials": int(payload.get("total_episodes", 1)),
    }
    payload["evaluation_fingerprint"] = build_evaluation_fingerprint(
        checkpoint_sha256=hashlib.sha256(checkpoint_bytes).hexdigest(),
        runtime_contract=runtime,
        result_contract=result,
    )
    return payload


def test_default_uses_fastest_overlapping_within_one_point():
    rows = [
        {"config": "best", "success_rate": 0.90, "ci_low": 0.86, "ci_high": 0.93, "p50_latency_s": 2.0},
        {"config": "fast", "success_rate": 0.895, "ci_low": 0.85, "ci_high": 0.93, "p50_latency_s": 1.0},
        {"config": "too-low", "success_rate": 0.88, "ci_low": 0.84, "ci_high": 0.91, "p50_latency_s": 0.5},
    ]
    assert pareto.choose_default(rows)["config"] == "fast"


def test_non_dominated_keeps_speed_success_tradeoff():
    rows = [
        {"config": "a", "success_rate": 0.9, "p50_latency_s": 2.0, "peak_gpu_gib": 10.0},
        {"config": "b", "success_rate": 0.89, "p50_latency_s": 1.0, "peak_gpu_gib": 8.0},
        {"config": "c", "success_rate": 0.88, "p50_latency_s": 2.5, "peak_gpu_gib": 11.0},
    ]
    assert {row["config"] for row in pareto.non_dominated(rows)} == {"a", "b"}


def test_aggregate_reports_latency_completion_and_per_task(tmp_path):
    payload = {
        "task_id": 3,
        "task_description": "test task",
        "checkpoint": "/tmp/model.pt",
        "memory_config": {
            "enabled": True,
            "causal_mode": "interleaved",
            "exit_depth": 30,
            "max_history_blocks": 70,
        },
        "successes": 1,
        "total_episodes": 2,
        "completion_steps": [100, 200],
        "memory_metrics": [
            {
                "peak_cache_bytes": 2**30,
                "peak_gpu_bytes": 3 * 2**30,
                "replans": [
                    {
                        "timing": {
                            "total_inference_s": 0.5,
                            "observation_prefill_s": 0.1,
                            "action_denoise_s": 0.4,
                        },
                        "memory": {
                            "completed_blocks": 4,
                            "cache_bytes": 2**29,
                        },
                        "commit": {
                            "commit_s": 0.05,
                            "completed_blocks": 5,
                            "cache_bytes": 3 * 2**28,
                        },
                    }
                ],
            }
        ],
    }
    _attach_fingerprint(payload)
    path = tmp_path / "gpu0_task3_results.json"
    path.write_text(json.dumps(payload))

    row = pareto.aggregate([path])[0]
    assert row["mean_completion_steps"] == 150
    assert row["p50_latency_s"] == 0.55
    assert row["p50_observation_prefill_s"] == 0.1
    assert row["p50_action_denoise_s"] == 0.4
    assert row["p50_action_commit_s"] == 0.05
    assert row["peak_cache_gib"] == 1
    assert row["peak_gpu_gib"] == 3

    task_row = pareto.aggregate_per_task([path])[0]
    assert task_row["task_id"] == 3
    assert task_row["success_rate"] == 0.5
    assert task_row["mean_completion_steps"] == 150

    history_row = pareto.aggregate_by_history([path])[0]
    assert history_row["history_blocks_before_replan"] == 4
    assert history_row["samples"] == 1
    assert history_row["p50_cache_after_observation_gib"] == 0.5
    assert history_row["p50_cache_after_commit_gib"] == 0.75
    assert history_row["p50_total_replan_s"] == 0.55


def test_optional_percentile_is_none_when_metric_is_unavailable():
    assert pareto.optional_percentile([], 0.5) is None


def test_config_key_separates_episode_capacity_from_retained_history():
    base = {
        "checkpoint": "/tmp/model.pt",
        "memory_config": {
            "enabled": True,
            "causal_mode": "action_aggregator",
            "exit_depth": 30,
            "max_history_blocks": 70,
            "retained_history_blocks": None,
        },
    }
    _attach_fingerprint(base)
    full = pareto.config_key(base, Path("full.json"))
    capped_payload = json.loads(json.dumps(base))
    capped_payload["memory_config"]["retained_history_blocks"] = 8
    _attach_fingerprint(capped_payload)
    capped = pareto.config_key(capped_payload, Path("capped.json"))

    assert "/h70/cap70/" in full
    assert "/h8/cap70/" in capped
    assert full != capped


def test_history_profile_uses_retained_window_not_absolute_episode_clock(tmp_path):
    payload = {
        "checkpoint": "/tmp/model.pt",
        "memory_config": {
            "enabled": True,
            "causal_mode": "action_aggregator",
            "exit_depth": 30,
            "max_history_blocks": 70,
            "retained_history_blocks": 8,
        },
        "memory_metrics": [
            {
                "replans": [
                    {
                        "memory": {
                            "completed_blocks": 20,
                            "retained_history_blocks": 8,
                            "cache_bytes": 123,
                        },
                        "timing": {"total_inference_s": 0.2},
                    }
                ]
            }
        ],
    }
    _attach_fingerprint(payload)
    path = tmp_path / "capped_results.json"
    path.write_text(json.dumps(payload))

    row = pareto.aggregate_by_history([path])[0]
    assert row["history_blocks_before_replan"] == 8
    assert row["p50_episode_blocks_before_replan"] == 20


def test_history_profile_rejects_runtime_retention_mismatch(tmp_path):
    payload = {
        "checkpoint": "/tmp/model.pt",
        "memory_config": {
            "enabled": True,
            "causal_mode": "action_aggregator",
            "exit_depth": 30,
            "max_history_blocks": 70,
            "retained_history_blocks": 8,
        },
        "memory_metrics": [
            {
                "replans": [
                    {
                        "memory": {
                            "completed_blocks": 20,
                            "retained_history_blocks": 9,
                        }
                    }
                ]
            }
        ],
    }
    _attach_fingerprint(payload)
    path = tmp_path / "corrupt_results.json"
    path.write_text(json.dumps(payload))

    try:
        pareto.aggregate_by_history([path])
    except ValueError as error:
        assert "runtime retained history 9" in str(error)
    else:
        raise AssertionError("retention mismatch unexpectedly passed aggregation")


def test_config_key_uses_checkpoint_and_runtime_hash_not_basename():
    first = _attach_fingerprint(
        {
            "checkpoint": "/one/step_001115.pt",
            "task_id": 0,
            "total_episodes": 1,
            "memory_config": {
                "enabled": True,
                "causal_mode": "action_aggregator",
                "exit_depth": 30,
                "max_history_blocks": 70,
                "retained_history_blocks": None,
            },
        },
        checkpoint_bytes=b"one",
    )
    second = _attach_fingerprint(
        json.loads(json.dumps(first)),
        checkpoint_bytes=b"two",
    )
    assert pareto.config_key(first, Path("first.json")) != pareto.config_key(
        second, Path("second.json")
    )


def test_memory_disabled_leapbot_is_not_grouped_as_fastwam():
    leapbot = _attach_fingerprint(
        {
            "task_id": 0,
            "total_episodes": 1,
            "memory_config": {"enabled": False},
        },
        config_name="sim_leapbot_libero",
    )
    fastwam = _attach_fingerprint(
        {
            "task_id": 0,
            "total_episodes": 1,
            "memory_config": {},
        },
        config_name="sim_libero",
    )
    assert pareto.config_key(leapbot, Path("leap.json")).startswith(
        "leapbot_no_memory/"
    )
    assert pareto.config_key(fastwam, Path("fast.json")).startswith(
        "fastwam_release/"
    )


def test_plot_labels_preserve_mode_depth_history_and_capacity():
    config = "action_aggregator/d16/h8/cap70/rt-abc/ckpt-def"
    assert plot_pareto._short_label(config) == "action_aggregator/d16/h8/cap70"
    assert plot_pareto._short_label(
        "fastwam_release/rt-abc/ckpt-def"
    ) == "FastWAM"


def test_validate_inputs_rejects_incomplete_or_unprofiled_results(tmp_path):
    payload = {
        "task_id": 0,
        "checkpoint": "/tmp/model.pt",
        "successes": 1,
        "total_episodes": 1,
        "completion_steps": [100],
        "memory_metrics": [{"peak_cache_bytes": 0, "replans": [{"timing": {}}]}],
    }
    path = tmp_path / "gpu0_task0_results.json"
    path.write_text(json.dumps(payload))

    try:
        pareto.validate_inputs(
            [path], expected_tasks=2, expected_trials_per_task=1, require_profiled=True
        )
    except ValueError as error:
        message = str(error)
        assert "task ids mismatch" in message
        assert "total_inference_s" in message
        assert "peak GPU metric missing" in message
    else:
        raise AssertionError("incomplete evaluation unexpectedly passed validation")


def test_validate_inputs_requires_causal_latency_breakdown(tmp_path):
    payload = {
        "task_id": 0,
        "checkpoint": "/tmp/model.pt",
        "memory_config": {"enabled": True, "causal_mode": "interleaved"},
        "successes": 1,
        "total_episodes": 1,
        "completion_steps": [100],
        "memory_metrics": [
            {
                "peak_cache_bytes": 1,
                "peak_gpu_bytes": 2,
                "replans": [{"timing": {"total_inference_s": 0.5}}],
            }
        ],
    }
    path = tmp_path / "gpu0_task0_results.json"
    path.write_text(json.dumps(payload))

    try:
        pareto.validate_inputs(
            [path], expected_tasks=1, expected_trials_per_task=1, require_profiled=True
        )
    except ValueError as error:
        message = str(error)
        assert "observation_prefill_s" in message
        assert "action_denoise_s" in message
        assert "action commit timing" in message
    else:
        raise AssertionError("causal result without latency breakdown unexpectedly passed")
