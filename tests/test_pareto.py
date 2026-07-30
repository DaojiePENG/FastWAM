import importlib.util
import json
from pathlib import Path


_PATH = Path(__file__).parents[1] / "experiments" / "leapbot" / "pareto.py"
_SPEC = importlib.util.spec_from_file_location("leapbot_pareto", _PATH)
pareto = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(pareto)


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
                        "commit": {"commit_s": 0.05},
                    }
                ],
            }
        ],
    }
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
