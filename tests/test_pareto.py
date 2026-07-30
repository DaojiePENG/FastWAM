import importlib.util
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
