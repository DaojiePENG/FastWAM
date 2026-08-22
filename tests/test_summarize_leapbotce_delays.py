import json

import pytest

from scripts.summarize_leapbotce_delays import summarize, wilson_interval


def _write_result(root, delay, task_id, successes, samples):
    path = root / f"delay_{delay}" / "libero_spatial" / f"gpu0_task{task_id}_results.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "task_suite": "libero_spatial",
                "task_id": task_id,
                "task_description": f"task {task_id}",
                "successes": successes,
                "total_episodes": 10,
                "delay_samples": samples,
            }
        ),
        encoding="utf-8",
    )


def test_delay_summary_aggregates_histograms_and_retention(tmp_path):
    _write_result(tmp_path, 0, 0, 8, [0, 0])
    _write_result(tmp_path, 0, 1, 4, [0, 0])
    _write_result(tmp_path, 5, 0, 6, [1, 5])
    _write_result(tmp_path, 5, 1, 2, [3, 5])
    (tmp_path / "delay_sweep_summary.json").write_text("{}", encoding="utf-8")
    (tmp_path / "delay_sweep_tasks.csv").write_text("", encoding="utf-8")

    windows = summarize(tmp_path)["windows"]

    assert [window["delay_max"] for window in windows] == [0, 5]
    assert windows[0]["success_rate"] == pytest.approx(0.6)
    assert windows[1]["actual_delay_mean"] == pytest.approx(3.5)
    assert windows[1]["delay_histogram"] == {"1": 1, "3": 1, "5": 2}
    assert windows[1]["retention_vs_delay_0"] == pytest.approx(2 / 3)
    assert windows[1]["tasks"][0]["retention_vs_delay_0"] == pytest.approx(0.75)
    assert windows[1]["tasks"][1]["retention_vs_delay_0"] == pytest.approx(0.5)


def test_wilson_interval_bounds_are_ordered():
    low, high = wilson_interval(5, 10)
    assert 0.0 < low < 0.5 < high < 1.0
