"""Summarize LeapBotCE delay-sweep result files.

The evaluator writes one JSON file per LIBERO task. This script combines those
files into task-level and overall delay-robustness measurements without relying
on the manager's per-window summary, which does not include realized delays.
"""

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path


def wilson_interval(successes: int, total: int, z: float = 1.96) -> list[float]:
    """Return a 95% Wilson interval for a binomial success rate."""
    if total <= 0:
        return [0.0, 0.0]
    rate = successes / total
    denominator = 1.0 + z * z / total
    center = (rate + z * z / (2.0 * total)) / denominator
    margin = z * math.sqrt(rate * (1.0 - rate) / total + z * z / (4.0 * total * total))
    return [max(0.0, center - margin / denominator), min(1.0, center + margin / denominator)]


def delay_value(path: Path) -> int:
    try:
        return int(path.name.removeprefix("delay_"))
    except ValueError as exc:
        raise ValueError(f"Expected a delay_<integer> directory, got {path}") from exc


def load_delay_dir(delay_dir: Path) -> tuple[list[dict], Counter]:
    task_rows = []
    aggregate_histogram: Counter = Counter()
    for path in sorted(delay_dir.glob("**/*_results.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        samples = [int(value) for value in result.get("delay_samples", [])]
        if not samples:
            raise ValueError(f"Missing delay samples in {path}")
        successes = int(result["successes"])
        total = int(result.get("total_episodes", 0))
        if total <= 0:
            raise ValueError(f"Invalid total episodes in {path}: {total}")
        histogram = Counter(samples)
        aggregate_histogram.update(histogram)
        task_rows.append(
            {
                "task_suite": str(result["task_suite"]),
                "task_id": int(result["task_id"]),
                "task_description": str(result.get("task_description", "")),
                "successes": successes,
                "total": total,
                "success_rate": successes / total,
                "success_rate_wilson95": wilson_interval(successes, total),
                "replans": len(samples),
                "actual_delay_mean": sum(samples) / len(samples),
                "delay_histogram": {str(delay): count for delay, count in sorted(histogram.items())},
            }
        )
    task_rows.sort(key=lambda row: (row["task_suite"], row["task_id"]))
    return task_rows, aggregate_histogram


def summarize(root: Path) -> dict:
    delay_dirs = sorted((path for path in root.glob("delay_*") if path.is_dir()), key=delay_value)
    if not delay_dirs:
        raise FileNotFoundError(f"No delay_* directories found under {root}")

    summary_rows = []
    baseline_by_task = {}
    baseline_rate = None
    for delay_dir in delay_dirs:
        task_rows, histogram = load_delay_dir(delay_dir)
        successes = sum(row["successes"] for row in task_rows)
        total = sum(row["total"] for row in task_rows)
        samples = sum(histogram.values())
        rate = successes / total if total else 0.0
        if baseline_rate is None:
            baseline_rate = rate
            baseline_by_task = {
                (row["task_suite"], row["task_id"]): row["success_rate"] for row in task_rows
            }
        for row in task_rows:
            baseline = baseline_by_task[(row["task_suite"], row["task_id"])]
            row["retention_vs_delay_0"] = row["success_rate"] / baseline if baseline else None
        summary_rows.append(
            {
                "delay_max": delay_value(delay_dir),
                "successes": successes,
                "total": total,
                "success_rate": rate,
                "success_rate_wilson95": wilson_interval(successes, total),
                "retention_vs_delay_0": rate / baseline_rate if baseline_rate else None,
                "replans": samples,
                "actual_delay_mean": (
                    sum(delay * count for delay, count in histogram.items()) / samples if samples else 0.0
                ),
                "delay_histogram": {str(delay): count for delay, count in sorted(histogram.items())},
                "tasks": task_rows,
            }
        )
    return {"root": str(root.resolve()), "windows": summary_rows}


def write_csv(summary: dict, path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "delay_max", "task_suite", "task_id", "successes", "total", "success_rate",
                "retention_vs_delay_0", "actual_delay_mean", "replans",
            ],
        )
        writer.writeheader()
        for window in summary["windows"]:
            for task in window["tasks"]:
                writer.writerow(
                    {
                        "delay_max": window["delay_max"],
                        "task_suite": task["task_suite"],
                        "task_id": task["task_id"],
                        "successes": task["successes"],
                        "total": task["total"],
                        "success_rate": f"{task['success_rate']:.6f}",
                        "retention_vs_delay_0": (
                            "" if task["retention_vs_delay_0"] is None
                            else f"{task['retention_vs_delay_0']:.6f}"
                        ),
                        "actual_delay_mean": f"{task['actual_delay_mean']:.6f}",
                        "replans": task["replans"],
                    }
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Directory containing delay_<N> result directories.")
    parser.add_argument("--output", type=Path, default=None, help="Output JSON path (default: <root>/delay_sweep_summary.json).")
    parser.add_argument("--csv", type=Path, default=None, help="Output task-level CSV path (default: <root>/delay_sweep_tasks.csv).")
    args = parser.parse_args()

    root = args.root.resolve()
    summary = summarize(root)
    output_json = args.output or root / "delay_sweep_summary.json"
    output_csv = args.csv or root / "delay_sweep_tasks.csv"
    output_json.write_text(json.dumps(summary, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    write_csv(summary, output_csv)

    print("delay\tsuccess\ttotal\trate\tCI95\tretention\tactual_delay\treplans")
    for window in summary["windows"]:
        low, high = window["success_rate_wilson95"]
        retention = window["retention_vs_delay_0"]
        print(
            f"{window['delay_max']}\t{window['successes']}\t{window['total']}\t"
            f"{window['success_rate']:.4f}\t[{low:.4f},{high:.4f}]\t"
            f"{retention:.4f}\t{window['actual_delay_mean']:.4f}\t{window['replans']}"
        )
    print(f"JSON: {output_json}")
    print(f"CSV: {output_csv}")


if __name__ == "__main__":
    main()
