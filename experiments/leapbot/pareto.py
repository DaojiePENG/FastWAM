#!/usr/bin/env python3
"""Aggregate LIBERO task JSON files and select LeapBot Pareto configurations."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median


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


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return center - radius, center + radius


def config_key(payload: dict, path: Path) -> str:
    memory = payload.get("memory_config") or {}
    if memory.get("enabled"):
        return "/".join(
            [
                str(memory.get("causal_mode", "unknown")),
                f"d{memory.get('exit_depth', 'unknown')}",
                f"h{memory.get('max_history_blocks', 'unknown')}",
                Path(str(payload.get("checkpoint", path.parent))).stem,
            ]
        )
    return f"fastwam/{Path(str(payload.get('checkpoint', path.parent))).stem}"


def aggregate(paths: list[Path]) -> list[dict]:
    groups = defaultdict(lambda: {"successes": 0, "episodes": 0, "latency": [], "cache": [], "gpu": []})
    for path in paths:
        payload = json.loads(path.read_text())
        key = config_key(payload, path)
        group = groups[key]
        group["successes"] += int(payload.get("successes", 0))
        group["episodes"] += int(payload.get("total_episodes", 0))
        for episode in payload.get("memory_metrics", []):
            group["cache"].append(float(episode.get("peak_cache_bytes", 0)))
            if "peak_gpu_bytes" in episode:
                group["gpu"].append(float(episode["peak_gpu_bytes"]))
            for replan in episode.get("replans", []):
                timing = replan.get("timing", {})
                total = float(timing.get("observation_prefill_s", 0)) + float(
                    timing.get("action_denoise_s", 0)
                ) + float(replan.get("commit", {}).get("commit_s", 0))
                group["latency"].append(total)

    rows = []
    for key, values in groups.items():
        rate = values["successes"] / values["episodes"] if values["episodes"] else 0.0
        ci_low, ci_high = wilson(values["successes"], values["episodes"])
        rows.append(
            {
                "config": key,
                "successes": values["successes"],
                "episodes": values["episodes"],
                "success_rate": rate,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "p50_latency_s": percentile(values["latency"], 0.50),
                "p95_latency_s": percentile(values["latency"], 0.95),
                "peak_cache_gib": max(values["cache"], default=0) / 2**30,
                "peak_gpu_gib": max(values["gpu"], default=0) / 2**30,
            }
        )
    return sorted(rows, key=lambda row: row["config"])


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("evaluate_results/leapbot/pareto"))
    args = parser.parse_args()
    paths = []
    for item in args.inputs:
        paths.extend(sorted(item.rglob("*_results.json")) if item.is_dir() else [item])
    rows = aggregate(paths)
    frontier = non_dominated(rows)
    default = choose_default(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ["config"])
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "pareto.json").write_text(
        json.dumps({"default": default, "frontier": frontier, "all": rows}, indent=2)
    )
    print("default:", None if default is None else default["config"])
    print("non-dominated:", len(frontier), "of", len(rows))


if __name__ == "__main__":
    main()
