#!/usr/bin/env python3
"""Render reproducible success/latency/memory and history-scaling figures."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize


def _float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value not in {"", None} else float("nan")


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def _short_label(config: str) -> str:
    if config.startswith("fastwam/"):
        return "FastWAM"
    return config.split("/", 1)[0]


def plot_pareto(artifact_dir: Path, output_dir: Path) -> list[Path]:
    rows = _read_csv(artifact_dir / "results.csv")
    if not rows:
        raise ValueError(f"no result rows in {artifact_dir / 'results.csv'}")
    metadata = json.loads((artifact_dir / "pareto.json").read_text())
    frontier = {row["config"] for row in metadata.get("frontier", [])}
    default = (metadata.get("default") or {}).get("config")

    cache_values = [_float(row, "peak_cache_gib") for row in rows]
    color_min = min(cache_values)
    color_max = max(cache_values)
    if color_max <= color_min:
        color_max = color_min + 1.0
    normalization = Normalize(vmin=color_min, vmax=color_max)
    colormap = plt.get_cmap("viridis")

    figure, axis = plt.subplots(figsize=(9.0, 6.0), constrained_layout=True)
    for row in rows:
        config = row["config"]
        latency = _float(row, "p50_latency_s")
        success = 100.0 * _float(row, "success_rate")
        gpu = _float(row, "peak_gpu_gib")
        cache = _float(row, "peak_cache_gib")
        edge = "gold" if config == default else ("black" if config in frontier else "white")
        width = 2.8 if config == default else (1.8 if config in frontier else 0.8)
        axis.scatter(
            latency,
            success,
            s=80.0 + 10.0 * gpu,
            c=[colormap(normalization(cache))],
            edgecolors=edge,
            linewidths=width,
            zorder=3,
        )
        axis.annotate(
            _short_label(config),
            (latency, success),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=9,
        )

    scalar = ScalarMappable(norm=normalization, cmap=colormap)
    scalar.set_array([])
    colorbar = figure.colorbar(scalar, ax=axis)
    colorbar.set_label("Peak causal KV cache (GiB)")
    axis.set_xlabel("P50 replanning latency including action commit (s)")
    axis.set_ylabel("LIBERO-Long success rate (%)")
    axis.set_title("LeapBot-VA success–latency–memory Pareto comparison")
    axis.grid(True, alpha=0.25)
    axis.text(
        0.01,
        0.01,
        "Marker area ∝ peak GPU memory; black edge = Pareto frontier; gold edge = default",
        transform=axis.transAxes,
        fontsize=8,
        va="bottom",
    )

    outputs = []
    for suffix in ("png", "svg"):
        path = output_dir / f"success_latency_memory.{suffix}"
        figure.savefig(path, dpi=220)
        outputs.append(path)
    plt.close(figure)
    return outputs


def plot_history_scaling(artifact_dir: Path, output_dir: Path) -> list[Path]:
    rows = _read_csv(artifact_dir / "history_profile.csv")
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["config"]].append(row)
    if not grouped:
        return []

    figure, (cache_axis, latency_axis) = plt.subplots(
        1, 2, figsize=(12.0, 5.0), constrained_layout=True
    )
    for config, config_rows in sorted(grouped.items()):
        ordered = sorted(
            config_rows,
            key=lambda row: int(row["history_blocks_before_replan"]),
        )
        blocks = [int(row["history_blocks_before_replan"]) for row in ordered]
        cache = [
            _float(row, "p50_cache_after_observation_gib") for row in ordered
        ]
        latency = [_float(row, "p50_total_replan_s") for row in ordered]
        label = _short_label(config)
        cache_axis.plot(blocks, cache, marker="o", markersize=3, label=label)
        latency_axis.plot(blocks, latency, marker="o", markersize=3, label=label)

    cache_axis.set_xlabel("Committed history blocks before replanning")
    cache_axis.set_ylabel("P50 causal KV cache after current observation (GiB)")
    cache_axis.set_title("KV-cache growth")
    cache_axis.grid(True, alpha=0.25)
    latency_axis.set_xlabel("Committed history blocks before replanning")
    latency_axis.set_ylabel("P50 total replanning latency (s)")
    latency_axis.set_title("Latency scaling with memory length")
    latency_axis.grid(True, alpha=0.25)
    latency_axis.legend(loc="best")

    outputs = []
    for suffix in ("png", "svg"):
        path = output_dir / f"history_cache_latency.{suffix}"
        figure.savefig(path, dpi=220)
        outputs.append(path)
    plt.close(figure)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact_dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    artifact_dir = args.artifact_dir.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else artifact_dir
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = plot_pareto(artifact_dir, output_dir)
    outputs.extend(plot_history_scaling(artifact_dir, output_dir))
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
