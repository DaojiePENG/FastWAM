#!/usr/bin/env python3
"""Create the hardware profiling figure from measured benchmark JSON files.

Reproduce with --input pointing to full-stage bench_client schema-v2 records.
The latency ECDF uses every successful measured attempt; run-level completion
rates retain failed-attempt time. Warm-up records remain excluded explicitly.
Pass --qa-tools to a directory containing audit_panel_alignment.py.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def summarize(values):
    a = np.asarray(values, dtype=float)
    return {"n": len(a), "mean": float(a.mean()),
            "p50": float(np.percentile(a, 50)), "p95": float(np.percentile(a, 95))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("Figures/fig_hardware_profile"))
    parser.add_argument("--qa-tools", type=Path, required=True)
    parser.add_argument("--expected-runs", type=int, default=3)
    args = parser.parse_args()
    sys.path.insert(0, str(args.qa_tools))
    from audit_panel_alignment import require_matplotlib_panel_alignment

    groups, samples, runs = {}, [], []
    for path in args.input:
        data = json.loads(path.read_text())
        if data.get("schema_version") != 2 or data.get("stage") != "full":
            raise ValueError(f"Expected full-stage schema-v2 measurements: {path}")
        records = [r for r in data["records"] if not r["warmup"]]
        if len(records) != data["n"] or not records:
            raise ValueError(f"Incomplete measured batch: {path}")
        group = (data["mode"], data["schedule"])
        item = groups.setdefault(group, {"records": [], "rates": [], "files": [],
                                         "attempted": 0, "failed": 0})
        ok = [r for r in records if r["ok"]]
        failed = len(records) - len(ok)
        if not ok:
            raise ValueError(f"No completed measurements: {path}")
        rate = data["throughput"]["completed_replans_per_s"]
        elapsed = data["throughput"]["measured_elapsed_s"]
        if not np.isclose(rate, len(ok) / elapsed):
            raise ValueError(f"Invalid completion-rate denominator: {path}")
        item["records"].extend(ok)
        item["rates"].append(rate)
        item["files"].append(path.name)
        item["attempted"] += len(records)
        item["failed"] += failed
        runs.append({"file": path.name, "mode": group[0], "schedule": group[1],
                     "attempted": len(records), "failed": failed, "elapsed_s": elapsed,
                     "completed_replans_per_s": rate})
        for r in records:
            samples.append({"file": path.name, "mode": group[0], "schedule": group[1],
                            "replan_id": r["replan_id"], "ok": r["ok"],
                            "round_trip_ms": r["round_trip_ms"]})

    order = [g for g in [("codec", "serial"), ("codec", "pipeline"),
                         ("raw", "serial"), ("raw", "pipeline")] if g in groups]
    if len(order) < 2:
        raise ValueError("At least two measured configurations are needed")
    for group in order:
        if len(groups[group]["rates"]) != args.expected_runs:
            raise ValueError(f"Expected {args.expected_runs} completed runs for {group}")
    matplotlib.rcParams.update({"font.family": "sans-serif", "font.size": 7,
                                "pdf.fonttype": 42, "svg.fonttype": "none",
                                "axes.spines.top": False, "axes.spines.right": False,
                                "axes.linewidth": .65, "xtick.labelsize": 6.5,
                                "ytick.labelsize": 6.5, "legend.fontsize": 6.3})
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.55))
    fig.subplots_adjust(left=.08, right=.985, bottom=.24, top=.77, wspace=.32)
    colors = {"codec": "#2878B5", "raw": "#7D7D7D"}
    labels = {"codec": "Codec", "raw": "Raw cache"}
    summary = {}
    for i, group in enumerate(order):
        mode, schedule = group
        item = groups[group]
        latency = np.sort([r["round_trip_ms"] / 1000 for r in item["records"]])
        axes[0].step(latency, np.arange(1, len(latency) + 1) / len(latency), where="post",
                     color=colors[mode], linestyle="-" if schedule == "serial" else "--",
                     linewidth=1.7, label=f"{labels[mode]}, {schedule}")
        rates = np.asarray(item["rates"])
        axes[1].bar(i, rates.mean(), color=colors[mode], alpha=.28, width=.6,
                    edgecolor=colors[mode], linewidth=.7)
        axes[1].vlines(i, rates.min(), rates.max(), colors=colors[mode], linewidth=1)
        offsets = np.linspace(-.10, .10, len(rates)) if len(rates) > 1 else [0]
        axes[1].scatter(i + np.asarray(offsets), rates, color=colors[mode], s=15, zorder=3)
        stage_keys = sorted({k for r in item["records"] for k, v in r.items()
                             if k.endswith("_ms") and isinstance(v, (float, int))})
        summary["/".join(group)] = {
            "attempted": item["attempted"], "failed": item["failed"],
            "files": item["files"], "run_rates": item["rates"],
            "stages_ms": {k: summarize([r[k] for r in item["records"] if k in r])
                          for k in stage_keys}}
    axes[0].set(xlabel="Completed replan latency (s)", ylabel="Empirical cumulative fraction",
                ylim=(0, 1.03))
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", bbox_to_anchor=(.5, 1.0),
               ncol=len(order), frameon=False)
    axes[1].set(ylabel="Completed replans / s", ylim=(0, None))
    axes[1].set_xticks(range(len(order)),
                       [f"{labels[m]}\n{s}" for m, s in order])
    for label, title, ax in zip(("a", "b"), ("Latency distribution", "Measured completion throughput"), axes):
        ax.annotate(label, (0, 1), xycoords="axes fraction", xytext=(-20, 9),
                    textcoords="offset points", fontweight="bold", fontsize=9)
        ax.set_title(title, fontsize=7.5, pad=11)
        ax.grid(axis="y", color="#E4E4E4", linewidth=.45)
        ax.set_axisbelow(True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    require_matplotlib_panel_alignment(fig, json_out=str(args.output) + ".alignment.json",
                                      tolerance_pt=1.5, strict=True)
    fig.savefig(str(args.output) + ".pdf")
    fig.savefig(str(args.output) + ".svg")
    fig.savefig(str(args.output) + ".png", dpi=600)
    plt.close(fig)
    for suffix, rows in (("samples", samples), ("runs", runs)):
        with Path(str(args.output) + f".{suffix}.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    Path(str(args.output) + ".summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
