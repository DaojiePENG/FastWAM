#!/usr/bin/env python3
"""Mirror an already-running FastWAM text log into a clearly labeled W&B run."""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import wandb


METRIC_PATTERN = re.compile(
    r"epoch=(?P<epoch>\d+)\s+step=(?P<step>\d+)/(?P<max_steps>\d+)\s+"
    r"loss=(?P<loss>[0-9.eE+-]+).*?"
    r"loss_action_d30=(?P<action>[0-9.eE+-]+).*?"
    r"loss_video_d30=(?P<video>[0-9.eE+-]+)\s+"
    r"lr=(?P<lr>[0-9.eE+-]+).*?"
    r"speed=(?P<steps_per_s>[0-9.eE+-]+)\s+step/s,\s+"
    r"(?P<samples_per_s>[0-9.eE+-]+)\s+samples/s",
    re.DOTALL,
)


def parse_metrics(text: str) -> list[dict[str, float | int]]:
    rows = []
    for match in METRIC_PATTERN.finditer(text):
        rows.append(
            {
                "epoch": int(match.group("epoch")),
                "step": int(match.group("step")),
                "max_steps": int(match.group("max_steps")),
                "train/loss": float(match.group("loss")),
                "train/loss_action_d30": float(match.group("action")),
                "train/loss_video_d30": float(match.group("video")),
                "train/lr": float(match.group("lr")),
                "performance/steps_per_s": float(match.group("steps_per_s")),
                "performance/samples_per_s": float(match.group("samples_per_s")),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--entity", required=True)
    parser.add_argument("--project", default="leapbot-va")
    parser.add_argument("--group", default="phase1-h8-d30-s1000-screening-seed42")
    parser.add_argument("--poll-seconds", type=float, default=20.0)
    args = parser.parse_args()

    run = wandb.init(
        entity=args.entity,
        project=args.project,
        group=args.group,
        name=f"screening-s1000-{args.mode}-seed42",
        id=f"screening-s1000-{args.mode}-seed42",
        resume="allow",
        job_type="screening",
        config={
            "causal_mode": args.mode,
            "seed": 42,
            "global_batch_size": 2,
            "max_steps": 1000,
            "status": "screening_not_full_finetune",
        },
    )
    logged_steps: set[int] = set()
    while True:
        text = args.log.read_text(errors="replace") if args.log.exists() else ""
        for row in parse_metrics(text):
            step = int(row.pop("step"))
            if step not in logged_steps:
                run.log(row, step=step)
                logged_steps.add(step)
        if "[done] max_steps reached step=1000" in text:
            break
        time.sleep(args.poll_seconds)
    run.finish()


if __name__ == "__main__":
    main()
