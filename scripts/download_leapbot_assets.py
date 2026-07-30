#!/usr/bin/env python3
"""Download the official FastWAM LIBERO LeRobot data and release checkpoint."""

from __future__ import annotations

import argparse
import tarfile
from pathlib import Path

from huggingface_hub import snapshot_download


DATASET_REPO = "yuanty/LIBERO-fastwam"
CHECKPOINT_REPO = "yuanty/fastwam"
EXPECTED_DATASETS = (
    "libero_10_no_noops_lerobot",
    "libero_goal_no_noops_lerobot",
    "libero_object_no_noops_lerobot",
    "libero_spatial_no_noops_lerobot",
)


def _safe_extract(archive: Path, output_dir: Path) -> None:
    output_root = output_dir.resolve()
    with tarfile.open(archive, "r:gz") as stream:
        for member in stream.getmembers():
            destination = (output_dir / member.name).resolve()
            if output_root not in destination.parents and destination != output_root:
                raise RuntimeError(f"unsafe archive member in {archive}: {member.name}")
        # Members were checked above; avoid Python-version-specific tar filters
        # because LeapBot's supported environment is Python 3.10.
        stream.extractall(output_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--dataset-only", action="store_true")
    parser.add_argument("--checkpoint-only", action="store_true")
    parser.add_argument("--no-extract", action="store_true")
    args = parser.parse_args()
    if args.dataset_only and args.checkpoint_only:
        parser.error("--dataset-only and --checkpoint-only are mutually exclusive")

    root = args.root.resolve()
    if not args.checkpoint_only:
        dataset_dir = root / "data" / "libero_mujoco3.3.2"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=DATASET_REPO,
            repo_type="dataset",
            local_dir=dataset_dir,
            allow_patterns=["*.tar.gz"],
        )
        if not args.no_extract:
            for archive in sorted(dataset_dir.glob("*.tar.gz")):
                print(f"extract {archive.name}")
                _safe_extract(archive, dataset_dir)
            missing = [name for name in EXPECTED_DATASETS if not (dataset_dir / name).is_dir()]
            if missing:
                raise RuntimeError(f"dataset extraction incomplete; missing {missing}")

    if not args.dataset_only:
        checkpoint_dir = root / "checkpoints" / "fastwam_release"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=CHECKPOINT_REPO,
            repo_type="model",
            local_dir=checkpoint_dir,
            allow_patterns=[
                "libero_uncond_2cam224.pt",
                "libero_uncond_2cam224_dataset_stats.json",
            ],
        )
        for filename in (
            "libero_uncond_2cam224.pt",
            "libero_uncond_2cam224_dataset_stats.json",
        ):
            if not (checkpoint_dir / filename).is_file():
                raise RuntimeError(f"checkpoint download incomplete: {filename}")

    print("LeapBot assets are ready under", root)


if __name__ == "__main__":
    main()
