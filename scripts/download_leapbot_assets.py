#!/usr/bin/env python3
"""Download the official FastWAM LIBERO LeRobot data and release checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path

from huggingface_hub import snapshot_download


DATASET_REPO = "yuanty/LIBERO-fastwam"
CHECKPOINT_REPO = "yuanty/fastwam"
DATASET_REVISION = "117413dc0ca99c7cd64036c4eaa4a316c537d692"
CHECKPOINT_REVISION = "139eebb6d90cdd9bdbbe465f72c6edc9ad5a518a"
EXPECTED_DATASETS = (
    "libero_10_no_noops_lerobot",
    "libero_goal_no_noops_lerobot",
    "libero_object_no_noops_lerobot",
    "libero_spatial_no_noops_lerobot",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_revision(path: Path) -> str:
    metadata = path.parent / ".cache" / "huggingface" / "download" / (
        path.name + ".metadata"
    )
    if not metadata.is_file():
        raise FileNotFoundError(f"Hugging Face download metadata missing: {metadata}")
    revision = metadata.read_text(encoding="utf-8").splitlines()[0]
    if len(revision) != 40:
        raise ValueError(f"invalid Hugging Face revision in {metadata}: {revision!r}")
    return revision


def _write_download_manifest(root: Path) -> Path:
    expected = _expected_assets(root)

    files = []
    for path, repo, revision in expected:
        if not path.is_file():
            raise FileNotFoundError(f"asset required for complete manifest: {path}")
        actual_revision = _download_revision(path)
        if actual_revision != revision:
            raise ValueError(
                f"download revision mismatch for {path}: "
                f"expected={revision} actual={actual_revision}"
            )
        files.append(
            {
                "path": str(path.relative_to(root)),
                "repo": repo,
                "revision": revision,
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    payload = {"schema_version": 1, "files": files}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["manifest_sha256"] = hashlib.sha256(canonical).hexdigest()
    output = root / "data" / "leapbot_asset_download_manifest.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)
    return output


def _expected_assets(root: Path) -> list[tuple[Path, str, str]]:
    expected: list[tuple[Path, str, str]] = []
    dataset_dir = root / "data" / "libero_mujoco3.3.2"
    for name in EXPECTED_DATASETS:
        expected.append((dataset_dir / f"{name}.tar.gz", DATASET_REPO, DATASET_REVISION))
    checkpoint_dir = root / "checkpoints" / "fastwam_release"
    for name in (
        "libero_uncond_2cam224.pt",
        "libero_uncond_2cam224_dataset_stats.json",
    ):
        expected.append((checkpoint_dir / name, CHECKPOINT_REPO, CHECKPOINT_REVISION))
    return expected


def _complete_asset_files_exist(root: Path) -> bool:
    return all(path.is_file() for path, _, _ in _expected_assets(root))


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
    parser.add_argument("--manifest-only", action="store_true")
    args = parser.parse_args()
    if args.dataset_only and args.checkpoint_only:
        parser.error("--dataset-only and --checkpoint-only are mutually exclusive")
    if args.manifest_only and (args.dataset_only or args.checkpoint_only):
        parser.error("--manifest-only cannot be combined with partial downloads")

    root = args.root.resolve()
    if not args.manifest_only and not args.checkpoint_only:
        dataset_dir = root / "data" / "libero_mujoco3.3.2"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=DATASET_REPO,
            repo_type="dataset",
            revision=DATASET_REVISION,
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

    if not args.manifest_only and not args.dataset_only:
        checkpoint_dir = root / "checkpoints" / "fastwam_release"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=CHECKPOINT_REPO,
            repo_type="model",
            revision=CHECKPOINT_REVISION,
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
    if args.manifest_only or (not args.dataset_only and not args.checkpoint_only):
        manifest = _write_download_manifest(root)
        print("Pinned download manifest:", manifest)
    elif _complete_asset_files_exist(root):
        manifest = _write_download_manifest(root)
        print("Pinned download manifest:", manifest)
    else:
        print("Pinned download manifest deferred until all six assets are present")


if __name__ == "__main__":
    main()
