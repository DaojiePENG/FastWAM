#!/usr/bin/env python3
"""Hash every immutable asset read by formal LeapBot training."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


PINNED_REVISIONS = {
    "yuanty/LIBERO-fastwam": "117413dc0ca99c7cd64036c4eaa4a316c537d692",
    "yuanty/fastwam": "139eebb6d90cdd9bdbbe465f72c6edc9ad5a518a",
}
EXPECTED_DOWNLOAD_ASSETS = {
    f"data/libero_mujoco3.3.2/{name}.tar.gz": (
        "yuanty/LIBERO-fastwam",
        PINNED_REVISIONS["yuanty/LIBERO-fastwam"],
    )
    for name in (
        "libero_10_no_noops_lerobot",
        "libero_goal_no_noops_lerobot",
        "libero_object_no_noops_lerobot",
        "libero_spatial_no_noops_lerobot",
    )
} | {
    f"checkpoints/fastwam_release/{name}": (
        "yuanty/fastwam",
        PINNED_REVISIONS["yuanty/fastwam"],
    )
    for name in (
        "libero_uncond_2cam224.pt",
        "libero_uncond_2cam224_dataset_stats.json",
    )
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def hash_tree(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    file_count = 0
    total_bytes = 0
    for item in sorted(path.rglob("*"), key=lambda value: value.relative_to(path).as_posix()):
        if item.is_symlink():
            raise ValueError(f"training asset trees must not contain symlinks: {item}")
        if not item.is_file():
            continue
        relative = item.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        size = item.stat().st_size
        digest.update(size.to_bytes(8, "big"))
        with item.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        file_count += 1
        total_bytes += size
    if file_count == 0:
        raise ValueError(f"training asset tree is empty: {path}")
    return {
        "path": str(path),
        "file_count": file_count,
        "bytes": total_bytes,
        "sha256": digest.hexdigest(),
    }


def _validated_download_manifest(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    claimed = payload.pop("manifest_sha256", None)
    actual = _canonical_sha256(payload)
    if claimed != actual:
        raise ValueError(
            f"download manifest digest mismatch: expected={claimed} actual={actual}"
        )
    files = payload.get("files")
    if (
        payload.get("schema_version") != 1
        or not isinstance(files, list)
        or len(files) != len(EXPECTED_DOWNLOAD_ASSETS)
    ):
        raise ValueError("download manifest must contain the six pinned FastWAM assets")
    seen_paths: set[str] = set()
    for item in files:
        if not isinstance(item, dict):
            raise ValueError(f"unexpected repository in download manifest: {item!r}")
        asset_path = item.get("path")
        if not isinstance(asset_path, str) or asset_path not in EXPECTED_DOWNLOAD_ASSETS:
            raise ValueError(f"unexpected asset path in download manifest: {item!r}")
        if asset_path in seen_paths:
            raise ValueError(f"duplicate asset path in download manifest: {asset_path}")
        seen_paths.add(asset_path)
        expected_repo, expected_revision = EXPECTED_DOWNLOAD_ASSETS[asset_path]
        if item.get("repo") != expected_repo:
            raise ValueError(
                f"repository mismatch for {asset_path}: "
                f"expected={expected_repo} actual={item.get('repo')}"
            )
        if item.get("revision") != expected_revision:
            raise ValueError(
                f"unpinned revision for {asset_path}: "
                f"expected={expected_revision} actual={item.get('revision')}"
            )
        if isinstance(item.get("bytes"), bool) or not isinstance(item.get("bytes"), int):
            raise ValueError(f"invalid asset byte count in download manifest: {item!r}")
        if item["bytes"] <= 0:
            raise ValueError(f"asset byte count must be positive: {item!r}")
        sha256 = item.get("sha256")
        if (
            not isinstance(sha256, str)
            or len(sha256) != 64
            or any(character not in "0123456789abcdef" for character in sha256)
        ):
            raise ValueError(f"invalid asset SHA-256 in download manifest: {item!r}")
    if seen_paths != set(EXPECTED_DOWNLOAD_ASSETS):
        raise ValueError("download manifest does not contain the exact pinned asset set")
    payload["manifest_sha256"] = claimed
    return {"path": str(path), "sha256": _sha256_file(path), "payload": payload}


def build_manifest(
    dataset_dirs: list[Path],
    text_embedding_cache: Path,
    vae_checkpoint: Path,
    download_manifest: Path,
) -> dict[str, Any]:
    if len(dataset_dirs) != 4:
        raise ValueError(f"formal LIBERO training requires exactly four datasets, got {len(dataset_dirs)}")
    datasets = [hash_tree(path) for path in dataset_dirs]
    names = [Path(item["path"]).name for item in datasets]
    if len(set(names)) != len(names):
        raise ValueError(f"dataset directory basenames must be unique: {names}")
    dataset_identity = [
        {key: item[key] for key in ("file_count", "bytes", "sha256")} | {"name": name}
        for name, item in sorted(zip(names, datasets))
    ]
    text_cache = hash_tree(text_embedding_cache)
    vae = vae_checkpoint.expanduser().resolve()
    if not vae.is_file():
        raise FileNotFoundError(vae)
    download_identity = _validated_download_manifest(download_manifest)
    result = {
        "schema_version": 1,
        "download_manifest": download_identity,
        "datasets": datasets,
        "dataset_file_count": sum(item["file_count"] for item in datasets),
        "dataset_bytes": sum(item["bytes"] for item in datasets),
        "dataset_content_sha256": _canonical_sha256(dataset_identity),
        "text_embedding_cache": text_cache,
        "vae_checkpoint": {
            "path": str(vae),
            "bytes": vae.stat().st_size,
            "sha256": _sha256_file(vae),
        },
    }
    result["manifest_sha256"] = _canonical_sha256(
        {
            "schema_version": result["schema_version"],
            "download_manifest_file_sha256": download_identity["sha256"],
            "download_manifest_payload_sha256": download_identity["payload"][
                "manifest_sha256"
            ],
            "dataset_content_sha256": result["dataset_content_sha256"],
            "dataset_file_count": result["dataset_file_count"],
            "dataset_bytes": result["dataset_bytes"],
            "text_embedding_cache_sha256": text_cache["sha256"],
            "text_embedding_cache_file_count": text_cache["file_count"],
            "text_embedding_cache_bytes": text_cache["bytes"],
            "vae_checkpoint_sha256": result["vae_checkpoint"]["sha256"],
            "vae_checkpoint_bytes": result["vae_checkpoint"]["bytes"],
        }
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", action="append", type=Path, required=True)
    parser.add_argument("--text-embedding-cache", type=Path, required=True)
    parser.add_argument("--vae-checkpoint", type=Path, required=True)
    parser.add_argument("--download-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_manifest(
        args.dataset_dir,
        args.text_embedding_cache,
        args.vae_checkpoint,
        args.download_manifest,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)


if __name__ == "__main__":
    main()
