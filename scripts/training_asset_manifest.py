#!/usr/bin/env python3
"""Hash every immutable asset read by formal LeapBot training."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from fastwam.datasets.lerobot.text_cache import (
    DEFAULT_PROMPT,
    wan_text_cache_filename,
)
from leapbot_va.conditioning_assets import (
    load_and_validate_text_cache_provenance,
    resolve_wan_conditioning_paths,
)


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


def expected_text_cache_files(
    dataset_dirs: list[Path], *, context_len: int, model_id: str
) -> list[str]:
    """Derive the exact cache key set consumed by the formal datasets."""
    prompts: set[str] = set()
    for dataset_dir in dataset_dirs:
        tasks_path = dataset_dir.expanduser().resolve() / "meta" / "tasks.jsonl"
        if not tasks_path.is_file():
            raise FileNotFoundError(f"formal dataset tasks metadata is missing: {tasks_path}")
        with tasks_path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                item = json.loads(line)
                if not isinstance(item, dict) or "task" not in item:
                    raise ValueError(f"missing task at {tasks_path}:{line_number}")
                prompts.add(DEFAULT_PROMPT.format(task=str(item["task"])))
    if not prompts:
        raise ValueError("formal datasets contain no text prompts")
    return sorted(
        wan_text_cache_filename(
            prompt, context_len=context_len, model_id=model_id
        )
        for prompt in prompts
    )


def hash_tree(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    file_count = 0
    total_bytes = 0
    for item in sorted(path.rglob("*"), key=lambda value: value.relative_to(path).as_posix()):
        # if item.is_symlink():
        #     raise ValueError(f"training asset trees must not contain symlinks: {item}")
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
    download_manifest: Path | None,
    text_encoder_checkpoint: Path,
    tokenizer_dir: Path,
    *,
    model_id: str = "Wan-AI/Wan2.2-TI2V-5B",
    tokenizer_model_id: str = "Wan-AI/Wan2.1-T2V-1.3B",
    redirect_common_files: bool = True,
    context_len: int = 128,
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
    text_cache_provenance = load_and_validate_text_cache_provenance(
        text_embedding_cache,
        text_encoder_path=text_encoder_checkpoint,
        tokenizer_path=tokenizer_dir,
        model_id=model_id,
        tokenizer_model_id=tokenizer_model_id,
        redirect_common_files=redirect_common_files,
        context_len=context_len,
        required_verification_method="online_source_forward_cache_tensor_exact",
    )
    cache_path = text_embedding_cache.expanduser().resolve()
    expected_cache_files = expected_text_cache_files(
        dataset_dirs, context_len=context_len, model_id=model_id
    )
    if text_cache_provenance["cache_files"] != expected_cache_files:
        missing = sorted(
            set(expected_cache_files) - set(text_cache_provenance["cache_files"])
        )
        extra = sorted(
            set(text_cache_provenance["cache_files"]) - set(expected_cache_files)
        )
        raise ValueError(
            f"text cache provenance does not match dataset prompts: "
            f"missing={missing} extra={extra}"
        )
    text_cache = dict(text_cache_provenance["cache"])
    text_cache["path"] = str(cache_path)
    vae = vae_checkpoint.expanduser().resolve()
    if not vae.is_file():
        raise FileNotFoundError(vae)
    download_identity = (
        _validated_download_manifest(download_manifest)
        if download_manifest is not None else None
    )
    result = {
        "schema_version": 3,
        "datasets": datasets,
        "dataset_file_count": sum(item["file_count"] for item in datasets),
        "dataset_bytes": sum(item["bytes"] for item in datasets),
        "dataset_content_sha256": _canonical_sha256(dataset_identity),
        "text_embedding_cache": text_cache,
        "text_cache_provenance": text_cache_provenance,
        "vae_checkpoint": {
            "path": str(vae),
            "bytes": vae.stat().st_size,
            "sha256": _sha256_file(vae),
        },
    }
    result["manifest_sha256"] = _canonical_sha256(
        {
            "schema_version": result["schema_version"],
            "dataset_content_sha256": result["dataset_content_sha256"],
            "dataset_file_count": result["dataset_file_count"],
            "dataset_bytes": result["dataset_bytes"],
            "text_embedding_cache_sha256": text_cache["sha256"],
            "text_embedding_cache_file_count": text_cache["file_count"],
            "text_embedding_cache_bytes": text_cache["bytes"],
            "text_cache_provenance_sha256": text_cache_provenance[
                "provenance_sha256"
            ],
            "text_encoder_checkpoint_sha256": text_cache_provenance[
                "source_assets"
            ]["text_encoder"]["sha256"],
            "tokenizer_sha256": text_cache_provenance["source_assets"][
                "tokenizer"
            ]["sha256"],
            "vae_checkpoint_sha256": result["vae_checkpoint"]["sha256"],
            "vae_checkpoint_bytes": result["vae_checkpoint"]["bytes"],
        }
    )
    if download_identity is not None:
        result["download_manifest"] = download_identity
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", action="append", type=Path, required=True)
    parser.add_argument("--text-embedding-cache", type=Path, required=True)
    parser.add_argument("--vae-checkpoint", type=Path, required=True)
    parser.add_argument("--download-manifest", type=Path)
    parser.add_argument("--model-id", default="Wan-AI/Wan2.2-TI2V-5B")
    parser.add_argument(
        "--tokenizer-model-id", default="Wan-AI/Wan2.1-T2V-1.3B"
    )
    parser.add_argument(
        "--redirect-common-files",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--context-len", type=int, default=128)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    resolved = resolve_wan_conditioning_paths(
        model_id=args.model_id,
        tokenizer_model_id=args.tokenizer_model_id,
        redirect_common_files=args.redirect_common_files,
        load_text_encoder=True,
    )
    configured_vae = args.vae_checkpoint.expanduser().resolve()
    resolved_vae = Path(str(resolved["vae"])).expanduser().resolve()
    if configured_vae != resolved_vae:
        raise ValueError(
            "--vae-checkpoint is not the VAE resolved by the configured Wan loader: "
            f"configured={configured_vae} resolved={resolved_vae}"
        )
    result = build_manifest(
        args.dataset_dir,
        args.text_embedding_cache,
        resolved_vae,
        args.download_manifest,
        Path(str(resolved["text_encoder"])),
        Path(str(resolved["tokenizer"])),
        model_id=args.model_id,
        tokenizer_model_id=args.tokenizer_model_id,
        redirect_common_files=args.redirect_common_files,
        context_len=args.context_len,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)


if __name__ == "__main__":
    main()
