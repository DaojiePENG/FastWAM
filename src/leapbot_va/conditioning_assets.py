"""Stable identities for external Wan conditioning assets and text caches."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

from leapbot_va.eval_fingerprint import canonical_json_sha256, sha256_file


CONDITIONING_ASSET_SCHEMA_VERSION = 1
TEXT_CACHE_PROVENANCE_SCHEMA_VERSION = 1
TEXT_CACHE_PROVENANCE_FILENAME = ".leapbot_text_cache_provenance.json"
TEXT_CACHE_VERIFICATION_METHODS = frozenset(
    {
        "source_forward_atomic_save_reload_tensor_exact",
        "online_source_forward_cache_tensor_exact",
    }
)


def _stat_signature(stat_result: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        stat_result.st_dev,
        stat_result.st_ino,
        stat_result.st_size,
        stat_result.st_mtime_ns,
        stat_result.st_ctime_ns,
    )


def file_identity(path: str | Path) -> dict[str, Any]:
    """Return a relocation-stable identity for one regular file."""
    unresolved = Path(path).expanduser()
    if unresolved.is_symlink():
        raise ValueError(f"conditioning assets must not be symlinks: {unresolved}")
    file_path = unresolved.resolve()
    if not file_path.is_file():
        raise FileNotFoundError(f"conditioning asset is not a regular file: {file_path}")
    before = file_path.stat()
    digest = sha256_file(file_path)
    after = file_path.stat()
    if _stat_signature(before) != _stat_signature(after):
        raise RuntimeError(f"conditioning asset changed while hashing: {file_path}")
    return {
        "kind": "file",
        "bytes": before.st_size,
        "sha256": digest,
    }


def tree_identity(
    path: str | Path,
    *,
    relative_files: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Hash a directory by relative path, size, and bytes in sorted order.

    ``relative_files`` defines an exact selected set. This is used for text
    caches so the provenance JSON never enters the digest it authenticates.
    """
    unresolved = Path(path).expanduser()
    if unresolved.is_symlink():
        raise ValueError(f"conditioning asset trees must not be symlinks: {unresolved}")
    root = unresolved.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"conditioning asset tree is not a directory: {root}")

    if relative_files is None:
        paths: list[Path] = []
        for item in root.rglob("*"):
            if item.is_symlink():
                raise ValueError(f"conditioning asset trees must not contain symlinks: {item}")
            if item.is_file():
                paths.append(item)
    else:
        names = list(relative_files)
        if len(names) != len(set(names)):
            raise ValueError("conditioning asset tree contains duplicate relative paths")
        paths = []
        for name in names:
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts or str(relative) in {"", "."}:
                raise ValueError(f"invalid relative conditioning asset path: {name!r}")
            item = root / relative
            if item.is_symlink() or not item.is_file():
                raise FileNotFoundError(f"conditioning asset tree entry is missing: {item}")
            paths.append(item)

    paths.sort(key=lambda item: item.relative_to(root).as_posix())
    if not paths:
        raise ValueError(f"conditioning asset tree is empty: {root}")

    import hashlib

    digest = hashlib.sha256()
    total_bytes = 0
    for item in paths:
        relative = item.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        before = item.stat()
        size = before.st_size
        digest.update(size.to_bytes(8, "big"))
        with item.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        after = item.stat()
        if _stat_signature(before) != _stat_signature(after):
            raise RuntimeError(f"conditioning asset changed while hashing: {item}")
        total_bytes += size
    return {
        "kind": "tree",
        "file_count": len(paths),
        "bytes": total_bytes,
        "sha256": digest.hexdigest(),
    }


def resolve_wan_conditioning_paths(
    *,
    model_id: str,
    tokenizer_model_id: str,
    redirect_common_files: bool,
    load_text_encoder: bool,
) -> dict[str, str | None]:
    """Resolve assets with the exact helper used by Wan model construction."""
    from fastwam.models.wan22.helpers.loader import _resolve_configs

    _, text_config, vae_config, tokenizer_config = _resolve_configs(
        model_id=str(model_id),
        tokenizer_model_id=str(tokenizer_model_id),
        redirect_common_files=bool(redirect_common_files),
    )
    vae_config.download_if_necessary()
    if load_text_encoder:
        text_config.download_if_necessary()
        tokenizer_config.download_if_necessary()
    return {
        "vae": str(vae_config.path),
        "text_encoder": str(text_config.path) if load_text_encoder else None,
        "tokenizer": str(tokenizer_config.path) if load_text_encoder else None,
    }


def build_wan_conditioning_identity(
    *,
    model_id: str,
    tokenizer_model_id: str,
    redirect_common_files: bool,
    load_text_encoder: bool,
    resolved_paths: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Hash the exact external assets used by a Wan model instance."""
    if resolved_paths is None:
        paths = resolve_wan_conditioning_paths(
            model_id=model_id,
            tokenizer_model_id=tokenizer_model_id,
            redirect_common_files=redirect_common_files,
            load_text_encoder=load_text_encoder,
        )
    else:
        paths = {
            "vae": resolved_paths.get("vae"),
            "text_encoder": resolved_paths.get("text_encoder"),
            "tokenizer": resolved_paths.get("tokenizer"),
        }
    if paths["vae"] is None:
        raise ValueError("resolved Wan model paths do not contain a VAE")
    if load_text_encoder and (
        paths["text_encoder"] is None or paths["tokenizer"] is None
    ):
        raise ValueError(
            "load_text_encoder=True requires resolved text_encoder and tokenizer paths"
        )

    identity = {
        "schema_version": CONDITIONING_ASSET_SCHEMA_VERSION,
        "loader": {
            "model_id": str(model_id),
            "tokenizer_model_id": str(tokenizer_model_id),
            "redirect_common_files": bool(redirect_common_files),
            "load_text_encoder": bool(load_text_encoder),
        },
        "vae": file_identity(str(paths["vae"])),
        "text_encoder": (
            file_identity(str(paths["text_encoder"])) if load_text_encoder else None
        ),
        "tokenizer": (
            tree_identity(str(paths["tokenizer"])) if load_text_encoder else None
        ),
    }
    identity["identity_sha256"] = canonical_json_sha256(identity)
    return identity


def _cache_file_names(cache_dir: Path) -> list[str]:
    names: list[str] = []
    for item in cache_dir.iterdir():
        if item.is_symlink():
            raise ValueError(f"text cache must not contain symlinks: {item}")
        if item.is_file() and item.suffix == ".pt":
            names.append(item.name)
    return sorted(names)


def _resolve_cache_dir(path: str | Path) -> Path:
    unresolved = Path(path).expanduser()
    if unresolved.is_symlink():
        raise ValueError(f"text cache directory must not be a symlink: {unresolved}")
    resolved = unresolved.resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"text cache directory does not exist: {resolved}")
    return resolved


def build_text_cache_provenance(
    *,
    cache_dir: str | Path,
    cache_file_names: Iterable[str],
    model_id: str,
    tokenizer_model_id: str,
    redirect_common_files: bool,
    context_len: int,
    text_encoder_path: str | Path,
    tokenizer_path: str | Path,
    verified_file_count: int,
    verification_method: str = "source_forward_atomic_save_reload_tensor_exact",
) -> dict[str, Any]:
    """Build provenance for a cache produced by one encoder/tokenizer pair."""
    cache = _resolve_cache_dir(cache_dir)
    expected_names = sorted(cache_file_names)
    if not expected_names or len(expected_names) != len(set(expected_names)):
        raise ValueError("text cache provenance requires unique cache file names")
    if any(Path(name).name != name or not name.endswith(".pt") for name in expected_names):
        raise ValueError("text cache provenance accepts only top-level .pt cache files")
    actual_names = _cache_file_names(cache)
    if actual_names != expected_names:
        missing = sorted(set(expected_names) - set(actual_names))
        extra = sorted(set(actual_names) - set(expected_names))
        raise ValueError(
            f"text cache file set mismatch: missing={missing} extra={extra}"
        )
    if int(context_len) <= 0:
        raise ValueError("text cache context_len must be positive")
    if (
        isinstance(verified_file_count, bool)
        or int(verified_file_count) != len(expected_names)
    ):
        raise ValueError(
            "text cache provenance requires exact source-forward/save/reload "
            f"verification for every file: expected={len(expected_names)} "
            f"actual={verified_file_count}"
        )
    if verification_method not in TEXT_CACHE_VERIFICATION_METHODS:
        raise ValueError(
            f"unsupported text cache verification method: {verification_method!r}"
        )

    payload = {
        "schema_version": TEXT_CACHE_PROVENANCE_SCHEMA_VERSION,
        "encoding_contract": {
            "model_id": str(model_id),
            "tokenizer_model_id": str(tokenizer_model_id),
            "redirect_common_files": bool(redirect_common_files),
            "context_len": int(context_len),
            "text_encoder_dtype": "bfloat16",
            "tokenizer_clean": "whitespace",
            "add_special_tokens": True,
            "cache_context_dtype": "bfloat16",
            "cache_mask_dtype": "bool",
        },
        "source_assets": {
            "text_encoder": file_identity(text_encoder_path),
            "tokenizer": tree_identity(tokenizer_path),
        },
        "verification": {
            "method": verification_method,
            "verified_file_count": len(expected_names),
            "context_tensor_exact": True,
            "mask_tensor_exact": True,
        },
        "cache_files": expected_names,
        "cache": tree_identity(cache, relative_files=expected_names),
    }
    payload["provenance_sha256"] = canonical_json_sha256(payload)
    return payload


def write_text_cache_provenance(
    cache_dir: str | Path, provenance: Mapping[str, Any]
) -> Path:
    """Atomically persist provenance outside the authenticated .pt file set."""
    cache = _resolve_cache_dir(cache_dir)
    output = cache / TEXT_CACHE_PROVENANCE_FILENAME
    temporary = output.with_suffix(output.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(dict(provenance), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)
    return output


def load_and_validate_text_cache_provenance(
    cache_dir: str | Path,
    *,
    text_encoder_path: str | Path,
    tokenizer_path: str | Path,
    model_id: str | None = None,
    tokenizer_model_id: str | None = None,
    redirect_common_files: bool | None = None,
    context_len: int | None = None,
    required_verification_method: str | None = None,
) -> dict[str, Any]:
    """Rehash cache and source assets and reject any stale provenance."""
    cache = _resolve_cache_dir(cache_dir)
    path = cache / TEXT_CACHE_PROVENANCE_FILENAME
    if not path.is_file():
        raise FileNotFoundError(
            f"text cache provenance is missing: {path}; rerun "
            "scripts/precompute_text_embeds.py with overwrite=true or run "
            "scripts/verify_text_cache_provenance.py"
        )
    raw = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "encoding_contract",
        "source_assets",
        "verification",
        "cache_files",
        "cache",
        "provenance_sha256",
    }
    if not isinstance(raw, dict) or set(raw) != required:
        raise ValueError("text cache provenance has an invalid field set")
    if raw["schema_version"] != TEXT_CACHE_PROVENANCE_SCHEMA_VERSION:
        raise ValueError("unsupported text cache provenance schema_version")
    claimed = raw.pop("provenance_sha256")
    actual = canonical_json_sha256(raw)
    raw["provenance_sha256"] = claimed
    if claimed != actual:
        raise ValueError(
            f"text cache provenance digest mismatch: expected={claimed} actual={actual}"
        )

    if not isinstance(raw["cache_files"], list):
        raise ValueError("text cache provenance cache_files is invalid")
    verification = raw["verification"]
    if (
        not isinstance(verification, dict)
        or set(verification)
        != {
            "method",
            "verified_file_count",
            "context_tensor_exact",
            "mask_tensor_exact",
        }
        or verification["method"] not in TEXT_CACHE_VERIFICATION_METHODS
        or verification["verified_file_count"] != len(raw["cache_files"])
        or verification["context_tensor_exact"] is not True
        or verification["mask_tensor_exact"] is not True
    ):
        raise ValueError("text cache provenance verification evidence is invalid")
    if (
        required_verification_method is not None
        and verification["method"] != required_verification_method
    ):
        raise ValueError(
            "text cache provenance verification method mismatch: "
            f"required={required_verification_method!r} "
            f"actual={verification['method']!r}"
        )

    contract = raw["encoding_contract"]
    contract_keys = {
        "model_id",
        "tokenizer_model_id",
        "redirect_common_files",
        "context_len",
        "text_encoder_dtype",
        "tokenizer_clean",
        "add_special_tokens",
        "cache_context_dtype",
        "cache_mask_dtype",
    }
    if not isinstance(contract, dict) or set(contract) != contract_keys:
        raise ValueError("text cache provenance encoding_contract is invalid")
    if (
        not isinstance(contract["model_id"], str)
        or not contract["model_id"]
        or not isinstance(contract["tokenizer_model_id"], str)
        or not contract["tokenizer_model_id"]
        or not isinstance(contract["redirect_common_files"], bool)
    ):
        raise ValueError("text cache provenance loader identity is invalid")
    fixed_contract = {
        "text_encoder_dtype": "bfloat16",
        "tokenizer_clean": "whitespace",
        "add_special_tokens": True,
        "cache_context_dtype": "bfloat16",
        "cache_mask_dtype": "bool",
    }
    for key, expected in fixed_contract.items():
        if contract[key] != expected:
            raise ValueError(
                f"text cache provenance {key} mismatch: "
                f"expected={expected!r} actual={contract[key]!r}"
            )
    if (
        isinstance(contract["context_len"], bool)
        or not isinstance(contract["context_len"], int)
        or contract["context_len"] <= 0
    ):
        raise ValueError("text cache provenance context_len must be a positive integer")
    expected_contract = {
        "model_id": model_id,
        "tokenizer_model_id": tokenizer_model_id,
        "redirect_common_files": redirect_common_files,
        "context_len": context_len,
    }
    for key, expected in expected_contract.items():
        if expected is not None and contract.get(key) != expected:
            raise ValueError(
                f"text cache provenance {key} mismatch: "
                f"expected={expected!r} actual={contract.get(key)!r}"
            )

    cache_files = raw["cache_files"]
    if (
        not isinstance(cache_files, list)
        or not cache_files
        or any(
            not isinstance(name, str)
            or Path(name).name != name
            or not name.endswith(".pt")
            for name in cache_files
        )
        or cache_files != sorted(set(cache_files))
    ):
        raise ValueError("text cache provenance cache_files is invalid")
    actual_cache = tree_identity(cache, relative_files=cache_files)
    actual_names = _cache_file_names(cache)
    if actual_names != cache_files or actual_cache != raw["cache"]:
        raise ValueError("text cache bytes or file set do not match provenance")
    actual_sources = {
        "text_encoder": file_identity(text_encoder_path),
        "tokenizer": tree_identity(tokenizer_path),
    }
    if actual_sources != raw["source_assets"]:
        raise ValueError("text cache encoder/tokenizer assets do not match provenance")
    return raw
