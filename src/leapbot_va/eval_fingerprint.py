"""Canonical, machine-checkable identity for LIBERO evaluation results.

Schema 3 deliberately separates the model/runtime behavior contract from the
result sampling contract.  Both expanded contracts are stored in the result so
that a rejected cache entry is inspectable, while their canonical hashes make
launcher-side comparisons cheap and unambiguous.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence


FINGERPRINT_SCHEMA_VERSION = 3
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_FINGERPRINT_KEYS = frozenset(
    {
        "schema_version",
        "checkpoint_sha256",
        "runtime_contract",
        "runtime_contract_sha256",
        "result_contract",
        "result_contract_sha256",
    }
)


def sha256_file(path: str | Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Hash a file without loading it into memory."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"file does not exist: {file_path}")
    digest = hashlib.sha256()
    with file_path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_json_value(value: Any, *, path: str = "$") -> Any:
    """Convert a JSON-like value into a deterministic, strict representation.

    Mapping keys must be strings, tuples become arrays, ``Path`` values become
    strings, non-finite floats are rejected, and negative zero is normalized.
    This keeps hashes stable across insertion order and Python container types.
    """
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"non-finite JSON number at {path}: {value!r}")
        return 0.0 if value == 0.0 else float(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key in sorted(value):
            if not isinstance(key, str):
                raise TypeError(f"JSON object key at {path} must be str, got {type(key)}")
            normalized[key] = normalize_json_value(value[key], path=f"{path}.{key}")
        return normalized
    if isinstance(value, (list, tuple)):
        return [
            normalize_json_value(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    raise TypeError(f"unsupported JSON value at {path}: {type(value)}")


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize with the one canonical representation used for all hashes."""
    normalized = normalize_json_value(value)
    return json.dumps(
        normalized,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _normalize_sha256(value: Any, *, field: str) -> str:
    digest = str(value).strip()
    if not _SHA256_RE.fullmatch(digest):
        raise ValueError(f"{field} must be exactly 64 lowercase hex digits")
    return digest


def build_evaluation_fingerprint(
    *,
    checkpoint_sha256: str,
    runtime_contract: Mapping[str, Any],
    result_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a schema-3 fingerprint from expanded behavior/result contracts."""
    checkpoint_digest = _normalize_sha256(
        checkpoint_sha256, field="checkpoint_sha256"
    )
    runtime = normalize_json_value(runtime_contract, path="$.runtime_contract")
    result = normalize_json_value(result_contract, path="$.result_contract")
    if not isinstance(runtime, dict) or not runtime:
        raise ValueError("runtime_contract must be a non-empty mapping")
    if not isinstance(result, dict) or not result:
        raise ValueError("result_contract must be a non-empty mapping")
    return {
        "schema_version": FINGERPRINT_SCHEMA_VERSION,
        "checkpoint_sha256": checkpoint_digest,
        "runtime_contract": runtime,
        "runtime_contract_sha256": canonical_json_sha256(runtime),
        "result_contract": result,
        "result_contract_sha256": canonical_json_sha256(result),
    }


def normalize_evaluation_fingerprint(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Strictly validate and canonicalize a fingerprint read from JSON."""
    if not isinstance(raw, Mapping):
        raise TypeError("evaluation fingerprint must be a mapping")
    keys = set(raw)
    missing = sorted(_FINGERPRINT_KEYS - keys)
    extra = sorted(keys - _FINGERPRINT_KEYS)
    if missing or extra:
        raise ValueError(
            f"evaluation fingerprint fields mismatch: missing={missing} extra={extra}"
        )
    if int(raw["schema_version"]) != FINGERPRINT_SCHEMA_VERSION:
        raise ValueError(
            "unsupported evaluation fingerprint schema_version: "
            f"{raw['schema_version']} (required {FINGERPRINT_SCHEMA_VERSION})"
        )
    normalized = build_evaluation_fingerprint(
        checkpoint_sha256=str(raw["checkpoint_sha256"]),
        runtime_contract=raw["runtime_contract"],
        result_contract=raw["result_contract"],
    )
    for field in ("runtime_contract_sha256", "result_contract_sha256"):
        supplied = _normalize_sha256(raw[field], field=field)
        if supplied != normalized[field]:
            raise ValueError(
                f"{field} does not match the expanded canonical contract: "
                f"supplied={supplied} actual={normalized[field]}"
            )
    return normalized


def load_evaluation_fingerprint(path: str | Path) -> dict[str, Any]:
    fingerprint_path = Path(path)
    with fingerprint_path.open("r", encoding="utf-8") as stream:
        raw = json.load(stream)
    return normalize_evaluation_fingerprint(raw)


def verify_expected_fingerprint(
    actual: Mapping[str, Any], expected: Mapping[str, Any]
) -> dict[str, Any]:
    """Return canonical actual identity or fail with a useful mismatch message."""
    actual_normalized = normalize_evaluation_fingerprint(actual)
    expected_normalized = normalize_evaluation_fingerprint(expected)
    if actual_normalized != expected_normalized:
        mismatches = [
            key
            for key in sorted(_FINGERPRINT_KEYS)
            if actual_normalized[key] != expected_normalized[key]
        ]
        raise ValueError(
            "actual evaluation fingerprint does not match expected fingerprint; "
            f"mismatched fields={mismatches}, "
            f"expected_checkpoint_sha256={expected_normalized['checkpoint_sha256']}, "
            f"actual_checkpoint_sha256={actual_normalized['checkpoint_sha256']}"
        )
    return actual_normalized


def build_verified_evaluation_fingerprint(
    *,
    checkpoint_path: str | Path,
    runtime_contract: Mapping[str, Any],
    result_contract: Mapping[str, Any],
    expected: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build identity from the checkpoint bytes and optionally verify expected.

    This function intentionally has no checkpoint-digest override.  The
    evaluator must hash the file it will load instead of trusting launcher
    input.  The preflight builder can use :func:`build_evaluation_fingerprint`
    directly when a digest was already computed once.
    """
    actual = build_evaluation_fingerprint(
        checkpoint_sha256=sha256_file(checkpoint_path),
        runtime_contract=runtime_contract,
        result_contract=result_contract,
    )
    if expected is not None:
        return verify_expected_fingerprint(actual, expected)
    return actual


def _is_complete_timing(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) >= 0.0
    )


def _timing_group_closes(
    timing: Mapping[str, Any],
    *,
    total_field: str,
    component_fields: Sequence[str],
) -> bool:
    fields = (total_field, *component_fields)
    if not all(_is_complete_timing(timing.get(field)) for field in fields):
        return False
    total = float(timing[total_field])
    component_sum = sum(float(timing[field]) for field in component_fields)
    return abs(total - component_sum) <= max(1e-6, total * 1e-5)


def _result_contract_trials(fingerprint: Mapping[str, Any]) -> int:
    trials = fingerprint["result_contract"].get("trials")
    if isinstance(trials, bool) or not isinstance(trials, int) or trials <= 0:
        raise ValueError("result_contract.trials must be a positive integer")
    return trials


def result_matches_fingerprint(
    result: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    require_profiled: bool = True,
) -> bool:
    """Accept only a complete result from the exact schema-3 evaluation run."""
    try:
        expected_normalized = normalize_evaluation_fingerprint(expected)
        actual_normalized = normalize_evaluation_fingerprint(
            result["evaluation_fingerprint"]
        )
        trials = _result_contract_trials(expected_normalized)
    except (KeyError, TypeError, ValueError):
        return False
    if actual_normalized != expected_normalized:
        return False
    if result.get("total_episodes") != trials:
        return False

    success_episodes = result.get("success_episodes")
    failure_episodes = result.get("failure_episodes")
    if not isinstance(success_episodes, list) or not isinstance(failure_episodes, list):
        return False
    episode_indices = success_episodes + failure_episodes
    if any(
        isinstance(index, bool) or not isinstance(index, int)
        for index in episode_indices
    ):
        return False
    if sorted(episode_indices) != list(range(trials)):
        return False
    if result.get("successes") != len(success_episodes):
        return False

    completion_steps = result.get("completion_steps")
    memory_metrics = result.get("memory_metrics")
    if not isinstance(completion_steps, list) or len(completion_steps) != trials:
        return False
    if not isinstance(memory_metrics, list) or len(memory_metrics) != trials:
        return False
    if any(
        isinstance(step, bool) or not isinstance(step, int) or step < 0
        for step in completion_steps
    ):
        return False
    if not require_profiled:
        return True

    memory_enabled = bool(
        expected_normalized["runtime_contract"].get("memory", {}).get("enabled")
    )
    for episode in memory_metrics:
        if not isinstance(episode, Mapping):
            return False
        if episode.get("enabled") is not memory_enabled:
            return False
        replans = episode.get("replans")
        if not isinstance(replans, list) or not replans:
            return False
        for replan in replans:
            if not isinstance(replan, Mapping):
                return False
            timing = replan.get("timing")
            if not isinstance(timing, Mapping) or not _timing_group_closes(
                timing,
                total_field="total_inference_s",
                component_fields=(
                    "input_preprocess_s",
                    "model_inference_s",
                    "action_postprocess_s",
                    "latency_residual_s",
                ),
            ):
                return False
            if memory_enabled:
                memory_contract = expected_normalized["runtime_contract"].get(
                    "memory", {}
                )
                causal_components = [
                    "conditioning_s",
                    "observation_prefill_s",
                    "future_video_setup_s",
                    "future_video_denoise_s",
                    "future_video_cache_s",
                    "action_setup_s",
                    "action_denoise_s",
                    "causal_model_residual_s",
                ]
                if memory_contract.get("history_storage_mode") == "strict_replay":
                    causal_components.insert(1, "history_replay_s")
                elif memory_contract.get("history_storage_mode") == "packed_replay":
                    causal_components.insert(1, "history_packed_rebuild_s")
                if not _timing_group_closes(
                    timing,
                    total_field="causal_model_s",
                    component_fields=tuple(causal_components),
                ):
                    return False
                commit = replan.get("commit")
                if not isinstance(commit, Mapping) or not _is_complete_timing(
                    commit.get("commit_s")
                ):
                    return False
    return True


def atomic_write_json(
    path: str | Path,
    payload: Any,
    *,
    indent: int | None = 2,
    encoder_cls: type[json.JSONEncoder] | None = None,
) -> None:
    """Durably write JSON via a same-directory temporary file and ``replace``."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(
                payload,
                stream,
                ensure_ascii=False,
                allow_nan=False,
                indent=indent,
                sort_keys=True,
                cls=encoder_cls,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    hash_parser = subparsers.add_parser("sha256")
    hash_parser.add_argument("file", type=Path)

    match_parser = subparsers.add_parser("matches")
    match_parser.add_argument("result", type=Path)
    match_parser.add_argument("--expected", type=Path, required=True)
    match_parser.add_argument("--allow-unprofiled", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "sha256":
        print(sha256_file(args.file))
        return 0
    expected = load_evaluation_fingerprint(args.expected)
    with args.result.open("r", encoding="utf-8") as stream:
        result = json.load(stream)
    return 0 if result_matches_fingerprint(
        result,
        expected,
        require_profiled=not args.allow_unprofiled,
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
