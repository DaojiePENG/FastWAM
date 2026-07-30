#!/usr/bin/env python3
"""Create and validate deterministic selections from fixed-noise history audits."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
MANIFEST_KIND = "history_audit_selection_manifest"
AUDIT_KIND = "paired_history_stratified_loss_audit"
REQUIRED_HISTORY_VARIANTS = frozenset({"correct", "masked", "shuffled"})
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
GIT_REVISION = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
ABLATION_FIELDS = {
    "learning_rate": "learning_rate",
    "initial_block_oversample": "initial_block_oversample",
}
CORRECT_LOSS_PATH = (
    "variant_summaries.correct.history_distribution_weighted.loss_action"
)
NATIVE_H0_MSE_PATH = (
    "native_diagnostic_summary.by_history.0."
    "action_raw_mse_executed10_all7"
)


class SelectionValidationError(ValueError):
    """Raised when an audit, contract, or manifest cannot be trusted."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = stream.name
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SelectionValidationError(f"invalid {label} JSON at {path}: {error}") from error
    if not isinstance(value, dict):
        raise SelectionValidationError(f"{label} must be a JSON object: {path}")
    return value


def _finite_number(value: Any, *, label: str, nonnegative: bool = True) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SelectionValidationError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        raise SelectionValidationError(f"{label} must be finite and non-negative")
    return result


def _require_metric_summary(
    checkpoint: dict[str, Any], path: tuple[str, ...], *, label: str
) -> dict[str, float]:
    node: Any = checkpoint
    for key in path:
        if not isinstance(node, dict) or key not in node:
            raise SelectionValidationError(f"{label} is missing {'.'.join(path)}")
        node = node[key]
    if not isinstance(node, dict):
        raise SelectionValidationError(f"{label} metric summary must be an object")
    mean = _finite_number(node.get("mean"), label=f"{label} mean")
    low = _finite_number(node.get("ci95_low"), label=f"{label} ci95_low")
    high = _finite_number(node.get("ci95_high"), label=f"{label} ci95_high")
    if low > high:
        raise SelectionValidationError(f"{label} requires ci95_low <= ci95_high")
    return {"mean": mean, "ci95_low": low, "ci95_high": high}


def _validate_source_identity(audit: dict[str, Any]) -> dict[str, Any]:
    identity = audit.get("source_identity")
    if not isinstance(identity, dict):
        raise SelectionValidationError("audit is missing source_identity")
    revision = identity.get("revision")
    if not isinstance(revision, str) or GIT_REVISION.fullmatch(revision) is None:
        raise SelectionValidationError("audit source revision is invalid")
    if identity.get("dirty") is not False:
        raise SelectionValidationError("audit source identity must be clean")
    if identity.get("worktree_sha256") != EMPTY_SHA256:
        raise SelectionValidationError(
            "clean audit source must have the empty-worktree SHA256"
        )
    return {
        "revision": revision,
        "dirty": False,
        "worktree_sha256": EMPTY_SHA256,
    }


def _validate_audit_protocol(audit: dict[str, Any]) -> dict[str, Any]:
    if audit.get("kind") != AUDIT_KIND:
        raise SelectionValidationError(
            f"expected audit kind {AUDIT_KIND!r}, got {audit.get('kind')!r}"
        )
    source_identity = _validate_source_identity(audit)
    fixed_u_values = audit.get("fixed_u_values")
    if not isinstance(fixed_u_values, list) or not fixed_u_values:
        raise SelectionValidationError("audit fixed_u_values must be non-empty")
    normalized_u: list[float] = []
    for index, value in enumerate(fixed_u_values):
        fixed_u = _finite_number(value, label=f"fixed_u_values[{index}]", nonnegative=False)
        if not 0.0 <= fixed_u < 1.0:
            raise SelectionValidationError("fixed_u_values must lie in [0, 1)")
        normalized_u.append(fixed_u)
    if len(set(normalized_u)) != len(normalized_u):
        raise SelectionValidationError("fixed_u_values must not contain duplicates")

    noise_repeats = audit.get("noise_repeats")
    if isinstance(noise_repeats, bool) or not isinstance(noise_repeats, int):
        raise SelectionValidationError("audit noise_repeats must be an integer")
    if noise_repeats <= 0:
        raise SelectionValidationError("audit noise_repeats must be positive")

    variants = audit.get("history_variants")
    if not isinstance(variants, list) or any(not isinstance(v, str) for v in variants):
        raise SelectionValidationError("audit history_variants must be a string list")
    if len(variants) != len(set(variants)):
        raise SelectionValidationError("audit history_variants contains duplicates")
    if set(variants) != REQUIRED_HISTORY_VARIANTS:
        raise SelectionValidationError(
            "audit must contain exactly correct, masked, and shuffled history variants"
        )
    return {
        "fixed_u_values": normalized_u,
        "noise_repeats": noise_repeats,
        "history_variants": sorted(REQUIRED_HISTORY_VARIANTS),
        "source_identity": source_identity,
    }


def _validate_draw_rows(
    rows: Any,
    *,
    checkpoint_label: str,
    variant: str,
    fixed_u_values: list[float],
    noise_repeats: int,
    require_complete: bool,
) -> None:
    if not isinstance(rows, list) or not rows:
        raise SelectionValidationError(
            f"checkpoint {checkpoint_label!r} has no {variant} history records"
        )
    actual_by_sample: dict[tuple[int, int], set[tuple[int, int]]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise SelectionValidationError(
                f"{checkpoint_label} {variant} record {index} is not an object"
            )
        try:
            dataset_index = int(row["dataset_index"])
            history_blocks = int(row["history_blocks"])
            u_index = int(row["u_index"])
            noise_replica = int(row["noise_replica"])
            fixed_u = float(row["fixed_u"])
        except (KeyError, TypeError, ValueError) as error:
            raise SelectionValidationError(
                f"{checkpoint_label} {variant} record {index} has invalid draw identity"
            ) from error
        if not 0 <= u_index < len(fixed_u_values):
            raise SelectionValidationError(f"{checkpoint_label} has out-of-range u_index")
        if fixed_u != fixed_u_values[u_index]:
            raise SelectionValidationError(f"{checkpoint_label} fixed_u/u_index disagree")
        if not 0 <= noise_replica < noise_repeats:
            raise SelectionValidationError(
                f"{checkpoint_label} has out-of-range noise_replica"
            )
        key = (dataset_index, history_blocks)
        draw = (u_index, noise_replica)
        sample_draws = actual_by_sample.setdefault(key, set())
        if draw in sample_draws:
            raise SelectionValidationError(
                f"{checkpoint_label} contains a duplicate {variant} draw {key + draw}"
            )
        sample_draws.add(draw)
    if require_complete:
        expected = {
            (u_index, noise_replica)
            for u_index in range(len(fixed_u_values))
            for noise_replica in range(noise_repeats)
        }
        incomplete = [key for key, draws in actual_by_sample.items() if draws != expected]
        if incomplete:
            raise SelectionValidationError(
                f"{checkpoint_label} has incomplete {variant} fixed-noise draws: "
                f"{incomplete[:3]}"
            )


def _resolve_checkpoint_path(raw: Any, audit_path: Path) -> Path:
    if not isinstance(raw, str) or not raw:
        raise SelectionValidationError("audit checkpoint path must be a non-empty string")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = audit_path.parent / path
    return path.resolve()


def _validated_checkpoint_entries(
    audit: dict[str, Any], audit_path: Path, protocol: dict[str, Any]
) -> list[tuple[dict[str, Any], Path]]:
    checkpoints = audit.get("checkpoints")
    if not isinstance(checkpoints, list) or not checkpoints:
        raise SelectionValidationError("audit checkpoints must be a non-empty list")
    result: list[tuple[dict[str, Any], Path]] = []
    seen: set[Path] = set()
    for index, checkpoint in enumerate(checkpoints):
        if not isinstance(checkpoint, dict):
            raise SelectionValidationError(f"audit checkpoint {index} is not an object")
        path = _resolve_checkpoint_path(checkpoint.get("checkpoint"), audit_path)
        if path in seen:
            raise SelectionValidationError(f"duplicate audit checkpoint path: {path}")
        seen.add(path)
        if not path.is_file():
            raise SelectionValidationError(f"audit checkpoint does not exist: {path}")
        declared_sha = checkpoint.get("checkpoint_sha256")
        if not isinstance(declared_sha, str) or HEX_SHA256.fullmatch(declared_sha) is None:
            raise SelectionValidationError(
                f"audit checkpoint is missing a valid checkpoint_sha256: {path}"
            )
        actual_sha = sha256_file(path)
        if actual_sha != declared_sha:
            raise SelectionValidationError(
                f"checkpoint SHA256 mismatch for {path}: "
                f"audit={declared_sha} current={actual_sha}"
            )
        summaries = checkpoint.get("variant_summaries")
        records = checkpoint.get("variant_records")
        if not isinstance(summaries, dict) or not isinstance(records, dict):
            raise SelectionValidationError(f"checkpoint {path} lacks variant audit data")
        for variant in sorted(REQUIRED_HISTORY_VARIANTS):
            if variant not in summaries or variant not in records:
                raise SelectionValidationError(
                    f"checkpoint {path} is missing {variant} history diagnostics"
                )
            _validate_draw_rows(
                records[variant],
                checkpoint_label=str(checkpoint.get("label", path.name)),
                variant=variant,
                fixed_u_values=protocol["fixed_u_values"],
                noise_repeats=protocol["noise_repeats"],
                require_complete=variant in {"correct", "masked"},
            )
        _require_metric_summary(
            checkpoint,
            ("variant_summaries", "correct", "history_distribution_weighted", "loss_action"),
            label=f"checkpoint {path} correct-history loss_action",
        )
        result.append((checkpoint, path))
    return result


def _find_adjacent_contract(checkpoint: Path) -> Path | None:
    for parent in checkpoint.parents:
        candidate = parent / "run_contract.txt"
        if candidate.is_file():
            return candidate.resolve()
    return None


def _read_contract(path: Path) -> tuple[dict[str, str], str, str]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise SelectionValidationError(f"cannot read run contract {path}: {error}") from error
    lines = text.splitlines()
    if not lines or not lines[0].startswith("run_contract_sha256="):
        raise SelectionValidationError(
            f"run contract must begin with run_contract_sha256: {path}"
        )
    values: dict[str, str] = {}
    for line_number, line in enumerate(lines, start=1):
        if not line or "=" not in line:
            raise SelectionValidationError(
                f"malformed run contract line {line_number} in {path}"
            )
        key, value = line.split("=", 1)
        if not key or key in values:
            raise SelectionValidationError(
                f"duplicate or empty run contract key {key!r} in {path}"
            )
        values[key] = value
    stored_sha = values["run_contract_sha256"]
    if HEX_SHA256.fullmatch(stored_sha) is None:
        raise SelectionValidationError(f"invalid run_contract_sha256 in {path}")
    payload_sha = hashlib.sha256("\n".join(lines[1:]).encode("utf-8")).hexdigest()
    if payload_sha != stored_sha:
        raise SelectionValidationError(
            f"run contract payload SHA256 mismatch for {path}: "
            f"stored={stored_sha} current={payload_sha}"
        )
    return values, sha256_file(path), payload_sha


def _normalize_ablation_value(kind: str, raw: str) -> tuple[str, Decimal]:
    if kind == "learning_rate":
        try:
            numeric = Decimal(raw)
        except InvalidOperation as error:
            raise SelectionValidationError(f"invalid learning_rate {raw!r}") from error
        if not numeric.is_finite() or numeric <= 0:
            raise SelectionValidationError("learning_rate must be finite and positive")
        return raw, numeric
    try:
        numeric_int = int(raw)
    except ValueError as error:
        raise SelectionValidationError(
            f"invalid initial_block_oversample {raw!r}"
        ) from error
    if str(numeric_int) != raw or numeric_int <= 0:
        raise SelectionValidationError(
            "initial_block_oversample must be a canonical positive integer"
        )
    return raw, Decimal(numeric_int)


def _candidate_manifest_entry(
    *,
    kind: str,
    checkpoint: dict[str, Any],
    checkpoint_path: Path,
    contract_path: Path,
    contract: dict[str, str],
    contract_file_sha: str,
) -> tuple[dict[str, Any], Decimal]:
    field = ABLATION_FIELDS[kind]
    if field not in contract:
        raise SelectionValidationError(f"run contract {contract_path} lacks {field}")
    value, numeric_value = _normalize_ablation_value(kind, contract[field])
    loss = _require_metric_summary(
        checkpoint,
        ("variant_summaries", "correct", "history_distribution_weighted", "loss_action"),
        label=f"checkpoint {checkpoint_path} correct-history loss_action",
    )
    entry: dict[str, Any] = {
        "label": str(checkpoint.get("label", checkpoint_path.name)),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": str(checkpoint["checkpoint_sha256"]),
        "run_contract": str(contract_path),
        "run_contract_file_sha256": contract_file_sha,
        "run_contract_sha256": contract["run_contract_sha256"],
        "value": value,
        "correct_history_loss_action": loss,
    }
    if kind == "initial_block_oversample":
        native = _require_metric_summary(
            checkpoint,
            (
                "native_diagnostic_summary",
                "by_history",
                "0",
                "action_raw_mse_executed10_all7",
            ),
            label=f"checkpoint {checkpoint_path} native H0 executed10 raw MSE",
        )
        entry["native_h0_executed10_raw_mse"] = native
    return entry, numeric_value


def _validate_contract_equivalence(
    contracts: list[tuple[Path, dict[str, str]]], *, kind: str
) -> None:
    field = ABLATION_FIELDS[kind]
    reference_path, reference = contracts[0]
    reference_shared = {
        key: value
        for key, value in reference.items()
        if key not in {field, "run_contract_sha256"}
    }
    for path, contract in contracts[1:]:
        shared = {
            key: value
            for key, value in contract.items()
            if key not in {field, "run_contract_sha256"}
        }
        if shared != reference_shared:
            differing = sorted(
                key
                for key in set(reference_shared).union(shared)
                if reference_shared.get(key) != shared.get(key)
            )
            raise SelectionValidationError(
                "candidate run contracts differ outside the ablation field "
                f"{field!r}: {reference_path} vs {path}; fields={differing}"
            )


def _closed_intervals_overlap(left: dict[str, float], right: dict[str, float]) -> bool:
    return max(left["ci95_low"], right["ci95_low"]) <= min(
        left["ci95_high"], right["ci95_high"]
    )


def build_manifest(audit_path: Path, *, kind: str) -> dict[str, Any]:
    if kind not in ABLATION_FIELDS:
        raise SelectionValidationError(f"unsupported selection kind: {kind!r}")
    audit_path = audit_path.expanduser().resolve()
    if not audit_path.is_file():
        raise SelectionValidationError(f"audit does not exist: {audit_path}")
    audit_sha = sha256_file(audit_path)
    audit = _load_json(audit_path, label="history audit")
    protocol = _validate_audit_protocol(audit)
    checkpoint_entries = _validated_checkpoint_entries(audit, audit_path, protocol)

    candidates: list[dict[str, Any]] = []
    numeric_values: dict[str, Decimal] = {}
    contracts: list[tuple[Path, dict[str, str]]] = []
    for checkpoint, checkpoint_path in checkpoint_entries:
        contract_path = _find_adjacent_contract(checkpoint_path)
        if contract_path is None:
            continue
        contract, contract_file_sha, _ = _read_contract(contract_path)
        entry, numeric = _candidate_manifest_entry(
            kind=kind,
            checkpoint=checkpoint,
            checkpoint_path=checkpoint_path,
            contract_path=contract_path,
            contract=contract,
            contract_file_sha=contract_file_sha,
        )
        candidates.append(entry)
        numeric_values[entry["checkpoint"]] = numeric
        contracts.append((contract_path, contract))
    if len(candidates) < 2:
        raise SelectionValidationError(
            "selection requires at least two audited checkpoints with adjacent run contracts"
        )
    _validate_contract_equivalence(contracts, kind=kind)
    if len({numeric_values[c["checkpoint"]] for c in candidates}) != len(candidates):
        raise SelectionValidationError("candidate ablation values must be unique")
    candidates.sort(key=lambda row: (numeric_values[row["checkpoint"]], row["checkpoint"]))

    if kind == "learning_rate":
        for candidate in candidates:
            candidate["eligible"] = True
        selected = min(
            candidates,
            key=lambda row: (
                row["correct_history_loss_action"]["mean"],
                numeric_values[row["checkpoint"]],
                row["checkpoint"],
            ),
        )
        rule = {
            "primary": f"minimum {CORRECT_LOSS_PATH}.mean",
            "tie_break": "lowest numeric learning_rate, then checkpoint path",
        }
    else:
        best = min(
            candidates,
            key=lambda row: (
                row["correct_history_loss_action"]["mean"],
                numeric_values[row["checkpoint"]],
                row["checkpoint"],
            ),
        )
        best_loss = best["correct_history_loss_action"]
        for candidate in candidates:
            candidate_loss = candidate["correct_history_loss_action"]
            within_five_percent = candidate_loss["mean"] <= best_loss["mean"] * 1.05
            ci_overlaps_best = _closed_intervals_overlap(candidate_loss, best_loss)
            candidate["within_best_loss_factor_1p05"] = within_five_percent
            candidate["ci95_overlaps_best"] = ci_overlaps_best
            candidate["eligible"] = within_five_percent and ci_overlaps_best
        eligible = [candidate for candidate in candidates if candidate["eligible"]]
        selected = min(
            eligible,
            key=lambda row: (
                row["native_h0_executed10_raw_mse"]["mean"],
                numeric_values[row["checkpoint"]],
                row["checkpoint"],
            ),
        )
        rule = {
            "best_definition": f"minimum {CORRECT_LOSS_PATH}.mean",
            "eligibility": (
                "correct-history mean <= best mean * 1.05 and closed ci95 "
                "interval overlaps the best candidate"
            ),
            "secondary": f"minimum {NATIVE_H0_MSE_PATH}.mean among eligible candidates",
            "tie_break": (
                "lowest numeric initial_block_oversample, then checkpoint path"
            ),
        }

    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": MANIFEST_KIND,
        "selection_kind": kind,
        "audit": str(audit_path),
        "audit_sha256": audit_sha,
        "audit_protocol": protocol,
        "candidates": candidates,
        "rule": rule,
        "selected_value": selected["value"],
        "selected_checkpoint": selected["checkpoint"],
        "selected_checkpoint_sha256": selected["checkpoint_sha256"],
    }
    manifest["manifest_sha256"] = canonical_json_sha256(manifest)
    return manifest


def create_manifest(audit_path: Path, *, kind: str, output: Path) -> dict[str, Any]:
    manifest = build_manifest(audit_path, kind=kind)
    atomic_write_json(output, manifest)
    return manifest


def validate_manifest(
    path: Path, *, expected_kind: str | None = None
) -> dict[str, Any]:
    path = path.expanduser().resolve()
    manifest = _load_json(path, label="selection manifest")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise SelectionValidationError("unsupported selection manifest schema_version")
    if manifest.get("kind") != MANIFEST_KIND:
        raise SelectionValidationError("invalid selection manifest kind")
    if expected_kind is not None and manifest.get("selection_kind") != expected_kind:
        raise SelectionValidationError(
            "selection manifest kind mismatch: "
            f"expected={expected_kind!r} actual={manifest.get('selection_kind')!r}"
        )
    declared_manifest_sha = manifest.get("manifest_sha256")
    if not isinstance(declared_manifest_sha, str) or HEX_SHA256.fullmatch(
        declared_manifest_sha
    ) is None:
        raise SelectionValidationError("selection manifest has no valid manifest_sha256")
    unhashed = dict(manifest)
    del unhashed["manifest_sha256"]
    current_manifest_sha = canonical_json_sha256(unhashed)
    if current_manifest_sha != declared_manifest_sha:
        raise SelectionValidationError(
            "selection manifest SHA256 mismatch: "
            f"stored={declared_manifest_sha} current={current_manifest_sha}"
        )

    audit_raw = manifest.get("audit")
    kind = manifest.get("selection_kind")
    if not isinstance(audit_raw, str) or not isinstance(kind, str):
        raise SelectionValidationError("selection manifest lacks audit/selection_kind")
    audit_path = Path(audit_raw).expanduser().resolve()
    if not audit_path.is_file():
        raise SelectionValidationError(f"manifest audit does not exist: {audit_path}")
    current_audit_sha = sha256_file(audit_path)
    if current_audit_sha != manifest.get("audit_sha256"):
        raise SelectionValidationError(
            "selection audit SHA256 mismatch: "
            f"stored={manifest.get('audit_sha256')} current={current_audit_sha}"
        )

    rebuilt = build_manifest(audit_path, kind=kind)
    if manifest != rebuilt:
        raise SelectionValidationError(
            "selection manifest no longer matches its checkpoint/contract inputs"
        )
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create", help="create a deterministic manifest")
    create.add_argument("--audit", type=Path, required=True)
    create.add_argument("--kind", choices=sorted(ABLATION_FIELDS), required=True)
    create.add_argument("--output", type=Path, required=True)

    validate = subparsers.add_parser("validate", help="revalidate a manifest")
    validate.add_argument("--manifest", type=Path, required=True)
    validate.add_argument("--expected-kind", choices=sorted(ABLATION_FIELDS))
    validate.add_argument("--selected-value-only", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "create":
        create_manifest(args.audit, kind=args.kind, output=args.output)
        print(args.output.expanduser().resolve())
        return
    manifest = validate_manifest(args.manifest, expected_kind=args.expected_kind)
    if args.selected_value_only:
        print(manifest["selected_value"])
    else:
        print(args.manifest.expanduser().resolve())


if __name__ == "__main__":
    main()
