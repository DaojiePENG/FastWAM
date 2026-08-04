#!/usr/bin/env python3
"""Validate self-identifying training contracts as one controlled group."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


_KEY_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
_MODES = frozenset({"action_aggregator", "interleaved", "vision_causal"})
_ALLOWED_DIFFERENCES = frozenset({"mode"})
_REQUIRED_FIELDS = (
    "code_commit",
    "release_checkpoint_sha256",
    "dataset_stats_sha256",
    "training_asset_manifest_sha256",
    "dataset_content_sha256",
    "dataset_file_count",
    "dataset_bytes",
    "text_embedding_cache_sha256",
    "text_embedding_cache_file_count",
    "text_cache_provenance_sha256",
    "text_cache_verification_method",
    "text_cache_verified_file_count",
    "text_encoder_checkpoint_sha256",
    "tokenizer_sha256",
    "vae_checkpoint_sha256",
    "mode",
    "num_processes",
    "batch_size",
    "gradient_accumulation_steps",
    "global_batch",
    "max_steps",
    "learning_rate",
    "lr_scheduler_type",
    "video_lora_multiplier",
    "history_vae_batch_chunk_size",
    "world_model_conditioning",
    "num_video_frames",
    "future_video_condition_noise_probability",
    "future_video_condition_min_u",
    "future_video_condition_max_u",
    "future_video_condition_clean_warmup_steps",
    "future_video_condition_noise_ramp_steps",
    "initial_block_oversample",
    "h0_anchor_mixing",
    "save_every",
    "seed",
    "padding_attention_mask",
    "history_training_mode",
    "history_sampling_mode",
    "history_window_blocks",
    "history_padding",
    "episode_anchor",
    "max_history_blocks",
    "replan_steps",
    "action_horizon",
    "training_exit_depths",
    "mixed_precision",
    "optimizer",
)


@dataclass(frozen=True)
class RunContract:
    path: Path
    expected_mode: str
    run_contract_sha256: str
    fields: tuple[tuple[str, str], ...]

    @property
    def values(self) -> dict[str, str]:
        return dict(self.fields)


def _positive_int(values: dict[str, str], key: str) -> int:
    try:
        value = int(values[key])
    except ValueError as error:
        raise ValueError(f"{key} must be an integer, got {values[key]!r}") from error
    if value <= 0:
        raise ValueError(f"{key} must be positive, got {value}")
    return value


def _positive_float(values: dict[str, str], key: str) -> float:
    try:
        value = float(values[key])
    except ValueError as error:
        raise ValueError(f"{key} must be numeric, got {values[key]!r}") from error
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{key} must be finite and positive, got {value}")
    return value


def _validate_semantics(contract: RunContract) -> None:
    values = contract.values
    missing = [key for key in _REQUIRED_FIELDS if key not in values]
    if missing:
        raise ValueError(f"missing required fields: {', '.join(missing)}")
    if values["mode"] != contract.expected_mode:
        raise ValueError(
            f"mode label mismatch: expected={contract.expected_mode!r} "
            f"actual={values['mode']!r}"
        )
    if values["mode"] not in _MODES:
        raise ValueError(f"unsupported causal mode: {values['mode']!r}")
    if not _COMMIT_RE.fullmatch(values["code_commit"]):
        raise ValueError(f"invalid code_commit: {values['code_commit']!r}")
    for key, value in values.items():
        if key.endswith("_sha256") and not _SHA256_RE.fullmatch(value):
            raise ValueError(f"invalid SHA-256 field {key}: {value!r}")

    num_processes = _positive_int(values, "num_processes")
    batch_size = _positive_int(values, "batch_size")
    grad_accum = _positive_int(values, "gradient_accumulation_steps")
    global_batch = _positive_int(values, "global_batch")
    expected_global_batch = num_processes * batch_size * grad_accum
    if global_batch != expected_global_batch:
        raise ValueError(
            "global_batch is inconsistent with the training topology: "
            f"expected={expected_global_batch} actual={global_batch}"
        )
    for key in (
        "max_steps",
        "dataset_file_count",
        "dataset_bytes",
        "text_embedding_cache_file_count",
        "text_cache_verified_file_count",
        "history_vae_batch_chunk_size",
        "num_video_frames",
        "future_video_condition_clean_warmup_steps",
        "future_video_condition_noise_ramp_steps",
        "initial_block_oversample",
        "save_every",
        "history_window_blocks",
        "max_history_blocks",
        "replan_steps",
        "action_horizon",
    ):
        _positive_int(values, key)
    if int(values["text_cache_verified_file_count"]) != int(
        values["text_embedding_cache_file_count"]
    ):
        raise ValueError(
            "text-cache verified/file counts differ: "
            f"verified={values['text_cache_verified_file_count']} "
            f"files={values['text_embedding_cache_file_count']}"
        )
    if values["text_cache_verification_method"] not in {
        "source_forward_atomic_save_reload_tensor_exact",
        "online_source_forward_cache_tensor_exact",
    }:
        raise ValueError(
            "unsupported text_cache_verification_method: "
            f"{values['text_cache_verification_method']!r}"
        )
    try:
        int(values["seed"])
    except ValueError as error:
        raise ValueError(f"seed must be an integer, got {values['seed']!r}") from error
    _positive_float(values, "learning_rate")
    _positive_float(values, "video_lora_multiplier")
    _positive_float(values, "future_video_condition_noise_probability")
    _positive_float(values, "future_video_condition_min_u")
    _positive_float(values, "future_video_condition_max_u")
    if values["world_model_conditioning"] != "lingbot_teacher_forced_v1":
        raise ValueError(
            "world_model_conditioning must be lingbot_teacher_forced_v1"
        )
    if int(values["num_video_frames"]) != 9:
        raise ValueError("formal LIBERO contract requires num_video_frames=9")
    if not (
        float(values["future_video_condition_noise_probability"]) == 0.5
        and float(values["future_video_condition_min_u"]) == 0.5
        and float(values["future_video_condition_max_u"]) == 1.0
    ):
        raise ValueError(
            "formal future-video condition contract requires prob=0.5 and u=[0.5,1.0]"
        )
    if values["padding_attention_mask"] != "true":
        raise ValueError(
            "padding_attention_mask must be exactly 'true', got "
            f"{values['padding_attention_mask']!r}"
        )
    if values["history_training_mode"] != "strict_replay_window_bptt":
        raise ValueError(
            "history_training_mode must be strict_replay_window_bptt"
        )
    if values["history_sampling_mode"] != "recent_window":
        raise ValueError("history_sampling_mode must be recent_window")
    if int(values["history_window_blocks"]) > int(values["max_history_blocks"]):
        raise ValueError(
            "history_window_blocks cannot exceed max_history_blocks"
        )
    if values["history_padding"] != "left_masked":
        raise ValueError("history_padding must be left_masked")
    if values["episode_anchor"] != "single_real_v0":
        raise ValueError("episode_anchor must be single_real_v0")
    if values["h0_anchor_mixing"] != "per_global_micro_batch":
        raise ValueError(
            "h0_anchor_mixing must be exactly 'per_global_micro_batch', got "
            f"{values['h0_anchor_mixing']!r}"
        )


def load_run_contract(path: Path, *, expected_mode: str) -> RunContract:
    path = path.expanduser().resolve()
    raw = path.read_bytes()
    if not raw:
        raise ValueError("contract is empty")
    if b"\r" in raw:
        raise ValueError("contract must use LF line endings")
    if not raw.endswith(b"\n"):
        raise ValueError("contract must end with exactly one LF")
    try:
        lines = raw[:-1].decode("ascii").split("\n")
    except UnicodeDecodeError as error:
        raise ValueError("contract must be ASCII") from error
    if not lines or any(not line for line in lines):
        raise ValueError("contract contains an empty line")

    parsed: list[tuple[str, str]] = []
    seen: set[str] = set()
    for line_number, line in enumerate(lines, start=1):
        if "=" not in line:
            raise ValueError(f"line {line_number} is not key=value: {line!r}")
        key, value = line.split("=", 1)
        if not _KEY_RE.fullmatch(key):
            raise ValueError(f"line {line_number} has invalid key: {key!r}")
        if not value:
            raise ValueError(f"line {line_number} has an empty value for {key}")
        if key in seen:
            raise ValueError(f"duplicate contract field: {key}")
        seen.add(key)
        parsed.append((key, value))

    if parsed[0][0] != "run_contract_sha256":
        raise ValueError("run_contract_sha256 must be the first field")
    claimed_hash = parsed[0][1]
    if not _SHA256_RE.fullmatch(claimed_hash):
        raise ValueError(f"invalid run_contract_sha256: {claimed_hash!r}")
    payload_lines = lines[1:]
    actual_hash = hashlib.sha256("\n".join(payload_lines).encode("ascii")).hexdigest()
    if actual_hash != claimed_hash:
        raise ValueError(
            "run_contract_sha256 mismatch: "
            f"claimed={claimed_hash} actual={actual_hash}"
        )

    contract = RunContract(
        path=path,
        expected_mode=expected_mode,
        run_contract_sha256=claimed_hash,
        fields=tuple(parsed[1:]),
    )
    try:
        _validate_semantics(contract)
    except ValueError as error:
        raise ValueError(f"{path}: {error}") from error
    return contract


def validate_contract_group(
    contracts: Sequence[RunContract],
    *,
    expected_fields: Iterable[tuple[str, str]] = (),
) -> dict[str, object]:
    if not contracts:
        raise ValueError("at least one run contract is required")
    modes = [contract.expected_mode for contract in contracts]
    if len(set(modes)) != len(modes):
        raise ValueError(f"duplicate causal modes in contract group: {modes}")

    expected_items = tuple(expected_fields)
    expected_keys = [key for key, _ in expected_items]
    if len(set(expected_keys)) != len(expected_keys):
        raise ValueError(f"duplicate expected fields: {expected_keys}")
    for contract in contracts:
        values = contract.values
        for key, expected in expected_items:
            actual = values.get(key)
            if actual != expected:
                raise ValueError(
                    f"{contract.path}: expected {key}={expected!r}, got {actual!r}"
                )

    reference = contracts[0]
    reference_common = tuple(
        item for item in reference.fields if item[0] not in _ALLOWED_DIFFERENCES
    )
    for contract in contracts[1:]:
        actual_common = tuple(
            item for item in contract.fields if item[0] not in _ALLOWED_DIFFERENCES
        )
        if actual_common != reference_common:
            reference_values = dict(reference_common)
            actual_values = dict(actual_common)
            differing_keys = sorted(
                key
                for key in reference_values.keys() | actual_values.keys()
                if reference_values.get(key) != actual_values.get(key)
            )
            if not differing_keys:
                differing_keys = ["<field-order>"]
            details = ", ".join(
                f"{key}: {reference_values.get(key)!r} != {actual_values.get(key)!r}"
                for key in differing_keys
            )
            raise ValueError(
                f"contract group mismatch between {reference.path} and "
                f"{contract.path}; only mode and run_contract_sha256 may differ; "
                f"{details}"
            )

    return {
        "schema_version": 1,
        "allowed_differences": ["mode", "run_contract_sha256"],
        "contracts": [
            {
                "mode": contract.expected_mode,
                "path": str(contract.path),
                "run_contract_sha256": contract.run_contract_sha256,
            }
            for contract in contracts
        ],
        "common_fields": dict(reference_common),
    }


def _parse_assignment(value: str, *, option: str) -> tuple[str, str]:
    if "=" not in value:
        raise ValueError(f"{option} requires KEY=VALUE, got {value!r}")
    key, assigned = value.split("=", 1)
    if not _KEY_RE.fullmatch(key) or not assigned:
        raise ValueError(f"invalid {option} assignment: {value!r}")
    return key, assigned


def _parse_contract_spec(value: str) -> tuple[str, Path]:
    mode, path = _parse_assignment(value, option="--contract")
    if mode not in _MODES:
        raise ValueError(f"invalid contract mode label: {mode!r}")
    return mode, Path(path)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contract",
        action="append",
        required=True,
        metavar="MODE=PATH",
        help="expected causal mode and its run_contract.txt (repeatable)",
    )
    parser.add_argument(
        "--expected-field",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="field that every contract must contain exactly (repeatable)",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    try:
        specs = [_parse_contract_spec(value) for value in args.contract]
        expected_fields = [
            _parse_assignment(value, option="--expected-field")
            for value in args.expected_field
        ]
        contracts = [
            load_run_contract(path, expected_mode=mode) for mode, path in specs
        ]
        result = validate_contract_group(
            contracts,
            expected_fields=expected_fields,
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))

    if args.output is not None:
        _write_json(args.output, result)
        print(args.output)
    else:
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
