from __future__ import annotations

import hashlib
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "validate_run_contract_group.py"
SPEC = importlib.util.spec_from_file_location("run_contract_group_validator", SCRIPT)
validator = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = validator
SPEC.loader.exec_module(validator)


BASE_FIELDS = (
    ("code_commit", "1" * 40),
    ("release_checkpoint_sha256", "2" * 64),
    ("dataset_stats_sha256", "3" * 64),
    ("training_asset_manifest_sha256", "4" * 64),
    ("asset_download_manifest_sha256", "5" * 64),
    ("dataset_content_sha256", "6" * 64),
    ("dataset_file_count", "5152"),
    ("dataset_bytes", "4732040000"),
    ("text_embedding_cache_sha256", "7" * 64),
    ("text_embedding_cache_file_count", "40"),
    ("vae_checkpoint_sha256", "8" * 64),
    ("mode", "action_aggregator"),
    ("num_processes", "8"),
    ("batch_size", "1"),
    ("gradient_accumulation_steps", "16"),
    ("global_batch", "128"),
    ("max_steps", "1115"),
    ("learning_rate", "1.0e-4"),
    ("lr_scheduler_type", "cosine"),
    ("video_lora_multiplier", "1.0"),
    ("history_vae_batch_chunk_size", "1"),
    ("initial_block_oversample", "4"),
    ("save_every", "223"),
    ("seed", "42"),
    ("padding_attention_mask", "true"),
    ("history_training_mode", "incremental_full_bptt"),
    ("full_episode_history", "true"),
    ("max_history_blocks", "70"),
    ("replan_steps", "10"),
    ("action_horizon", "32"),
    ("training_exit_depths", "30"),
    ("mixed_precision", "bf16"),
    ("optimizer", "adamw_beta0.9_0.95_wd0.01_clip1.0"),
)


def _write_contract(
    path: Path,
    mode: str,
    *,
    overrides: dict[str, str] | None = None,
    fields: tuple[tuple[str, str], ...] = BASE_FIELDS,
) -> Path:
    replacements = {"mode": mode, **(overrides or {})}
    payload_fields = tuple((key, replacements.get(key, value)) for key, value in fields)
    payload = "\n".join(f"{key}={value}" for key, value in payload_fields)
    digest = hashlib.sha256(payload.encode("ascii")).hexdigest()
    path.write_text(f"run_contract_sha256={digest}\n{payload}\n", encoding="ascii")
    return path


def _load(path: Path, mode: str):
    return validator.load_run_contract(path, expected_mode=mode)


def test_group_accepts_only_mode_and_derived_hash_differences(tmp_path):
    contracts = [
        _load(_write_contract(tmp_path / f"{mode}.txt", mode), mode)
        for mode in ("action_aggregator", "interleaved", "vision_causal")
    ]
    result = validator.validate_contract_group(
        contracts,
        expected_fields=(("max_steps", "1115"),),
    )
    assert [item["mode"] for item in result["contracts"]] == [
        "action_aggregator",
        "interleaved",
        "vision_causal",
    ]
    assert len({item["run_contract_sha256"] for item in result["contracts"]}) == 3
    assert result["common_fields"]["learning_rate"] == "1.0e-4"
    assert "mode" not in result["common_fields"]


def test_contract_rejects_a_tampered_payload(tmp_path):
    path = _write_contract(tmp_path / "contract.txt", "action_aggregator")
    path.write_text(
        path.read_text().replace("learning_rate=1.0e-4", "learning_rate=9.0e-4"),
        encoding="ascii",
    )
    with pytest.raises(ValueError, match="run_contract_sha256 mismatch"):
        _load(path, "action_aggregator")


def test_contract_rejects_duplicate_fields_even_with_a_matching_hash(tmp_path):
    fields = BASE_FIELDS + (("seed", "42"),)
    path = _write_contract(
        tmp_path / "duplicate.txt",
        "action_aggregator",
        fields=fields,
    )
    with pytest.raises(ValueError, match="duplicate contract field: seed"):
        _load(path, "action_aggregator")


def test_contract_rejects_mode_label_mismatch(tmp_path):
    path = _write_contract(tmp_path / "wrong-mode.txt", "vision_causal")
    with pytest.raises(ValueError, match="mode label mismatch"):
        _load(path, "interleaved")


def test_contract_rejects_internally_inconsistent_topology(tmp_path):
    path = _write_contract(
        tmp_path / "bad-topology.txt",
        "action_aggregator",
        overrides={"num_processes": "4"},
    )
    with pytest.raises(ValueError, match="global_batch is inconsistent"):
        _load(path, "action_aggregator")


def test_contract_requires_padding_attention_mask(tmp_path):
    fields = tuple(item for item in BASE_FIELDS if item[0] != "padding_attention_mask")
    path = _write_contract(
        tmp_path / "missing-padding-mask.txt",
        "action_aggregator",
        fields=fields,
    )
    with pytest.raises(ValueError, match="missing required fields: padding_attention_mask"):
        _load(path, "action_aggregator")


def test_contract_requires_padding_attention_mask_to_be_true(tmp_path):
    path = _write_contract(
        tmp_path / "disabled-padding-mask.txt",
        "action_aggregator",
        overrides={"padding_attention_mask": "false"},
    )
    with pytest.raises(ValueError, match="padding_attention_mask must be exactly 'true'"):
        _load(path, "action_aggregator")


@pytest.mark.parametrize(
    "overrides",
    (
        {"code_commit": "a" * 40},
        {"learning_rate": "5.0e-5"},
        {"seed": "7"},
        {"dataset_stats_sha256": "b" * 64},
        {"num_processes": "4", "global_batch": "64"},
        {"max_steps": "2230"},
    ),
    ids=("commit", "learning-rate", "seed", "stats", "topology", "step"),
)
def test_group_rejects_mixed_controlled_fields(tmp_path, overrides):
    reference = _load(
        _write_contract(tmp_path / "action.txt", "action_aggregator"),
        "action_aggregator",
    )
    different = _load(
        _write_contract(
            tmp_path / "vision.txt",
            "vision_causal",
            overrides=overrides,
        ),
        "vision_causal",
    )
    with pytest.raises(ValueError, match="only mode and run_contract_sha256"):
        validator.validate_contract_group((reference, different))


def test_group_rejects_field_order_drift(tmp_path):
    reordered = BASE_FIELDS[:5] + (BASE_FIELDS[6], BASE_FIELDS[5]) + BASE_FIELDS[7:]
    reference = _load(
        _write_contract(tmp_path / "action.txt", "action_aggregator"),
        "action_aggregator",
    )
    different = _load(
        _write_contract(
            tmp_path / "interleaved.txt",
            "interleaved",
            fields=reordered,
        ),
        "interleaved",
    )
    with pytest.raises(ValueError, match="<field-order>"):
        validator.validate_contract_group((reference, different))


def test_cli_fails_without_writing_output_for_a_mixed_group(tmp_path):
    action = _write_contract(tmp_path / "action.txt", "action_aggregator")
    vision = _write_contract(
        tmp_path / "vision.txt",
        "vision_causal",
        overrides={"seed": "123"},
    )
    output = tmp_path / "validation.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--contract",
            f"action_aggregator={action}",
            "--contract",
            f"vision_causal={vision}",
            "--output",
            str(output),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    assert completed.returncode == 2
    assert "contract group mismatch" in completed.stdout
    assert not output.exists()


@pytest.mark.parametrize(
    "launcher",
    (
        "run_causal_full_bptt_comparison.sh",
        "run_phase1_eval_after_training.sh",
        "run_final_50_trial_comparison.sh",
    ),
)
def test_formal_launchers_use_strict_contract_group_validation(launcher):
    source = (ROOT / "scripts" / launcher).read_text()
    assert "validate_run_contract_group.py" in source
    assert "--contract" in source
    assert "--expected-field" in source
    if "eval_after_training" in launcher or "final_50_trial" in launcher:
        group_validation = source.index("validate_run_contract_group.py")
        checkpoint_validation = source.index("validate_leapbot_checkpoint.py")
        fingerprint_build = source.index("build_mode_fingerprint")
        assert group_validation < checkpoint_validation < fingerprint_build
        assert '--expected-field "release_checkpoint_sha256=' in source
        assert '--expected-field "dataset_stats_sha256=' in source


def test_batched_training_validates_existing_contracts_before_each_mode():
    source = (ROOT / "scripts" / "run_causal_full_bptt_comparison.sh").read_text()
    loop = source.index('for mode in "${MODES[@]}"; do', source.index("CODE_COMMIT="))
    assert source.index("validate_existing_contract_group", source.index("CODE_COMMIT=")) < loop
    loop_body = source[loop:]
    assert loop_body.index("validate_existing_contract_group") < loop_body.index(
        'log "complete controlled full-BPTT mode=$mode"'
    )
