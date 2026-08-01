from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from leapbot_va.positions import TEMPORAL_POSITION_SCHEME


SCRIPT = Path(__file__).parents[1] / "scripts" / "history_audit_selection.py"
SPEC = importlib.util.spec_from_file_location("history_audit_selection", SCRIPT)
selection = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(selection)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_contract(root: Path, *, field: str, value: str, seed: int = 42) -> Path:
    payload = [
        "code_commit=" + "a" * 40,
        "dataset_stats_sha256=" + "b" * 64,
        "mode=action_aggregator",
        f"seed={seed}",
        "learning_rate=1.0e-4",
        "initial_block_oversample=1",
    ]
    replacement = f"{field}="
    payload = [replacement + value if line.startswith(replacement) else line for line in payload]
    payload_text = "\n".join(payload)
    contract_sha = hashlib.sha256(payload_text.encode()).hexdigest()
    path = root / "run_contract.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"run_contract_sha256={contract_sha}\n{payload_text}\n", encoding="utf-8"
    )
    return path


def _draw_rows() -> list[dict[str, object]]:
    rows = []
    for dataset_index, history in ((10, 0), (18, 8)):
        for u_index, fixed_u in enumerate((0.1, 0.9)):
            for noise_replica in range(2):
                rows.append(
                    {
                        "dataset_index": dataset_index,
                        "history_blocks": history,
                        "u_index": u_index,
                        "fixed_u": fixed_u,
                        "noise_replica": noise_replica,
                    }
                )
    return rows


def _metric(mean: float, low: float | None = None, high: float | None = None):
    return {
        "mean": mean,
        "ci95_low": mean if low is None else low,
        "ci95_high": mean if high is None else high,
    }


def _checkpoint_result(
    path: Path,
    *,
    label: str,
    loss: float,
    loss_ci: tuple[float, float] | None = None,
    native_h0_mse: float = 0.2,
) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((label + "-checkpoint").encode())
    low, high = loss_ci or (loss, loss)
    rows = _draw_rows()
    return {
        "label": label,
        "checkpoint": str(path),
        "checkpoint_sha256": _sha256(path),
        "variant_records": {
            "correct": rows,
            "masked": rows,
            "shuffled": rows,
        },
        "variant_summaries": {
            "correct": {
                "history_distribution_weighted": {
                    "loss_action": _metric(loss, low, high)
                }
            },
            "masked": {},
            "shuffled": {},
        },
        "native_diagnostic_summary": {
            "by_history": {
                "0": {
                    "action_raw_mse_executed10_all7": _metric(native_h0_mse)
                }
            }
        },
    }


def _write_audit(
    tmp_path: Path,
    candidates: list[dict[str, object]],
    *,
    dirty: bool = False,
) -> Path:
    reference = _checkpoint_result(
        tmp_path / "release" / "release.pt",
        label="release",
        loss=0.8,
    )
    payload = {
        "kind": "paired_history_stratified_loss_audit",
        "source_identity": {
            "revision": "c" * 40,
            "dirty": dirty,
            "worktree_sha256": (
                "d" * 64 if dirty else hashlib.sha256(b"").hexdigest()
            ),
        },
        "fixed_u_values": [0.1, 0.9],
        "noise_repeats": 2,
        "history_variants": ["correct", "masked", "shuffled"],
        "checkpoints": [reference, *candidates],
    }
    path = tmp_path / "audit.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _candidate(
    tmp_path: Path,
    name: str,
    *,
    field: str,
    value: str,
    loss: float,
    loss_ci: tuple[float, float] | None = None,
    native_h0_mse: float = 0.2,
    seed: int = 42,
) -> dict[str, object]:
    root = tmp_path / name
    _write_contract(root, field=field, value=value, seed=seed)
    return _checkpoint_result(
        root / "checkpoints" / "weights" / "step_000100.pt",
        label=name,
        loss=loss,
        loss_ci=loss_ci,
        native_h0_mse=native_h0_mse,
    )


def _write_complete_lr_screen_source(
    tmp_path: Path,
    *,
    learning_rate: str = "1.0e-4",
    trainer_step: int = 100,
    checkpoint_contract_sha: str | None = None,
) -> tuple[Path, Path, Path]:
    root = tmp_path / ("lr1p0e-4" if learning_rate == "1.0e-4" else "lr1p0e-5")
    payload_lines = [
        "code_commit=" + "a" * 40,
        "release_checkpoint_sha256=" + "b" * 64,
        "dataset_stats_sha256=" + "c" * 64,
        "mode=action_aggregator",
        "num_processes=4",
        "batch_size=10",
        "gradient_accumulation_steps=2",
        "global_batch=80",
        "max_steps=100",
        f"learning_rate={learning_rate}",
        "lr_scheduler_type=constant",
        "video_lora_multiplier=1.0",
        "history_vae_batch_chunk_size=1",
        "world_model_conditioning=lingbot_teacher_forced_v1",
        "num_video_frames=9",
        "future_video_condition_noise_probability=0.5",
        "future_video_condition_min_u=0.5",
        "future_video_condition_max_u=1.0",
        "initial_block_oversample=1",
        "save_every=100",
        "seed=42",
        "padding_attention_mask=true",
        "history_training_mode=incremental_full_bptt",
        "full_episode_history=true",
        "max_history_blocks=70",
        "replan_steps=10",
        "action_horizon=32",
        "training_exit_depths=30",
    ]
    payload_text = "\n".join(payload_lines)
    contract_sha = hashlib.sha256(payload_text.encode()).hexdigest()
    contract = root / "run_contract.txt"
    contract.parent.mkdir(parents=True, exist_ok=True)
    contract.write_text(
        f"run_contract_sha256={contract_sha}\n{payload_text}\n", encoding="utf-8"
    )

    final_tag = "step_000100"
    checkpoint = root / "checkpoints" / "weights" / f"{final_tag}.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_payload = {
        "step": 100,
        "causal_mode": "action_aggregator",
        "history_training_mode": "incremental_full_bptt",
        "training_strategy": "video_lora_action_full",
        "training_replan_steps": 10,
        "training_action_horizon": 32,
        "training_num_video_frames": 9,
        "future_video_conditioning": "lingbot_teacher_forced_v1",
        "future_video_condition_noise_probability": 0.5,
        "future_video_condition_min_u": 0.5,
        "future_video_condition_max_u": 1.0,
        "temporal_position_scheme": TEMPORAL_POSITION_SCHEME,
        "history_vae_batch_chunk_size": 1,
        "training_exit_depths": (30,),
        "trained_exit_depths": (30,),
        "run_contract_sha256": checkpoint_contract_sha or contract_sha,
        "code_commit": "a" * 40,
        "video_lora_config": {
            "enabled": True,
            "rank": 16,
            "alpha": 16.0,
            "dropout": 0.0,
            "learning_rate_multiplier": 1.0,
        },
        "mot": {"weight": torch.zeros(1)},
        "action_exit_heads": {"weight": torch.zeros(1)},
        "video_exit_heads": {"weight": torch.zeros(1)},
    }
    torch.save(checkpoint_payload, checkpoint)

    state_dir = root / "checkpoints" / "state" / final_tag
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "trainer_state.json").write_text(
        json.dumps(
            {
                "global_step": trainer_step,
                "run_contract_sha256": contract_sha,
                "code_commit": "a" * 40,
            }
        ),
        encoding="utf-8",
    )
    (state_dir / "optimizer_state.bin").write_bytes(b"complete-state-shard")
    return checkpoint, state_dir, contract


def test_learning_rate_manifest_selects_minimum_correct_history_loss(tmp_path):
    low = _candidate(
        tmp_path,
        "low",
        field="learning_rate",
        value="1.0e-5",
        loss=0.3,
    )
    high = _candidate(
        tmp_path,
        "high",
        field="learning_rate",
        value="1.0e-4",
        loss=0.2,
    )
    audit = _write_audit(tmp_path, [high, low])
    output = tmp_path / "selection.json"

    manifest = selection.create_manifest(
        audit, kind="learning_rate", output=output
    )

    assert manifest["selected_value"] == "1.0e-4"
    assert manifest["selected_checkpoint"].endswith("high/checkpoints/weights/step_000100.pt")
    assert [candidate["value"] for candidate in manifest["candidates"]] == [
        "1.0e-5",
        "1.0e-4",
    ]
    assert manifest["audit_sha256"] == _sha256(audit)
    unhashed = dict(manifest)
    assert unhashed.pop("manifest_sha256") == selection.canonical_json_sha256(unhashed)
    assert selection.validate_manifest(output) == manifest
    with pytest.raises(selection.SelectionValidationError, match="kind mismatch"):
        selection.validate_manifest(output, expected_kind="initial_block_oversample")


def test_h0_selection_uses_loss_band_ci_then_native_h0_mse(tmp_path):
    best = _candidate(
        tmp_path,
        "h0x1",
        field="initial_block_oversample",
        value="1",
        loss=1.0,
        loss_ci=(0.90, 1.10),
        native_h0_mse=0.30,
    )
    retained = _candidate(
        tmp_path,
        "h0x4",
        field="initial_block_oversample",
        value="4",
        loss=1.04,
        loss_ci=(1.00, 1.08),
        native_h0_mse=0.10,
    )
    excluded = _candidate(
        tmp_path,
        "h0x8",
        field="initial_block_oversample",
        value="8",
        loss=1.04,
        loss_ci=(1.20, 1.30),
        native_h0_mse=0.01,
    )
    audit = _write_audit(tmp_path, [excluded, retained, best])

    manifest = selection.build_manifest(
        audit, kind="initial_block_oversample"
    )

    assert manifest["selected_value"] == "4"
    by_value = {row["value"]: row for row in manifest["candidates"]}
    assert by_value["1"]["eligible"] is True
    assert by_value["4"]["eligible"] is True
    assert by_value["8"]["eligible"] is False
    assert by_value["8"]["ci95_overlaps_best"] is False


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda payload: payload.update(source_identity={"dirty": True}), "source"),
        (lambda payload: payload.update(fixed_u_values=[]), "fixed_u_values"),
        (lambda payload: payload.update(noise_repeats=0), "noise_repeats"),
        (lambda payload: payload.update(history_variants=["correct"]), "variants"),
    ],
)
def test_create_rejects_incomplete_or_dirty_audit_protocol(
    tmp_path, mutation, match
):
    one = _candidate(
        tmp_path, "one", field="learning_rate", value="1e-5", loss=0.2
    )
    two = _candidate(
        tmp_path, "two", field="learning_rate", value="1e-4", loss=0.1
    )
    audit = _write_audit(tmp_path, [one, two])
    payload = json.loads(audit.read_text())
    mutation(payload)
    audit.write_text(json.dumps(payload))

    with pytest.raises(selection.SelectionValidationError, match=match):
        selection.build_manifest(audit, kind="learning_rate")


def test_create_rejects_checkpoint_sha_and_non_ablation_contract_drift(tmp_path):
    first = _candidate(
        tmp_path, "first", field="learning_rate", value="1e-5", loss=0.2
    )
    second = _candidate(
        tmp_path,
        "second",
        field="learning_rate",
        value="1e-4",
        loss=0.1,
        seed=43,
    )
    audit = _write_audit(tmp_path, [first, second])
    with pytest.raises(selection.SelectionValidationError, match="outside the ablation"):
        selection.build_manifest(audit, kind="learning_rate")

    second_path = Path(second["checkpoint"])
    second_path.write_bytes(b"tampered")
    with pytest.raises(selection.SelectionValidationError, match="checkpoint SHA256"):
        selection.build_manifest(audit, kind="learning_rate")


def test_validate_rejects_manifest_audit_checkpoint_and_contract_tampering(tmp_path):
    first = _candidate(
        tmp_path, "first", field="learning_rate", value="1e-5", loss=0.2
    )
    second = _candidate(
        tmp_path, "second", field="learning_rate", value="1e-4", loss=0.1
    )
    audit = _write_audit(tmp_path, [first, second])
    output = tmp_path / "selection.json"
    selection.create_manifest(audit, kind="learning_rate", output=output)

    manifest = json.loads(output.read_text())
    manifest["selected_value"] = "1e-5"
    output.write_text(json.dumps(manifest))
    with pytest.raises(selection.SelectionValidationError, match="manifest SHA256"):
        selection.validate_manifest(output)

    selection.create_manifest(audit, kind="learning_rate", output=output)
    audit.write_text(audit.read_text() + "\n")
    with pytest.raises(selection.SelectionValidationError, match="audit SHA256"):
        selection.validate_manifest(output)

    audit.write_text(audit.read_text().rstrip() + "\n")
    selection.create_manifest(audit, kind="learning_rate", output=output)
    Path(first["checkpoint"]).write_bytes(b"new checkpoint")
    with pytest.raises(selection.SelectionValidationError, match="checkpoint SHA256"):
        selection.validate_manifest(output)

    Path(first["checkpoint"]).write_bytes(b"first-checkpoint")
    selection.create_manifest(audit, kind="learning_rate", output=output)
    contract = tmp_path / "first" / "run_contract.txt"
    contract.write_text(contract.read_text().replace("seed=42", "seed=43"))
    with pytest.raises(selection.SelectionValidationError, match="payload SHA256"):
        selection.validate_manifest(output)


def test_validate_cli_can_print_only_selected_value(tmp_path):
    first = _candidate(
        tmp_path, "first", field="learning_rate", value="1e-5", loss=0.2
    )
    second = _candidate(
        tmp_path, "second", field="learning_rate", value="1e-4", loss=0.1
    )
    audit = _write_audit(tmp_path, [first, second])
    output = tmp_path / "selection.json"
    selection.create_manifest(audit, kind="learning_rate", output=output)

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "validate",
            "--manifest",
            str(output),
            "--selected-value-only",
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert completed.stdout == "1e-4\n"


def test_user_directed_lr_manifest_is_explicit_non_statistical_and_revalidated(
    tmp_path,
):
    checkpoint, state_dir, contract = _write_complete_lr_screen_source(tmp_path)
    output = tmp_path / "user_directed_selection.json"

    manifest = selection.create_user_directed_learning_rate_manifest(
        selected_value="1.0e-4",
        checkpoint_path=checkpoint,
        state_dir=state_dir,
        contract_path=contract,
        selection_reason="The user explicitly chose the higher screened learning rate.",
        user_selection_note="User stated that 1p0e-4 was clearly better and skipped the LR audit.",
        selected_at_utc="2026-07-31T14:00:00Z",
        output=output,
    )

    assert manifest["selection_basis"] == "user_directed"
    assert manifest["statistical_audit_performed"] is False
    assert manifest["allowed_candidate_values"] == ["1.0e-5", "1.0e-4"]
    assert manifest["selected_value"] == "1.0e-4"
    assert manifest["selected_at_utc"] == "2026-07-31T14:00:00Z"
    assert manifest["source"]["step"] == 100
    assert manifest["selected_checkpoint_sha256"] == _sha256(checkpoint)
    assert manifest["selected_run_contract_file_sha256"] == _sha256(contract)
    assert "audit" not in manifest
    assert (
        manifest["limitations"]["statistical_learning_rate_comparison_claim_supported"]
        is False
    )
    assert selection.validate_manifest(
        output,
        expected_kind="learning_rate",
        allowed_bases={"user_directed"},
    ) == manifest
    with pytest.raises(selection.SelectionValidationError, match="not accepted"):
        selection.validate_manifest(
            output,
            expected_kind="learning_rate",
            allowed_bases={"fixed_noise_audit"},
        )


@pytest.mark.parametrize("learning_rate", ["1e-4", "2.0e-4", "1.0e-3"])
def test_user_directed_lr_rejects_values_outside_exact_screen_candidates(
    tmp_path, learning_rate
):
    checkpoint, state_dir, contract = _write_complete_lr_screen_source(tmp_path)
    with pytest.raises(selection.SelectionValidationError, match="exactly one of"):
        selection.build_user_directed_learning_rate_manifest(
            selected_value=learning_rate,
            checkpoint_path=checkpoint,
            state_dir=state_dir,
            contract_path=contract,
            selection_reason="Explicit user decision.",
            user_selection_note="No fixed-noise LR audit was requested.",
            selected_at_utc="2026-07-31T14:00:00Z",
        )


def test_user_directed_lr_does_not_require_unselected_run_but_rejects_terminated_run(
    tmp_path,
):
    high_checkpoint, high_state, high_contract = _write_complete_lr_screen_source(
        tmp_path, learning_rate="1.0e-4"
    )
    manifest = selection.build_user_directed_learning_rate_manifest(
        selected_value="1.0e-4",
        checkpoint_path=high_checkpoint,
        state_dir=high_state,
        contract_path=high_contract,
        selection_reason="Explicit user decision.",
        user_selection_note="The unselected 1.0e-5 run was stopped.",
        selected_at_utc="2026-07-31T14:00:00Z",
    )
    assert manifest["selected_value"] == "1.0e-4"

    low_checkpoint, low_state, low_contract = _write_complete_lr_screen_source(
        tmp_path / "terminated", learning_rate="1.0e-5"
    )
    low_checkpoint.unlink()
    with pytest.raises(selection.SelectionValidationError, match="complete self-identifying"):
        selection.build_user_directed_learning_rate_manifest(
            selected_value="1.0e-5",
            checkpoint_path=low_checkpoint,
            state_dir=low_state,
            contract_path=low_contract,
            selection_reason="Invalid attempt.",
            user_selection_note="A terminated run cannot be selected.",
            selected_at_utc="2026-07-31T14:00:00Z",
        )


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("state_step", "trainer state/checkpoint step mismatch"),
        ("checkpoint_identity", "run_contract_sha256"),
    ],
)
def test_user_directed_lr_requires_step100_state_and_self_identity(
    tmp_path, mutation, match
):
    checkpoint, state_dir, contract = _write_complete_lr_screen_source(
        tmp_path,
        trainer_step=99 if mutation == "state_step" else 100,
        checkpoint_contract_sha=("f" * 64 if mutation == "checkpoint_identity" else None),
    )
    with pytest.raises(selection.SelectionValidationError, match=match):
        selection.build_user_directed_learning_rate_manifest(
            selected_value="1.0e-4",
            checkpoint_path=checkpoint,
            state_dir=state_dir,
            contract_path=contract,
            selection_reason="Explicit user decision.",
            user_selection_note="The source still requires strict validation.",
            selected_at_utc="2026-07-31T14:00:00Z",
        )


def test_user_directed_manifest_detects_source_tampering(tmp_path):
    checkpoint, state_dir, contract = _write_complete_lr_screen_source(tmp_path)
    output = tmp_path / "selection.json"
    selection.create_user_directed_learning_rate_manifest(
        selected_value="1.0e-4",
        checkpoint_path=checkpoint,
        state_dir=state_dir,
        contract_path=contract,
        selection_reason="Explicit user decision.",
        user_selection_note="No statistical LR comparison is claimed.",
        selected_at_utc="2026-07-31T14:00:00Z",
        output=output,
    )

    trainer_state = state_dir / "trainer_state.json"
    trainer_state.write_text(trainer_state.read_text() + "\n", encoding="utf-8")
    with pytest.raises(selection.SelectionValidationError, match="no longer matches"):
        selection.validate_manifest(output, allowed_bases={"user_directed"})
