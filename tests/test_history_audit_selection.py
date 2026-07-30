from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


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
