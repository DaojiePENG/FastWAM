import importlib.util
from pathlib import Path

import pytest
import torch

from leapbot_va.positions import TEMPORAL_POSITION_SCHEME


_PATH = Path(__file__).parents[1] / "scripts" / "validate_leapbot_checkpoint.py"
_SPEC = importlib.util.spec_from_file_location("checkpoint_validator", _PATH)
validator = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(validator)


def _checkpoint(path: Path, exits=(8, 16, 24, 30)) -> Path:
    payload = {
        "step": 100,
        "causal_mode": "action_aggregator",
        "history_training_mode": "incremental_full_bptt",
        "training_strategy": "video_lora_action_full",
        "training_replan_steps": 10,
        "training_action_horizon": 32,
        "temporal_position_scheme": TEMPORAL_POSITION_SCHEME,
        "training_exit_depths": exits,
        "trained_exit_depths": exits,
        "run_contract_sha256": "a" * 64,
        "code_commit": "b" * 40,
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
    torch.save(payload, path)
    return path


def test_validator_accepts_declared_multi_exit_checkpoint(tmp_path):
    checkpoint = _checkpoint(tmp_path / "multi.pt")
    result = validator.validate_checkpoint(
        checkpoint,
        expected_step=100,
        expected_mode="action_aggregator",
        expected_trained_exit_depths=(8, 16, 24, 30),
        expected_run_contract_sha256="a" * 64,
        expected_code_commit="b" * 40,
    )
    assert result["trained_exit_depths"] == [8, 16, 24, 30]


def test_validator_rejects_using_d30_checkpoint_as_multi_exit(tmp_path):
    checkpoint = _checkpoint(tmp_path / "d30.pt", exits=(30,))
    with pytest.raises(ValueError, match="training_exit_depths"):
        validator.validate_checkpoint(
            checkpoint,
            expected_step=100,
            expected_mode="action_aggregator",
            expected_trained_exit_depths=(8, 16, 24, 30),
        )


def test_validator_rejects_wrong_training_identity(tmp_path):
    checkpoint = _checkpoint(tmp_path / "identity.pt")
    with pytest.raises(ValueError, match="run_contract_sha256"):
        validator.validate_checkpoint(
            checkpoint,
            expected_step=100,
            expected_mode="action_aggregator",
            expected_trained_exit_depths=(8, 16, 24, 30),
            expected_run_contract_sha256="c" * 64,
            expected_code_commit="b" * 40,
        )
