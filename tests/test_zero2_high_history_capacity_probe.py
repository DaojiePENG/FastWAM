from __future__ import annotations

import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from torch.utils.data import Dataset

from scripts.probe_zero2_high_history_capacity import (
    FORMAL_OPTIMIZER_UPDATES,
    FORMAL_WORLD_SIZE,
    FixedRealPrefixProbeDataset,
    find_checkpoint_files,
    read_deepspeed_engine_step_state,
    read_formal_optimizer_parameter_groups,
    select_real_high_history_prefixes,
    validate_deepspeed_engine_step_transition,
    validate_capacity_probe_contract,
)


def _cfg(*, batch_size: int = 20, causal_mode: str = "action_aggregator"):
    return OmegaConf.create(
        {
            "batch_size": batch_size,
            "gradient_accumulation_steps": 1,
            "mixed_precision": "bf16",
            "max_steps": 2,
            "lr_scheduler_type": "constant",
            "weight_decay": 1.0e-2,
            "max_grad_norm": 1.0,
            "save_every": 0,
            "eval_every": 0,
            "wandb": {"enabled": False},
            "model": {
                "causal_mode": causal_mode,
                "history_training_mode": "incremental_full_bptt",
                "history_vae_batch_chunk_size": 1,
                "training_strategy": "video_lora_action_full",
                "training_exit_depths": [30],
                "mot_checkpoint_mixed_attn": True,
                "replan_steps": 10,
                "action_horizon": 32,
                "video_lora": {
                    "enabled": True,
                    "rank": 16,
                    "alpha": 16.0,
                    "dropout": 0.0,
                    "learning_rate_multiplier": 1.0,
                },
            },
            "data": {
                "train": {
                    "full_episode_history": True,
                    "min_history_blocks": 0,
                    "max_history_blocks": 70,
                    "replan_steps": 10,
                }
            },
            "capacity_probe": {
                "optimizer_updates": 2,
                "history_min": 41,
                "history_max": 50,
            },
        }
    )


@pytest.mark.parametrize(
    "causal_mode", ["action_aggregator", "interleaved", "vision_causal"]
)
def test_capacity_contract_is_formal_checkpoint_free_zero2(causal_mode):
    contract = validate_capacity_probe_contract(_cfg(causal_mode=causal_mode))

    assert contract == {
        "batch_size_per_rank": 20,
        "world_size": 8,
        "gradient_accumulation_steps": 1,
        "global_batch_size": 160,
        "optimizer_updates": 2,
        "history_min": 41,
        "history_max": 50,
        "mixed_precision": "bf16",
        "zero_stage": 2,
        "causal_mode": causal_mode,
        "history_training_mode": "incremental_full_bptt",
        "history_vae_batch_chunk_size": 1,
        "training_strategy": "video_lora_action_full",
        "video_lora": {
            "rank": 16,
            "alpha": 16.0,
            "dropout": 0.0,
            "learning_rate_multiplier": 1.0,
        },
        "training_exit_depths": [30],
        "optimizer": "adamw_beta0.9_0.95_clip1.0",
        "optimizer_parameter_groups": {
            "action_and_aux": {"weight_decay": 1.0e-2},
            "video_lora": {"weight_decay": 0.0},
        },
        "checkpoints_forbidden": True,
        "wandb_forbidden": True,
    }


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        ("batch_size", 19, "batch_size must be one of"),
        ("gradient_accumulation_steps", 2, "gradient_accumulation_steps=1"),
        ("wandb.enabled", True, "forbids W&B"),
        ("save_every", 2, "forbids checkpoint"),
        ("model.causal_mode", "invalid_mode", "causal_mode must be one of"),
        ("model.history_vae_batch_chunk_size", 2, "chunk1"),
        ("model.training_exit_depths", [8, 16, 24, 30], "D30 exit only"),
        ("data.train.full_episode_history", False, "full_episode_history=true"),
        ("capacity_probe.history_min", 40, "H41-H50"),
    ],
)
def test_capacity_contract_rejects_nonformal_variants(path, value, message):
    cfg = _cfg()
    OmegaConf.update(cfg, path, value)
    with pytest.raises(ValueError, match=message):
        validate_capacity_probe_contract(cfg)


class _HistoryIndexDataset(Dataset):
    replan_steps = 10

    def __init__(self):
        histories = [40, 41, 44, 50, 43, 49, 42, 45]
        self._valid_replan_indices = [1000 + index for index in range(len(histories))]
        self._episode_step = {
            frame: history * self.replan_steps
            for frame, history in zip(self._valid_replan_indices, histories)
        }

    def __len__(self):
        return len(self._valid_replan_indices)

    def __getitem__(self, index):
        return {"source_dataset_index": int(index)}


def test_real_prefix_selection_uses_distinct_largest_genuine_rows():
    dataset = _HistoryIndexDataset()
    selected = select_real_high_history_prefixes(dataset, batch_size=4)

    assert [item.history_blocks for item in selected] == [50, 49, 45, 44]
    assert len({item.dataset_index for item in selected}) == 4
    assert all(item.episode_step == item.history_blocks * 10 for item in selected)
    with pytest.raises(ValueError, match="too few distinct real H41-H50"):
        select_real_high_history_prefixes(dataset, batch_size=8)


def test_probe_dataset_repeats_only_the_fixed_distinct_real_schedule():
    dataset = _HistoryIndexDataset()
    selected = select_real_high_history_prefixes(dataset, batch_size=4)
    probe = FixedRealPrefixProbeDataset(
        dataset,
        selected,
        world_size=FORMAL_WORLD_SIZE,
        optimizer_updates=FORMAL_OPTIMIZER_UPDATES,
    )

    expected_batch = [item.dataset_index for item in selected]
    assert len(probe) == 4 * FORMAL_WORLD_SIZE * FORMAL_OPTIMIZER_UPDATES
    assert [probe[index]["source_dataset_index"] for index in range(4)] == expected_batch
    assert [probe[index]["source_dataset_index"] for index in range(4, 8)] == expected_batch
    assert [
        int(probe[index]["capacity_probe_source_dataset_index"]) for index in range(4)
    ] == expected_batch
    assert [
        int(probe[index]["capacity_probe_expected_history_blocks"])
        for index in range(4)
    ] == [item.history_blocks for item in selected]


def test_deepspeed_engine_step_evidence_requires_one_applied_update():
    before = read_deepspeed_engine_step_state(
        SimpleNamespace(global_steps=7, skipped_steps=2, _step_applied=False)
    )
    after = read_deepspeed_engine_step_state(
        SimpleNamespace(global_steps=8, skipped_steps=2, _step_applied=True)
    )

    evidence = validate_deepspeed_engine_step_transition(
        before, after, optimizer_step_was_skipped=False
    )
    assert evidence["engine_global_steps_delta"] == 1
    assert evidence["engine_skipped_steps_delta"] == 0
    assert evidence["engine_step_applied"] is True
    assert evidence["accelerator_optimizer_step_was_skipped"] is False


@pytest.mark.parametrize(
    ("after", "accelerator_skipped", "message"),
    [
        (
            {"global_steps": 7, "skipped_steps": 2, "_step_applied": True},
            False,
            "did not advance exactly once",
        ),
        (
            {"global_steps": 8, "skipped_steps": 3, "_step_applied": False},
            True,
            "was skipped or not applied",
        ),
        (
            {"global_steps": 8, "skipped_steps": 2, "_step_applied": False},
            False,
            "was skipped or not applied",
        ),
    ],
)
def test_deepspeed_engine_step_evidence_rejects_missing_update(
    after, accelerator_skipped, message
):
    before = {"global_steps": 7, "skipped_steps": 2, "_step_applied": False}
    with pytest.raises(RuntimeError, match=message):
        validate_deepspeed_engine_step_transition(
            before, after, optimizer_step_was_skipped=accelerator_skipped
        )


def test_deepspeed_engine_state_reader_fails_closed_on_missing_api():
    with pytest.raises(RuntimeError, match="integer global_steps"):
        read_deepspeed_engine_step_state(
            SimpleNamespace(skipped_steps=0, _step_applied=False)
        )
    with pytest.raises(RuntimeError, match="boolean _step_applied"):
        read_deepspeed_engine_step_state(
            SimpleNamespace(global_steps=0, skipped_steps=0)
        )


def test_formal_optimizer_groups_preserve_distinct_weight_decay():
    action = torch.nn.Parameter(torch.zeros(3))
    lora = torch.nn.Parameter(torch.zeros(5))
    optimizer = SimpleNamespace(
        param_groups=[
            {
                "group_name": "action_and_aux",
                "weight_decay": 0.01,
                "params": [action],
            },
            {
                "group_name": "video_lora",
                "weight_decay": 0.0,
                "params": [lora],
            },
        ]
    )
    assert read_formal_optimizer_parameter_groups(optimizer) == {
        "action_and_aux": {"weight_decay": 0.01, "parameters": 3},
        "video_lora": {"weight_decay": 0.0, "parameters": 5},
    }


def test_checkpoint_scan_reports_actual_files(tmp_path):
    checkpoint = tmp_path / "checkpoints/state/step_000001/rank0.bin"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"state")
    assert find_checkpoint_files(tmp_path) == [
        "checkpoints/state/step_000001/rank0.bin"
    ]


def test_capacity_launcher_has_valid_bash_syntax():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ["bash", "-n", str(root / "scripts/probe_zero2_high_history_capacity.sh")],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_capacity_launcher_rejects_invalid_mode_before_hardware_access():
    root = Path(__file__).resolve().parents[1]
    environment = dict(os.environ)
    environment["MODE"] = "not_a_causal_mode"
    result = subprocess.run(
        ["bash", str(root / "scripts/probe_zero2_high_history_capacity.sh")],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert result.returncode == 2
    assert "MODE must be action_aggregator, interleaved or vision_causal" in (
        result.stdout + result.stderr
    )


def test_capacity_launcher_rejects_invalid_timeout_before_hardware_access():
    root = Path(__file__).resolve().parents[1]
    environment = dict(os.environ)
    environment["PROBE_TIMEOUT_SECONDS"] = "0"
    result = subprocess.run(
        ["bash", str(root / "scripts/probe_zero2_high_history_capacity.sh")],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert result.returncode == 2
    assert "PROBE_TIMEOUT_SECONDS must be a positive integer" in (
        result.stdout + result.stderr
    )


def test_capacity_launcher_binds_assets_lock_timeout_and_checkpoint_scan():
    root = Path(__file__).resolve().parents[1]
    source = (root / "scripts/probe_zero2_high_history_capacity.sh").read_text()
    assert 'PROBE_TIMEOUT_SECONDS="${PROBE_TIMEOUT_SECONDS:-7200}"' in source
    assert 'flock -s "$TEXT_CACHE_LOCK_FD"' in source
    assert "training_asset_manifest.py" in source
    assert 'training_asset_manifest_sha256=%s' in source
    assert 'timeout --signal=TERM --kill-after=60s "$PROBE_TIMEOUT_SECONDS"' in source
    assert 'checkpoint_files_written": checkpoint_files' in source
    assert 'status = "contract_violation"' in source
