from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path

import torch


_PATH = Path(__file__).resolve().parents[1] / "scripts" / "history_stratified_loss.py"
_SPEC = importlib.util.spec_from_file_location("history_stratified_loss", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
select_history_samples = _MODULE.select_history_samples
summarize_records = _MODULE.summarize_records
summarize_paired_variant_delta = _MODULE.summarize_paired_variant_delta
local_rope_history_sample = _MODULE.local_rope_history_sample


@dataclass
class _FakeDataset:
    _valid_replan_indices: list[int]
    _episode_step: dict[int, int]
    replan_steps: int = 10


def test_local_rope_history_sample_preserves_history_and_resets_only_positions():
    sample = {
        "history_video": torch.randn(1, 3, 2, 4, 4),
        "history_action": torch.randn(1, 2, 10, 7),
        "history_valid_blocks": torch.tensor([[True, True]]),
        "history_block_positions": torch.tensor([[3, 4]]),
        "current_block_position": torch.tensor([5]),
        "episode_step": torch.tensor([50]),
    }

    local = local_rope_history_sample(sample)

    assert local is not sample
    assert local["history_video"] is sample["history_video"]
    assert local["history_action"] is sample["history_action"]
    assert local["history_valid_blocks"] is sample["history_valid_blocks"]
    assert torch.equal(
        local["history_block_positions"], torch.zeros((1, 2), dtype=torch.long)
    )
    assert torch.equal(
        local["current_block_position"], torch.zeros((1,), dtype=torch.long)
    )
    assert torch.equal(local["episode_step"], torch.zeros((1,), dtype=torch.long))
    assert torch.equal(sample["history_block_positions"], torch.tensor([[3, 4]]))


def test_select_history_samples_is_exact_deterministic_and_caps_rare_lengths():
    dataset = _FakeDataset(
        _valid_replan_indices=list(range(10)),
        _episode_step={index: (index % 3) * 10 for index in range(10)},
    )
    first = select_history_samples(
        dataset, history_lengths=[0, 1, 2], samples_per_history=3, seed=42
    )
    second = select_history_samples(
        dataset, history_lengths=[0, 1, 2], samples_per_history=3, seed=42
    )
    assert first == second
    assert [row["history_blocks"] for row in first].count(0) == 3
    assert [row["history_blocks"] for row in first].count(1) == 3
    assert [row["history_blocks"] for row in first].count(2) == 3
    for row in first:
        frame = dataset._valid_replan_indices[row["dataset_index"]]
        assert dataset._episode_step[frame] // dataset.replan_steps == row["history_blocks"]


def test_summarize_records_preserves_exact_history_groups():
    records = [
        {"history_blocks": 0, "loss": 3.0, "loss_video": 2.0, "loss_action": 1.0},
        {"history_blocks": 0, "loss": 5.0, "loss_video": 3.0, "loss_action": 2.0},
        {"history_blocks": 8, "loss": 9.0, "loss_video": 4.0, "loss_action": 5.0},
    ]
    summary = summarize_records(records)
    assert summary["overall"]["loss_action"]["mean"] == 8.0 / 3.0
    assert summary["by_history"]["0"]["loss_action"]["mean"] == 1.5
    assert summary["by_history"]["8"]["loss_action"]["mean"] == 5.0


def test_paired_variant_delta_separates_history_groups():
    reference = [
        {
            "dataset_index": 10,
            "noise_seed": 20,
            "history_blocks": 0,
            "loss": 3.0,
            "loss_video": 2.0,
            "loss_action": 1.0,
        },
        {
            "dataset_index": 11,
            "noise_seed": 21,
            "history_blocks": 8,
            "loss": 7.0,
            "loss_video": 4.0,
            "loss_action": 3.0,
        },
    ]
    candidate = [
        {**reference[0], "loss": 4.5, "loss_video": 2.5, "loss_action": 2.0},
        {**reference[1], "loss": 10.0, "loss_video": 5.0, "loss_action": 5.0},
    ]
    delta = summarize_paired_variant_delta(candidate, reference)
    assert delta["overall"]["loss_action_delta_mean"] == 1.5
    assert delta["by_history"]["0"]["loss_action_delta_mean"] == 1.0
    assert delta["by_history"]["8"]["loss_action_delta_mean"] == 2.0
