from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
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
build_shuffled_history_donors = _MODULE.build_shuffled_history_donors
shuffled_history_sample = _MODULE.shuffled_history_sample
masked_history_sample = _MODULE.masked_history_sample
fixed_flow_draw = _MODULE.fixed_flow_draw
compute_action_diagnostics = _MODULE.compute_action_diagnostics
episode_cluster_bootstrap_ci = _MODULE.episode_cluster_bootstrap_ci
summarize_diagnostic_records = _MODULE.summarize_diagnostic_records
paired_rows = _MODULE._paired_rows
stable_seed = _MODULE._stable_seed
checkpoint_decompositions = _MODULE._checkpoint_decompositions


@dataclass
class _FakeDataset:
    _valid_replan_indices: list[int]
    _episode_step: dict[int, int]
    replan_steps: int = 10
    _episode_id: dict[int, int] | None = None


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


def test_selection_records_episode_identity_and_population():
    dataset = _FakeDataset(
        _valid_replan_indices=[0, 10, 20, 30, 40, 50],
        _episode_step={0: 0, 10: 10, 20: 20, 30: 0, 40: 10, 50: 20},
        _episode_id={0: 0, 10: 0, 20: 0, 30: 1, 40: 1, 50: 1},
    )
    selected = select_history_samples(
        dataset, history_lengths=[0, 1, 2], samples_per_history=2, seed=7
    )

    assert {row["episode_id"] for row in selected if row["history_blocks"] == 0} == {
        0,
        1,
    }
    assert all(row["history_population_count"] == 2 for row in selected)
    assert all(row["episode_step"] == row["history_blocks"] * 10 for row in selected)


def test_cross_episode_shuffle_is_deterministic_exact_h_and_never_self_maps():
    dataset = _FakeDataset(
        _valid_replan_indices=[0, 10, 100, 110, 200, 210, 300, 310, 320, 330],
        _episode_step={
            0: 0,
            10: 10,
            100: 0,
            110: 10,
            200: 0,
            210: 10,
            300: 0,
            310: 10,
            320: 20,
            330: 30,
        },
        _episode_id={
            0: 0,
            10: 0,
            100: 1,
            110: 1,
            200: 2,
            210: 2,
            300: 3,
            310: 3,
            320: 3,
            330: 3,
        },
    )
    selected = select_history_samples(
        dataset, history_lengths=[0, 1, 3], samples_per_history=3, seed=11
    )
    first = build_shuffled_history_donors(dataset, selected, seed=99)
    second = build_shuffled_history_donors(dataset, list(reversed(selected)), seed=99)

    assert first == second
    h1_donor_indices = []
    for recipient in selected:
        donor = first[recipient["dataset_index"]]
        if recipient["history_blocks"] in {0, 3}:
            assert donor is None
        else:
            assert donor["history_blocks"] == recipient["history_blocks"]
            assert donor["episode_id"] != recipient["episode_id"]
            h1_donor_indices.append(donor["dataset_index"])
    assert len(h1_donor_indices) == len(set(h1_donor_indices))


def _history_sample(fill: float, *, history_blocks: int = 2) -> dict:
    return {
        "video": torch.full((1, 3, 2, 2, 2), fill),
        "action": torch.full((1, 32, 7), fill),
        "proprio": torch.full((1, 32, 8), fill),
        "context": torch.full((1, 4, 5), fill),
        "history_video": torch.full((1, 3, 4, 2, 2), fill),
        "history_action": torch.full((1, 4, 10, 7), fill),
        "history_proprio": torch.full((1, 4, 8), fill),
        "history_valid_blocks": torch.tensor(
            [[index < history_blocks for index in range(4)]]
        ),
        "history_block_positions": torch.tensor([[0, 1, -1, -1]]),
        "current_block_position": torch.tensor([history_blocks]),
        "episode_step": torch.tensor([history_blocks * 10]),
        "full_episode_history": torch.tensor([True]),
    }


def test_shuffled_sample_replaces_only_history_and_mask_preserves_current_time():
    recipient = _history_sample(1.0)
    donor = _history_sample(9.0)
    shuffled = shuffled_history_sample(
        recipient,
        donor,
        history_blocks=2,
        recipient_episode_id=1,
        donor_episode_id=2,
    )

    for key in ("history_video", "history_action", "history_proprio"):
        if key == "history_video":
            assert torch.all(shuffled[key][:, :, :2] == 9.0)
        else:
            assert torch.all(shuffled[key][:, :2] == 9.0)
    for key in (
        "video",
        "action",
        "proprio",
        "context",
        "history_valid_blocks",
        "history_block_positions",
        "current_block_position",
        "episode_step",
    ):
        assert torch.equal(shuffled[key], recipient[key])

    masked = masked_history_sample(recipient)
    assert not masked["history_valid_blocks"].any()
    assert not masked["full_episode_history"].any()
    assert torch.equal(masked["current_block_position"], recipient["current_block_position"])
    assert torch.equal(masked["episode_step"], recipient["episode_step"])


class _ToyScheduler:
    shift = 5.0
    num_train_timesteps = 1000

    def sample_training_t(self, batch_size, device, dtype):
        raise AssertionError("fixed-u mode must bypass the random timestep sampler")

    @staticmethod
    def training_target(sample, noise, timestep):
        del timestep
        return noise - sample


class _ToyActionExpert:
    @staticmethod
    def post_dit(value):
        return value


def _run_toy_draw(video_seed: int, action_seed: int):
    model = SimpleNamespace(
        train_video_scheduler=_ToyScheduler(),
        train_action_scheduler=_ToyScheduler(),
        action_expert=_ToyActionExpert(),
    )
    with fixed_flow_draw(
        model,
        fixed_u=0.375,
        video_noise_seed=video_seed,
        action_noise_seed=action_seed,
    ) as captured:
        video_noise = torch.randn_like(torch.zeros(1, 2))
        model.train_video_scheduler.sample_training_t(1, torch.device("cpu"), torch.float32)
        action_noise = torch.randn_like(torch.zeros(1, 32, 7))
        action_t = model.train_action_scheduler.sample_training_t(
            1, torch.device("cpu"), torch.float32
        )
        target = model.train_action_scheduler.training_target(
            torch.zeros_like(action_noise), action_noise, action_t
        )
        model.action_expert.post_dit(target)
    return video_noise, action_noise, captured


def test_fixed_flow_draw_is_stateless_exact_and_reproducible():
    global_state = torch.random.get_rng_state().clone()
    first = _run_toy_draw(123, 456)
    second = _run_toy_draw(123, 456)

    assert torch.equal(torch.random.get_rng_state(), global_state)
    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])
    assert float(first[2]["timestep_action"].item()) == pytest.approx(750.0)
    assert float(first[2]["timestep_video"].item()) == pytest.approx(750.0)
    assert not torch.equal(first[0].flatten()[:2], first[1].flatten()[:2])


def test_action_split_known_values_padding_dimensions_and_weight():
    target = torch.zeros(1, 32, 7)
    per_step_error = torch.arange(1, 33, dtype=torch.float32).sqrt()
    pred = per_step_error.view(1, 32, 1).expand(-1, -1, 7)
    result = compute_action_diagnostics(
        pred_action=pred,
        target_action=target,
        action_is_pad=torch.zeros(1, 32, dtype=torch.bool),
        scheduler_weight=torch.tensor([1.0]),
        loss_lambda_action=1.0,
        executed_action_steps=10,
        continuous_action_dims=6,
        gripper_action_index=6,
    )
    flat = result["flat"]
    assert flat["action_raw_mse_executed10_all7"] == pytest.approx(5.5)
    assert flat["action_raw_mse_tail22_all7"] == pytest.approx(21.5)
    assert flat["action_raw_mse_full32_all7"] == pytest.approx(16.5)
    assert 32 * flat["action_raw_mse_full32_all7"] == pytest.approx(
        10 * flat["action_raw_mse_executed10_all7"]
        + 22 * flat["action_raw_mse_tail22_all7"]
    )

    dimensional_pred = torch.ones(1, 32, 7)
    dimensional_pred[:, :, 6] = 3.0
    padded = torch.zeros(1, 32, dtype=torch.bool)
    padded[:, 10:] = True
    result = compute_action_diagnostics(
        pred_action=dimensional_pred,
        target_action=target,
        action_is_pad=padded,
        scheduler_weight=torch.tensor([2.0]),
        loss_lambda_action=3.0,
        executed_action_steps=10,
        continuous_action_dims=6,
        gripper_action_index=6,
    )
    flat = result["flat"]
    assert flat["action_raw_mse_executed10_continuous6dof"] == pytest.approx(1.0)
    assert flat["action_raw_mse_executed10_gripper1"] == pytest.approx(9.0)
    assert flat["action_raw_mse_executed10_all7"] == pytest.approx(15.0 / 7.0)
    assert flat["action_weighted_fm_executed10_all7"] == pytest.approx(
        (15.0 / 7.0) * 6.0
    )
    legacy_full = float(
        ((dimensional_pred - target).square().mean() * 2.0 * 3.0).item()
    )
    assert abs(flat["action_weighted_fm_full32_all7"] - legacy_full) <= 1.0e-7
    assert flat["action_raw_mse_tail22_all7"] is None
    assert flat["action_valid_count_tail22_all7"] == 0


def _metric_row(episode, history, value, population=1, u_index=0, noise_replica=0):
    return {
        "dataset_index": episode * 10 + history,
        "episode_id": episode,
        "history_blocks": history,
        "history_population_count": population,
        "noise_seed": 100,
        "u_index": u_index,
        "noise_replica": noise_replica,
        "loss": value,
        "loss_video": value,
        "loss_action": value,
    }


def test_macro_and_history_distribution_weighted_summaries_differ():
    rows = [_metric_row(0, 0, 1.0, population=9), _metric_row(1, 8, 9.0)]
    summary = summarize_diagnostic_records(
        rows, bootstrap_iterations=0, bootstrap_seed=123
    )
    assert summary["sample_weighted"]["loss_action"]["mean"] == pytest.approx(5.0)
    assert summary["macro_h"]["loss_action"]["mean"] == pytest.approx(5.0)
    assert summary["history_distribution_weighted"]["loss_action"][
        "mean"
    ] == pytest.approx(1.8)


def test_episode_cluster_bootstrap_is_deterministic_order_invariant_and_degenerate():
    rows = []
    for episode, value in ((0, 0.0), (1, 10.0)):
        for noise_replica in range(4):
            rows.append(_metric_row(episode, 4, value, noise_replica=noise_replica))
    first = episode_cluster_bootstrap_ci(
        rows,
        metric="loss_action",
        mode="sample_weighted",
        iterations=1000,
        seed=8,
    )
    second = episode_cluster_bootstrap_ci(
        list(reversed(rows)),
        metric="loss_action",
        mode="sample_weighted",
        iterations=1000,
        seed=8,
    )
    assert first == second
    assert first["ci95_low"] == pytest.approx(0.0)
    assert first["ci95_high"] == pytest.approx(10.0)
    assert first["num_clusters"] == 2

    constant = episode_cluster_bootstrap_ci(
        [_metric_row(0, 4, 2.0)],
        metric="loss_action",
        mode="sample_weighted",
        iterations=100,
        seed=8,
    )
    assert constant["mean"] == constant["ci95_low"] == constant["ci95_high"] == 2.0
    assert constant["degenerate"] is True


def test_complete_draw_pairing_is_order_independent_and_fails_on_bad_key_sets():
    reference = [
        _metric_row(0, 0, 1.0, u_index=0),
        _metric_row(0, 0, 2.0, u_index=1),
    ]
    candidate = [
        {**reference[1], "loss_action": 5.0},
        {**reference[0], "loss_action": 3.0},
    ]
    rows = paired_rows(candidate, reference, metric_keys=["loss_action"])
    assert [row["loss_action"] for row in rows] == [2.0, 3.0]

    with pytest.raises(ValueError, match="duplicate paired draw key"):
        paired_rows(reference + [dict(reference[0])], reference, metric_keys=["loss_action"])
    with pytest.raises(ValueError, match="draw keys differ"):
        paired_rows(candidate[:1], reference, metric_keys=["loss_action"])


def test_stable_draw_seed_is_keyed_not_iteration_order():
    keys = [(index, u, noise) for index in (4, 9) for u in range(4) for noise in range(2)]
    forward = {
        key: stable_seed(42, *key, "action")
        for key in keys
    }
    reverse = {
        key: stable_seed(42, *key, "action")
        for key in reversed(keys)
    }
    assert forward == reverse
    assert len(set(forward.values())) == len(keys)


def test_checkpoint_decomposition_separates_prefix_penalty_and_parameter_drift():
    base = _metric_row(0, 4, 0.0, population=3)
    release_native = {**base, "loss": 2.0, "loss_video": 2.0, "loss_action": 2.0}
    release_incremental = {**base, "loss": 3.0, "loss_video": 3.0, "loss_action": 3.0}
    candidate_native = {**base, "loss": 4.0, "loss_video": 4.0, "loss_action": 4.0}
    candidate_incremental = {**base, "loss": 5.0, "loss_video": 5.0, "loss_action": 5.0}
    decomposition = checkpoint_decompositions(
        [
            {
                "label": "release",
                "records": [release_incremental],
                "native_records": [release_native],
            },
            {
                "label": "candidate",
                "records": [candidate_incremental],
                "native_records": [candidate_native],
            },
        ],
        bootstrap_iterations=0,
        bootstrap_seed=9,
    )[1]

    def action_mean(section):
        return section["summary"]["sample_weighted"]["loss_action"]["mean"]

    assert action_mean(
        decomposition["incremental_minus_candidate_native"]
    ) == pytest.approx(1.0)
    assert action_mean(
        decomposition["candidate_native_minus_release_native"]
    ) == pytest.approx(2.0)
    assert action_mean(
        decomposition["incremental_minus_release_native"]
    ) == pytest.approx(3.0)
