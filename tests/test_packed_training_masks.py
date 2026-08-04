import inspect

import pytest
import torch

from leapbot_va.training import (
    build_packed_history_attention_mask,
    build_query_context_masks,
    history_window_indices,
    resolve_full_episode_history_batch,
    validate_packed_history_metadata,
)
from leapbot_va.data import (
    LeapRobotVideoDataset,
    _assert_finite_training_tensor,
    full_episode_sparse_offsets,
    oversample_episode_starts,
)


def test_causal_dataset_has_an_explicit_recent_window_sampling_mode():
    signature = inspect.signature(LeapRobotVideoDataset)
    assert signature.parameters["full_episode_history"].default is True
    assert signature.parameters["history_sampling_mode"].default is None
    assert signature.parameters["max_history_blocks"].default == 70


def _indices():
    # H=2, one token per video frame, current has real+future; one action per
    # historical block and two current actions.
    return {
        "v0": 0,
        "v1": 1,
        "vcur": 2,
        "vfuture": 3,
        "a0": 4,
        "a1": 5,
        "acur0": 6,
        "acur1": 7,
    }


def _mask(mode, valid=(True, True)):
    return build_packed_history_attention_mask(
        torch.tensor([valid], dtype=torch.bool),
        video_tokens_per_frame=1,
        current_video_frames=2,
        replan_steps=1,
        action_horizon=2,
        causal_mode=mode,
    )[0]


def test_current_action_never_reads_future_video_supervision():
    i = _indices()
    for mode in ("interleaved", "vision_causal", "action_aggregator"):
        mask = _mask(mode)
        assert mask[i["acur0"], i["vcur"]]
        assert not mask[i["acur0"], i["vfuture"]]
        assert mask[i["acur0"], i["acur1"]]  # current action block is bidirectional


def test_visual_history_rules_match_three_ablation_modes():
    i = _indices()
    assert _mask("interleaved")[i["vcur"], i["a1"]]
    assert not _mask("vision_causal")[i["vcur"], i["a1"]]
    assert _mask("vision_causal")[i["vcur"], i["v1"]]
    assert not _mask("action_aggregator")[i["vcur"], i["v1"]]


def test_padding_cannot_be_used_as_key_by_current_tokens():
    i = _indices()
    mask = _mask("interleaved", valid=(True, False))
    assert not mask[i["acur0"], i["v1"]]
    assert not mask[i["acur0"], i["a1"]]
    assert mask[i["v1"], i["v1"]]  # safe self row avoids all-masked SDPA rows


def test_each_query_sees_language_and_only_its_proprio():
    video, action = build_query_context_masks(
        torch.tensor([[True, True, False]]),
        torch.tensor([[True, False]]),
        video_tokens_per_frame=1,
        current_video_frames=2,
        replan_steps=1,
        action_horizon=2,
    )
    # context columns: 3 language, then block-0, block-1(pad), current.
    assert video[0, 0].tolist() == [True, True, False, True, False, False]
    assert not video[0, 1].any()
    assert video[0, 2].tolist() == [True, True, False, False, False, True]
    assert action[0, -1].tolist() == [True, True, False, False, False, True]


def test_history_window_is_aligned_and_cannot_cross_episode():
    observations, actions, positions = history_window_indices(
        current_episode_step=50,
        history_blocks=3,
        replan_steps=10,
        current_window_offset=80,
    )
    assert observations == [50, 60, 70]
    assert actions == slice(50, 80)
    assert positions == [2, 3, 4]
    with pytest.raises(ValueError, match="cross"):
        history_window_indices(
            current_episode_step=10,
            history_blocks=2,
            replan_steps=10,
            current_window_offset=80,
        )


def test_nonfinite_causal_training_values_fail_before_model_forward():
    with pytest.raises(RuntimeError, match="non-finite.*history_action"):
        _assert_finite_training_tensor(
            torch.tensor([float("nan")]),
            mapped_idx=12,
            episode_step=20,
            field="history_action",
        )
    _assert_finite_training_tensor(
        torch.empty(0),
        mapped_idx=0,
        episode_step=0,
        field="empty_history",
    )


def test_episode_start_oversampling_repeats_only_true_h0_samples():
    indices = [10, 20, 30, 100, 110]
    steps = {10: 0, 20: 10, 30: 20, 100: 0, 110: 10}
    assert oversample_episode_starts(indices, steps, 3) == [
        10,
        20,
        30,
        100,
        110,
        10,
        100,
        10,
        100,
    ]
    with pytest.raises(ValueError, match="positive integer"):
        oversample_episode_starts(indices, steps, 0)


def test_causal_dataset_rejects_nonunit_global_sample_stride(monkeypatch):
    def fail_if_base_constructor_runs(*args, **kwargs):
        raise AssertionError("stride validation must run before loading the dataset")

    monkeypatch.setattr(
        "fastwam.datasets.lerobot.robot_video_dataset.RobotVideoDataset.__init__",
        fail_if_base_constructor_runs,
    )
    common = {
        "dataset_dirs": [],
        "shape_meta": {},
        "replan_steps": 10,
    }
    for stride in (True, 2, 0):
        with pytest.raises(ValueError, match="global_sample_stride=1"):
            LeapRobotVideoDataset(global_sample_stride=stride, **common)


def test_full_episode_metadata_requires_complete_left_aligned_prefix():
    valid = torch.tensor([[True, True, False], [False, False, False]])
    positions = torch.tensor([[0, 1, -1], [-1, -1, -1]])
    counts = validate_packed_history_metadata(
        valid,
        positions,
        torch.tensor([2, 0]),
        torch.tensor([20, 0]),
        replan_steps=10,
        full_episode_history=True,
    )
    assert counts.tolist() == [2, 0]

    bad_gap = valid.clone()
    bad_gap[0] = torch.tensor([True, False, True])
    import pytest

    with pytest.raises(ValueError, match="left-aligned"):
        validate_packed_history_metadata(
            bad_gap,
            positions,
            torch.tensor([2, 0]),
            torch.tensor([20, 0]),
            replan_steps=10,
            full_episode_history=True,
        )

    with pytest.raises(ValueError, match="every preceding"):
        validate_packed_history_metadata(
            valid,
            positions,
            torch.tensor([3, 0]),
            torch.tensor([30, 0]),
            replan_steps=10,
            full_episode_history=True,
        )


def test_packed_batch_rejects_mixed_full_and_short_history_contracts():
    assert resolve_full_episode_history_batch(
        torch.tensor([True, True]),
        batch_size=2,
        device=torch.device("cpu"),
    )
    assert not resolve_full_episode_history_batch(
        torch.tensor([False, False]),
        batch_size=2,
        device=torch.device("cpu"),
    )

    import pytest

    with pytest.raises(ValueError, match="homogeneous"):
        resolve_full_episode_history_batch(
            torch.tensor([True, False]),
            batch_size=2,
            device=torch.device("cpu"),
        )
    with pytest.raises(ValueError, match="one flag per sample"):
        resolve_full_episode_history_batch(
            torch.tensor([True, True, True]),
            batch_size=2,
            device=torch.device("cpu"),
        )

def test_full_episode_offsets_decode_only_replan_observations():
    observations, actions = full_episode_sparse_offsets(
        max_history_blocks=70,
        replan_steps=10,
        current_action_horizon=32,
        current_video_offsets=[0, 4, 8, 12, 16, 20, 24, 28, 32],
    )
    assert len(observations) == 79
    assert observations[:3] == [-700, -690, -680]
    assert observations[69:72] == [-10, 0, 4]
    assert len(actions) == 732
    assert actions[0] == -700
    assert actions[-1] == 31


class _FixedLeRobotSample:
    def __init__(self, sample):
        self.sample = sample

    def __getitem__(self, mapped_idx):
        assert mapped_idx == 17
        return self.sample


def _minimal_causal_dataset(sample):
    dataset = object.__new__(LeapRobotVideoDataset)
    dataset.lerobot_dataset = _FixedLeRobotSample(sample)
    dataset._valid_replan_indices = [17]
    dataset._episode_step = {17: 4}
    dataset.max_history_blocks = 3
    dataset.min_history_blocks = 0
    dataset.replan_steps = 2
    dataset.history_action_steps = 6
    dataset.full_episode_history = True
    dataset.video_sample_indices = [0, 2, 4]
    dataset.num_frames = 5
    dataset.override_instruction = None
    dataset._format_camera_video = lambda full_video, indices: full_video[
        0, indices
    ].permute(1, 0, 2, 3)
    dataset._get_cached_text_context = lambda instruction: (
        torch.zeros(3, 4),
        torch.tensor([True, True, False]),
    )
    return dataset


def _minimal_causal_sample():
    # H=2 at episode step 4. Observation slots 1/2 are the two historical
    # replanning frames and slot 3 is current. Slots 4/5 are future supervision
    # and are deliberately padded to prove that the fail-fast is causal-only.
    return {
        "idx": 17,
        "pixel_values": torch.arange(1 * 6 * 3 * 2 * 2, dtype=torch.float32).reshape(
            1, 6, 3, 2, 2
        ),
        "image_is_pad": torch.tensor([False, False, False, False, True, True]),
        "action": torch.zeros(10, 7),
        # History uses slots 2:6. Current slots 8/9 are allowed future padding.
        "action_is_pad": torch.tensor(
            [False, False, False, False, False, False, False, False, True, True]
        ),
        "proprio": torch.zeros(6, 8),
        "proprio_is_pad": torch.tensor([False, False, False, False, True, True]),
        "instruction": "move the object",
    }


def test_causal_dataset_accepts_real_prefix_and_padded_future_supervision():
    output = _minimal_causal_dataset(_minimal_causal_sample())._get(0)
    assert output["history_valid_blocks"].tolist() == [True, True, False]
    assert output["image_is_pad"].tolist() == [False, True, True]
    assert output["action_is_pad"].tolist() == [False, False, True, True]


def test_causal_dataset_rejects_padding_in_any_real_history_source():
    import pytest

    injected = [
        ("image_is_pad", 1, "history_observation.image_is_pad"),
        ("action_is_pad", 2, "history_action.action_is_pad"),
        ("proprio_is_pad", 2, "history_proprio.proprio_is_pad"),
        ("image_is_pad", 3, "current_observation.image_is_pad"),
        ("proprio_is_pad", 3, "current_proprio.proprio_is_pad"),
    ]
    for mask_name, source_slot, field in injected:
        sample = _minimal_causal_sample()
        sample[mask_name][source_slot] = True
        with pytest.raises(RuntimeError) as error:
            _minimal_causal_dataset(sample)._get(0)
        message = str(error.value)
        assert "mapped_idx=17" in message
        assert "episode_step=4" in message
        assert f"field={field}" in message
        assert f"source_slots=[{source_slot}]" in message


class _MappedLeRobotSamples:
    def __init__(self, samples):
        self.samples = samples

    def __getitem__(self, mapped_idx):
        return self.samples[mapped_idx]


def test_strict_dataset_loads_recent_suffix_and_v0_once_without_frame_repetition():
    current = {
        "idx": 17,
        "pixel_values": torch.arange(
            1 * 6 * 3 * 2 * 2, dtype=torch.float32
        ).reshape(1, 6, 3, 2, 2),
        "image_is_pad": torch.tensor([False, False, False, True, True, True]),
        "action": torch.arange(6 * 3, dtype=torch.float32).reshape(6, 3),
        "action_is_pad": torch.tensor([False, False, False, False, True, True]),
        "proprio": torch.arange(6 * 2, dtype=torch.float32).reshape(6, 2),
        "proprio_is_pad": torch.tensor([False, False, False, True, True, True]),
        "instruction": "move the object",
    }
    anchor = {
        **current,
        "idx": 10,
        "pixel_values": current["pixel_values"] + 1000.0,
        "image_is_pad": torch.tensor([True, False, False, True, True, True]),
        "proprio_is_pad": torch.tensor([True, False, False, True, True, True]),
    }
    dataset = object.__new__(LeapRobotVideoDataset)
    dataset.lerobot_dataset = _MappedLeRobotSamples({17: current, 10: anchor})
    dataset._valid_replan_indices = [17]
    dataset._episode_step = {17: 4}
    dataset._episode_start = {17: 10}
    dataset.max_history_blocks = 70
    dataset.min_history_blocks = 0
    dataset.history_window_blocks = 1
    dataset.history_storage_blocks = 1
    dataset.replan_steps = 2
    dataset.history_action_steps = 2
    dataset.full_episode_history = True
    dataset.use_episode_anchor = True
    dataset.video_sample_indices = [0, 2, 4]
    dataset.num_frames = 5
    dataset.override_instruction = None
    dataset._format_camera_video = lambda full_video, indices: full_video[
        0, indices
    ].permute(1, 0, 2, 3)
    dataset._get_cached_text_context = lambda instruction: (
        torch.zeros(3, 4),
        torch.tensor([True, True, False]),
    )

    output = dataset._get(0)
    assert output["history_valid_blocks"].tolist() == [True]
    assert output["history_block_positions"].tolist() == [1]
    assert output["current_block_position"].item() == 2
    assert not output["full_episode_history"].item()
    assert output["history_window_blocks"].item() == 1
    assert output["episode_anchor_valid"].item()
    torch.testing.assert_close(
        output["episode_anchor_video"],
        dataset._format_camera_video(anchor["pixel_values"], [1]),
    )
    assert not torch.equal(
        output["episode_anchor_video"], output["history_video"]
    )
