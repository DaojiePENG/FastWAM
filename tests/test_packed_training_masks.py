import torch

from leapbot_va.training import (
    build_packed_history_attention_mask,
    build_query_context_masks,
    current_video_segment_attention_mask,
    history_window_indices,
    validate_packed_history_metadata,
)
from leapbot_va.data import full_episode_sparse_offsets


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
    import pytest

    with pytest.raises(ValueError, match="cross"):
        history_window_indices(
            current_episode_step=10,
            history_blocks=2,
            replan_steps=10,
            current_window_offset=80,
        )


def test_incremental_current_real_frame_cannot_read_future_supervision():
    mask = current_video_segment_attention_mask(
        tokens_per_frame=2,
        num_frames=3,
        device=torch.device("cpu"),
    )
    assert mask[:2, :2].all()
    assert not mask[:2, 2:].any()
    assert mask[2:, :].all()


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
