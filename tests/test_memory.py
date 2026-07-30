import pytest
import torch

from leapbot_va.memory import (
    KVSegment,
    LeapMemoryConfig,
    LeapMemoryState,
    MemoryPhase,
)


def _segment(modality: str, block: int, positions, depth: int = 8) -> KVSegment:
    positions = torch.as_tensor(positions, dtype=torch.long)
    seq = positions.numel()
    keys = [torch.full((1, seq, 4), float(layer)) for layer in range(depth)]
    values = [torch.full((1, seq, 4), float(layer + 10)) for layer in range(depth)]
    return KVSegment(modality, block, positions, keys, values)


def _state(mode="interleaved", capacity=70, retained=None):
    return LeapMemoryState(
        LeapMemoryConfig(
            exit_depth=8,
            causal_mode=mode,
            max_history_blocks=capacity,
            retained_history_blocks=retained,
            action_horizon=32,
            replan_steps=10,
        )
    )


def _commit_full_block(state: LeapMemoryState, block: int) -> None:
    context = torch.zeros(1, 2, 3)
    mask = torch.ones(1, 2, dtype=torch.bool)
    state.append_observation(
        _segment("video", block, [block]),
        context=context,
        context_mask=mask,
    )
    action_start = block * state.config.replan_steps
    state.append_actions(
        _segment(
            "action",
            block,
            range(action_start, action_start + state.config.replan_steps),
        )
    )


def test_state_machine_commits_only_executed_actions():
    state = _state()
    context = torch.zeros(1, 2, 3)
    mask = torch.ones(1, 2, dtype=torch.bool)
    state.append_observation(
        _segment("video", 0, [0, 0, 0]), context=context, context_mask=mask
    )
    assert state.phase is MemoryPhase.EXPECT_ACTION_COMMIT
    assert state.completed_blocks == 0

    # The model may predict 32 actions, but only these ten executed positions
    # are eligible for the persistent cache.
    state.append_actions(_segment("action", 0, range(10)))
    assert state.phase is MemoryPhase.EXPECT_OBSERVATION
    assert state.completed_blocks == 1
    assert state.next_action_position == 10
    assert state.token_counts == {"video": 3, "action": 10}


def test_next_observation_is_rejected_until_action_commit():
    state = _state()
    context = torch.zeros(1, 2, 3)
    mask = torch.ones(1, 2, dtype=torch.bool)
    state.append_observation(
        _segment("video", 0, [0]), context=context, context_mask=mask
    )
    with pytest.raises(RuntimeError, match="executed actions"):
        state.begin_observation()


def test_partial_action_commit_requires_episode_reset_before_next_observation():
    state = _state()
    context = torch.zeros(1, 2, 3)
    mask = torch.ones(1, 2, dtype=torch.bool)
    state.append_observation(
        _segment("video", 0, [0]), context=context, context_mask=mask
    )
    state.append_actions(_segment("action", 0, range(4)))
    assert state.next_action_position == 4
    with pytest.raises(RuntimeError, match="partial action commit is terminal"):
        state.begin_observation()
    state.reset()
    assert state.begin_observation() == 0


@pytest.mark.parametrize(
    ("mode", "expected_modalities"),
    [
        ("interleaved", ["video", "action"]),
        ("vision_causal", ["video"]),
        ("action_aggregator", []),
    ],
)
def test_video_history_selection(mode, expected_modalities):
    state = _state(mode)
    state.segments = [
        _segment("video", 0, [0]),
        _segment("action", 0, [0]),
    ]
    assert [s.modality for s in state.selected_segments_for_video()] == expected_modalities
    assert [s.modality for s in state.selected_segments_for_action()] == ["video", "action"]


def test_capacity_is_measured_in_completed_replan_blocks():
    state = _state(capacity=1)
    context = torch.zeros(1, 2, 3)
    mask = torch.ones(1, 2, dtype=torch.bool)
    state.append_observation(
        _segment("video", 0, [0]), context=context, context_mask=mask
    )
    state.append_actions(_segment("action", 0, range(10)))
    with pytest.raises(RuntimeError, match="capacity exceeded"):
        state.begin_observation()


def test_capacity_70_allows_700_actions_and_rejects_71st_observation():
    state = _state(capacity=70)
    context = torch.zeros(1, 2, 3)
    mask = torch.ones(1, 2, dtype=torch.bool)

    for block in range(70):
        assert state.begin_observation() == block
        state.append_observation(
            _segment("video", block, [block]),
            context=context,
            context_mask=mask,
        )
        action_start = block * state.config.replan_steps
        state.append_actions(
            _segment(
                "action",
                block,
                range(action_start, action_start + state.config.replan_steps),
            )
        )

    assert state.completed_blocks == 70
    assert state.next_action_position == 700
    assert state.token_counts == {"video": 70, "action": 700}
    with pytest.raises(RuntimeError, match="capacity exceeded"):
        state.begin_observation()


@pytest.mark.parametrize("retained", [-1, 71, 2.5, True])
def test_retention_window_must_fit_inside_episode_capacity(retained):
    with pytest.raises(ValueError, match="retained_history_blocks"):
        _state(capacity=70, retained=retained)


def test_retention_zero_evicts_only_after_full_action_commit():
    state = _state(retained=0)
    context = torch.randn(1, 2, 3)
    mask = torch.ones(1, 2, dtype=torch.bool)

    state.append_observation(
        _segment("video", 0, [0]), context=context, context_mask=mask
    )
    # The current real observation must remain available to action prediction
    # and action-KV prefill until the complete executed block is committed.
    assert [(s.modality, s.block_index) for s in state.segments] == [("video", 0)]
    assert state.token_counts == {"video": 1, "action": 0}
    state.append_actions(_segment("action", 0, range(10)))

    assert state.segments == []
    assert state.token_counts == {"video": 0, "action": 0}
    assert state.cache_nbytes == 0
    assert state.completed_blocks == 1
    assert state.next_action_position == 10

    # Absolute episode clocks continue even though no KV was retained.
    assert state.begin_observation() == 1
    state.append_observation(
        _segment("video", 1, [1]), context=context, context_mask=mask
    )
    assert state.segments[0].block_index == 1


def test_retention_two_keeps_complete_pairs_and_absolute_clocks():
    state = _state(retained=2)
    for block in range(4):
        _commit_full_block(state, block)

    assert [(s.modality, s.block_index) for s in state.segments] == [
        ("video", 2),
        ("action", 2),
        ("video", 3),
        ("action", 3),
    ]
    assert state.token_counts == {"video": 2, "action": 20}
    assert state.cache_nbytes == sum(segment.nbytes for segment in state.segments)
    assert state.completed_blocks == 4
    assert state.retained_completed_blocks == 2
    assert state.next_action_position == 40
    assert state.begin_observation() == 4


def test_none_retention_keeps_the_full_episode():
    state = _state(retained=None)
    for block in range(4):
        _commit_full_block(state, block)

    assert [segment.block_index for segment in state.segments] == [
        0,
        0,
        1,
        1,
        2,
        2,
        3,
        3,
    ]
    assert state.token_counts == {"video": 4, "action": 40}
    assert state.retained_completed_blocks == 4


@pytest.mark.parametrize(
    ("mode", "expected_video"),
    [
        (
            "interleaved",
            [("video", 1), ("action", 1), ("video", 2), ("action", 2)],
        ),
        ("vision_causal", [("video", 1), ("video", 2)]),
        ("action_aggregator", []),
    ],
)
def test_causal_mode_selection_operates_on_retained_window(mode, expected_video):
    state = _state(mode=mode, retained=2)
    for block in range(3):
        _commit_full_block(state, block)

    selected_video = [
        (segment.modality, segment.block_index)
        for segment in state.selected_segments_for_video()
    ]
    selected_action = [
        (segment.modality, segment.block_index)
        for segment in state.selected_segments_for_action()
    ]
    assert selected_video == expected_video
    assert selected_action == [
        ("video", 1),
        ("action", 1),
        ("video", 2),
        ("action", 2),
    ]


def test_retention_does_not_bypass_70_block_episode_capacity():
    state = _state(capacity=70, retained=0)
    for block in range(70):
        _commit_full_block(state, block)

    assert state.segments == []
    assert state.completed_blocks == 70
    assert state.next_action_position == 700
    with pytest.raises(RuntimeError, match="70/70 blocks"):
        state.begin_observation()


def test_rollback_restores_front_segments_evicted_by_completed_commit():
    state = _state(retained=2)
    _commit_full_block(state, 0)
    _commit_full_block(state, 1)
    context = torch.randn(1, 2, 3)
    mask = torch.ones(1, 2, dtype=torch.bool)
    state.append_observation(
        _segment("video", 2, [2]), context=context, context_mask=mask
    )
    snapshot = state.snapshot()
    before = [(segment.modality, segment.block_index) for segment in state.segments]

    state.append_actions(_segment("action", 2, range(20, 30)))
    assert state.segments[0].block_index == 1
    state.rollback(snapshot)

    restored = [
        (segment.modality, segment.block_index) for segment in state.segments
    ]
    assert restored == before
    assert state.phase is MemoryPhase.EXPECT_ACTION_COMMIT
    assert state.completed_blocks == 2
    assert state.next_action_position == 20
    assert torch.equal(state.pending_context, context)


def test_partial_commit_is_not_evicted_even_with_zero_retention():
    state = _state(retained=0)
    context = torch.zeros(1, 2, 3)
    mask = torch.ones(1, 2, dtype=torch.bool)
    state.append_observation(
        _segment("video", 0, [0]), context=context, context_mask=mask
    )
    state.append_actions(_segment("action", 0, range(4)))

    assert [(s.modality, s.block_index) for s in state.segments] == [
        ("video", 0),
        ("action", 0),
    ]
    assert state.token_counts == {"video": 1, "action": 4}
    with pytest.raises(RuntimeError, match="partial action commit is terminal"):
        state.begin_observation()
    state.reset()
    assert state.segments == []
    assert state.completed_blocks == 0
    assert state.next_action_position == 0
    assert state.config.retained_history_blocks == 0


def test_snapshot_rollback_restores_pending_context():
    state = _state()
    context = torch.randn(1, 2, 3)
    mask = torch.ones(1, 2, dtype=torch.bool)
    state.append_observation(
        _segment("video", 0, [0]), context=context, context_mask=mask
    )
    snapshot = state.snapshot()
    state.pending_context = None
    state.rollback(snapshot)
    assert state.phase is MemoryPhase.EXPECT_ACTION_COMMIT
    assert torch.equal(state.pending_context, context)


def test_materialized_cache_preserves_chronological_order():
    state = _state()
    first = _segment("video", 0, [0, 0])
    second = _segment("action", 0, [0])
    state.segments = [first, second]
    cache = state.materialize(state.segments)
    assert cache is not None
    assert cache[0]["k"].shape == (1, 3, 4)
    assert torch.equal(cache[0]["k"][:, :2], first.keys[0])
    assert torch.equal(cache[0]["k"][:, 2:], second.keys[0])


def test_reset_releases_episode_state():
    state = _state()
    state.segments = [_segment("video", 0, [0])]
    state.completed_blocks = 1
    state.next_action_position = 10
    state.prompt_fingerprint = "abc"
    state.reset()
    assert state.segments == []
    assert state.completed_blocks == 0
    assert state.next_action_position == 0
    assert state.prompt_fingerprint is None
    assert state.phase is MemoryPhase.EXPECT_OBSERVATION
