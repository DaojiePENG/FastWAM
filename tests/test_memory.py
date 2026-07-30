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


def _state(mode="interleaved", capacity=70):
    return LeapMemoryState(
        LeapMemoryConfig(
            exit_depth=8,
            causal_mode=mode,
            max_history_blocks=capacity,
            action_horizon=32,
            replan_steps=10,
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
