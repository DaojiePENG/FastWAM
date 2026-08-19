import pytest
import torch

from leapbot_va.episode_memory import (
    EpisodeAffineUpdate,
    EpisodeChunkUpdater,
    EpisodeMemoryConfig,
    EpisodeMemoryReader,
    apply_prefix_updates,
    associative_affine_scan,
    build_episode_prefix_states,
)
from leapbot_va.memory import KVSegment, LeapMemoryConfig, LeapMemoryState


def _operator(matrix: torch.Tensor, bias: torch.Tensor) -> EpisodeAffineUpdate:
    return EpisodeAffineUpdate(matrix=matrix, bias=bias)


def test_prediction_correction_expands_to_same_affine_operator():
    torch.manual_seed(3)
    batch, slots, groups, dim = 2, 3, 4, 2
    state = torch.randn(batch, slots, groups, dim)
    diagonal = 0.9 + 0.2 * torch.rand(batch, slots, groups, dim)
    offset = torch.randn_like(state)
    direction = torch.nn.functional.normalize(torch.randn_like(state), dim=-1)
    gain = 0.2 * torch.rand(batch, slots, groups)
    observed = torch.randn(batch, slots, groups)

    predicted = diagonal * state + offset
    evidence_prediction = (direction * predicted).sum(dim=-1)
    corrected = predicted + (
        gain[..., None]
        * direction
        * (observed - evidence_prediction)[..., None]
    )

    eye = torch.eye(dim)
    erase = eye - gain[..., None, None] * (
        direction[..., :, None] * direction[..., None, :]
    )
    matrix = erase @ torch.diag_embed(diagonal)
    bias = (erase @ offset.unsqueeze(-1)).squeeze(-1)
    bias = bias + gain[..., None] * direction * observed[..., None]
    affine = _operator(matrix, bias).apply(state.flatten(-2))
    torch.testing.assert_close(affine, corrected.flatten(-2))


def _sequential_prefix(initial, matrices, biases):
    states = []
    state = initial
    for index in range(matrices.shape[1]):
        state = EpisodeAffineUpdate(
            matrices[:, index], biases[:, index]
        ).apply(state)
        states.append(state)
    return torch.stack(states, dim=1)


def test_associative_scan_matches_sequential_forward_and_gradient_non_power_of_two():
    torch.manual_seed(5)
    batch, chunks, slots, groups, dim = 2, 5, 2, 3, 2
    initial_a = torch.randn(batch, slots, groups * dim, requires_grad=True)
    matrix_a = (
        torch.eye(dim)
        .reshape(1, 1, 1, 1, dim, dim)
        .expand(batch, chunks, slots, groups, dim, dim)
        .clone()
        + 0.02
        * torch.randn(batch, chunks, slots, groups, dim, dim)
    ).requires_grad_()
    bias_a = (
        0.02 * torch.randn(batch, chunks, slots, groups, dim)
    ).requires_grad_()
    sequential = _sequential_prefix(initial_a, matrix_a, bias_a)
    sequential.square().mean().backward()
    sequential_grads = (
        initial_a.grad.clone(),
        matrix_a.grad.clone(),
        bias_a.grad.clone(),
    )

    initial_b = initial_a.detach().clone().requires_grad_()
    matrix_b = matrix_a.detach().clone().requires_grad_()
    bias_b = bias_a.detach().clone().requires_grad_()
    prefixes = associative_affine_scan(_operator(matrix_b, bias_b))
    scanned = apply_prefix_updates(initial_b, prefixes)
    torch.testing.assert_close(scanned, sequential.detach(), rtol=1e-5, atol=1e-6)
    scanned.square().mean().backward()
    torch.testing.assert_close(initial_b.grad, sequential_grads[0], rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(matrix_b.grad, sequential_grads[1], rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(bias_b.grad, sequential_grads[2], rtol=2e-5, atol=2e-6)


def test_chunk_operator_depends_only_on_closed_chunk_not_input_h():
    config = EpisodeMemoryConfig(
        enabled=True,
        window_blocks=4,
        chunk_blocks=2,
        num_slots=3,
        state_dim=8,
        group_dim=2,
        updater_dim=16,
        updater_heads=4,
        reader_rank=4,
    )
    updater = EpisodeChunkUpdater(config, video_dim=6, action_dim=5)
    video = torch.randn(2, 3, 4, 6)
    action = torch.randn(2, 2, 3, 5)
    update, _ = updater(video, action)
    first = torch.randn(2, 3, 8)
    second = torch.randn(2, 3, 8)
    delta = update.apply(first) - update.apply(second)
    linear_delta = EpisodeAffineUpdate(
        update.matrix, torch.zeros_like(update.bias)
    ).apply(first - second)
    torch.testing.assert_close(delta, linear_delta, rtol=1e-5, atol=1e-6)


def test_training_scan_and_online_left_fold_use_identical_chunk_operators():
    torch.manual_seed(11)
    config = EpisodeMemoryConfig(
        enabled=True,
        window_blocks=4,
        chunk_blocks=2,
        num_slots=3,
        state_dim=8,
        group_dim=2,
        updater_dim=16,
        updater_heads=4,
        reader_rank=4,
    )
    updater = EpisodeChunkUpdater(config, video_dim=6, action_dim=5)
    initial = torch.randn(1, 3, 8)
    video = torch.randn(1, 11, 2, 6)
    action = torch.randn(1, 10, 3, 5)
    states, _, _ = build_episode_prefix_states(
        updater, initial, video, action
    )
    online = initial
    for chunk in range(5):
        update, _ = updater(
            video[:, chunk * 2 : chunk * 2 + 3],
            action[:, chunk * 2 : chunk * 2 + 2],
        )
        online = update.apply(online)
    torch.testing.assert_close(online, states[:, -1], rtol=2e-5, atol=2e-6)



def test_reader_zero_gate_preserves_existing_model_behavior():
    config = EpisodeMemoryConfig(
        num_slots=4,
        state_dim=8,
        group_dim=2,
        updater_dim=16,
        updater_heads=4,
        reader_rank=2,
    )
    reader = EpisodeMemoryReader(
        config,
        video_dim=12,
        action_dim=8,
        attention_dim=12,
        num_heads=3,
        num_layers=2,
    )
    query = torch.randn(2, 5, 12)
    state = torch.randn(2, 4, 8)
    output = reader("video", 0, query, state)
    assert torch.count_nonzero(output) == 0
    output.sum().backward()
    assert reader.gates["video"].grad is not None
    assert reader.gates["video"].grad[0].abs() > 0


def _segment(modality: str, block: int, length: int) -> KVSegment:
    positions = torch.arange(length)
    return KVSegment(
        modality=modality,
        block_index=block,
        positions=positions,
        keys=[torch.zeros(1, length, 4) for _ in range(8)],
        values=[torch.zeros(1, length, 4) for _ in range(8)],
    )


def _observation(block: int) -> torch.Tensor:
    return torch.full((1, 2, 1, 2, 2), float(block))


def test_h_q_pch_partition_is_complete_disjoint_and_transactional():
    episode = EpisodeMemoryConfig(enabled=True)
    initial = torch.zeros(1, episode.num_slots, episode.state_dim)
    state = LeapMemoryState(
        LeapMemoryConfig(
            exit_depth=8,
            history_storage_mode="packed_replay",
            history_window_blocks=8,
            replan_steps=2,
            action_horizon=2,
        ),
        episode_memory_config=episode,
        episode_state=initial.clone(),
        initial_episode_state=initial.clone(),
    )
    context = torch.zeros(1, 2, 3)
    mask = torch.ones(1, 2, dtype=torch.bool)
    for block in range(13):
        chunk = state.close_previous_transition(
            next_observation_latents=_observation(block),
            next_context=context,
            next_context_mask=mask,
        )
        if chunk is not None:
            state.commit_handoff(state.episode_state + 1)
        state.append_observation(
            _segment("video", block, 1), context=context, context_mask=mask
        )
        state.stage_replay_observation(
            observation_latents=_observation(block),
            context=context,
            context_mask=mask,
        )
        state.commit_packed_replay_actions(torch.zeros(1, 2, 7))

    assert state.episode_anchor is not None
    assert state.episode_anchor.block_index == 0

    snapshot = state.snapshot()
    chunk = state.close_previous_transition(
        next_observation_latents=_observation(13),
        next_context=context,
        next_context_mask=mask,
    )
    assert chunk is None
    assert state.episode_partition == {
        "H": (0, 4),
        "Q": (4, 5),
        "PCH": (5, 13),
    }
    assert [item.block_index for item in state.handoff_blocks] == [4]
    assert [item.block_index for item in state.replay_blocks] == list(range(5, 13))
    exact = [item.block_index for item in state.replay_prefix()]
    assert exact == list(range(4, 13))
    state.rollback(snapshot)
    assert state.handoff_blocks == []
    assert [item.block_index for item in state.replay_blocks] == list(range(4, 13))


def test_episode_memory_rejects_non_packed_storage():
    episode = EpisodeMemoryConfig(enabled=True)
    initial = torch.zeros(1, episode.num_slots, episode.state_dim)
    with pytest.raises(ValueError, match="packed_replay"):
        LeapMemoryState(
            LeapMemoryConfig(exit_depth=8),
            episode_memory_config=episode,
            episode_state=initial,
            initial_episode_state=initial,
        )
