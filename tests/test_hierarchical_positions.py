import pytest
import torch

from leapbot_va.positions import (
    HierarchicalTemporalPositionEmbedding,
    sinusoidal_episode_features,
)


def test_local_rope_coordinates_remain_native_with_batched_option():
    assert HierarchicalTemporalPositionEmbedding.local_video_rope_ids(4).tolist() == [
        0,
        1,
        2,
        3,
    ]
    action_ids = HierarchicalTemporalPositionEmbedding.local_action_rope_ids(
        3,
        batch_size=2,
    )
    assert action_ids.tolist() == [[0, 1, 2], [0, 1, 2]]


def test_zero_initialization_is_exact_identity_for_video_and_action():
    torch.manual_seed(3)
    module = HierarchicalTemporalPositionEmbedding(
        video_dim=8,
        action_dim=6,
        feature_dim=16,
    )
    video = torch.randn(2, 6, 8)
    action = torch.randn(2, 4, 6)

    positioned_video = module.add_video(
        video,
        torch.tensor([[0, 1000], [19, 2_000_000]]),
        tokens_per_frame=3,
    )
    positioned_action = module.add_action(
        action,
        torch.tensor([[0, 1, 2, 3], [500, 501, 502, 503]]),
        torch.tensor([[0, 0, 0, 0], [50, 50, 50, 50]]),
    )

    assert torch.equal(positioned_video, video)
    assert torch.equal(positioned_action, action)
    assert torch.count_nonzero(module.video_projection.weight) == 0
    assert torch.count_nonzero(module.action_block_projection.weight) == 0
    assert torch.count_nonzero(module.action_control_projection.weight) == 0


def test_offsets_follow_token_shape_dtype_device_and_expand_video_frames():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float64
    module = HierarchicalTemporalPositionEmbedding(
        video_dim=5,
        action_dim=7,
        feature_dim=12,
    ).to(device=device)
    video = torch.randn(2, 6, 5, device=device, dtype=dtype)
    action = torch.randn(2, 4, 7, device=device, dtype=dtype)

    with torch.no_grad():
        module.video_projection.weight[:, 0].fill_(0.25)
        module.action_block_projection.weight[:, 0].fill_(0.25)
        module.action_control_projection.weight[:, 1].fill_(0.5)

    video_offsets = module.video_offsets(
        video,
        torch.tensor([[7, 8], [90, 91]]),
        tokens_per_frame=3,
    )
    action_offsets = module.action_offsets(
        action,
        torch.tensor([100, 101, 102, 103]),
        torch.tensor([10, 10, 10, 10]),
    )

    assert video_offsets.shape == video.shape
    assert action_offsets.shape == action.shape
    assert video_offsets.dtype == video.dtype
    assert action_offsets.dtype == action.dtype
    assert video_offsets.device == video.device
    assert action_offsets.device == action.device
    torch.testing.assert_close(video_offsets[:, 0], video_offsets[:, 1])
    torch.testing.assert_close(video_offsets[:, 1], video_offsets[:, 2])
    assert not torch.equal(video_offsets[:, 2], video_offsets[:, 3])


def test_table_free_features_support_large_dynamic_positions():
    ids = torch.tensor([0, 70, 10**6, 10**12], dtype=torch.long)
    features = sinusoidal_episode_features(ids, 32, dtype=torch.float64)

    assert features.shape == (4, 32)
    assert features.dtype == torch.float64
    assert torch.isfinite(features).all()
    assert not torch.equal(features[0], features[-1])


def test_zero_initialized_projections_receive_gradients_and_can_distinguish_ids():
    module = HierarchicalTemporalPositionEmbedding(
        video_dim=3,
        action_dim=2,
        feature_dim=8,
        dtype=torch.float64,
    )
    video = torch.randn(1, 2, 3, dtype=torch.float64, requires_grad=True)
    action = torch.randn(1, 2, 2, dtype=torch.float64, requires_grad=True)
    loss = module.add_video(
        video,
        torch.tensor([0, 1]),
        tokens_per_frame=1,
    ).sum() + module.add_action(
        action,
        torch.tensor([20, 21]),
        torch.tensor([2, 2]),
    ).sum()
    loss.backward()

    assert module.video_projection.weight.grad is not None
    assert module.action_block_projection.weight.grad is not None
    assert module.action_control_projection.weight.grad is not None
    assert torch.count_nonzero(module.video_projection.weight.grad) > 0
    assert torch.count_nonzero(module.action_block_projection.weight.grad) > 0
    assert torch.count_nonzero(module.action_control_projection.weight.grad) > 0
    torch.testing.assert_close(video.grad, torch.ones_like(video))
    torch.testing.assert_close(action.grad, torch.ones_like(action))

    with torch.no_grad():
        module.video_projection.weight[0, 0] = 1.0
    learned_offsets = module.video_offsets(
        torch.zeros_like(video),
        torch.tensor([0, 1]),
        tokens_per_frame=1,
    )
    assert not torch.equal(learned_offsets[:, 0], learned_offsets[:, 1])


def test_trained_episode_offsets_are_exactly_zero_at_first_replan():
    module = HierarchicalTemporalPositionEmbedding(
        video_dim=4,
        action_dim=3,
        feature_dim=8,
        dtype=torch.float64,
    )
    with torch.no_grad():
        module.video_projection.weight.normal_(std=0.2)
        module.action_block_projection.weight.normal_(std=0.2)
        module.action_control_projection.weight.normal_(std=0.2)

    video = torch.randn(1, 2, 4, dtype=torch.float64)
    action = torch.randn(1, 4, 3, dtype=torch.float64)
    local = torch.arange(4)
    torch.testing.assert_close(
        module.add_video(video, torch.tensor([0, 0]), tokens_per_frame=1),
        video,
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        module.add_action(
            action,
            local,
            torch.zeros_like(local),
            local_control_ids=local,
        ),
        action,
        atol=0,
        rtol=0,
    )

    assert not torch.equal(
        module.add_video(video, torch.tensor([1, 1]), tokens_per_frame=1),
        video,
    )
    assert not torch.equal(
        module.add_action(
            action,
            local + 10,
            torch.ones_like(local),
            local_control_ids=local,
        ),
        action,
    )


def test_pre_dit_helpers_copy_state_and_preserve_metadata():
    module = HierarchicalTemporalPositionEmbedding(4, 3, feature_dim=8)
    video_pre = {
        "tokens": torch.randn(1, 4, 4),
        "freqs": object(),
        "meta": {"tokens_per_frame": 2, "grid_size": (2, 1, 2)},
    }
    action_pre = {
        "tokens": torch.randn(1, 3, 3),
        "freqs": object(),
        "meta": {"seq_len": 3},
    }
    block_ids = torch.tensor([11, 12])
    control_ids = torch.tensor([110, 111, 112])
    action_block_ids = torch.tensor([11, 11, 11])

    positioned_video = module.apply_video_pre_dit(video_pre, block_ids)
    positioned_action = module.apply_action_pre_dit(
        action_pre,
        control_ids,
        action_block_ids,
    )

    assert positioned_video is not video_pre
    assert positioned_video["meta"] is not video_pre["meta"]
    assert positioned_video["freqs"] is video_pre["freqs"]
    assert positioned_video["meta"]["absolute_block_ids"] is block_ids
    assert positioned_action["meta"]["absolute_control_ids"] is control_ids
    assert positioned_action["meta"]["absolute_block_ids"] is action_block_ids
    assert "absolute_block_ids" not in video_pre["meta"]
    assert "absolute_control_ids" not in action_pre["meta"]


def test_action_coarse_block_and_fine_control_clocks_are_independent():
    module = HierarchicalTemporalPositionEmbedding(4, 3, feature_dim=8)
    tokens = torch.zeros(1, 3, 3)
    controls = torch.tensor([100, 101, 102])
    one_block = torch.tensor([7, 7, 7])
    with torch.no_grad():
        module.action_block_projection.weight[:, 0].fill_(1.0)

    coarse_only = module.action_offsets(tokens, controls, one_block)
    torch.testing.assert_close(coarse_only[:, :1].expand_as(coarse_only), coarse_only)

    with torch.no_grad():
        module.action_block_projection.weight.zero_()
        module.action_control_projection.weight[:, 0].fill_(1.0)
    fine_only = module.action_offsets(tokens, controls, one_block)
    assert not torch.equal(fine_only[:, 0], fine_only[:, 1])

    with torch.no_grad():
        module.action_control_projection.weight.zero_()
        module.action_block_projection.weight[:, 0].fill_(1.0)
    different_blocks = module.action_offsets(
        tokens,
        torch.tensor([100, 100, 100]),
        torch.tensor([7, 8, 9]),
    )
    assert not torch.equal(different_blocks[:, 0], different_blocks[:, 1])


@pytest.mark.parametrize(
    ("ids", "message"),
    [
        (torch.tensor([-1]), "non-negative"),
        (torch.tensor([1.0]), "integer dtype"),
    ],
)
def test_episode_feature_validation(ids, message):
    with pytest.raises((TypeError, ValueError), match=message):
        sinusoidal_episode_features(ids, 8)
