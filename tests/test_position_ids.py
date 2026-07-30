import torch

from fastwam.models.wan22.action_dit import ActionDiT
from fastwam.models.wan22.wan_video_dit import WanVideoDiT


def test_action_position_ids_extend_beyond_original_rope_table():
    model = ActionDiT(
        hidden_dim=12,
        action_dim=3,
        ffn_dim=24,
        text_dim=6,
        freq_dim=8,
        eps=1e-6,
        num_heads=2,
        attn_head_dim=6,
        num_layers=1,
    )
    pre = model.pre_dit(
        action_tokens=torch.randn(1, 2, 3),
        timestep=torch.zeros(1),
        context=torch.randn(1, 3, 6),
        context_mask=torch.ones(1, 3, dtype=torch.bool),
        position_ids=torch.tensor([1200, 1201]),
    )
    assert pre["freqs"].shape[0] == 2
    assert pre["meta"]["position_ids"].tolist() == [1200, 1201]


def test_action_tokenwise_timesteps_support_clean_history_and_noisy_target():
    model = ActionDiT(
        hidden_dim=12,
        action_dim=3,
        ffn_dim=24,
        text_dim=6,
        freq_dim=8,
        eps=1e-6,
        num_heads=2,
        attn_head_dim=6,
        num_layers=1,
    )
    pre = model.pre_dit(
        action_tokens=torch.randn(2, 4, 3),
        timestep=torch.zeros(2),
        context=torch.randn(2, 3, 6),
        token_timesteps=torch.tensor([[0.0, 0.0, 500.0, 500.0]]).expand(2, -1),
        position_ids=torch.tensor([[10, 11, 12, 13], [20, 21, 22, 23]]),
    )
    assert pre["t_mod"].shape == (2, 4, 6, 12)
    assert pre["freqs"].shape[:3] == (2, 4, 1)


def test_video_frame_position_ids_are_explicit_and_dynamic():
    model = WanVideoDiT(
        hidden_dim=12,
        in_dim=4,
        ffn_dim=24,
        freq_dim=8,
        text_dim=6,
        out_dim=4,
        num_heads=2,
        num_layers=1,
        patch_size=(1, 2, 2),
        eps=1e-6,
        has_image_input=False,
        seperated_timestep=True,
        require_vae_embedding=False,
        require_clip_embedding=False,
        fuse_vae_embedding_in_latents=True,
        attn_head_dim=6,
    )
    pre = model.pre_dit(
        x=torch.randn(1, 4, 1, 4, 4),
        timestep=torch.zeros(1),
        context=torch.randn(1, 3, 6),
        context_mask=torch.ones(1, 3, dtype=torch.bool),
        fuse_vae_embedding_in_latents=True,
        frame_position_ids=torch.tensor([1200]),
    )
    assert pre["meta"]["frame_position_ids"].tolist() == [1200]
    assert pre["freqs"].shape[0] == 4


def test_video_history_frames_can_have_clean_token_timesteps():
    model = WanVideoDiT(
        hidden_dim=12,
        in_dim=4,
        ffn_dim=24,
        freq_dim=8,
        text_dim=6,
        out_dim=4,
        num_heads=2,
        num_layers=1,
        patch_size=(1, 2, 2),
        eps=1e-6,
        has_image_input=False,
        seperated_timestep=True,
        require_vae_embedding=False,
        require_clip_embedding=False,
        fuse_vae_embedding_in_latents=True,
        attn_head_dim=6,
    )
    frame_timesteps = torch.tensor([[0.0, 0.0, 750.0]])
    pre = model.pre_dit(
        x=torch.randn(1, 4, 3, 4, 4),
        timestep=torch.tensor([750.0]),
        context=torch.randn(1, 3, 6),
        fuse_vae_embedding_in_latents=True,
        frame_position_ids=torch.tensor([3, 4, 4]),
        frame_timesteps=frame_timesteps,
    )
    assert torch.equal(pre["meta"]["frame_timesteps"], frame_timesteps)
    assert pre["t_mod"].shape == (1, 12, 6, 12)
