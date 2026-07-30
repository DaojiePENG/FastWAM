import torch
from torch import nn

from fastwam.models.wan22.action_dit import ActionDiT
from fastwam.models.wan22.mot import MoT
from fastwam.models.wan22.wan_video_dit import WanVideoDiT
from leapbot_va.models.leapbot import LeapBotVA


def _model():
    video = WanVideoDiT(
        hidden_dim=12,
        in_dim=4,
        ffn_dim=24,
        out_dim=4,
        text_dim=6,
        freq_dim=8,
        eps=1e-6,
        patch_size=(1, 1, 1),
        num_heads=2,
        attn_head_dim=6,
        num_layers=30,
        has_image_input=False,
        seperated_timestep=True,
        fuse_vae_embedding_in_latents=True,
    )
    action = ActionDiT(
        hidden_dim=10,
        action_dim=3,
        ffn_dim=20,
        text_dim=6,
        freq_dim=8,
        eps=1e-6,
        num_heads=2,
        attn_head_dim=6,
        num_layers=30,
    )
    mot = MoT({"video": video, "action": action}, mot_checkpoint_mixed_attn=False)
    return LeapBotVA(
        video_expert=video,
        action_expert=action,
        mot=mot,
        vae=nn.Identity(),
        text_dim=6,
        device="cpu",
    )


def test_release_or_full_depth_checkpoint_reinitializes_shallow_heads(tmp_path):
    source = _model()
    nn.init.constant_(source.action_expert.head.weight, 2.0)
    nn.init.constant_(source.action_exit_heads["8"].weight, -3.0)
    path = tmp_path / "full_depth.pt"
    source.save_checkpoint(path)

    loaded = _model()
    loaded.load_checkpoint(path)
    torch.testing.assert_close(
        loaded.action_exit_heads["8"].weight, loaded.action_expert.head.weight
    )
    assert loaded.action_exit_heads["8"].weight[0, 0].item() == 2.0


def test_trained_multi_exit_checkpoint_preserves_exit_heads(tmp_path):
    source = _model()
    source.configure_causal_training(training_exit_depths=(8, 16, 24, 30))
    nn.init.constant_(source.action_exit_heads["8"].weight, 4.0)
    path = tmp_path / "multi_exit.pt"
    source.save_checkpoint(path)

    loaded = _model()
    loaded.load_checkpoint(path)
    assert loaded.action_exit_heads["8"].weight[0, 0].item() == 4.0
