import torch

from fastwam.models.wan22.action_dit import ActionDiT
from fastwam.models.wan22.mot import MoT
from fastwam.models.wan22.wan_video_dit import WanVideoDiT, precompute_freqs_cis


def _experts(layers=3):
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
        num_layers=layers,
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
        num_layers=layers,
    )
    return video, action


def test_incremental_kv_matches_one_shot_causal_forward():
    torch.manual_seed(7)
    video, action = _experts()
    mot = MoT({"video": video, "action": action}, mot_checkpoint_mixed_attn=False).eval()
    video_tokens = torch.randn(1, 2, 12)
    action_tokens = torch.randn(1, 3, 10)
    video_freqs = precompute_freqs_cis(6, end=2).view(2, 1, -1)
    action_freqs = precompute_freqs_cis(6, end=3).view(3, 1, -1)
    video_t = torch.zeros(1, 6, 12)
    action_t = torch.zeros(1, 6, 10)

    full_mask = torch.zeros(5, 5, dtype=torch.bool)
    full_mask[:2, :2] = True
    full_mask[2:, :] = True
    one_shot = mot(
        embeds_all={"video": video_tokens, "action": action_tokens},
        attention_mask=full_mask,
        freqs_all={"video": video_freqs, "action": action_freqs},
        context_all={"video": None, "action": None},
        t_mod_all={"video": video_t, "action": action_t},
    )["action"]

    _, video_kv = mot.prefill_expert_segment(
        expert_name="video",
        tokens=video_tokens,
        freqs=video_freqs,
        t_mod=video_t,
        context_payload=None,
    )
    incremental = mot.forward_action_with_history(
        action_tokens=action_tokens,
        action_freqs=action_freqs,
        action_t_mod=action_t,
        action_context_payload=None,
        history_kv=video_kv,
    )
    torch.testing.assert_close(incremental, one_shot, atol=1e-5, rtol=1e-5)
    assert len(video_kv) == 3
    assert all(layer["k"].shape == (1, 2, 12) for layer in video_kv)


def test_one_forward_returns_all_requested_shared_depth_exits():
    torch.manual_seed(11)
    video, action = _experts(layers=4)
    mot = MoT({"video": video, "action": action}, mot_checkpoint_mixed_attn=False).eval()
    outputs = mot(
        embeds_all={"video": torch.randn(1, 1, 12), "action": torch.randn(1, 1, 10)},
        attention_mask=torch.ones(2, 2, dtype=torch.bool),
        freqs_all={
            "video": precompute_freqs_cis(6, end=1).view(1, 1, -1),
            "action": precompute_freqs_cis(6, end=1).view(1, 1, -1),
        },
        context_all={"video": None, "action": None},
        t_mod_all={"video": torch.zeros(1, 6, 12), "action": torch.zeros(1, 6, 10)},
        exit_depths=(1, 2, 4),
    )
    assert tuple(outputs) == (1, 2, 4)
    assert outputs[2]["video"].shape == (1, 1, 12)
    assert outputs[2]["action"].shape == (1, 1, 10)
