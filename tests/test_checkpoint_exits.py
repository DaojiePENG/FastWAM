import torch
import pytest
from torch import nn

from fastwam.models.wan22.action_dit import ActionDiT
from fastwam.models.wan22.mot import MoT
from fastwam.models.wan22.wan_video_dit import WanVideoDiT
from leapbot_va.models.leapbot import LeapBotVA
from leapbot_va.lora import VideoLoRAConfig
from fastwam.trainer import Wan22Trainer


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
    assert loaded.trained_exit_depths == (30,)
    with pytest.raises(ValueError, match="was not trained"):
        loaded.create_memory(exit_depth=8, action_horizon=2, replan_steps=1)


def test_trained_multi_exit_checkpoint_preserves_exit_heads(tmp_path):
    source = _model()
    source.configure_causal_training(training_exit_depths=(8, 16, 24, 30))
    nn.init.constant_(source.action_exit_heads["8"].weight, 4.0)
    path = tmp_path / "multi_exit.pt"
    source.save_checkpoint(path)

    loaded = _model()
    loaded.load_checkpoint(path)
    assert loaded.action_exit_heads["8"].weight[0, 0].item() == 4.0
    assert loaded.trained_exit_depths == (8, 16, 24, 30)
    memory = loaded.create_memory(exit_depth=8, action_horizon=2, replan_steps=1)
    assert memory.config.exit_depth == 8


def test_detached_history_training_is_rejected():
    model = _model()
    with pytest.raises(ValueError, match="drops historical gradients"):
        model.configure_causal_training(
            training_exit_depths=(30,),
            history_training_mode="incremental_detached_prefix",
        )


def test_hybrid_strategy_freezes_video_base_and_fully_trains_action():
    model = _model()
    model.configure_finetuning(
        training_strategy="video_lora_action_full",
        video_lora_config=VideoLoRAConfig(
            enabled=True,
            rank=2,
            alpha=2,
            learning_rate_multiplier=10,
        ),
    )
    Wan22Trainer._apply_dit_only_train_mode(model)

    video_trainable = {
        name for name, parameter in model.video_expert.named_parameters() if parameter.requires_grad
    }
    assert video_trainable
    assert all(name.endswith(("lora_A", "lora_B")) for name in video_trainable)
    assert all(parameter.requires_grad for parameter in model.action_expert.parameters())

    groups = model.optimizer_parameter_groups(learning_rate=1e-5, weight_decay=1e-2)
    assert [group["group_name"] for group in groups] == ["action_and_aux", "video_lora"]
    assert groups[0]["lr"] == 1e-5
    assert groups[1]["lr"] == 1e-4
    assert groups[1]["weight_decay"] == 0


def test_lora_checkpoint_roundtrip_requires_matching_model(tmp_path):
    config = VideoLoRAConfig(enabled=True, rank=2, alpha=2)
    source = _model()
    source.configure_finetuning(
        training_strategy="video_lora_action_full",
        video_lora_config=config,
    )
    nn.init.constant_(source.video_expert.blocks[0].self_attn.q.lora_B, 0.25)
    path = tmp_path / "lora.pt"
    source.save_checkpoint(path)

    loaded = _model()
    loaded.configure_finetuning(
        training_strategy="full_dit",
        video_lora_config=config,
    )
    loaded.load_checkpoint(path)
    torch.testing.assert_close(
        loaded.video_expert.blocks[0].self_attn.q.lora_B,
        source.video_expert.blocks[0].self_attn.q.lora_B,
    )

    incompatible = _model()
    try:
        incompatible.load_checkpoint(path)
    except ValueError as error:
        assert "LoRA mismatch" in str(error)
    else:
        raise AssertionError("LoRA checkpoint loaded without LoRA modules")
