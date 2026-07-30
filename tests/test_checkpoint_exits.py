import copy

import torch
import pytest
from torch import nn

from fastwam.models.wan22.action_dit import ActionDiT
from fastwam.models.wan22.mot import MoT
from fastwam.models.wan22.wan_video_dit import WanVideoDiT
from leapbot_va.models.leapbot import LeapBotVA
from leapbot_va.lora import VideoLoRAConfig
from fastwam.trainer import Wan22Trainer


def _model(*, proprio_dim=None):
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
        proprio_dim=proprio_dim,
        device="cpu",
    )


_ATOMIC_CHECKPOINT_ATTRIBUTES = (
    "causal_mode",
    "training_exit_depths",
    "trained_exit_depths",
    "training_strategy",
    "history_training_mode",
    "training_replan_steps",
    "training_action_horizon",
    "history_vae_batch_chunk_size",
    "video_lora_config",
    "video_lora_merged",
)


def _snapshot_model_state(model):
    tensor_bytes = {
        name: value.detach().cpu().contiguous().view(torch.uint8).clone()
        for name, value in model.state_dict().items()
    }
    return {
        "tensor_bytes": tensor_bytes,
        "attributes": {
            name: copy.deepcopy(getattr(model, name))
            for name in _ATOMIC_CHECKPOINT_ATTRIBUTES
        },
        "module_training": {
            name: module.training for name, module in model.named_modules()
        },
        "requires_grad": {
            name: parameter.requires_grad
            for name, parameter in model.named_parameters()
        },
    }


def _assert_model_state_unchanged(model, snapshot):
    current = model.state_dict()
    assert set(current) == set(snapshot["tensor_bytes"])
    for name, value in current.items():
        actual_bytes = value.detach().cpu().contiguous().view(torch.uint8)
        assert torch.equal(actual_bytes, snapshot["tensor_bytes"][name]), name
    assert {
        name: getattr(model, name) for name in _ATOMIC_CHECKPOINT_ATTRIBUTES
    } == snapshot["attributes"]
    assert {
        name: module.training for name, module in model.named_modules()
    } == snapshot["module_training"]
    assert {
        name: parameter.requires_grad
        for name, parameter in model.named_parameters()
    } == snapshot["requires_grad"]


def _truncate_first_state_tensor(state_dict):
    key = next(
        name
        for name, value in state_dict.items()
        if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] > 1
    )
    state_dict[key] = state_dict[key][0:1].clone()


@pytest.mark.parametrize(
    ("corruption", "expected_error"),
    [
        ("causal_mode", "causal mode mismatch"),
        ("position_scheme", "temporal-position scheme mismatch"),
        ("mot_missing_key", "MoT key mismatch"),
        ("mot_shape", "MoT shape mismatch"),
        ("temporal_shape", "temporal positions shape mismatch"),
        ("action_exit_shape", "action exit heads shape mismatch"),
        ("late_vae_contract", "history VAE batch chunk mismatch"),
        ("trained_exits", "trained exits are incompatible"),
        ("exit_metadata_disagreement", "training/trained exit metadata mismatch"),
    ],
)
def test_rejected_native_checkpoint_is_bitwise_atomic(
    tmp_path, corruption, expected_error
):
    torch.manual_seed(101)
    source = _model(proprio_dim=2)
    source.configure_causal_training(
        causal_mode="action_aggregator",
        training_exit_depths=(30,),
        replan_steps=1,
        action_horizon=2,
    )
    checkpoint = tmp_path / f"atomic_{corruption}.pt"
    source.save_checkpoint(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)

    if corruption == "causal_mode":
        payload["causal_mode"] = "vision_causal"
    elif corruption == "position_scheme":
        payload["temporal_position_scheme"] = "obsolete_position_scheme"
    elif corruption == "mot_missing_key":
        del payload["mot"][next(iter(payload["mot"]))]
    elif corruption == "mot_shape":
        _truncate_first_state_tensor(payload["mot"])
    elif corruption == "temporal_shape":
        _truncate_first_state_tensor(payload["temporal_positions"])
    elif corruption == "action_exit_shape":
        _truncate_first_state_tensor(payload["action_exit_heads"])
    elif corruption == "late_vae_contract":
        # The checkpoint clock is valid and adoptable, but no attribute may be
        # assigned until this later contract check has also succeeded.
        payload["history_vae_batch_chunk_size"] = 2
    elif corruption == "trained_exits":
        payload["trained_exit_depths"] = (7, 30)
    elif corruption == "exit_metadata_disagreement":
        payload["trained_exit_depths"] = (8, 30)
    else:  # pragma: no cover - keeps additions to the table explicit
        raise AssertionError(corruption)
    torch.save(payload, checkpoint)

    torch.manual_seed(202)
    target = _model(proprio_dim=2)
    target.configure_causal_training(
        causal_mode="action_aggregator",
        training_exit_depths=(30,),
    )
    before = _snapshot_model_state(target)
    with pytest.raises(ValueError, match=expected_error):
        target.load_checkpoint(checkpoint)
    _assert_model_state_unchanged(target, before)


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
    with pytest.raises(ValueError, match="do not match runtime BF16 execution"):
        model.configure_causal_training(
            training_exit_depths=(30,),
            history_training_mode="incremental_detached_prefix",
        )


def test_packed_history_training_is_rejected():
    model = _model()
    with pytest.raises(ValueError, match="do not match runtime BF16 execution"):
        model.configure_causal_training(
            training_exit_depths=(30,),
            history_training_mode="packed_full_bptt",
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
        training_strategy="video_lora_action_full",
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
        assert "training strategy mismatch" in str(error)
    else:
        raise AssertionError("LoRA checkpoint loaded without LoRA modules")


def test_native_checkpoint_rejects_strategy_and_complete_lora_contract(tmp_path):
    config = VideoLoRAConfig(
        enabled=True,
        rank=2,
        alpha=2,
        dropout=0.0,
        learning_rate_multiplier=3,
    )
    source = _model()
    source.configure_finetuning(
        training_strategy="video_lora_action_full",
        video_lora_config=config,
    )
    path = tmp_path / "native.pt"
    source.save_checkpoint(path)

    wrong_strategy = _model()
    wrong_strategy.configure_finetuning(
        training_strategy="full_dit",
        video_lora_config=config,
    )
    with pytest.raises(ValueError, match="training strategy mismatch"):
        wrong_strategy.load_checkpoint(path)

    wrong_multiplier = _model()
    wrong_multiplier.configure_finetuning(
        training_strategy="video_lora_action_full",
        video_lora_config=VideoLoRAConfig(
            enabled=True,
            rank=2,
            alpha=2,
            dropout=0.0,
            learning_rate_multiplier=4,
        ),
    )
    with pytest.raises(ValueError, match="LoRA configuration mismatch"):
        wrong_multiplier.load_checkpoint(path)


def test_native_checkpoint_rejects_missing_mot_weight(tmp_path):
    source = _model()
    path = tmp_path / "native.pt"
    source.save_checkpoint(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    removed_key = next(iter(payload["mot"]))
    del payload["mot"][removed_key]
    torch.save(payload, path)

    with pytest.raises(ValueError, match="MoT key mismatch"):
        _model().load_checkpoint(path)


@pytest.mark.parametrize(
    "missing_field",
    [
        "causal_mode",
        "temporal_positions",
        "temporal_position_scheme",
        "proprio_encoder",
    ],
)
def test_native_checkpoint_rejects_missing_learned_auxiliary_state(
    tmp_path, missing_field
):
    source = _model(proprio_dim=2)
    path = tmp_path / "native.pt"
    source.save_checkpoint(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    del payload[missing_field]
    torch.save(payload, path)

    with pytest.raises(ValueError, match=f"missing fields=.*{missing_field}"):
        _model(proprio_dim=2).load_checkpoint(path)


def test_native_checkpoint_rejects_obsolete_temporal_position_semantics(tmp_path):
    source = _model()
    path = tmp_path / "native.pt"
    source.save_checkpoint(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["temporal_position_scheme"] = "absolute_episode_v1"
    torch.save(payload, path)

    with pytest.raises(ValueError, match="temporal-position scheme mismatch"):
        _model().load_checkpoint(path)
