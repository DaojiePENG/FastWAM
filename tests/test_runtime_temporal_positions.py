import types

import pytest
import torch
from torch import nn

from fastwam.models.wan22.action_dit import ActionDiT
from fastwam.models.wan22.mot import MoT
from fastwam.models.wan22.wan_video_dit import WanVideoDiT
from leapbot_va.lora import VideoLoRAConfig
from leapbot_va.memory import LeapMemoryConfig, LeapMemoryState
from leapbot_va.models.leapbot import LeapBotVA


def _model(*, layers: int = 2) -> LeapBotVA:
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
    mot = MoT({"video": video, "action": action}, mot_checkpoint_mixed_attn=False)
    return LeapBotVA(
        video_expert=video,
        action_expert=action,
        mot=mot,
        vae=nn.Identity(),
        text_dim=6,
        device="cpu",
        exit_depths=(layers,),
    )


def test_temporal_positions_checkpoint_roundtrip_and_legacy_reset(tmp_path):
    source = _model()
    with torch.no_grad():
        source.temporal_positions.video_projection.weight.fill_(0.125)
        source.temporal_positions.action_block_projection.weight.fill_(-0.25)
        source.temporal_positions.action_control_projection.weight.fill_(0.375)
    checkpoint = tmp_path / "temporal.pt"
    source.save_checkpoint(checkpoint)

    loaded = _model()
    loaded.load_checkpoint(checkpoint)
    torch.testing.assert_close(
        loaded.temporal_positions.video_projection.weight,
        source.temporal_positions.video_projection.weight,
    )
    torch.testing.assert_close(
        loaded.temporal_positions.action_block_projection.weight,
        source.temporal_positions.action_block_projection.weight,
    )
    torch.testing.assert_close(
        loaded.temporal_positions.action_control_projection.weight,
        source.temporal_positions.action_control_projection.weight,
    )

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload.pop("temporal_positions")
    legacy_checkpoint = tmp_path / "legacy.pt"
    torch.save(payload, legacy_checkpoint)
    loaded.load_checkpoint(legacy_checkpoint)
    assert torch.count_nonzero(loaded.temporal_positions.video_projection.weight) == 0
    assert (
        torch.count_nonzero(
            loaded.temporal_positions.action_block_projection.weight
        )
        == 0
    )
    assert (
        torch.count_nonzero(
            loaded.temporal_positions.action_control_projection.weight
        )
        == 0
    )


@pytest.mark.parametrize("training_strategy", ["full_dit", "video_lora_action_full"])
def test_temporal_positions_train_for_full_and_hybrid_strategies(training_strategy):
    model = _model()
    if training_strategy == "video_lora_action_full":
        model.configure_finetuning(
            training_strategy=training_strategy,
            video_lora_config=VideoLoRAConfig(enabled=True, rank=2, alpha=2),
        )
    model.requires_grad_(False)
    model.configure_trainable_parameters()

    temporal_parameters = list(model.temporal_positions.parameters())
    assert temporal_parameters
    assert all(parameter.requires_grad for parameter in temporal_parameters)
    trainable_ids = {
        id(parameter)
        for group in model.optimizer_parameter_groups(
            learning_rate=1e-5,
            weight_decay=1e-2,
        )
        for parameter in group["params"]
    }
    assert all(id(parameter) in trainable_ids for parameter in temporal_parameters)


def test_memory_causal_mode_must_match_model_configuration():
    model = _model(layers=30)
    model.configure_causal_training(
        causal_mode="action_aggregator",
        training_exit_depths=(30,),
    )
    memory = model.create_memory(action_horizon=2, replan_steps=1)
    assert memory.config.causal_mode == "action_aggregator"

    with pytest.raises(ValueError, match="memory/model causal mode mismatch"):
        model.create_memory(
            causal_mode="interleaved",
            action_horizon=2,
            replan_steps=1,
        )

    incompatible_memory = LeapMemoryState(
        LeapMemoryConfig(
            exit_depth=30,
            causal_mode="vision_causal",
            action_horizon=2,
            replan_steps=1,
        )
    )
    with pytest.raises(ValueError, match="memory/model causal mode mismatch"):
        model._validate_memory_compatibility(incompatible_memory)


def test_training_temporal_contract_is_checkpointed_and_enforced(tmp_path):
    source = _model(layers=30)
    source.configure_causal_training(
        causal_mode="action_aggregator",
        training_exit_depths=(30,),
        replan_steps=1,
        action_horizon=2,
    )
    memory = source.create_memory()
    assert memory.config.replan_steps == 1
    assert memory.config.action_horizon == 2
    with pytest.raises(ValueError, match="replan_steps differs"):
        source.create_memory(replan_steps=2, action_horizon=2)
    with pytest.raises(ValueError, match="action_horizon differs"):
        source.create_memory(replan_steps=1, action_horizon=3)

    checkpoint = tmp_path / "clock.pt"
    source.save_checkpoint(checkpoint)
    loaded = _model(layers=30)
    loaded.configure_causal_training(
        causal_mode="action_aggregator",
        training_exit_depths=(30,),
        replan_steps=1,
        action_horizon=2,
    )
    loaded.load_checkpoint(checkpoint)
    assert loaded.training_replan_steps == 1
    assert loaded.training_action_horizon == 2

    incompatible = _model(layers=30)
    incompatible.configure_causal_training(
        causal_mode="action_aggregator",
        training_exit_depths=(30,),
        replan_steps=1,
        action_horizon=3,
    )
    with pytest.raises(ValueError, match="action_horizon differs"):
        incompatible.load_checkpoint(checkpoint)


def test_memory_runtime_uses_local_rope_and_absolute_additive_positions():
    model = _model(layers=30).eval()
    feature_half = model.temporal_positions.feature_dim // 2
    with torch.no_grad():
        model.temporal_positions.video_projection.weight[:, 0].fill_(0.5)
        model.temporal_positions.video_projection.weight[:, feature_half].fill_(0.25)
        model.temporal_positions.action_block_projection.weight[:, 0].fill_(0.3)
        model.temporal_positions.action_block_projection.weight[:, feature_half].fill_(0.1)
        model.temporal_positions.action_control_projection.weight[:, 0].fill_(0.4)
        model.temporal_positions.action_control_projection.weight[:, feature_half].fill_(0.2)

    model._encode_input_image_latents_tensor = types.MethodType(
        lambda self, input_image, tiled=False: torch.ones(1, 4, 1, 1, 1),
        model,
    )
    calls = {
        "video_local": [],
        "video_base": [],
        "video_positioned": [],
        "action_local": [],
        "action_base": [],
        "action_positioned": [],
        "action_absolute_controls": [],
        "action_absolute_blocks": [],
        "raw_history_kwargs": [],
    }
    original_video_pre_dit = model.video_expert.pre_dit
    original_action_pre_dit = model.action_expert.pre_dit

    def video_pre_dit(*args, **kwargs):
        calls["video_local"].append(kwargs["frame_position_ids"].detach().clone())
        result = original_video_pre_dit(*args, **kwargs)
        calls["video_base"].append(result["tokens"].detach().clone())
        return result

    def action_pre_dit(*args, **kwargs):
        calls["action_local"].append(kwargs["position_ids"].detach().clone())
        result = original_action_pre_dit(*args, **kwargs)
        calls["action_base"].append(result["tokens"].detach().clone())
        return result

    model.video_expert.pre_dit = video_pre_dit
    model.action_expert.pre_dit = action_pre_dit
    original_apply_action = model.temporal_positions.apply_action_pre_dit

    def apply_action_pre_dit(pre_state, absolute_control_ids, absolute_block_ids):
        calls["action_absolute_controls"].append(
            absolute_control_ids.detach().clone()
        )
        calls["action_absolute_blocks"].append(absolute_block_ids.detach().clone())
        return original_apply_action(
            pre_state,
            absolute_control_ids,
            absolute_block_ids,
        )

    model.temporal_positions.apply_action_pre_dit = apply_action_pre_dit

    def fake_prefill(_self, **kwargs):
        calls["raw_history_kwargs"].append(set(kwargs))
        if kwargs["expert_name"] == "video":
            calls["video_positioned"].append(kwargs["tokens"].detach().clone())
        else:
            calls["action_positioned"].append(kwargs["tokens"].detach().clone())
        batch, length = kwargs["tokens"].shape[:2]
        kv = [
            {
                "k": torch.zeros(batch, length, 12),
                "v": torch.zeros(batch, length, 12),
            }
            for _ in range(kwargs["max_layers"])
        ]
        return kwargs["tokens"], kv

    def fake_action_forward(_self, **kwargs):
        calls["raw_history_kwargs"].append(set(kwargs))
        calls["action_positioned"].append(kwargs["action_tokens"].detach().clone())
        return kwargs["action_tokens"]

    model.mot.prefill_expert_segment = types.MethodType(fake_prefill, model.mot)
    model.mot.forward_action_with_history = types.MethodType(
        fake_action_forward,
        model.mot,
    )

    context = torch.randn(1, 2, 6)
    context_mask = torch.ones(1, 2, dtype=torch.bool)
    image = torch.zeros(1, 3, 16, 16)
    memory = model.create_memory(action_horizon=2, replan_steps=1)

    model.infer_action(
        prompt=None,
        input_image=image,
        action_horizon=2,
        context=context,
        context_mask=context_mask,
        num_inference_steps=1,
        seed=11,
        memory=memory,
    )
    model.commit_executed_actions(memory, torch.zeros(1, 3))
    model.infer_action(
        prompt=None,
        input_image=image,
        action_horizon=2,
        context=context,
        context_mask=context_mask,
        num_inference_steps=1,
        seed=12,
        memory=memory,
    )

    assert [ids.tolist() for ids in calls["video_local"]] == [[0], [0]]
    assert [ids.tolist() for ids in calls["action_local"]] == [[0, 1], [0], [0, 1]]
    assert [ids.tolist() for ids in calls["action_absolute_controls"]] == [
        [0, 1],
        [0],
        [1, 2],
    ]
    assert [ids.tolist() for ids in calls["action_absolute_blocks"]] == [
        [0, 0],
        [0],
        [1, 1],
    ]
    assert not torch.equal(calls["video_positioned"][0], calls["video_base"][0])
    assert not torch.equal(calls["video_positioned"][1], calls["video_base"][1])
    assert not torch.equal(calls["video_positioned"][0], calls["video_positioned"][1])
    for positioned, base in zip(calls["action_positioned"], calls["action_base"]):
        assert not torch.equal(positioned, base)
    assert all("native_kv" not in keys for keys in calls["raw_history_kwargs"])
    assert all(
        "separate_history_attention" not in keys
        for keys in calls["raw_history_kwargs"]
    )
    assert memory.segments[0].positions.tolist() == [0]
    assert memory.segments[1].positions.tolist() == [0]
    assert memory.segments[2].positions.tolist() == [1]
