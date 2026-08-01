import types

import pytest
import torch
from torch import nn

from fastwam.models.wan22.action_dit import ActionDiT
from fastwam.models.wan22.fastwam import FastWAM
from fastwam.models.wan22.mot import MoT
from fastwam.models.wan22.wan_video_dit import WanVideoDiT
from leapbot_va.lora import VideoLoRAConfig
from leapbot_va.memory import LeapMemoryConfig, LeapMemoryState
from leapbot_va.models.leapbot import LeapBotVA


class _TinyVideoVAE(nn.Module):
    temporal_downsample_factor = 1

    def encode(self, video, device=None, tiled=False, **kwargs):
        del tiled, kwargs
        if isinstance(video, list):
            if len(video) != 1:
                raise ValueError("tiny VAE supports one runtime observation")
            video = video[0].unsqueeze(0)
        pooled = video.to(device=device).mean(dim=(-2, -1), keepdim=True)
        return torch.cat((pooled, pooled[:, :1]), dim=1)


def _model(
    *,
    layers: int = 2,
    proprio_dim: int | None = None,
    vae: nn.Module | None = None,
    video_attention_mask_mode: str = "bidirectional",
) -> LeapBotVA:
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
        video_attention_mask_mode=video_attention_mask_mode,
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
        vae=nn.Identity() if vae is None else vae,
        text_dim=6,
        proprio_dim=proprio_dim,
        device="cpu",
        exit_depths=(layers,),
    )


@pytest.mark.parametrize(
    "causal_mode",
    ["interleaved", "vision_causal", "action_aggregator"],
)
def test_h0_preserves_video_flow_but_uses_future_conditioned_action(
    causal_mode,
):
    """H0 changes only the intended ActionDiT factorization."""

    torch.manual_seed(321)
    model = _model(
        layers=8,
        proprio_dim=2,
        vae=_TinyVideoVAE(),
        video_attention_mask_mode="first_frame_causal",
    )
    model.configure_causal_training(
        causal_mode=causal_mode,
        training_exit_depths=(8,),
        replan_steps=2,
        action_horizon=4,
        num_video_frames=5,
    )
    with torch.no_grad():
        for parameter in model.temporal_positions.parameters():
            parameter.normal_(std=0.3)
            assert torch.count_nonzero(parameter) > 0

    batch_size = 1
    base_sample = {
        "video": torch.randn(batch_size, 3, 5, 16, 16),
        "action": torch.randn(batch_size, 4, 3),
        "proprio": torch.randn(batch_size, 4, 2),
        "context": torch.randn(batch_size, 3, 6),
        "context_mask": torch.ones(batch_size, 3, dtype=torch.bool),
        "image_is_pad": torch.zeros(batch_size, 5, dtype=torch.bool),
        "action_is_pad": torch.zeros(batch_size, 4, dtype=torch.bool),
    }
    causal_sample = {
        **base_sample,
        "history_video": torch.empty(batch_size, 3, 0, 16, 16),
        "history_action": torch.empty(batch_size, 0, 2, 3),
        "history_proprio": torch.empty(batch_size, 0, 2),
        "history_valid_blocks": torch.empty(batch_size, 0, dtype=torch.bool),
        "history_block_positions": torch.empty(batch_size, 0, dtype=torch.long),
        "current_block_position": torch.zeros(batch_size, dtype=torch.long),
        "episode_step": torch.zeros(batch_size, dtype=torch.long),
        "full_episode_history": torch.ones(batch_size, dtype=torch.bool),
    }

    model.train()
    torch.manual_seed(999)
    fastwam_total, fastwam_metrics = FastWAM.training_loss(model, base_sample)
    torch.manual_seed(999)
    causal_total, causal_metrics = model.training_loss(causal_sample)

    assert torch.isfinite(causal_total)
    assert torch.isfinite(fastwam_total)
    assert causal_metrics["loss_video_d8"] == pytest.approx(
        fastwam_metrics["loss_video"], abs=5e-7, rel=0
    )
    assert causal_metrics["future_video_condition_noised_fraction"] in (0.0, 1.0)

    model.eval()
    model._encode_input_image_latents_tensor = types.MethodType(
        lambda self, input_image, tiled=False: torch.tensor(
            [-0.7, -0.2, 0.3, 0.8], dtype=self.torch_dtype
        ).view(1, 4, 1, 1, 1),
        model,
    )
    image = torch.zeros(1, 3, 16, 16)
    context = base_sample["context"]
    context_mask = base_sample["context_mask"]
    proprio = base_sample["proprio"][:, 0]
    memoryless_action = model.infer_action(
        prompt=None,
        input_image=image,
        action_horizon=4,
        proprio=proprio,
        context=context,
        context_mask=context_mask,
        num_inference_steps=2,
        seed=77,
        memory=None,
    )["action"]

    def forbidden_decode(*args, **kwargs):
        raise AssertionError("memory action inference decoded imagined video")

    model.infer_joint = types.MethodType(forbidden_decode, model)
    model._decode_latents = types.MethodType(forbidden_decode, model)
    memory = model.create_memory(exit_depth=8, max_history_blocks=2)
    memory_result = model.infer_action(
        prompt=None,
        input_image=image,
        action_horizon=4,
        num_video_frames=5,
        proprio=proprio,
        context=context,
        context_mask=context_mask,
        num_inference_steps=2,
        seed=77,
        memory=memory,
        profile=True,
    )
    memory_action = memory_result["action"]
    assert memory_action.shape == memoryless_action.shape
    assert torch.isfinite(memory_action).all()
    assert not torch.equal(memory_action, memoryless_action)
    assert memory.token_counts == {"video": 1, "action": 0}
    assert len(memory.segments) == 1
    assert memory_result["memory"]["transient_future_video_cache_bytes"] > 0
    timing = memory_result["timing"]
    assert set(timing) == {
        "conditioning_s",
        "observation_prefill_s",
        "future_video_setup_s",
        "future_video_denoise_s",
        "future_video_cache_s",
        "action_setup_s",
        "action_denoise_s",
        "causal_model_s",
        "causal_model_residual_s",
    }
    assert all(value >= 0 for value in timing.values())
    assert timing["causal_model_s"] == pytest.approx(
        timing["conditioning_s"]
        + timing["observation_prefill_s"]
        + timing["future_video_setup_s"]
        + timing["future_video_denoise_s"]
        + timing["future_video_cache_s"]
        + timing["action_setup_s"]
        + timing["action_denoise_s"]
        + timing["causal_model_residual_s"],
        abs=1e-9,
    )


def test_memory_inference_requires_finite_training_conditioning_inputs():
    model = _model(layers=30, proprio_dim=2)
    context = torch.zeros(1, 2, 6)
    context_mask = torch.ones(1, 2, dtype=torch.bool)

    with pytest.raises(ValueError, match="proprio is required"):
        model._prepare_inference_context(
            prompt=None,
            context=context,
            context_mask=context_mask,
            proprio=None,
        )
    with pytest.raises(ValueError, match="proprio contains non-finite"):
        model._prepare_inference_context(
            prompt=None,
            context=context,
            context_mask=context_mask,
            proprio=torch.tensor([0.0, float("nan")]),
        )
    with pytest.raises(ValueError, match="language context contains non-finite"):
        model._prepare_inference_context(
            prompt=None,
            context=torch.full_like(context, float("inf")),
            context_mask=context_mask,
            proprio=torch.zeros(2),
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
    # A true FastWAM release contains only the shared MoT payload and base
    # metadata.  Removing one learned LeapBot field from an otherwise native
    # checkpoint is corruption and is tested separately.
    payload = {
        key: value
        for key, value in payload.items()
        if key in {"mot", "step", "torch_dtype"}
    }
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

    untrained_depth_memory = LeapMemoryState(
        LeapMemoryConfig(
            exit_depth=8,
            causal_mode="action_aggregator",
            action_horizon=2,
            replan_steps=1,
        )
    )
    with pytest.raises(ValueError, match="unsupported"):
        model._validate_memory_compatibility(untrained_depth_memory)
    model.exit_depths = (8, 30)
    with pytest.raises(ValueError, match="was not trained"):
        model._validate_memory_compatibility(untrained_depth_memory)


def test_training_temporal_contract_is_checkpointed_and_enforced(tmp_path):
    source = _model(layers=30)
    source.history_vae_batch_chunk_size = 3
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
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert "retained_history_blocks" not in payload
    loaded = _model(layers=30)
    loaded.history_vae_batch_chunk_size = 3
    loaded.configure_causal_training(
        causal_mode="action_aggregator",
        training_exit_depths=(30,),
        replan_steps=1,
        action_horizon=2,
    )
    loaded.load_checkpoint(checkpoint)
    assert loaded.training_replan_steps == 1
    assert loaded.training_action_horizon == 2
    assert loaded.history_vae_batch_chunk_size == 3
    # Retention is an inference ablation, not a learned/checkpoint contract.
    # The same loaded weights may therefore be evaluated with any window.
    for retained in (None, 0, 2):
        ablation_memory = loaded.create_memory(retained_history_blocks=retained)
        assert ablation_memory.config.retained_history_blocks == retained

    incompatible = _model(layers=30)
    incompatible.configure_causal_training(
        causal_mode="action_aggregator",
        training_exit_depths=(30,),
        replan_steps=1,
        action_horizon=3,
    )
    with pytest.raises(ValueError, match="action_horizon differs"):
        incompatible.load_checkpoint(checkpoint)

    wrong_vae_chunk = _model(layers=30)
    wrong_vae_chunk.configure_causal_training(
        causal_mode="action_aggregator",
        training_exit_depths=(30,),
        replan_steps=1,
        action_horizon=2,
    )
    with pytest.raises(ValueError, match="history VAE batch chunk mismatch"):
        wrong_vae_chunk.load_checkpoint(checkpoint)


@pytest.mark.parametrize(
    ("replan_steps", "action_horizon", "expected_error"),
    [
        (2, 2, "replan_steps differs"),
        (1, 3, "action_horizon differs"),
    ],
)
def test_handcrafted_memory_cannot_bypass_model_temporal_contract(
    replan_steps,
    action_horizon,
    expected_error,
):
    model = _model(layers=30)
    model.configure_causal_training(
        causal_mode="action_aggregator",
        training_exit_depths=(30,),
        replan_steps=1,
        action_horizon=2,
    )
    memory = LeapMemoryState(
        LeapMemoryConfig(
            exit_depth=30,
            causal_mode="action_aggregator",
            replan_steps=replan_steps,
            action_horizon=action_horizon,
        )
    )

    with pytest.raises(ValueError, match=expected_error):
        model._validate_memory_compatibility(memory)


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
        hidden = kwargs["tokens"]
        if kwargs.get("exit_depths") is not None:
            hidden = {int(depth): kwargs["tokens"] for depth in kwargs["exit_depths"]}
        return hidden, kv

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

    assert [ids.tolist() for ids in calls["video_local"]] == [
        [0],
        [1, 2],
        [1, 2],
        [0],
        [1, 2],
        [1, 2],
    ]
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
    # The relative-position extension contributes exactly zero at the first
    # replan. This compares token inputs, not trained weights with a release.
    assert torch.equal(calls["video_positioned"][0], calls["video_base"][0])
    assert torch.equal(calls["video_positioned"][1], calls["video_base"][1])
    assert not torch.equal(calls["video_positioned"][3], calls["video_base"][3])
    assert not torch.equal(calls["video_positioned"][0], calls["video_positioned"][3])
    assert torch.equal(calls["action_positioned"][0], calls["action_base"][0])
    assert torch.equal(calls["action_positioned"][1], calls["action_base"][1])
    assert not torch.equal(calls["action_positioned"][2], calls["action_base"][2])
    assert all("native_kv" not in keys for keys in calls["raw_history_kwargs"])
    assert all(
        "separate_history_attention" not in keys
        for keys in calls["raw_history_kwargs"]
    )
    assert memory.segments[0].positions.tolist() == [0]
    assert memory.segments[1].positions.tolist() == [0]
    assert memory.segments[2].positions.tolist() == [1]
