from __future__ import annotations

from collections import Counter

import pytest
import torch
from torch import nn

from fastwam.models.wan22.action_dit import ActionDiT
from fastwam.models.wan22.mot import MoT
from fastwam.models.wan22.wan_video_dit import WanVideoDiT
from leapbot_va.episode_memory import EpisodeMemoryConfig
from leapbot_va.lora import VideoLoRAConfig
from leapbot_va.models.leapbot import LeapBotVA
from leapbot_va.training import (
    _packed_causal_history_reference_loss,
    _use_segment_activation_checkpointing,
    sample_lingbot_future_video_condition,
)
from scripts.validate_real_6b_runtime_training_equivalence import (
    validate_incremental_action_equivalence,
)


CAUSAL_MODES = ("interleaved", "vision_causal", "action_aggregator")
REPLAN_STEPS = 2
ACTION_HORIZON = 4


class _FrameIndependentVAE(nn.Module):
    """Tiny frozen VAE that preserves the temporal length and frame identity."""

    temporal_downsample_factor = 1

    def encode(self, video, device=None, tiled=False, **kwargs):
        del tiled, kwargs
        if isinstance(video, list):
            if len(video) != 1:
                raise ValueError("the tiny runtime VAE accepts one observation")
            video = video[0].unsqueeze(0)
        pooled = video.to(device=device).mean(dim=(-2, -1), keepdim=True)
        return torch.cat((pooled, pooled[:, :1]), dim=1)


def _model(
    causal_mode: str,
    *,
    layers: int = 2,
    exit_depths: tuple[int, ...] | None = None,
    training_exit_depths: tuple[int, ...] | None = None,
) -> LeapBotVA:
    torch.manual_seed(1701)
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
        video_attention_mask_mode="first_frame_causal",
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
    mot = MoT(
        {"video": video, "action": action},
        mot_checkpoint_mixed_attn=False,
    )
    available_exits = exit_depths or (layers,)
    model = LeapBotVA(
        video_expert=video,
        action_expert=action,
        mot=mot,
        vae=_FrameIndependentVAE(),
        text_dim=6,
        proprio_dim=2,
        device="cpu",
        torch_dtype=torch.float32,
        exit_depths=available_exits,
    )
    model.configure_causal_training(
        causal_mode=causal_mode,
        training_exit_depths=training_exit_depths or (layers,),
        replan_steps=REPLAN_STEPS,
        action_horizon=ACTION_HORIZON,
        num_video_frames=ACTION_HORIZON + 1,
    )
    return model.train()


def _sample(history_counts: tuple[int, ...]) -> dict[str, torch.Tensor]:
    batch = len(history_counts)
    max_history = max(history_counts, default=0)
    generator = torch.Generator(device="cpu").manual_seed(8803)
    history_valid = torch.zeros(batch, max_history, dtype=torch.bool)
    history_positions = torch.full((batch, max_history), -1, dtype=torch.long)
    for row, count in enumerate(history_counts):
        history_valid[row, :count] = True
        history_positions[row, :count] = torch.arange(count)

    return {
        "video": torch.randn(batch, 3, 5, 16, 16, generator=generator),
        "action": torch.randn(batch, ACTION_HORIZON, 3, generator=generator),
        "proprio": torch.randn(batch, ACTION_HORIZON, 2, generator=generator),
        "context": torch.randn(batch, 3, 6, generator=generator),
        "context_mask": torch.ones(batch, 3, dtype=torch.bool),
        "image_is_pad": torch.zeros(batch, 5, dtype=torch.bool),
        "action_is_pad": torch.zeros(batch, ACTION_HORIZON, dtype=torch.bool),
        "history_video": torch.randn(
            batch, 3, max_history, 16, 16, generator=generator
        ),
        "history_action": torch.randn(
            batch, max_history, REPLAN_STEPS, 3, generator=generator
        ),
        "history_proprio": torch.randn(
            batch, max_history, 2, generator=generator
        ),
        "history_valid_blocks": history_valid,
        "history_block_positions": history_positions,
        "current_block_position": torch.tensor(history_counts, dtype=torch.long),
        "episode_step": torch.tensor(history_counts, dtype=torch.long)
        * REPLAN_STEPS,
        "full_episode_history": torch.ones(batch, dtype=torch.bool),
    }


def _record_pre_dit_tokens(model: LeapBotVA, monkeypatch):
    records: dict[str, list[torch.Tensor]] = {"video": [], "action": []}

    original_video_pre = model.video_expert.pre_dit
    original_action_pre = model.action_expert.pre_dit

    def video_pre(*args, **kwargs):
        state = original_video_pre(*args, **kwargs)
        state["tokens"].retain_grad()
        records["video"].append(state["tokens"])
        return state

    def action_pre(*args, **kwargs):
        state = original_action_pre(*args, **kwargs)
        state["tokens"].retain_grad()
        records["action"].append(state["tokens"])
        return state

    monkeypatch.setattr(model.video_expert, "pre_dit", video_pre)
    monkeypatch.setattr(model.action_expert, "pre_dit", action_pre)
    return records


def _has_nonzero_gradient(tensor: torch.Tensor) -> bool:
    return tensor.grad is not None and bool(torch.count_nonzero(tensor.grad).item())


def test_lingbot_teacher_forcing_clean_and_high_noise_branches():
    model = _model("interleaved")
    clean = torch.linspace(-1.0, 1.0, 32).view(1, 4, 2, 2, 2)

    model.future_video_condition_noise_probability = 0.0
    condition, timesteps, was_noised = sample_lingbot_future_video_condition(
        model, clean
    )
    assert not was_noised
    assert torch.equal(condition, clean)
    assert torch.count_nonzero(timesteps) == 0

    model.future_video_condition_noise_probability = 1.0
    model.future_video_condition_min_u = 0.5
    model.future_video_condition_max_u = 0.5
    torch.manual_seed(77)
    condition, timesteps, was_noised = sample_lingbot_future_video_condition(
        model, clean
    )
    expected_sigma = model.train_video_scheduler._phi(
        torch.tensor(0.5), model.train_video_scheduler.shift
    )
    assert was_noised
    torch.testing.assert_close(
        timesteps,
        torch.full_like(
            timesteps,
            expected_sigma * model.train_video_scheduler.num_train_timesteps,
        ),
    )
    assert not torch.equal(condition, clean)


def test_future_video_condition_curriculum_is_driven_by_optimizer_step():
    model = _model("interleaved")
    model.configure_causal_training(
        causal_mode="interleaved",
        training_exit_depths=(2,),
        future_video_condition_noise_probability=0.5,
        future_video_condition_clean_warmup_steps=2,
        future_video_condition_noise_ramp_steps=4,
    )
    expected = {
        0: 0.0,
        1: 0.0,
        2: 0.0,
        4: 0.25,
        6: 0.5,
        100: 0.5,
    }
    for step, probability in expected.items():
        model.set_training_step(step)
        assert model.effective_future_video_condition_noise_probability() == pytest.approx(
            probability
        )


def test_training_rejects_video_length_outside_checkpoint_contract():
    model = _model("interleaved")
    sample = _sample((0,))
    model.training_num_video_frames = 9
    with pytest.raises(ValueError, match="video length differs"):
        model.training_loss(sample)


def test_segment_checkpoint_uses_fastwam_dit_only_training_state():
    model = _model("interleaved")
    model.history_segment_activation_checkpointing = True

    model.eval()
    model.mot.train()

    assert not model.training
    assert model.mot.training
    assert _use_segment_activation_checkpointing(model)


def test_outer_segment_checkpoint_disables_all_nested_mot_checkpoints(monkeypatch):
    """Temporary prefix cats must never become inputs to an inner checkpoint."""

    model = _model("interleaved")
    model.history_segment_activation_checkpointing = True
    model.mot.mot_checkpoint_mixed_attn = True
    model.video_expert.use_gradient_checkpointing = True
    model.action_expert.use_gradient_checkpointing = True
    attention_switches: list[bool | None] = []
    post_switches: list[bool] = []
    original_attention = model.mot._mixed_attention
    original_post = model.mot._apply_post_with_optional_checkpoint

    def record_attention(*args, **kwargs):
        attention_switches.append(kwargs.get("checkpoint_attention"))
        return original_attention(*args, **kwargs)

    def record_post(*args, **kwargs):
        post_switches.append(bool(kwargs["use_gradient_checkpointing"]))
        return original_post(*args, **kwargs)

    monkeypatch.setattr(model.mot, "_mixed_attention", record_attention)
    monkeypatch.setattr(model.mot, "_apply_post_with_optional_checkpoint", record_post)

    torch.manual_seed(7302)
    loss, _ = model.training_loss(_sample((1,)))

    assert torch.isfinite(loss)
    assert attention_switches and set(attention_switches) == {False}
    assert post_switches and set(post_switches) == {False}


@pytest.mark.parametrize("causal_mode", CAUSAL_MODES)
def test_action_loss_reaches_real_history_and_future_video_condition_only(
    causal_mode,
    monkeypatch,
):
    """ActionDiT reads the condition branch, never the video flow target."""

    model = _model(causal_mode)
    model.history_segment_activation_checkpointing = True
    model.loss_lambda_video = 0.0
    model.loss_lambda_action = 1.0
    records = _record_pre_dit_tokens(model, monkeypatch)
    action_predictions: list[torch.Tensor] = []
    original_action_post = model.action_expert.post_dit

    def action_post(*args, **kwargs):
        prediction = original_action_post(*args, **kwargs)
        action_predictions.append(prediction)
        return prediction

    monkeypatch.setattr(model.action_expert, "post_dit", action_post)

    torch.manual_seed(4401)
    total, _ = model.training_loss(_sample((2,)))

    # Calls are history V0, history V1, current real V, teacher-forced future
    # condition V, then the independently noised video flow target V.
    assert len(records["video"]) == 5
    assert len(records["action"]) == 3
    future_condition_tokens = records["video"][-2]
    future_target_tokens = records["video"][-1]
    condition_to_action = torch.autograd.grad(
        action_predictions[-1].sum(),
        future_condition_tokens,
        allow_unused=True,
        retain_graph=True,
    )[0]
    target_to_action = torch.autograd.grad(
        action_predictions[-1].sum(),
        future_target_tokens,
        allow_unused=True,
        retain_graph=True,
    )[0]
    assert condition_to_action is not None
    assert torch.count_nonzero(condition_to_action) > 0
    assert target_to_action is None

    total.backward()
    for block, token in enumerate(records["video"][:2]):
        assert _has_nonzero_gradient(token), (
            causal_mode,
            "history_video",
            block,
        )
    for block, token in enumerate(records["action"][:2]):
        assert _has_nonzero_gradient(token), (
            causal_mode,
            "history_action",
            block,
        )
    assert _has_nonzero_gradient(future_condition_tokens)
    assert not _has_nonzero_gradient(future_target_tokens)


@pytest.mark.parametrize(
    ("causal_mode", "video_has_past_video", "video_has_past_action"),
    [
        ("interleaved", True, True),
        ("vision_causal", True, False),
        ("action_aggregator", False, False),
    ],
)
def test_video_loss_history_gradients_follow_the_selected_causal_mode(
    causal_mode,
    video_has_past_video,
    video_has_past_action,
    monkeypatch,
):
    model = _model(causal_mode)
    model.history_segment_activation_checkpointing = True
    model.loss_lambda_video = 1.0
    model.loss_lambda_action = 0.0
    records = _record_pre_dit_tokens(model, monkeypatch)

    torch.manual_seed(4402)
    total, _ = model.training_loss(_sample((2,)))
    total.backward()

    assert all(
        _has_nonzero_gradient(token) is video_has_past_video
        for token in records["video"][:2]
    )
    assert all(
        _has_nonzero_gradient(token) is video_has_past_action
        for token in records["action"][:2]
    )
    # Every mode conditions future-video denoising on this block's real frame.
    assert _has_nonzero_gradient(records["video"][2])


def test_mixed_history_batch_runs_each_episode_separately_and_never_reads_padding(
    monkeypatch,
):
    model = _model("interleaved")
    sample = _sample((2, 1))
    # Distinct language inputs remain distinguishable after each expert's
    # deterministic context projection.
    sample["context"][0].fill_(0.125)
    sample["context"][1].fill_(0.875)
    # Invalid values are intentional: a vectorized/padded history program would
    # immediately contaminate attention, while per-episode execution ignores it.
    sample["history_video"][1, :, 1].fill_(float("nan"))
    sample["history_action"][1, 1].fill_(float("nan"))
    sample["history_proprio"][1, 1].fill_(float("nan"))

    prefill_by_expert_and_episode: dict[str, Counter[float]] = {
        "video": Counter(),
        "action": Counter(),
    }
    action_by_episode: Counter[float] = Counter()
    original_prefill = model.mot.prefill_expert_segment
    original_action = model.mot.forward_action_with_history

    def marker(payload: dict[str, torch.Tensor]) -> float:
        return round(float(payload["context"][0, 0, 0].item()), 3)

    def prefill(*args, **kwargs):
        assert kwargs["tokens"].shape[0] == 1
        assert torch.isfinite(kwargs["tokens"]).all()
        prefill_by_expert_and_episode[kwargs["expert_name"]][
            marker(kwargs["context_payload"])
        ] += 1
        return original_prefill(*args, **kwargs)

    def action_forward(*args, **kwargs):
        assert kwargs["action_tokens"].shape[0] == 1
        assert torch.isfinite(kwargs["action_tokens"]).all()
        action_by_episode[marker(kwargs["action_context_payload"])] += 1
        return original_action(*args, **kwargs)

    monkeypatch.setattr(model.mot, "prefill_expert_segment", prefill)
    monkeypatch.setattr(model.mot, "forward_action_with_history", action_forward)

    torch.manual_seed(4403)
    total, metrics = model.training_loss(sample)

    assert torch.isfinite(total)
    assert metrics["history_blocks_mean"] == pytest.approx(1.5)
    assert metrics["history_h0_fraction"] == 0.0
    assert metrics["__metric_weight__loss_action_d2_h0"] == 0.0
    assert metrics["__metric_weight__loss_action_d2_h1_4"] == 2.0
    assert metrics["loss_action_d2_h1_4"] == pytest.approx(
        metrics["loss_action_d2"]
    )
    # Each valid history contributes V+A; each episode also contributes current
    # real V, teacher-forced condition V, and video-flow target V. The invalid
    # second slot of episode 1 adds 0.
    # Video and Action experts apply different context projections, so the
    # numeric marker differs by expert. Counts still identify both episodes:
    # H=2 emits 5 video + 2 committed-action prefills; H=1 emits 4 + 1.
    assert sorted(prefill_by_expert_and_episode["video"].values()) == [4, 5]
    assert sorted(prefill_by_expert_and_episode["action"].values()) == [1, 2]
    assert sorted(action_by_episode.values()) == [1, 1]


def test_four_depth_losses_share_one_action_and_one_video_transformer_traversal(
    monkeypatch,
):
    depths = (8, 16, 24, 30)
    model = _model(
        "interleaved",
        layers=30,
        exit_depths=depths,
        training_exit_depths=depths,
    )
    model.history_segment_activation_checkpointing = True
    action_calls: list[tuple[int, ...]] = []
    video_exit_calls: list[tuple[int, ...]] = []
    original_action = model.mot.forward_action_with_history
    original_prefill = model.mot.prefill_expert_segment

    def action_forward(*args, **kwargs):
        requested = tuple(kwargs.get("exit_depths") or ())
        action_calls.append(requested)
        result = original_action(*args, **kwargs)
        assert set(result) == set(depths)
        return result

    def prefill(*args, **kwargs):
        requested = tuple(kwargs.get("exit_depths") or ())
        result = original_prefill(*args, **kwargs)
        if requested:
            video_exit_calls.append(requested)
            assert set(result[0]) == set(depths)
        return result

    monkeypatch.setattr(model.mot, "forward_action_with_history", action_forward)
    monkeypatch.setattr(model.mot, "prefill_expert_segment", prefill)

    torch.manual_seed(4404)
    total, metrics = model.training_loss(_sample((1,)))
    # One forward traversal produces every requested exit. Backward is expected
    # to run the same checkpointed traversal again for activation recomputation.
    assert action_calls == [depths]
    assert video_exit_calls == [depths]
    total.backward()
    for depth in depths:
        assert torch.isfinite(torch.tensor(metrics[f"loss_video_d{depth}"]))
        assert torch.isfinite(torch.tensor(metrics[f"loss_action_d{depth}"]))
    for depth in depths[:-1]:
        assert any(
            parameter.grad is not None
            and bool(torch.count_nonzero(parameter.grad).item())
            for parameter in model.action_exit_heads[str(depth)].parameters()
        )
        assert any(
            parameter.grad is not None
            and bool(torch.count_nonzero(parameter.grad).item())
            for parameter in model.video_exit_heads[str(depth)].parameters()
        )


@pytest.mark.parametrize("causal_mode", CAUSAL_MODES)
def test_segment_checkpoint_loss_and_parameter_gradients_match_reference(
    causal_mode,
    monkeypatch,
):
    """Rematerializing prefix cats must preserve the full-BPTT objective exactly."""

    reference = _model(causal_mode)
    checkpointed = _model(causal_mode)
    # The production model normally enables both inner attention/post-block
    # checkpointing and the new outer segment checkpoint.  The outer closure
    # deliberately suppresses its nested checkpoints, so compare that path to
    # the original inner-checkpoint-only reference as well.
    for model in (reference, checkpointed):
        model.mot.mot_checkpoint_mixed_attn = True
        model.video_expert.use_gradient_checkpointing = True
        model.action_expert.use_gradient_checkpointing = True
    reference.history_segment_activation_checkpointing = False
    checkpointed.history_segment_activation_checkpointing = True
    reference_tokens = _record_pre_dit_tokens(reference, monkeypatch)
    checkpointed_tokens = _record_pre_dit_tokens(checkpointed, monkeypatch)
    sample = _sample((2,))

    torch.manual_seed(7401)
    reference_loss, reference_metrics = reference.training_loss(sample)
    torch.manual_seed(7401)
    checkpointed_loss, checkpointed_metrics = checkpointed.training_loss(sample)

    torch.testing.assert_close(
        checkpointed_loss,
        reference_loss,
        rtol=0,
        atol=0,
    )
    assert checkpointed_metrics == reference_metrics

    reference_loss.backward()
    checkpointed_loss.backward()
    reference_gradients = {
        name: parameter.grad
        for name, parameter in reference.named_parameters()
    }
    checkpointed_gradients = {
        name: parameter.grad
        for name, parameter in checkpointed.named_parameters()
    }
    assert checkpointed_gradients.keys() == reference_gradients.keys()
    for name, reference_gradient in reference_gradients.items():
        checkpointed_gradient = checkpointed_gradients[name]
        assert (reference_gradient is None) is (checkpointed_gradient is None), name
        if reference_gradient is not None:
            torch.testing.assert_close(
                checkpointed_gradient,
                reference_gradient,
                rtol=1e-5,
                atol=1e-6,
                msg=lambda message, name=name: f"{name}: {message}",
            )
    for modality in ("video", "action"):
        assert len(checkpointed_tokens[modality]) == len(reference_tokens[modality])
        for index, reference_token in enumerate(reference_tokens[modality]):
            checkpointed_token = checkpointed_tokens[modality][index]
            assert reference_token.grad is not None, (modality, index)
            assert checkpointed_token.grad is not None, (modality, index)
            torch.testing.assert_close(
                checkpointed_token.grad,
                reference_token.grad,
                rtol=1e-5,
                atol=1e-6,
                msg=lambda message, modality=modality, index=index: (
                    f"{modality} pre-DiT segment {index}: {message}"
                ),
            )


@pytest.mark.parametrize("history_blocks", (0, 1, 2))
@pytest.mark.parametrize("causal_mode", CAUSAL_MODES)
def test_incremental_video_flow_matches_one_shot_packed_causal_reference(
    causal_mode,
    history_blocks,
    monkeypatch,
):
    """The separate video-flow target keeps the packed causal objective."""

    reference = _model(causal_mode)
    incremental = _model(causal_mode)
    # The action factorization intentionally changed: it now includes a second
    # teacher-forced future-video branch that this legacy packed reference does
    # not contain. Isolate the still-identical video flow objective here.
    reference.loss_lambda_action = 0.0
    incremental.loss_lambda_action = 0.0
    sample = _sample((history_blocks,))

    # Use distinct, explicit diffusion times for the two modalities. Resetting
    # the RNG before each path then supplies bitwise-identical video/action
    # noise without coupling this equivalence check to scheduler sampling.
    fixed_timesteps = (
        (reference.train_video_scheduler, 237.5),
        (incremental.train_video_scheduler, 237.5),
        (reference.train_action_scheduler, 681.25),
        (incremental.train_action_scheduler, 681.25),
    )
    for scheduler, value in fixed_timesteps:
        monkeypatch.setattr(
            scheduler,
            "sample_training_t",
            lambda batch_size, device, dtype, value=value: torch.full(
                (batch_size,), value, device=device, dtype=dtype
            ),
        )

    torch.manual_seed(9127)
    reference_loss, reference_metrics = _packed_causal_history_reference_loss(
        reference,
        sample,
    )
    torch.manual_seed(9127)
    incremental_loss, incremental_metrics = incremental.training_loss(sample)

    final_depth = reference.mot.num_layers
    metric = f"loss_video_d{final_depth}"
    assert incremental_metrics[metric] == pytest.approx(
        reference_metrics[metric],
        rel=1e-6,
        abs=1e-6,
    )
    assert incremental_metrics["history_blocks_mean"] == history_blocks
    assert incremental_metrics["history_blocks_max"] == history_blocks
    torch.testing.assert_close(
        incremental_loss,
        reference_loss,
        rtol=1e-6,
        atol=1e-6,
    )

    reference_loss.backward()
    incremental_loss.backward()
    reference_parameters = dict(reference.named_parameters())
    incremental_parameters = dict(incremental.named_parameters())
    assert incremental_parameters.keys() == reference_parameters.keys()

    saw_gradient = False
    for name, reference_parameter in reference_parameters.items():
        incremental_parameter = incremental_parameters[name]
        reference_gradient = reference_parameter.grad
        incremental_gradient = incremental_parameter.grad
        assert (incremental_gradient is None) is (reference_gradient is None), name
        if reference_gradient is None:
            continue
        saw_gradient = True
        assert torch.isfinite(reference_gradient).all(), name
        assert torch.isfinite(incremental_gradient).all(), name
        torch.testing.assert_close(
            incremental_gradient,
            reference_gradient,
            rtol=1e-5,
            atol=1e-6,
            msg=lambda message, name=name: f"{name}: {message}",
        )
    assert saw_gradient


@pytest.mark.parametrize("causal_mode", CAUSAL_MODES)
def test_training_and_runtime_share_persistent_prefix_and_condition_shape(causal_mode):
    """GT and generated conditions differ, but their causal contracts match."""

    # LeapMemoryConfig exposes the production exit depths, so D=8 is the
    # smallest inexpensive model that can exercise the public memory API.
    model = _model(causal_mode, layers=8)
    assert model.training

    result = validate_incremental_action_equivalence(
        model,
        _sample((2,)),
        seed=1203,
    )

    assert model.training
    assert result["sequence"]["runtime_conditioning_valid"]
    assert result["prefix_kv"]["bitwise_equal"]
    assert result["transient_future_video"]["shape_match"]
    assert result["transient_future_video"]["finite"]
    assert not result["transient_future_video"]["numeric_equality_expected"]
    assert result["bitwise_pass"]


def test_strict_window_training_and_runtime_rebuild_the_same_prefix():
    model = _model("action_aggregator", layers=8)
    model.configure_causal_training(
        causal_mode="action_aggregator",
        training_exit_depths=(8,),
        history_training_mode="strict_replay_window_bptt",
        history_window_blocks=2,
        replan_steps=REPLAN_STEPS,
        action_horizon=ACTION_HORIZON,
        num_video_frames=ACTION_HORIZON + 1,
    )
    sample = _sample((2,))
    sample["history_window_blocks"] = torch.tensor([2])
    sample["full_episode_history"] = torch.tensor([False])
    sample["episode_anchor_video"] = torch.zeros(1, 3, 1, 16, 16)
    sample["episode_anchor_proprio"] = torch.zeros(1, 2)
    sample["episode_anchor_valid"] = torch.tensor([False])

    result = validate_incremental_action_equivalence(model, sample, seed=1203)

    assert result["history_storage_mode"] == "strict_replay"
    assert result["sequence"]["runtime_conditioning_valid"]
    assert result["prefix_kv"]["bitwise_equal"]
    assert result["bitwise_pass"]



@pytest.mark.parametrize("causal_mode", CAUSAL_MODES)
def test_pch_training_runs_one_real_batch_for_every_dit_stage(
    causal_mode, monkeypatch
):
    model = _model(causal_mode)
    model.configure_causal_training(
        causal_mode=causal_mode,
        training_exit_depths=(2,),
        history_training_mode="packed_causal_history_bptt",
        packed_history_attention_backend="dense",
        history_window_blocks=2,
        replan_steps=REPLAN_STEPS,
        action_horizon=ACTION_HORIZON,
        num_video_frames=ACTION_HORIZON + 1,
    )
    sample = _sample((0, 2))
    sample["history_window_blocks"] = torch.tensor([2, 2])
    sample["full_episode_history"] = torch.tensor([False, False])
    sample["episode_anchor_video"] = torch.zeros(2, 3, 1, 16, 16)
    sample["episode_anchor_proprio"] = torch.zeros(2, 2)
    sample["episode_anchor_valid"] = torch.tensor([False, False])

    packed_batches = []
    segment_batches = []
    action_batches = []
    original_packed = model.mot.prefill_packed_history
    original_segment = model.mot.prefill_expert_segment
    original_action = model.mot.forward_action_with_history

    def packed(*args, **kwargs):
        packed_batches.append(int(kwargs["video_tokens"].shape[0]))
        return original_packed(*args, **kwargs)

    def segment(*args, **kwargs):
        segment_batches.append(int(kwargs["tokens"].shape[0]))
        return original_segment(*args, **kwargs)

    def action(*args, **kwargs):
        action_batches.append(int(kwargs["action_tokens"].shape[0]))
        return original_action(*args, **kwargs)

    monkeypatch.setattr(model.mot, "prefill_packed_history", packed)
    monkeypatch.setattr(model.mot, "prefill_expert_segment", segment)
    monkeypatch.setattr(model.mot, "forward_action_with_history", action)

    loss, metrics = model.training_loss(sample)
    assert torch.isfinite(loss)
    loss.backward()
    assert packed_batches == [2]
    assert segment_batches == [2, 2, 2]
    assert action_batches == [2]
    assert metrics["pch_packed_tokens"] > metrics["pch_valid_tokens"]



def test_pch_all_h0_skips_history_forward(monkeypatch):
    model = _model("interleaved")
    model.configure_causal_training(
        causal_mode="interleaved",
        training_exit_depths=(2,),
        history_training_mode="packed_causal_history_bptt",
        packed_history_attention_backend="dense",
        history_window_blocks=2,
        replan_steps=REPLAN_STEPS,
        action_horizon=ACTION_HORIZON,
        num_video_frames=ACTION_HORIZON + 1,
    )
    sample = _sample((0,))
    sample["history_video"] = torch.zeros(1, 3, 2, 16, 16)
    sample["history_action"] = torch.zeros(1, 2, REPLAN_STEPS, 3)
    sample["history_proprio"] = torch.zeros(1, 2, 2)
    sample["history_valid_blocks"] = torch.zeros(1, 2, dtype=torch.bool)
    sample["history_block_positions"] = torch.full((1, 2), -1, dtype=torch.long)
    sample["history_window_blocks"] = torch.tensor([2])
    sample["full_episode_history"] = torch.tensor([False])
    sample["episode_anchor_video"] = torch.zeros(1, 3, 1, 16, 16)
    sample["episode_anchor_proprio"] = torch.zeros(1, 2)
    sample["episode_anchor_valid"] = torch.tensor([False])

    def forbidden(*args, **kwargs):
        raise AssertionError("all-H0 batch must skip packed history MoT")

    monkeypatch.setattr(model.mot, "prefill_packed_history", forbidden)
    loss, metrics = model.training_loss(sample)
    loss.backward()
    assert torch.isfinite(loss)
    assert metrics["history_blocks_max"] == 0

def test_pch_multi_exit_trains_shallow_action_and_video_heads():
    model = _model(
        "interleaved",
        layers=2,
        exit_depths=(1, 2),
        training_exit_depths=(1, 2),
    )
    model.configure_causal_training(
        causal_mode="interleaved",
        training_exit_depths=(1, 2),
        history_training_mode="packed_causal_history_bptt",
        packed_history_attention_backend="dense",
        history_window_blocks=1,
        replan_steps=REPLAN_STEPS,
        action_horizon=ACTION_HORIZON,
        num_video_frames=ACTION_HORIZON + 1,
    )
    sample = _sample((1,))
    sample["history_window_blocks"] = torch.tensor([1])
    sample["full_episode_history"] = torch.tensor([False])
    sample["episode_anchor_video"] = torch.zeros(1, 3, 1, 16, 16)
    sample["episode_anchor_proprio"] = torch.zeros(1, 2)
    sample["episode_anchor_valid"] = torch.tensor([False])
    loss, metrics = model.training_loss(sample)
    loss.backward()
    assert "loss_action_d1" in metrics and "loss_video_d1" in metrics
    assert any(parameter.grad is not None for parameter in model.action_exit_heads["1"].parameters())
    assert any(parameter.grad is not None for parameter in model.video_exit_heads["1"].parameters())

def test_runtime_equivalence_rejects_padded_action_target_contract():
    model = _model("interleaved")
    sample = _sample((1,))
    sample["action_is_pad"][0, -1] = True

    with pytest.raises(ValueError, match="complete, unpadded action horizon"):
        validate_incremental_action_equivalence(model, sample)


def test_padded_current_action_and_video_tokens_are_masked_before_attention(
    monkeypatch,
):
    model = _model("interleaved")
    sample = _sample((1,))
    sample["action_is_pad"][0, 2:] = True
    sample["image_is_pad"][0, 3:] = True
    captured_action_masks = []
    captured_video_masks = []
    original_action = model.mot.forward_action_with_history
    original_prefill = model.mot.prefill_expert_segment

    def action_forward(*args, **kwargs):
        captured_action_masks.append(kwargs.get("action_valid_mask"))
        return original_action(*args, **kwargs)

    def prefill(*args, **kwargs):
        if kwargs["expert_name"] == "video" and kwargs.get("exit_depths"):
            captured_video_masks.append(kwargs.get("segment_valid_mask"))
        return original_prefill(*args, **kwargs)

    monkeypatch.setattr(model.mot, "forward_action_with_history", action_forward)
    monkeypatch.setattr(model.mot, "prefill_expert_segment", prefill)

    torch.manual_seed(8804)
    loss, _ = model.training_loss(sample)
    assert torch.isfinite(loss)
    assert len(captured_action_masks) == 1
    assert captured_action_masks[0].tolist() == [[True, True, False, False]]
    assert len(captured_video_masks) == 1
    assert captured_video_masks[0].tolist() == [[True, True, False, False]]


def test_padded_action_values_cannot_change_valid_action_loss():
    reference = _model("interleaved")
    perturbed = _model("interleaved")
    for model in (reference, perturbed):
        model.loss_lambda_video = 0.0
        model.loss_lambda_action = 1.0
    sample = _sample((1,))
    sample["action_is_pad"][0, 2:] = True
    changed = {key: value.clone() for key, value in sample.items()}
    changed["action"][0, 2:] += 1000.0

    torch.manual_seed(8805)
    reference_loss, _ = reference.training_loss(sample)
    torch.manual_seed(8805)
    changed_loss, _ = perturbed.training_loss(changed)
    torch.testing.assert_close(changed_loss, reference_loss, rtol=0, atol=0)


def test_fully_padded_video_values_cannot_change_valid_video_loss():
    reference = _model("interleaved")
    perturbed = _model("interleaved")
    for model in (reference, perturbed):
        model.loss_lambda_video = 1.0
        model.loss_lambda_action = 0.0
    sample = _sample((1,))
    sample["image_is_pad"][0, 3:] = True
    changed = {key: value.clone() for key, value in sample.items()}
    changed["video"][0, :, 3:] += 1000.0

    torch.manual_seed(8806)
    reference_loss, _ = reference.training_loss(sample)
    torch.manual_seed(8806)
    changed_loss, _ = perturbed.training_loss(changed)
    torch.testing.assert_close(changed_loss, reference_loss, rtol=0, atol=0)


def test_checkpointed_padding_masks_survive_recompute_and_isolate_padded_tails(
    monkeypatch,
):
    reference = _model("interleaved")
    perturbed = _model("interleaved")
    for model in (reference, perturbed):
        model.history_segment_activation_checkpointing = True
        model.mot.mot_checkpoint_mixed_attn = True
        model.video_expert.use_gradient_checkpointing = True
        model.action_expert.use_gradient_checkpointing = True
        assert _use_segment_activation_checkpointing(model)

    sample = _sample((1,))
    sample["action_is_pad"][0, 2:] = True
    sample["image_is_pad"][0, 3:] = True
    changed = {key: value.clone() for key, value in sample.items()}
    changed["action"][0, 2:] += 1000.0
    changed["video"][0, :, 3:] += 1000.0

    expected_valid_mask = torch.tensor([[True, True, False, False]])
    action_mask_calls: list[tuple[torch.Tensor, bool]] = []
    video_mask_calls: list[tuple[torch.Tensor, bool]] = []
    original_action = reference.mot.forward_action_with_history
    original_prefill = reference.mot.prefill_expert_segment

    def record_action_mask(*args, **kwargs):
        valid_mask = kwargs.get("action_valid_mask")
        if valid_mask is not None:
            action_mask_calls.append(
                (valid_mask.detach().cpu().clone(), kwargs["checkpoint_internal"])
            )
        return original_action(*args, **kwargs)

    def record_video_mask(*args, **kwargs):
        valid_mask = kwargs.get("segment_valid_mask")
        if kwargs["expert_name"] == "video" and valid_mask is not None:
            video_mask_calls.append(
                (valid_mask.detach().cpu().clone(), kwargs["checkpoint_internal"])
            )
        return original_prefill(*args, **kwargs)

    monkeypatch.setattr(
        reference.mot,
        "forward_action_with_history",
        record_action_mask,
    )
    monkeypatch.setattr(
        reference.mot,
        "prefill_expert_segment",
        record_video_mask,
    )

    torch.manual_seed(8807)
    reference_loss, reference_metrics = reference.training_loss(sample)
    torch.manual_seed(8807)
    changed_loss, changed_metrics = perturbed.training_loss(changed)

    torch.testing.assert_close(changed_loss, reference_loss, rtol=0, atol=0)
    assert changed_metrics == reference_metrics
    assert torch.isfinite(reference_loss)
    assert torch.isfinite(changed_loss)

    reference_loss.backward()
    changed_loss.backward()

    # Non-reentrant checkpointing invokes these methods once in the original
    # forward and again while rematerializing activations during backward.
    assert len(action_mask_calls) >= 2
    assert len(video_mask_calls) >= 2
    for valid_mask, checkpoint_internal in action_mask_calls + video_mask_calls:
        torch.testing.assert_close(valid_mask, expected_valid_mask)
        assert checkpoint_internal is False

    saw_finite_gradient = False
    for (reference_name, reference_parameter), (
        changed_name,
        changed_parameter,
    ) in zip(reference.named_parameters(), perturbed.named_parameters(), strict=True):
        assert changed_name == reference_name
        assert (reference_parameter.grad is None) is (changed_parameter.grad is None)
        if reference_parameter.grad is None:
            continue
        saw_finite_gradient = True
        assert torch.isfinite(reference_parameter.grad).all(), reference_name
        assert torch.isfinite(changed_parameter.grad).all(), changed_name
        torch.testing.assert_close(
            changed_parameter.grad,
            reference_parameter.grad,
            rtol=1e-5,
            atol=1e-6,
            msg=lambda message, name=reference_name: f"{name}: {message}",
        )
    assert saw_finite_gradient


def test_strict_window_accepts_only_right_aligned_recent_suffix_and_v0_anchor():
    model = _model("interleaved")
    model.configure_causal_training(
        causal_mode="interleaved",
        training_exit_depths=(2,),
        history_training_mode="strict_replay_window_bptt",
        history_window_blocks=4,
        replan_steps=REPLAN_STEPS,
        action_horizon=ACTION_HORIZON,
        num_video_frames=ACTION_HORIZON + 1,
    )

    early = _sample((2,))
    for key, dim in (
        ("history_video", 2),
        ("history_action", 1),
        ("history_proprio", 1),
    ):
        value = early[key]
        shape = list(value.shape)
        shape[dim] = 2
        early[key] = torch.cat([value.new_zeros(shape), value], dim=dim)
    early["history_valid_blocks"] = torch.tensor(
        [[False, False, True, True]]
    )
    early["history_block_positions"] = torch.tensor([[-1, -1, 0, 1]])
    early["history_window_blocks"] = torch.tensor([4])
    early["full_episode_history"] = torch.tensor([False])
    early["episode_anchor_video"] = torch.zeros(1, 3, 1, 16, 16)
    early["episode_anchor_proprio"] = torch.zeros(1, 2)
    early["episode_anchor_valid"] = torch.tensor([False])
    model.future_video_condition_noise_probability = 0.0
    torch.manual_seed(9901)
    early_loss, early_metrics = model.training_loss(early)
    assert torch.isfinite(early_loss)
    assert early_metrics["history_blocks_mean"] == 2.0
    assert early_metrics["episode_anchor_fraction"] == 0.0
    assert early_metrics["__metric_weight__loss_action_d2_condition_clean"] == 1.0
    assert early_metrics["__metric_weight__loss_action_d2_condition_noised"] == 0.0

    anchored = _sample((4,))
    anchored["history_block_positions"] = torch.tensor([[1, 2, 3, 4]])
    anchored["current_block_position"] = torch.tensor([5])
    anchored["episode_step"] = torch.tensor([5 * REPLAN_STEPS])
    anchored["history_window_blocks"] = torch.tensor([4])
    anchored["full_episode_history"] = torch.tensor([False])
    anchored["episode_anchor_video"] = torch.randn(1, 3, 1, 16, 16)
    anchored["episode_anchor_proprio"] = torch.randn(1, 2)
    anchored["episode_anchor_valid"] = torch.tensor([True])
    model.future_video_condition_noise_probability = 1.0
    torch.manual_seed(9902)
    anchor_loss, anchor_metrics = model.training_loss(anchored)
    assert torch.isfinite(anchor_loss)
    assert anchor_metrics["history_blocks_mean"] == 4.0
    assert anchor_metrics["episode_anchor_fraction"] == 1.0
    assert anchor_metrics["__metric_weight__loss_action_d2_condition_clean"] == 0.0
    assert anchor_metrics["__metric_weight__loss_action_d2_condition_noised"] == 1.0

    broken = dict(early)
    broken["history_valid_blocks"] = torch.tensor(
        [[True, True, False, False]]
    )
    broken["history_block_positions"] = torch.tensor([[0, 1, -1, -1]])
    with pytest.raises(ValueError, match="left padding"):
        model.training_loss(broken)

def test_episode_memory_scan_training_step_backpropagates_through_h():
    model = _model("interleaved")
    config = EpisodeMemoryConfig(
        enabled=True,
        window_blocks=8,
        chunk_blocks=4,
        num_slots=4,
        state_dim=8,
        group_dim=2,
        updater_dim=16,
        updater_heads=4,
        reader_rank=2,
    )
    model.configure_causal_training(
        causal_mode="interleaved",
        training_exit_depths=(2,),
        history_training_mode="episode_memory_scan_bptt",
        packed_history_attention_backend="dense",
        history_window_blocks=8,
        replan_steps=REPLAN_STEPS,
        action_horizon=ACTION_HORIZON,
        num_video_frames=ACTION_HORIZON + 1,
        episode_memory_config=config,
    )
    sample = _sample((12,))
    loss, metrics = model.training_loss(sample)
    assert torch.isfinite(loss)
    assert metrics["loss_episode_memory_aux"] >= 0
    assert metrics["episode_anchor_fraction"] == 1.0
    loss.backward()
    updater_grad = sum(
        float(parameter.grad.abs().sum())
        for parameter in model.episode_memory.updater.parameters()
        if parameter.grad is not None
    )
    assert updater_grad > 0
    assert model.episode_memory.reader.gates["action"].grad is not None
    assert model.episode_memory.reader.gates["action"].grad.abs().sum() > 0


def test_episode_memory_scan_supports_mixed_chunk_counts_in_one_batch():
    model = _model("interleaved")
    config = EpisodeMemoryConfig(
        enabled=True,
        window_blocks=8,
        chunk_blocks=4,
        num_slots=4,
        state_dim=8,
        group_dim=2,
        updater_dim=16,
        updater_heads=4,
        reader_rank=2,
    )
    model.configure_causal_training(
        causal_mode="interleaved",
        training_exit_depths=(2,),
        history_training_mode="episode_memory_scan_bptt",
        packed_history_attention_backend="dense",
        history_window_blocks=8,
        replan_steps=REPLAN_STEPS,
        action_horizon=ACTION_HORIZON,
        num_video_frames=ACTION_HORIZON + 1,
        episode_memory_config=config,
    )

    # The first row uses H0 (zero complete chunks); the second scans one chunk.
    loss, metrics = model.training_loss(_sample((8, 12)))
    assert torch.isfinite(loss)
    assert metrics["history_blocks_mean"] == 8.0
    loss.backward()
    updater_grad = sum(
        float(parameter.grad.abs().sum())
        for parameter in model.episode_memory.updater.parameters()
        if parameter.grad is not None
    )
    assert updater_grad > 0


@pytest.mark.parametrize(
    ("mode", "expected_video_h"),
    [
        ("interleaved", True),
        ("vision_causal", False),
        ("action_aggregator", False),
    ],
)
def test_episode_memory_reader_routing_follows_causal_mode(mode, expected_video_h):
    model = _model(mode)
    config = EpisodeMemoryConfig(
        enabled=True,
        window_blocks=8,
        chunk_blocks=4,
        num_slots=4,
        state_dim=8,
        group_dim=2,
        updater_dim=16,
        updater_heads=4,
        reader_rank=2,
    )
    model.configure_causal_training(
        causal_mode=mode,
        training_exit_depths=(2,),
        history_training_mode="episode_memory_scan_bptt",
        history_window_blocks=8,
        replan_steps=REPLAN_STEPS,
        action_horizon=ACTION_HORIZON,
        num_video_frames=ACTION_HORIZON + 1,
        episode_memory_config=config,
    )
    memory = type("ToyMemory", (), {})()
    memory.episode_memory_config = config
    memory.episode_state = model.episode_memory.initial_state(1)
    assert bool(model._episode_memory_kwargs(memory, "video")) is expected_video_h
    assert bool(model._episode_memory_kwargs(memory, "action")) is True

def test_episode_memory_checkpoint_round_trip(tmp_path):
    config = EpisodeMemoryConfig(
        enabled=True,
        window_blocks=8,
        chunk_blocks=4,
        num_slots=4,
        state_dim=8,
        group_dim=2,
        updater_dim=16,
        updater_heads=4,
        reader_rank=2,
    )

    def configured_model():
        instance = _model("interleaved")
        instance.configure_causal_training(
            causal_mode="interleaved",
            training_exit_depths=(2,),
            history_training_mode="episode_memory_scan_bptt",
            history_window_blocks=8,
            replan_steps=REPLAN_STEPS,
            action_horizon=ACTION_HORIZON,
            num_video_frames=ACTION_HORIZON + 1,
            episode_memory_config=config,
        )
        return instance

    source = configured_model()
    with torch.no_grad():
        source.episode_memory.empty_state.fill_(0.25)
        source.episode_memory.reader.gates["action"][0] = 0.5
    checkpoint_path = tmp_path / "episode-memory.pt"
    source.save_checkpoint(checkpoint_path, step=7)

    restored = configured_model()
    restored.load_checkpoint(checkpoint_path)
    torch.testing.assert_close(
        restored.episode_memory.empty_state,
        source.episode_memory.empty_state,
    )
    torch.testing.assert_close(
        restored.episode_memory.reader.gates["action"],
        source.episode_memory.reader.gates["action"],
    )

    legacy_payload = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    legacy_payload["episode_memory_config"]["video_reads"] = False
    legacy_payload["episode_memory_config"]["action_reads"] = False
    legacy_path = tmp_path / "episode-memory-legacy-reader-switches.pt"
    torch.save(legacy_payload, legacy_path)
    configured_model().load_checkpoint(legacy_path)

    lora = VideoLoRAConfig(enabled=True, rank=2, alpha=2.0)
    source.configure_finetuning(
        training_strategy="episode_memory_only",
        video_lora_config=lora,
    )
    staged_path = tmp_path / "episode-memory-stage1.pt"
    source.save_checkpoint(staged_path, step=8)

    joint = configured_model()
    joint.configure_finetuning(
        training_strategy="video_lora_action_full",
        video_lora_config=lora,
    )
    joint.load_checkpoint(staged_path)
    torch.testing.assert_close(
        joint.episode_memory.empty_state,
        source.episode_memory.empty_state,
    )
