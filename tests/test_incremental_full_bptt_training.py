from __future__ import annotations

from collections import Counter

import pytest
import torch
from torch import nn

from fastwam.models.wan22.action_dit import ActionDiT
from fastwam.models.wan22.mot import MoT
from fastwam.models.wan22.wan_video_dit import WanVideoDiT
from leapbot_va.models.leapbot import LeapBotVA
from leapbot_va.training import _use_segment_activation_checkpointing
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
def test_action_loss_reaches_every_real_history_segment_and_not_future_video(
    causal_mode,
    monkeypatch,
):
    """ActionDiT uses every real V/A block while future supervision is transient."""

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

    # Calls are history V0, history V1, current real V, then future-video V.
    # The Action prediction was completed before the future token existed, so
    # the latter must not be an ancestor in its autograd graph.
    assert len(records["video"]) == 4
    assert len(records["action"]) == 3
    future_video_tokens = records["video"][-1]
    future_to_action = torch.autograd.grad(
        action_predictions[-1].sum(),
        future_video_tokens,
        allow_unused=True,
        retain_graph=True,
    )[0]
    assert future_to_action is None

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
    assert not _has_nonzero_gradient(future_video_tokens)


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
    # Each valid history contributes V+A; each episode also contributes current
    # real V and transient future V. The invalid second slot of episode 1 adds 0.
    # Video and Action experts apply different context projections, so the
    # numeric marker differs by expert. Counts still identify both episodes:
    # H=2 emits 4 video + 2 committed-action prefills; H=1 emits 3 + 1.
    assert sorted(prefill_by_expert_and_episode["video"].values()) == [3, 4]
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


@pytest.mark.parametrize("causal_mode", CAUSAL_MODES)
def test_incremental_training_action_is_bitwise_public_runtime(causal_mode):
    """The validator must compare the real training and public runtime paths."""

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
    assert result["sequence"]["action_path_isomorphic"]
    assert result["prefix_kv"]["bitwise_equal"]
    assert result["hidden"]["bitwise_equal"]
    assert result["head"]["bitwise_equal"]
    assert result["bitwise_pass"]


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
