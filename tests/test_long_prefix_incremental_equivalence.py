import types

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from fastwam.models.wan22.action_dit import ActionDiT
from fastwam.models.wan22.mot import MoT
from fastwam.models.wan22.wan_video_dit import WanVideoDiT
from leapbot_va.models.leapbot import LeapBotVA
from leapbot_va.training import build_packed_history_attention_mask


HISTORY_BLOCKS = 8
REPLAN_STEPS = 1
ACTION_HORIZON = 2
LAYERS = 8


def _require_cpu_bf16() -> None:
    """Skip only when the primitive CPU operators needed here lack BF16."""

    try:
        value = torch.ones(1, 2, 4, dtype=torch.bfloat16)
        nn.Linear(4, 4).to(dtype=torch.bfloat16)(value)
        nn.Conv3d(1, 1, 1).to(dtype=torch.bfloat16)(
            torch.ones(1, 1, 1, 1, 1, dtype=torch.bfloat16)
        )
        query = torch.ones(1, 2, 2, 4, dtype=torch.bfloat16)
        F.scaled_dot_product_attention(query, query, query)
    except (RuntimeError, TypeError, NotImplementedError) as error:
        pytest.skip(f"CPU BF16 kernels required by the tiny MoT are unavailable: {error}")


def _tiny_model(causal_mode: str, dtype: torch.dtype) -> LeapBotVA:
    torch.manual_seed(1203)
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
        num_layers=LAYERS,
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
        num_layers=LAYERS,
    )
    mot = MoT(
        {"video": video, "action": action},
        mot_checkpoint_mixed_attn=False,
    )
    model = LeapBotVA(
        video_expert=video,
        action_expert=action,
        mot=mot,
        vae=nn.Identity(),
        text_dim=6,
        device="cpu",
        torch_dtype=dtype,
        exit_depths=(LAYERS,),
    ).to(dtype=dtype)
    model.configure_causal_training(
        causal_mode=causal_mode,
        training_exit_depths=(LAYERS,),
        replan_steps=REPLAN_STEPS,
        action_horizon=ACTION_HORIZON,
    )

    # Non-zero episode positions ensure equivalence is not obtained merely
    # because the checkpoint-compatible temporal extension starts as identity.
    generator = torch.Generator(device="cpu").manual_seed(991)
    with torch.no_grad():
        for projection, std in (
            (model.temporal_positions.video_projection, 0.07),
            (model.temporal_positions.action_block_projection, 0.05),
            (model.temporal_positions.action_control_projection, 0.03),
        ):
            values = torch.randn(
                projection.weight.shape,
                generator=generator,
                dtype=torch.float32,
            )
            projection.weight.copy_(values.to(dtype=dtype) * std)

    # A one-token deterministic latent keeps this a transformer/KV regression,
    # independent of heavyweight VAE weights.
    def encode_real_observation(self, input_image, tiled=False):
        del tiled
        base = torch.tensor(
            [-0.75, -0.25, 0.25, 0.75],
            device=self.device,
            dtype=self.torch_dtype,
        ).view(1, 4, 1, 1, 1)
        return base + input_image.mean().to(dtype=self.torch_dtype)

    model._encode_input_image_latents_tensor = types.MethodType(
        encode_real_observation,
        model,
    )
    return model.eval()


def _clone_runtime_call(kwargs, *, hidden=None):
    context_payload = kwargs.get("context_payload") or kwargs.get(
        "action_context_payload"
    )
    record = {
        "tokens": kwargs.get("tokens", kwargs.get("action_tokens")).detach().clone(),
        "freqs": kwargs.get("freqs", kwargs.get("action_freqs")).detach().clone(),
        "t_mod": kwargs.get("t_mod", kwargs.get("action_t_mod")).detach().clone(),
        "context": context_payload["context"].detach().clone(),
        "context_mask": context_payload["mask"].detach().clone(),
    }
    if hidden is not None:
        record["hidden"] = hidden.detach().clone()
    return record


def _packed_t_mod(records):
    expanded = []
    for record in records:
        t_mod = record["t_mod"]
        sequence = int(record["tokens"].shape[1])
        if t_mod.ndim == 3:
            t_mod = t_mod.unsqueeze(1).expand(-1, sequence, -1, -1)
        else:
            assert t_mod.ndim == 4
            assert t_mod.shape[1] == sequence
        expanded.append(t_mod)
    return torch.cat(expanded, dim=1)


def _packed_context(records):
    reference = records[0]["context"]
    for record in records[1:]:
        torch.testing.assert_close(record["context"], reference, atol=0, rtol=0)
    return {
        "context": reference,
        "mask": torch.cat([record["context_mask"] for record in records], dim=1),
    }


@pytest.mark.parametrize(
    "causal_mode",
    ["interleaved", "vision_causal", "action_aggregator"],
)
@pytest.mark.parametrize(
    "dtype",
    [torch.float32, torch.bfloat16],
    ids=["fp32", "bf16"],
)
def test_long_real_prefix_packed_matches_runtime_kv_and_prediction_is_transient(
    causal_mode,
    dtype,
    monkeypatch,
):
    """H=8 packed causal execution must equal the public runtime KV program."""

    if dtype is torch.bfloat16:
        _require_cpu_bf16()
    model = _tiny_model(causal_mode, dtype)
    memory = model.create_memory(
        exit_depth=LAYERS,
        max_history_blocks=HISTORY_BLOCKS + 1,
    )

    video_records = []
    committed_action_records = []
    transient_action_records = []
    original_prefill = model.mot.prefill_expert_segment
    original_action_forward = model.mot.forward_action_with_history

    def record_prefill(*args, **kwargs):
        result = original_prefill(*args, **kwargs)
        record = _clone_runtime_call(kwargs)
        if kwargs["expert_name"] == "video":
            video_records.append(record)
        else:
            committed_action_records.append(record)
        return result

    def record_transient_action(*args, **kwargs):
        hidden = original_action_forward(*args, **kwargs)
        transient_action_records.append(
            _clone_runtime_call(kwargs, hidden=hidden)
        )
        return hidden

    monkeypatch.setattr(model.mot, "prefill_expert_segment", record_prefill)
    monkeypatch.setattr(
        model.mot,
        "forward_action_with_history",
        record_transient_action,
    )

    context = torch.tensor(
        [[[0.2, -0.4, 0.6, -0.8, 1.0, -1.2], [0.3, 0.1, -0.2, 0.7, -0.5, 0.9]]],
        dtype=torch.float32,
    )
    context_mask = torch.ones(1, 2, dtype=torch.bool)
    executed_actions = torch.linspace(
        -0.8,
        0.9,
        HISTORY_BLOCKS * 3,
        dtype=torch.float32,
    ).view(HISTORY_BLOCKS, REPLAN_STEPS, 3)

    for block in range(HISTORY_BLOCKS):
        image = torch.full((1, 3, 16, 16), block / 16.0)
        model.infer_action(
            prompt=None,
            input_image=image,
            action_horizon=ACTION_HORIZON,
            context=context,
            context_mask=context_mask,
            num_inference_steps=1,
            seed=500 + block,
            memory=memory,
        )
        # The two predicted tokens are transient; only the single command below
        # is allowed to become the action history for this block.
        assert memory.token_counts["action"] == block * REPLAN_STEPS
        model.commit_executed_actions(memory, executed_actions[block])

    assert memory.completed_blocks == HISTORY_BLOCKS
    assert memory.token_counts == {
        "video": HISTORY_BLOCKS,
        "action": HISTORY_BLOCKS * REPLAN_STEPS,
    }
    action_tokens_before_prediction = memory.token_counts["action"]
    segment_count_before_prediction = len(memory.segments)

    prediction = model.infer_action(
        prompt=None,
        input_image=torch.full((1, 3, 16, 16), HISTORY_BLOCKS / 16.0),
        action_horizon=ACTION_HORIZON,
        context=context,
        context_mask=context_mask,
        num_inference_steps=1,
        seed=900,
        memory=memory,
    )

    assert prediction["action"].shape == (ACTION_HORIZON, 3)
    assert len(video_records) == HISTORY_BLOCKS + 1
    assert len(committed_action_records) == HISTORY_BLOCKS
    assert len(transient_action_records) == HISTORY_BLOCKS + 1
    assert len(memory.segments) == segment_count_before_prediction + 1
    assert memory.token_counts == {
        "video": HISTORY_BLOCKS + 1,
        "action": action_tokens_before_prediction,
    }
    assert memory.segments[-1].modality == "video"
    assert not any(
        segment.modality == "action"
        and segment.block_index == HISTORY_BLOCKS
        for segment in memory.segments
    )

    current_action = transient_action_records[-1]
    packed_action_records = [*committed_action_records, current_action]
    packed_mask = build_packed_history_attention_mask(
        torch.ones(1, HISTORY_BLOCKS, dtype=torch.bool),
        video_tokens_per_frame=1,
        current_video_frames=1,
        replan_steps=REPLAN_STEPS,
        action_horizon=ACTION_HORIZON,
        causal_mode=causal_mode,
    )[0]

    with torch.no_grad():
        packed_hidden = model.mot(
            embeds_all={
                "video": torch.cat(
                    [record["tokens"] for record in video_records], dim=1
                ),
                "action": torch.cat(
                    [record["tokens"] for record in packed_action_records], dim=1
                ),
            },
            attention_mask=packed_mask,
            freqs_all={
                "video": torch.cat(
                    [record["freqs"] for record in video_records], dim=0
                ),
                "action": torch.cat(
                    [record["freqs"] for record in packed_action_records], dim=0
                ),
            },
            context_all={
                "video": _packed_context(video_records),
                "action": _packed_context(packed_action_records),
            },
            t_mod_all={
                "video": _packed_t_mod(video_records),
                "action": _packed_t_mod(packed_action_records),
            },
        )["action"][:, -ACTION_HORIZON:]

    atol, rtol = ((3e-5, 3e-5) if dtype is torch.float32 else (2e-2, 2e-2))
    torch.testing.assert_close(
        packed_hidden.float(),
        current_action["hidden"].float(),
        atol=atol,
        rtol=rtol,
    )
