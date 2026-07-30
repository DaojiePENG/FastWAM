import pytest
import torch

from fastwam.models.wan22.action_dit import ActionDiT
from fastwam.models.wan22.mot import MoT
from fastwam.models.wan22.wan_video_dit import WanVideoDiT, precompute_freqs_cis
from leapbot_va.positions import HierarchicalTemporalPositionEmbedding
from leapbot_va.training import (
    build_packed_history_attention_mask,
    build_query_context_masks,
)


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


def test_masked_real_frame_kv_matches_standalone_inference_prefill():
    torch.manual_seed(19)
    video, action = _experts()
    mot = MoT({"video": video, "action": action}, mot_checkpoint_mixed_attn=False).eval()
    tokens = torch.randn(1, 3, 12)
    freqs = precompute_freqs_cis(6, end=3).view(3, 1, -1)
    t_mod = torch.randn(1, 6, 12)

    standalone_hidden, standalone_kv = mot.prefill_expert_segment(
        expert_name="video",
        tokens=tokens[:, :1],
        freqs=freqs[:1],
        t_mod=t_mod,
        context_payload=None,
    )
    segment_mask = torch.ones(3, 3, dtype=torch.bool)
    segment_mask[0, 1:] = False
    training_hidden, training_kv = mot.prefill_expert_segment(
        expert_name="video",
        tokens=tokens,
        freqs=freqs,
        t_mod=t_mod,
        context_payload=None,
        segment_attention_mask=segment_mask,
    )

    torch.testing.assert_close(
        training_hidden[:, :1], standalone_hidden, atol=1e-5, rtol=1e-5
    )
    for training_layer, standalone_layer in zip(training_kv, standalone_kv):
        torch.testing.assert_close(
            training_layer["k"][:, :1], standalone_layer["k"], atol=1e-5, rtol=1e-5
        )
        torch.testing.assert_close(
            training_layer["v"][:, :1], standalone_layer["v"], atol=1e-5, rtol=1e-5
        )


def test_consecutive_absolute_video_positions_preserve_native_relative_geometry():
    torch.manual_seed(23)
    video, action = _experts()
    mot = MoT({"video": video, "action": action}, mot_checkpoint_mixed_attn=False).eval()
    latents = torch.randn(1, 4, 3, 1, 1)
    timestep = torch.tensor([0.37])
    context = torch.randn(1, 2, 6)
    context_mask = torch.ones(1, 2, dtype=torch.bool)

    def run(frame_positions: torch.Tensor) -> torch.Tensor:
        pre = video.pre_dit(
            x=latents,
            timestep=timestep,
            context=context,
            context_mask=context_mask,
            fuse_vae_embedding_in_latents=True,
            frame_position_ids=frame_positions,
        )
        hidden, _ = mot.prefill_expert_segment(
            expert_name="video",
            tokens=pre["tokens"],
            freqs=pre["freqs"],
            t_mod=pre["t_mod"],
            context_payload={"context": pre["context"], "mask": pre["context_mask"]},
            segment_attention_mask=torch.ones(3, 3, dtype=torch.bool),
        )
        return hidden

    native = run(torch.tensor([0, 1, 2]))
    shifted = run(torch.tensor([17, 18, 19]))
    torch.testing.assert_close(shifted, native, atol=2e-5, rtol=2e-5)


def _materialize_segment_kv(segments, *, modality=None):
    selected = [
        segment
        for segment in segments
        if modality is None or segment["modality"] == modality
    ]
    if not selected:
        return None
    layers = len(selected[0]["kv"])
    return [
        {
            "k": torch.cat([segment["kv"][layer]["k"] for segment in selected], dim=1),
            "v": torch.cat([segment["kv"][layer]["v"] for segment in selected], dim=1),
        }
        for layer in range(layers)
    ]


@pytest.mark.parametrize(
    "causal_mode",
    ["interleaved", "vision_causal", "action_aggregator"],
)
def test_multiblock_packed_matches_incremental_raw_kv_with_hierarchical_positions(
    causal_mode,
    monkeypatch,
):
    """Packed causal attention and runtime KV prefill must be the same program.

    The sequence contains two fully executed history blocks and one current
    observation/action block.  It deliberately contains no future-video
    supervision.  Every block resets native RoPE, while non-zero additive
    positions carry both the shared coarse block clock and the action-specific
    fine control clock.
    """

    torch.manual_seed(211)
    history_blocks = 2
    total_blocks = history_blocks + 1
    replan_steps = 2
    action_horizon = 3
    layers = 3
    video, action = _experts(layers=layers)
    mot = MoT(
        {"video": video, "action": action},
        mot_checkpoint_mixed_attn=False,
    ).eval()
    positions = HierarchicalTemporalPositionEmbedding(
        video_dim=video.hidden_dim,
        action_dim=action.hidden_dim,
        feature_dim=8,
    )
    with torch.no_grad():
        # Exercise every hierarchical branch; this is intentionally not the
        # checkpoint-compatible zero initialization used by default.
        torch.nn.init.normal_(positions.video_projection.weight, std=0.07)
        torch.nn.init.normal_(positions.action_block_projection.weight, std=0.05)
        torch.nn.init.normal_(positions.action_control_projection.weight, std=0.03)

    raw_video = [torch.randn(1, 1, video.hidden_dim) for _ in range(total_blocks)]
    raw_action = [
        torch.randn(
            1,
            replan_steps if block < history_blocks else action_horizon,
            action.hidden_dim,
        )
        for block in range(total_blocks)
    ]
    video_segments = []
    action_segments = []
    video_freq_segments = []
    action_freq_segments = []
    for block in range(total_blocks):
        action_length = replan_steps if block < history_blocks else action_horizon
        video_segments.append(
            positions.add_video(
                raw_video[block],
                torch.tensor([block]),
                tokens_per_frame=1,
            )
        )
        control_start = block * replan_steps
        action_segments.append(
            positions.add_action(
                raw_action[block],
                torch.arange(control_start, control_start + action_length),
                torch.full((action_length,), block, dtype=torch.long),
            )
        )
        # Native RoPE explicitly restarts inside every observation/action block.
        video_freq_segments.append(
            precompute_freqs_cis(6, end=1).view(1, 1, -1)
        )
        action_freq_segments.append(
            precompute_freqs_cis(6, end=action_length).view(action_length, 1, -1)
        )

    assert not torch.equal(video_segments[1], raw_video[1])
    assert not torch.equal(action_segments[1], raw_action[1])
    torch.testing.assert_close(action_freq_segments[0], action_freq_segments[1])
    assert torch.equal(video_freq_segments[0], video_freq_segments[2])

    packed_video = torch.cat(video_segments, dim=1)
    packed_action = torch.cat(action_segments, dim=1)
    packed_video_freqs = torch.cat(video_freq_segments, dim=0)
    packed_action_freqs = torch.cat(action_freq_segments, dim=0)
    packed_mask = build_packed_history_attention_mask(
        torch.ones(1, history_blocks, dtype=torch.bool),
        video_tokens_per_frame=1,
        current_video_frames=1,
        replan_steps=replan_steps,
        action_horizon=action_horizon,
        causal_mode=causal_mode,
    )[0]
    video_t_mod = torch.zeros(1, 6, video.hidden_dim)
    action_t_mod = torch.zeros(1, 6, action.hidden_dim)
    language_video = torch.randn(1, 2, video.hidden_dim)
    language_action = torch.randn(1, 2, action.hidden_dim)
    proprio_video = torch.randn(1, total_blocks, video.hidden_dim)
    proprio_action = torch.randn(1, total_blocks, action.hidden_dim)
    packed_video_context = torch.cat([language_video, proprio_video], dim=1)
    packed_action_context = torch.cat([language_action, proprio_action], dim=1)
    packed_video_context_mask, packed_action_context_mask = build_query_context_masks(
        torch.ones(1, 2, dtype=torch.bool),
        torch.ones(1, history_blocks, dtype=torch.bool),
        video_tokens_per_frame=1,
        current_video_frames=1,
        replan_steps=replan_steps,
        action_horizon=action_horizon,
    )

    def block_context(expert_name: str, block: int) -> dict[str, torch.Tensor]:
        if expert_name == "video":
            context = torch.cat(
                [language_video, proprio_video[:, block : block + 1]], dim=1
            )
        else:
            context = torch.cat(
                [language_action, proprio_action[:, block : block + 1]], dim=1
            )
        return {
            "context": context,
            "mask": torch.ones(1, context.shape[1], dtype=torch.bool),
        }

    # Capture the exact layer K/V generated by the one-shot packed path.  The
    # public packed forward returns hidden states only.
    packed_kv = {"video": {}, "action": {}}
    incremental_current_action_kv = {}
    block_identity = {
        id(block): ("video", layer)
        for layer, block in enumerate(video.blocks)
    }
    block_identity.update(
        {
            id(block): ("action", layer)
            for layer, block in enumerate(action.blocks)
        }
    )
    capture_phase = {"name": "packed"}
    original_build_attention_io = mot._build_expert_attention_io

    def capture_attention_io(*args, **kwargs):
        result = original_build_attention_io(*args, **kwargs)
        expert_name, layer = block_identity[id(kwargs["block"])]
        if capture_phase["name"] == "packed":
            assert layer not in packed_kv[expert_name]
            packed_kv[expert_name][layer] = {
                "k": result[1].detach().clone(),
                "v": result[2].detach().clone(),
            }
        elif (
            capture_phase["name"] == "incremental_current_action"
            and expert_name == "action"
        ):
            incremental_current_action_kv[layer] = {
                "k": result[1].detach().clone(),
                "v": result[2].detach().clone(),
            }
        return result

    monkeypatch.setattr(mot, "_build_expert_attention_io", capture_attention_io)
    with torch.no_grad():
        packed_hidden = mot(
            embeds_all={"video": packed_video, "action": packed_action},
            attention_mask=packed_mask,
            freqs_all={"video": packed_video_freqs, "action": packed_action_freqs},
            context_all={
                "video": {
                    "context": packed_video_context,
                    "mask": packed_video_context_mask,
                },
                "action": {
                    "context": packed_action_context,
                    "mask": packed_action_context_mask,
                },
            },
            t_mod_all={"video": video_t_mod, "action": action_t_mod},
        )["action"][:, history_blocks * replan_steps :]

        capture_phase["name"] = "incremental_prefix"
        cached_segments = []
        for block in range(history_blocks):
            if causal_mode == "interleaved":
                video_history = _materialize_segment_kv(cached_segments)
            elif causal_mode == "vision_causal":
                video_history = _materialize_segment_kv(
                    cached_segments,
                    modality="video",
                )
            else:
                video_history = None
            _, video_kv = mot.prefill_expert_segment(
                expert_name="video",
                tokens=video_segments[block],
                freqs=video_freq_segments[block],
                t_mod=video_t_mod,
                context_payload=block_context("video", block),
                history_kv=video_history,
            )
            cached_segments.append(
                {"modality": "video", "block": block, "kv": video_kv}
            )

            _, action_kv = mot.prefill_expert_segment(
                expert_name="action",
                tokens=action_segments[block],
                freqs=action_freq_segments[block],
                t_mod=action_t_mod,
                context_payload=block_context("action", block),
                history_kv=_materialize_segment_kv(cached_segments),
            )
            cached_segments.append(
                {"modality": "action", "block": block, "kv": action_kv}
            )

        current_block = history_blocks
        if causal_mode == "interleaved":
            current_video_history = _materialize_segment_kv(cached_segments)
        elif causal_mode == "vision_causal":
            current_video_history = _materialize_segment_kv(
                cached_segments,
                modality="video",
            )
        else:
            current_video_history = None
        _, current_video_kv = mot.prefill_expert_segment(
            expert_name="video",
            tokens=video_segments[current_block],
            freqs=video_freq_segments[current_block],
            t_mod=video_t_mod,
            context_payload=block_context("video", current_block),
            history_kv=current_video_history,
        )
        cached_segments.append(
            {
                "modality": "video",
                "block": current_block,
                "kv": current_video_kv,
            }
        )

        capture_phase["name"] = "incremental_current_action"
        incremental_hidden = mot.forward_action_with_history(
            action_tokens=action_segments[current_block],
            action_freqs=action_freq_segments[current_block],
            action_t_mod=action_t_mod,
            action_context_payload=block_context("action", current_block),
            history_kv=_materialize_segment_kv(cached_segments),
        )

    torch.testing.assert_close(
        incremental_hidden,
        packed_hidden,
        atol=2e-5,
        rtol=2e-5,
    )

    # Packed expert order is [all video][all action], whereas persistent memory
    # is chronological [v0,a0,v1,a1,vcur].  Compare corresponding slices, not
    # concatenation order; attention itself is invariant to that key ordering.
    for segment in cached_segments:
        modality = segment["modality"]
        block = segment["block"]
        if modality == "video":
            token_slice = slice(block, block + 1)
        else:
            start = block * replan_steps
            token_slice = slice(start, start + replan_steps)
        for layer in range(layers):
            torch.testing.assert_close(
                segment["kv"][layer]["k"],
                packed_kv[modality][layer]["k"][:, token_slice],
                atol=2e-5,
                rtol=2e-5,
            )
            torch.testing.assert_close(
                segment["kv"][layer]["v"],
                packed_kv[modality][layer]["v"][:, token_slice],
                atol=2e-5,
                rtol=2e-5,
            )

    current_action_start = history_blocks * replan_steps
    for layer in range(layers):
        torch.testing.assert_close(
            incremental_current_action_kv[layer]["k"],
            packed_kv["action"][layer]["k"][:, current_action_start:],
            atol=2e-5,
            rtol=2e-5,
        )
        torch.testing.assert_close(
            incremental_current_action_kv[layer]["v"],
            packed_kv["action"][layer]["v"][:, current_action_start:],
            atol=2e-5,
            rtol=2e-5,
        )


@pytest.mark.parametrize(
    "causal_mode",
    ["interleaved", "vision_causal", "action_aggregator"],
)
def test_future_video_tokens_cannot_change_or_receive_gradient_from_current_action(
    causal_mode,
):
    torch.manual_seed(307)
    video, action = _experts(layers=2)
    mot = MoT(
        {"video": video, "action": action},
        mot_checkpoint_mixed_attn=False,
    ).eval()
    # Packed order is Vhist, Vcurrent-real, Vcurrent-future(2), Ahist,
    # Acurrent(3). Only the current action output is used as the objective.
    history_video = torch.randn(1, 1, video.hidden_dim)
    current_real = torch.randn(1, 1, video.hidden_dim)
    future_video = torch.randn(
        1, 2, video.hidden_dim, requires_grad=True
    )
    history_action = torch.randn(1, 2, action.hidden_dim)
    current_action = torch.randn(1, 3, action.hidden_dim)
    mask = build_packed_history_attention_mask(
        torch.ones(1, 1, dtype=torch.bool),
        video_tokens_per_frame=1,
        current_video_frames=3,
        replan_steps=2,
        action_horizon=3,
        causal_mode=causal_mode,
    )
    video_freqs = torch.cat(
        [
            precompute_freqs_cis(6, end=1).view(1, 1, -1),
            precompute_freqs_cis(6, end=3).view(3, 1, -1),
        ],
        dim=0,
    )
    action_freqs = torch.cat(
        [
            precompute_freqs_cis(6, end=2).view(2, 1, -1),
            precompute_freqs_cis(6, end=3).view(3, 1, -1),
        ],
        dim=0,
    )

    def run(future: torch.Tensor) -> torch.Tensor:
        output = mot(
            embeds_all={
                "video": torch.cat(
                    [history_video, current_real, future], dim=1
                ),
                "action": torch.cat([history_action, current_action], dim=1),
            },
            attention_mask=mask,
            freqs_all={"video": video_freqs, "action": action_freqs},
            context_all={"video": None, "action": None},
            t_mod_all={
                "video": torch.zeros(1, 6, video.hidden_dim),
                "action": torch.zeros(1, 6, action.hidden_dim),
            },
        )["action"]
        return output[:, -3:]

    action_output = run(future_video)
    future_gradient = torch.autograd.grad(
        action_output.square().sum(), future_video
    )[0]
    assert torch.count_nonzero(future_gradient) == 0

    with torch.no_grad():
        perturbed_output = run(future_video.detach() + 100.0 * torch.randn_like(future_video))
    torch.testing.assert_close(
        perturbed_output,
        action_output.detach(),
        atol=2e-5,
        rtol=2e-5,
    )


@pytest.mark.parametrize(
    "causal_mode",
    ["interleaved", "vision_causal", "action_aggregator"],
)
def test_padded_history_contents_cannot_affect_valid_current_action(causal_mode):
    torch.manual_seed(401)
    video, action = _experts(layers=2)
    mot = MoT(
        {"video": video, "action": action},
        mot_checkpoint_mixed_attn=False,
    ).eval()
    # Two allocated history slots, but slot 1 is padding. The mask must make
    # arbitrary content in Vpad/Apad observationally irrelevant.
    valid_video = torch.randn(1, 1, video.hidden_dim)
    padded_video = torch.randn(1, 1, video.hidden_dim, requires_grad=True)
    current_video = torch.randn(1, 1, video.hidden_dim)
    valid_action = torch.randn(1, 2, action.hidden_dim)
    padded_action = torch.randn(1, 2, action.hidden_dim, requires_grad=True)
    current_action = torch.randn(1, 3, action.hidden_dim)
    mask = build_packed_history_attention_mask(
        torch.tensor([[True, False]]),
        video_tokens_per_frame=1,
        current_video_frames=1,
        replan_steps=2,
        action_horizon=3,
        causal_mode=causal_mode,
    )
    video_freqs = precompute_freqs_cis(6, end=1).view(1, 1, -1).expand(3, -1, -1)
    action_freqs = torch.cat(
        [
            precompute_freqs_cis(6, end=2).view(2, 1, -1),
            precompute_freqs_cis(6, end=2).view(2, 1, -1),
            precompute_freqs_cis(6, end=3).view(3, 1, -1),
        ],
        dim=0,
    )

    def run(video_pad: torch.Tensor, action_pad: torch.Tensor) -> torch.Tensor:
        return mot(
            embeds_all={
                "video": torch.cat(
                    [valid_video, video_pad, current_video], dim=1
                ),
                "action": torch.cat(
                    [valid_action, action_pad, current_action], dim=1
                ),
            },
            attention_mask=mask,
            freqs_all={"video": video_freqs, "action": action_freqs},
            context_all={"video": None, "action": None},
            t_mod_all={
                "video": torch.zeros(1, 6, video.hidden_dim),
                "action": torch.zeros(1, 6, action.hidden_dim),
            },
        )["action"][:, -3:]

    output = run(padded_video, padded_action)
    gradients = torch.autograd.grad(
        output.square().sum(),
        (padded_video, padded_action),
    )
    assert all(torch.count_nonzero(gradient) == 0 for gradient in gradients)

    with torch.no_grad():
        perturbed = run(
            padded_video.detach() + 100.0 * torch.randn_like(padded_video),
            padded_action.detach() + 100.0 * torch.randn_like(padded_action),
        )
    torch.testing.assert_close(perturbed, output.detach(), atol=2e-5, rtol=2e-5)
