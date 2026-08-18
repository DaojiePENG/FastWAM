import pytest
import torch

from fastwam.models.wan22.action_dit import ActionDiT
from fastwam.models.wan22.mot import MoT
from fastwam.models.wan22.wan_video_dit import WanVideoDiT, precompute_freqs_cis
from leapbot_va.pch import (
    PCHLayout,
    build_pch_attention_mask,
    build_pch_context_masks,
    build_pch_dense_attention_mask,
    build_pch_validity_signature,
    encode_pch_slot_validity_metadata,
)


def _layout(valid, anchor, *, video_tokens=1, action_tokens=2):
    valid = torch.tensor(valid, dtype=torch.bool)
    batch, window = valid.shape
    positions = torch.arange(window).expand(batch, -1).clone()
    positions[~valid] = 0
    anchor_valid = torch.tensor(anchor, dtype=torch.bool)
    validity_signature = build_pch_validity_signature(
        encode_pch_slot_validity_metadata(valid, anchor_valid),
        batch_size=batch,
        window_blocks=window,
        video_tokens_per_slot=video_tokens,
        action_tokens_per_slot=action_tokens,
    )
    return PCHLayout(
        history_valid_blocks=valid,
        anchor_valid=anchor_valid,
        history_block_positions=positions,
        anchor_block_positions=torch.zeros(batch, dtype=torch.long),
        video_tokens_per_slot=video_tokens,
        action_tokens_per_slot=action_tokens,
        validity_signature=validity_signature,
    )


def test_pch_cpu_signature_matches_key_validity_bytes():
    layout = _layout(
        [[False, True], [True, True]],
        [True, False],
        video_tokens=3,
        action_tokens=2,
    )
    expected = layout.key_valid_mask.contiguous().numpy().tobytes()
    assert layout.validity_signature == expected
    compact = encode_pch_slot_validity_metadata(
        layout.history_valid_blocks, layout.anchor_valid
    )
    rows = [compact[:3], compact[3:]]
    assert build_pch_validity_signature(
        rows,
        batch_size=2,
        window_blocks=2,
        video_tokens_per_slot=3,
        action_tokens_per_slot=2,
    ) == expected


@pytest.mark.parametrize("window", [1, 4, 8, 16])
def test_pch_fixed_padding_geometry_is_not_hardcoded(window):
    layout = _layout([[False] * (window - 1) + [True]], [True])
    mask = build_pch_dense_attention_mask(layout, "interleaved")
    expected = 1 + window + 2 * window
    assert mask.shape == (1, expected, expected)
    assert layout.valid_blocks.tolist() == [2]


@pytest.mark.parametrize(
    "mode,video_reads_old_video,video_reads_old_action",
    [
        ("interleaved", True, True),
        ("vision_causal", True, False),
        ("action_aggregator", False, False),
    ],
)
def test_pch_three_causal_modes(mode, video_reads_old_video, video_reads_old_action):
    layout = _layout([[True, True]], [False], action_tokens=1)
    mask = build_pch_dense_attention_mask(layout, mode)[0]
    # Global order: anchor,V0,V1,A0,A1. Anchor is invalid in this case.
    old_video, current_video, old_action, current_action = 1, 2, 3, 4
    assert bool(mask[current_video, old_video]) is video_reads_old_video
    assert bool(mask[current_video, old_action]) is video_reads_old_action
    assert mask[current_action, old_video]
    assert mask[current_action, old_action]
    assert mask[current_action, current_video]
    assert mask[current_action, current_action]
    assert not mask[current_video, current_action]


def test_pch_mixed_h_anchor_padding_and_safe_invalid_rows():
    layout = _layout(
        [
            [False, False, False, False],
            [False, False, False, True],
            [False, False, True, True],
            [True, True, True, True],
        ],
        [False, False, True, True],
    )
    mask = build_pch_dense_attention_mask(layout, "interleaved")
    valid = layout.key_valid_mask
    for row in range(layout.batch_size):
        valid_queries = torch.nonzero(valid[row], as_tuple=False).flatten()
        invalid_keys = ~valid[row]
        if valid_queries.numel():
            assert not mask[row, valid_queries][:, invalid_keys].any()
        invalid_queries = torch.nonzero(~valid[row], as_tuple=False).flatten()
        for query in invalid_queries.tolist():
            assert mask[row, query].sum() == 1
            assert mask[row, query, query]


def test_pch_context_is_language_plus_own_proprio_only():
    layout = _layout([[False, True]], [True], video_tokens=2, action_tokens=1)
    video, action = build_pch_context_masks(
        torch.tensor([[True, False, True]]), layout
    )
    # Context: 3 language tokens, then anchor/history0/history1 proprio.
    assert video[0, 0].tolist() == [True, False, True, True, False, False]
    assert not video[0, 2].any()  # invalid history0 video slot
    assert action[0, 1].tolist() == [True, False, True, False, False, True]


def _tiny_mot():
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
        attn_head_dim=16,
        num_layers=2,
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
        attn_head_dim=16,
        num_layers=2,
    )
    return MoT({"video": video, "action": action}, mot_checkpoint_mixed_attn=False)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FlexAttention comparison needs CUDA")
@pytest.mark.parametrize("mode", ["interleaved", "vision_causal", "action_aggregator"])
def test_pch_dense_and_flex_kv_and_gradients_match(mode):
    torch.manual_seed(31)
    device = torch.device("cuda")
    mot = _tiny_mot().to(device).train()
    mot.mot_checkpoint_mixed_attn = True
    layout = _layout([[False, True], [True, True]], [True, False])
    for name, value in vars(layout).items():
        if isinstance(value, torch.Tensor):
            setattr(layout, name, value.to(device))
    video = torch.randn(2, 3, 12, device=device, requires_grad=True)
    action = torch.randn(2, 4, 10, device=device, requires_grad=True)
    kwargs = dict(
        video_tokens=video,
        action_tokens=action,
        video_freqs=precompute_freqs_cis(16, end=3).to(device).view(3, 1, -1),
        action_freqs=precompute_freqs_cis(16, end=4).to(device).view(4, 1, -1),
        video_t_mod=torch.zeros(2, 6, 12, device=device),
        action_t_mod=torch.zeros(2, 6, 10, device=device),
        video_context=None,
        action_context=None,
        layout=layout,
        max_layers=2,
    )
    dense = mot.prefill_packed_history(
        **kwargs,
        attention_mask=build_pch_attention_mask(layout, mode, "dense"),
        attention_backend="dense",
    )
    dense_loss = dense.video_kv[-1]["k"].sum() + dense.action_kv[-1]["k"].sum()
    dense_grads = torch.autograd.grad(dense_loss, (video, action), retain_graph=True)
    flex = mot.prefill_packed_history(
        **kwargs,
        attention_mask=build_pch_attention_mask(layout, mode, "flex"),
        attention_backend="flex",
    )
    flex_loss = flex.video_kv[-1]["k"].sum() + flex.action_kv[-1]["k"].sum()
    flex_grads = torch.autograd.grad(flex_loss, (video, action))
    for dense_layer, flex_layer in zip(dense.video_kv, flex.video_kv):
        torch.testing.assert_close(dense_layer["k"], flex_layer["k"], atol=1e-3, rtol=1e-2)
        torch.testing.assert_close(dense_layer["v"], flex_layer["v"], atol=1e-3, rtol=1e-2)
    for dense_grad, flex_grad in zip(dense_grads, flex_grads):
        torch.testing.assert_close(dense_grad, flex_grad, atol=2e-3, rtol=3e-2)
