"""Unit tests for the eval-time cloud-cache controls, in particular `swap`.

Motivation (2026-11-04, docs/leapbotce_ops_log.md):

* ``shuffled`` is a mathematical no-op -- the cached ``k`` already carries RoPE
  and the denoiser feeds ``cat([cache_k, edge_k, action_k])`` into
  ``flash_attention``, whose output is a softmax-weighted sum over V and thus
  invariant to a joint (K,V) permutation.
* ``constant`` zeroes ``k``, so hundreds of slots score ``exp(0)=1`` and steal
  softmax mass -- the collapse measures attention poisoning, not information
  loss.

``swap`` replaces the cache with the one prefilled for the previous episode:
slot count, per-slot norms, RoPE positions and attention geometry all stay
those of a genuine payload, while the correspondence to the current scene (and
to its prompt, which the cloud prefill folds into K/V) is destroyed.

These tests exercise the two methods on a bare namespace -- no model weights,
no GPU, no data.
"""

from types import MethodType, SimpleNamespace

import pytest
import torch

from fastwam.models.wan22.leapbotce import CloudPlanningCache, LeapBotCE


def _stub(mode="swap"):
    ns = SimpleNamespace(
        cloud_cache_mode=mode,
        _swap_bank=None,
        _swap_used_count=0,
        _swap_warmup_count=0,
    )
    ns._apply_cache_token_ablation = MethodType(
        LeapBotCE._apply_cache_token_ablation, ns)
    ns._swap_cloud_cache = MethodType(LeapBotCE._swap_cloud_cache, ns)
    return ns


def _cache(value, seq=6, layers=2, dim=4, dtype=torch.float32):
    k = [torch.full((1, seq, dim), float(value) + layer, dtype=dtype)
         for layer in range(layers)]
    v = [torch.full((1, seq, dim), -float(value) - layer, dtype=dtype)
         for layer in range(layers)]
    context = torch.ones(1, 3, dim, dtype=dtype)
    mask = torch.ones(1, 3, dtype=torch.bool)
    return CloudPlanningCache(k, v, context, mask, seq, 2)


def test_swap_first_episode_is_warm_up_and_passes_real_cache_through():
    model = _stub()
    real = _cache(1.0)
    out = model._apply_cache_token_ablation(real)
    assert out is real
    assert model._swap_warmup_count == 1
    assert model._swap_used_count == 0


def test_swap_second_episode_receives_previous_episode_payload():
    model = _stub()
    first = _cache(1.0)
    second = _cache(2.0)

    model._apply_cache_token_ablation(first)
    out = model._apply_cache_token_ablation(second)

    for got, want in zip(out.video_cache_k, first.video_cache_k):
        assert torch.equal(got, want)
    for got, want in zip(out.video_cache_v, first.video_cache_v):
        assert torch.equal(got, want)
    # ... and NOT the current episode's own payload.
    for got, own in zip(out.video_cache_k, second.video_cache_k):
        assert not torch.equal(got, own)
    assert model._swap_used_count == 1
    assert model._swap_warmup_count == 1


def test_swap_advances_the_bank_so_each_episode_sees_the_immediately_previous_one():
    model = _stub()
    caches = [_cache(float(i)) for i in range(1, 5)]
    outs = [model._apply_cache_token_ablation(c) for c in caches]
    # outs[0] warm-up (real), outs[1] == caches[0], outs[2] == caches[1], ...
    assert outs[0] is caches[0]
    for i in range(1, 4):
        for got, want in zip(outs[i].video_cache_k, caches[i - 1].video_cache_k):
            assert torch.equal(got, want)
    assert model._swap_used_count == 3
    assert model._swap_warmup_count == 1


def test_swap_preserves_layout_norms_and_geometry():
    model = _stub()
    first = _cache(1.0, seq=6)
    model._apply_cache_token_ablation(first)
    out = model._apply_cache_token_ablation(_cache(2.0, seq=6))

    assert out.video_seq_len == 6
    assert out.video_tokens_per_frame == first.video_tokens_per_frame
    assert len(out.video_cache_k) == len(first.video_cache_k)
    for got, want in zip(out.video_cache_k, first.video_cache_k):
        assert got.shape == want.shape
        assert torch.allclose(got.norm(), want.norm())
    # The language context stays the CURRENT episode's, only K/V are swapped.
    assert out.context is not None


def test_swap_falls_back_to_warm_up_on_shape_mismatch():
    model = _stub()
    model._apply_cache_token_ablation(_cache(1.0, seq=6))
    # A codec change would alter the slot count; never swap incompatible caches.
    real = _cache(2.0, seq=12)
    out = model._apply_cache_token_ablation(real)
    assert out is real
    assert model._swap_warmup_count == 2
    assert model._swap_used_count == 0


def test_swap_does_not_mutate_the_incoming_cache():
    model = _stub()
    model._apply_cache_token_ablation(_cache(1.0))
    second = _cache(2.0)
    before = [k.clone() for k in second.video_cache_k]
    model._apply_cache_token_ablation(second)
    for now, then in zip(second.video_cache_k, before):
        assert torch.equal(now, then)


def test_swap_bank_is_detached_from_autograd():
    model = _stub()
    first = _cache(1.0)
    first.video_cache_k = [k.requires_grad_(True) for k in first.video_cache_k]
    model._apply_cache_token_ablation(first)
    out = model._apply_cache_token_ablation(_cache(2.0))
    assert all(not k.requires_grad for k in out.video_cache_k)


@pytest.mark.parametrize("mode", ["full", "constant", "shuffled"])
def test_other_modes_are_unaffected(mode):
    model = _stub(mode)
    real = _cache(1.0)
    out = model._apply_cache_token_ablation(real)
    assert model._swap_used_count == 0
    assert model._swap_warmup_count == 0
    if mode == "full":
        assert out is real
    elif mode == "constant":
        assert all(torch.count_nonzero(k) == 0 for k in out.video_cache_k)
        assert out.video_seq_len == real.video_seq_len
    else:  # shuffled keeps the multiset of slots, only the order changes
        assert out.video_seq_len == real.video_seq_len
        for got, want in zip(out.video_cache_k, real.video_cache_k):
            assert torch.equal(got.sort(dim=1).values, want.sort(dim=1).values)


def test_unknown_mode_is_rejected():
    # The mode string is validated in __init__, which needs the full model; this
    # guards the tuple that the validation reads.
    import inspect

    src = inspect.getsource(LeapBotCE.__init__)
    assert '"swap"' in src
    assert "constant/shuffled/swap" in src
