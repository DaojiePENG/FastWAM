import pytest
import torch

from leapbot_va.models.leapbot import LeapBotVA


def test_prompt_fingerprint_is_exact_for_context_and_mask():
    first = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]], dtype=torch.bfloat16)
    permuted = torch.tensor([[[2.0, 1.0], [4.0, 3.0]]], dtype=torch.bfloat16)
    mask = torch.tensor([[True, False]])

    first_fingerprint = LeapBotVA._prompt_fingerprint(None, first, mask)
    assert first_fingerprint == LeapBotVA._prompt_fingerprint(
        None, first.clone(), mask.clone()
    )
    # These contexts have identical sum and squared sum, so this specifically
    # protects against the former moment-summary collision.
    assert first_fingerprint != LeapBotVA._prompt_fingerprint(None, permuted, mask)
    assert first_fingerprint != LeapBotVA._prompt_fingerprint(
        None, first, torch.tensor([[False, True]])
    )


def test_prompt_fingerprint_rejects_ambiguous_or_incomplete_inputs():
    context = torch.zeros(1, 1, 2)
    mask = torch.ones(1, 1, dtype=torch.bool)

    with pytest.raises(ValueError, match="mutually exclusive"):
        LeapBotVA._prompt_fingerprint("task", context, mask)
    with pytest.raises(ValueError, match="both context/context_mask"):
        LeapBotVA._prompt_fingerprint(None, context, None)
    with pytest.raises(ValueError, match="both context/context_mask"):
        LeapBotVA._prompt_fingerprint(None, None, mask)

