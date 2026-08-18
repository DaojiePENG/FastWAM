import torch
from torch import nn

from fastwam.models.wan22.wan_video_vae import WanVideoVAE
from leapbot_va.training import encode_independent_history_video_latents


class _FakeWanVAE(WanVideoVAE):
    """Exercise the real Wan fast-path without constructing its large network."""

    def __init__(self):
        nn.Module.__init__(self)
        self.calls = []

    def single_encode(self, videos, device):
        assert device == torch.device("cpu")
        assert videos.ndim == 5
        assert videos.shape[2] == 1
        identifiers = videos[:, 0, 0, 0, 0]
        self.calls.append(
            {
                "shape": tuple(videos.shape),
                "identifiers": identifiers.tolist(),
            }
        )
        return torch.stack([identifiers, -identifiers], dim=1).view(
            videos.shape[0], 2, 1, 1, 1
        )


class _FakeModel:
    def __init__(self):
        self.device = torch.device("cpu")
        self.vae = _FakeWanVAE()

    def _encode_video_latents(self, videos, tiled=False):
        raise AssertionError(
            "non-tiled Wan history must use the batch-capable single_encode path"
        )


def _history_with_identifiers(batch, history):
    video = torch.zeros(batch, 1, history, 2, 2)
    for batch_index in range(batch):
        for history_index in range(history):
            video[batch_index, :, history_index].fill_(
                1 + batch_index * 100 + history_index
            )
    return video


def test_history_vae_batches_only_valid_t1_frames_and_restores_bch_order():
    model = _FakeModel()
    history = _history_with_identifiers(batch=4, history=4)
    valid = torch.tensor(
        [
            [True, True, True, True],
            [True, True, False, False],
            [True, True, True, False],
            [False, False, False, False],
        ]
    )
    reference = torch.empty(4, 2, 3, 1, 1)

    latents = encode_independent_history_video_latents(
        model,
        history,
        valid,
        empty_latent_reference=reference,
        chunk_size=3,
    )

    assert latents.shape == (4, 2, 4, 1, 1)
    assert [call["shape"] for call in model.vae.calls] == [
        (3, 1, 1, 2, 2),
        (3, 1, 1, 2, 2),
        (3, 1, 1, 2, 2),
    ]
    # Row-major (B,H) order is preserved even when a chunk crosses a batch
    # boundary: the second call ends B0 and continues with B1.
    assert [call["identifiers"] for call in model.vae.calls] == [
        [1.0, 2.0, 3.0],
        [4.0, 101.0, 102.0],
        [201.0, 202.0, 203.0],
    ]
    expected = torch.zeros(4, 2, 4, 1, 1)
    for batch_index, history_index in torch.nonzero(valid).tolist():
        identifier = 1 + batch_index * 100 + history_index
        expected[batch_index, :, history_index, 0, 0] = torch.tensor(
            [identifier, -identifier]
        )
    torch.testing.assert_close(latents, expected)


def test_history_vae_chunk_one_and_four_are_semantically_identical():
    history = _history_with_identifiers(batch=4, history=4)
    valid = torch.tensor([
        [True, True, True, True],
        [False, True, True, True],
        [False, False, True, True],
        [False, False, False, True],
    ])
    outputs = []
    call_batches = []
    for chunk_size in (1, 4):
        model = _FakeModel()
        outputs.append(encode_independent_history_video_latents(
            model,
            history,
            valid,
            empty_latent_reference=torch.empty(4, 2, 1, 1, 1),
            chunk_size=chunk_size,
        ))
        call_batches.append([call["shape"][0] for call in model.vae.calls])
    torch.testing.assert_close(outputs[0], outputs[1])
    assert call_batches == [[1] * 10, [4, 4, 2]]


def test_pch_b4_w8_chunk_four_reduces_independent_vae_calls_to_ten():
    model = _FakeModel()
    history = _history_with_identifiers(batch=4, history=8)
    valid = torch.ones(4, 8, dtype=torch.bool)
    reference = torch.empty(4, 2, 1, 1, 1)
    encode_independent_history_video_latents(model, history, valid, empty_latent_reference=reference, chunk_size=4)
    anchor = _history_with_identifiers(batch=4, history=1)
    encode_independent_history_video_latents(model, anchor, torch.ones(4, 1, dtype=torch.bool), empty_latent_reference=reference, chunk_size=4)
    encode_independent_history_video_latents(model, anchor, torch.ones(4, 1, dtype=torch.bool), empty_latent_reference=reference, chunk_size=4)
    assert len(model.vae.calls) == 10
    assert all(call["shape"][2] == 1 for call in model.vae.calls)


def test_history_vae_chunk_tail_is_bounded_and_keeps_t_equal_to_one():
    model = _FakeModel()
    history = _history_with_identifiers(batch=2, history=5)
    valid = torch.tensor(
        [
            [True, True, True, True, True],
            [True, True, True, False, False],
        ]
    )

    encode_independent_history_video_latents(
        model,
        history,
        valid,
        empty_latent_reference=torch.empty(2, 2, 1, 1, 1),
        chunk_size=3,
    )

    assert [call["shape"][0] for call in model.vae.calls] == [3, 3, 2]
    assert all(call["shape"][2] == 1 for call in model.vae.calls)


def test_history_vae_h0_does_not_call_encoder():
    model = _FakeModel()
    reference = torch.empty(2, 2, 3, 1, 1)

    latents = encode_independent_history_video_latents(
        model,
        torch.empty(2, 1, 0, 2, 2),
        torch.empty(2, 0, dtype=torch.bool),
        empty_latent_reference=reference,
        chunk_size=2,
    )

    assert latents.shape == (2, 2, 0, 1, 1)
    assert model.vae.calls == []
