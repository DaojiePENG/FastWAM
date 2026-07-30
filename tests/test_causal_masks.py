import pytest

from leapbot_va.memory import build_block_causal_mask


@pytest.mark.parametrize(
    ("mode", "expected_video_row"),
    [
        ("interleaved", [True, True, True, False]),
        ("vision_causal", [True, False, True, False]),
        ("action_aggregator", [False, False, True, False]),
    ],
)
def test_three_causal_ablation_masks(mode, expected_video_row):
    # Keys: V0, A0, V1, A2(future).  The V1 query exposes the exact
    # difference among the three ablations.
    mask = build_block_causal_mask(
        query_modalities=["video", "action"],
        query_blocks=[1, 1],
        key_modalities=["video", "action", "video", "action"],
        key_blocks=[0, 0, 1, 2],
        mode=mode,
    )
    assert mask[0].tolist() == expected_video_row
    assert mask[1].tolist() == [True, True, True, False]


def test_future_information_never_leaks():
    mask = build_block_causal_mask(
        query_modalities=["action"],
        query_blocks=[3],
        key_modalities=["video", "action", "video"],
        key_blocks=[2, 3, 4],
        mode="interleaved",
    )
    assert mask.tolist() == [[True, True, False]]


def test_visual_query_never_reads_action_from_its_own_block():
    for mode in ("interleaved", "vision_causal", "action_aggregator"):
        mask = build_block_causal_mask(
            query_modalities=["video"],
            query_blocks=[2],
            key_modalities=["video", "action"],
            key_blocks=[2, 2],
            mode=mode,
        )
        assert mask.tolist() == [[True, False]]


@pytest.mark.parametrize("mode", ["interleaved", "vision_causal", "action_aggregator"])
def test_same_block_future_video_is_invisible_to_real_and_action_queries(mode):
    mask = build_block_causal_mask(
        query_modalities=["video", "action", "video"],
        query_blocks=[2, 2, 2],
        query_is_future_video=[False, False, True],
        key_modalities=["video", "video", "action"],
        key_blocks=[2, 2, 2],
        key_is_future_video=[False, True, False],
        mode=mode,
    )
    # Current real V reads only the real V; current A reads real V and its own
    # bidirectional action block; transient future V reads real/future V only.
    assert mask.tolist() == [
        [True, False, False],
        [True, False, True],
        [True, True, False],
    ]


def test_future_video_flags_reject_action_tokens():
    with pytest.raises(ValueError, match="only video keys"):
        build_block_causal_mask(
            query_modalities=["action"],
            query_blocks=[0],
            key_modalities=["action"],
            key_blocks=[0],
            key_is_future_video=[True],
            mode="interleaved",
        )
