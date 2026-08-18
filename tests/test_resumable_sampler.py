import pytest
import torch
from accelerate.data_loader import DataLoaderShard

from fastwam.utils.samplers import ResumableEpochSampler


class _Dataset:
    def __init__(self, length: int):
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> int:
        return int(index)


class _LengthDataset(_Dataset):
    def __init__(self, lengths):
        super().__init__(len(lengths))
        self.lengths = tuple(lengths)

    def sampler_grouping_lengths(self):
        return self.lengths


class _AnchoredLengthDataset(_LengthDataset):
    def __init__(self, lengths, anchor_flags):
        super().__init__(lengths)
        self.anchor_flags = tuple(anchor_flags)

    def sampler_anchor_flags(self):
        return self.anchor_flags


def test_epoch_sampler_is_deterministic_full_permutation():
    dataset = _Dataset(28523)
    sampler = ResumableEpochSampler(
        dataset=dataset,
        seed=42,
        batch_size=16,
        num_processes=8,
    )

    epoch_zero = list(sampler)
    assert len(epoch_zero) == len(dataset)
    assert sorted(epoch_zero) == list(range(len(dataset)))

    expected = torch.randperm(
        len(dataset), generator=torch.Generator(device="cpu").manual_seed(42)
    ).tolist()
    assert epoch_zero == expected
    assert epoch_zero.index(21686) == 28169

    sampler.set_epoch(1)
    epoch_one = list(sampler)
    assert sorted(epoch_one) == list(range(len(dataset)))
    assert epoch_one != epoch_zero


def test_resume_skips_only_consumed_global_batch_prefix():
    dataset = _Dataset(1000)
    sampler = ResumableEpochSampler(
        dataset=dataset,
        seed=7,
        batch_size=4,
        num_processes=8,
    )
    full_epoch = list(sampler)

    sampler.set_resume_batch_offset(3)
    resumed = list(sampler)
    assert resumed == full_epoch[3 * 4 * 8 :]
    assert set(resumed).isdisjoint(full_epoch[: 3 * 4 * 8])

    sampler.clear_resume_batch_offset()
    assert list(sampler) == full_epoch


def test_resume_offset_applies_to_nonzero_epoch():
    dataset = _Dataset(1000)
    sampler = ResumableEpochSampler(
        dataset=dataset,
        seed=7,
        batch_size=4,
        num_processes=8,
    )
    sampler.set_epoch(3)
    full_epoch = list(sampler)
    sampler.set_resume_batch_offset(5)
    assert list(sampler) == full_epoch[5 * 4 * 8 :]


def test_length_grouping_is_full_deterministic_permutation_without_curriculum():
    global_batch_size = 8
    lengths = [length for length in range(6) for _ in range(global_batch_size * 2)]
    dataset = _LengthDataset(lengths)
    sampler = ResumableEpochSampler(
        dataset=dataset,
        seed=19,
        batch_size=2,
        num_processes=4,
    )

    epoch_zero = list(sampler)
    assert sorted(epoch_zero) == list(range(len(dataset)))
    assert epoch_zero == list(sampler)
    grouped_costs = [
        {lengths[index] for index in epoch_zero[offset : offset + global_batch_size]}
        for offset in range(0, len(epoch_zero), global_batch_size)
    ]
    assert all(len(costs) == 1 for costs in grouped_costs)
    # Complete global batches are shuffled rather than presented in increasing
    # history order.
    first_costs = [next(iter(costs)) for costs in grouped_costs]
    assert first_costs != sorted(first_costs)

    sampler.set_epoch(1)
    epoch_one = list(sampler)
    assert sorted(epoch_one) == list(range(len(dataset)))
    assert epoch_one != epoch_zero


def test_length_grouping_keeps_incomplete_tail_at_end_and_resume_exact():
    lengths = [index // 3 for index in range(37)]
    dataset = _LengthDataset(lengths)
    sampler = ResumableEpochSampler(
        dataset=dataset,
        seed=23,
        batch_size=2,
        num_processes=4,
    )
    full = list(sampler)
    assert len(full) == 40
    assert set(full) == set(range(len(dataset)))

    # 37 real samples are deterministically padded to five full distributed
    # batches. The exact padding is part of the absolute epoch order, so resume
    # cannot change it.
    assert full[-3:] == full[:3]
    assert len(sampler) == 40
    sampler.set_resume_batch_offset(2)
    assert list(sampler) == full[16:]


def test_length_grouping_pads_to_complete_gradient_accumulation_windows():
    lengths = [index // 3 for index in range(37)]
    dataset = _LengthDataset(lengths)
    sampler = ResumableEpochSampler(
        dataset=dataset,
        seed=23,
        batch_size=2,
        num_processes=4,
        gradient_accumulation_steps=3,
    )

    epoch_zero = list(sampler)
    assert len(epoch_zero) == 48
    assert len(sampler) == 48
    assert set(epoch_zero) == set(range(len(dataset)))
    assert epoch_zero[-11:] == epoch_zero[:11]
    assert (len(epoch_zero) // (2 * 4)) % 3 == 0

    # Resume offsets remain global-microbatch offsets, not optimizer-step offsets.
    sampler.set_resume_batch_offset(2)
    assert list(sampler) == epoch_zero[2 * 2 * 4 :]

    sampler.clear_resume_batch_offset()
    sampler.set_epoch(1)
    epoch_one = list(sampler)
    assert len(epoch_one) == 48
    assert set(epoch_one) == set(range(len(dataset)))
    assert epoch_one != epoch_zero


def test_length_grouping_default_ga_one_preserves_existing_order_and_length():
    lengths = [index // 3 for index in range(37)]
    dataset = _LengthDataset(lengths)
    implicit = ResumableEpochSampler(
        dataset=dataset,
        seed=23,
        batch_size=2,
        num_processes=4,
    )
    explicit = ResumableEpochSampler(
        dataset=dataset,
        seed=23,
        batch_size=2,
        num_processes=4,
        gradient_accumulation_steps=1,
    )

    assert list(explicit) == list(implicit)
    assert len(explicit) == len(implicit) == 40


def test_anchor_mixing_puts_h0_and_full_history_on_every_rank_and_update():
    # Four local B5 batches form one global B20 update.  The x4-like 20/80
    # population split gives every rank exactly one H0 and four H>0 samples.
    anchor_flags = [True] * 20 + [False] * 80
    lengths = [0] * 20 + [1 + index // 16 for index in range(80)]
    dataset = _AnchoredLengthDataset(lengths, anchor_flags)
    sampler = ResumableEpochSampler(
        dataset=dataset,
        seed=31,
        batch_size=5,
        num_processes=4,
    )

    epoch = list(sampler)
    assert len(epoch) == len(sampler) == 100
    assert sorted(epoch) == list(range(100))
    for global_offset in range(0, len(epoch), 20):
        global_batch = epoch[global_offset : global_offset + 20]
        assert sum(anchor_flags[index] for index in global_batch) == 4
        for local_offset in range(0, 20, 5):
            local_batch = global_batch[local_offset : local_offset + 5]
            assert sum(anchor_flags[index] for index in local_batch) == 1
            assert any(not anchor_flags[index] for index in local_batch)


def test_anchor_mixing_is_deterministic_resumable_and_accumulation_aligned():
    anchor_flags = [True] * 23 + [False] * 94
    lengths = [0] * 23 + [1 + index // 7 for index in range(94)]
    dataset = _AnchoredLengthDataset(lengths, anchor_flags)
    sampler = ResumableEpochSampler(
        dataset=dataset,
        seed=37,
        batch_size=3,
        num_processes=4,
        gradient_accumulation_steps=2,
    )

    full = list(sampler)
    assert full == list(sampler)
    assert len(full) == len(sampler)
    assert (len(full) // (3 * 4)) % 2 == 0
    assert set(full) == set(range(len(dataset)))
    for offset in range(0, len(full), 12):
        batch = full[offset : offset + 12]
        assert any(anchor_flags[index] for index in batch)
        assert any(not anchor_flags[index] for index in batch)

    sampler.set_resume_batch_offset(3)
    assert list(sampler) == full[3 * 3 * 4 :]
    sampler.clear_resume_batch_offset()
    sampler.set_epoch(1)
    assert list(sampler) != full


def test_anchor_mixing_falls_back_for_single_sample_global_batch():
    anchor_flags = [True, True, False, False, False, False]
    dataset = _AnchoredLengthDataset([0, 0, 1, 2, 3, 4], anchor_flags)
    sampler = ResumableEpochSampler(
        dataset=dataset,
        seed=41,
        batch_size=1,
        num_processes=1,
    )

    epoch = list(sampler)
    assert len(epoch) == len(sampler) == len(dataset)
    assert sorted(epoch) == list(range(len(dataset)))
    assert epoch == list(sampler)
    assert sampler.anchor_batch_contract() is None


@pytest.mark.parametrize(
    "anchor_flags,match",
    (
        ([False] * 7, "both anchor and non-anchor"),
        ([True] * 7, "both anchor and non-anchor"),
        ([True, False], "count must match"),
    ),
)
def test_anchor_mixing_rejects_invalid_metadata(anchor_flags, match):
    dataset = _AnchoredLengthDataset([0] * 7, anchor_flags)
    with pytest.raises(ValueError, match=match):
        ResumableEpochSampler(
            dataset=dataset,
            seed=42,
            batch_size=2,
            num_processes=1,
        )


@pytest.mark.parametrize("value", (0, -1, True, 1.5, "2"))
def test_sampler_rejects_invalid_gradient_accumulation_steps(value):
    with pytest.raises(
        ValueError, match="gradient_accumulation_steps must be a positive integer"
    ):
        ResumableEpochSampler(
            dataset=_Dataset(8),
            seed=42,
            batch_size=2,
            num_processes=1,
            gradient_accumulation_steps=value,
        )


def test_accelerate_dataloader_shard_advances_sampler_epoch():
    dataset = _Dataset(17)
    sampler = ResumableEpochSampler(
        dataset=dataset,
        seed=42,
        batch_size=2,
        num_processes=1,
    )
    loader = DataLoaderShard(dataset, batch_size=2, sampler=sampler)
    first = [int(index) for batch in loader for index in batch]
    assert sampler.epoch == 0
    second = [int(index) for batch in loader for index in batch]
    assert sampler.epoch == 1
    assert first != second
