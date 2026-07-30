import torch

from fastwam.utils.samplers import ResumableEpochSampler


class _Dataset:
    def __init__(self, length: int):
        self.length = length

    def __len__(self) -> int:
        return self.length


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
