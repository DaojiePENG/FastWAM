from collections.abc import Sequence
from typing import Iterator, Sized

import torch
from torch.utils.data import Sampler


class ResumableEpochSampler(Sampler[int]):
    LENGTH_BUCKET_GLOBAL_BATCHES = 32

    def __init__(self, dataset: Sized, seed: int, batch_size: int, num_processes: int):
        self.dataset = dataset
        self.seed = int(seed)
        self.batch_size = int(batch_size)
        self.num_processes = int(num_processes)
        self.epoch = 0
        self.epoch_offset = 0
        self.resume_batch_offset = 0
        self._grouping_lengths = self._read_grouping_lengths(dataset)

    @staticmethod
    def _read_grouping_lengths(dataset: Sized) -> tuple[int, ...] | None:
        """Read an optional deterministic per-sample compute-cost hint.

        Variable-length causal prefixes create severe distributed stragglers when
        unrelated lengths land in the same global micro-batch.  LeapBot datasets
        expose ``sampler_grouping_lengths()`` so the sampler can place similarly
        sized prefixes together while preserving a full random permutation of
        the real dataset.  The unavoidable distributed tail padding is made
        explicit and deterministic so resume is exact. Ordinary FastWAM datasets
        do not expose the hook and retain the original behavior exactly.
        """

        getter = getattr(dataset, "sampler_grouping_lengths", None)
        if getter is None:
            return None
        if not callable(getter):
            raise TypeError("dataset.sampler_grouping_lengths must be callable")
        values = getter()
        if values is None:
            return None
        if not isinstance(values, Sequence):
            raise TypeError("sampler_grouping_lengths() must return a sequence or None")
        if len(values) != len(dataset):
            raise ValueError(
                "sampler grouping length count must match dataset length: "
                f"{len(values)} != {len(dataset)}"
            )
        normalized = tuple(int(value) for value in values)
        if any(value < 0 for value in normalized):
            raise ValueError("sampler grouping lengths must be non-negative")
        return normalized

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def set_epoch_offset(self, epoch_offset: int):
        self.epoch_offset = int(epoch_offset)

    def set_resume_batch_offset(self, batch_in_epoch: int):
        self.resume_batch_offset = int(batch_in_epoch)

    def clear_resume_batch_offset(self):
        self.resume_batch_offset = 0

    def _epoch_indices(self, generator: torch.Generator) -> list[int]:
        indices = torch.randperm(len(self.dataset), generator=generator).tolist()
        if self._grouping_lengths is None:
            return indices

        # Start from a full random permutation, then sort only within moderately
        # sized mega-buckets.  This preserves stochastic mixing while placing
        # similarly expensive prefixes in the same distributed micro-batch.
        # Complete global batches are shuffled once more to avoid a length
        # curriculum.  The only incomplete tail remains at the physical end,
        # where DataLoader/Accelerate expect it.
        global_batch_size = self.batch_size * self.num_processes
        mega_bucket_size = global_batch_size * self.LENGTH_BUCKET_GLOBAL_BATCHES
        complete_chunks: list[list[int]] = []
        tail: list[int] = []
        for mega_offset in range(0, len(indices), mega_bucket_size):
            mega_bucket = indices[mega_offset : mega_offset + mega_bucket_size]
            mega_bucket.sort(key=self._grouping_lengths.__getitem__)
            complete_count = len(mega_bucket) // global_batch_size
            for offset in range(0, complete_count * global_batch_size, global_batch_size):
                chunk = mega_bucket[offset : offset + global_batch_size]
                # Accelerate gives each rank one consecutive local batch.  Round
                # robin assignment keeps every rank's local maximum similar.
                chunk.sort(key=self._grouping_lengths.__getitem__, reverse=True)
                rank_bins = [[] for _ in range(self.num_processes)]
                for item_offset, index in enumerate(chunk):
                    rank_bins[item_offset % self.num_processes].append(index)
                complete_chunks.append(
                    [index for rank_bin in rank_bins for index in rank_bin]
                )
            remainder = mega_bucket[complete_count * global_batch_size :]
            if remainder:
                if tail:
                    raise RuntimeError("only the final length mega-bucket may have a tail")
                tail = remainder
        if complete_chunks:
            chunk_order = torch.randperm(
                len(complete_chunks), generator=generator
            ).tolist()
            ordered = [
                index
                for chunk_index in chunk_order
                for index in complete_chunks[chunk_index]
            ]
        else:
            ordered = []
        ordered.extend(tail)
        remainder = len(ordered) % global_batch_size
        if remainder:
            # Accelerate otherwise pads its distributed tail from the beginning
            # of the *post-resume* iterator, which changes duplicated samples
            # after a restart.  Padding the absolute epoch order here makes an
            # uninterrupted and resumed run consume exactly the same indices.
            needed = global_batch_size - remainder
            repeats = (needed + len(ordered) - 1) // len(ordered)
            ordered.extend((ordered * repeats)[:needed])
        return ordered

    def __iter__(self) -> Iterator[int]:
        g = torch.Generator(device="cpu")
        g.manual_seed(self.seed + self.epoch + self.epoch_offset)
        indices = self._epoch_indices(g)
        if self.resume_batch_offset > 0:
            sample_offset = self.resume_batch_offset * self.batch_size * self.num_processes
            indices = indices[sample_offset:]
        return iter(indices)

    def __len__(self) -> int:
        if self._grouping_lengths is None:
            return len(self.dataset)
        global_batch_size = self.batch_size * self.num_processes
        return (
            (len(self.dataset) + global_batch_size - 1) // global_batch_size
        ) * global_batch_size
