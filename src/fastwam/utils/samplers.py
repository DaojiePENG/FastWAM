from collections.abc import Sequence
from typing import Iterator, Sized

import torch
from torch.utils.data import Sampler


class ResumableEpochSampler(Sampler[int]):
    LENGTH_BUCKET_GLOBAL_BATCHES = 32

    def __init__(
        self,
        dataset: Sized,
        seed: int,
        batch_size: int,
        num_processes: int,
        gradient_accumulation_steps: int = 1,
    ):
        if (
            isinstance(gradient_accumulation_steps, bool)
            or not isinstance(gradient_accumulation_steps, int)
            or gradient_accumulation_steps <= 0
        ):
            raise ValueError(
                "gradient_accumulation_steps must be a positive integer"
            )
        self.dataset = dataset
        self.seed = int(seed)
        self.batch_size = int(batch_size)
        self.num_processes = int(num_processes)
        self.gradient_accumulation_steps = int(gradient_accumulation_steps)
        self.epoch = 0
        self.epoch_offset = 0
        self.resume_batch_offset = 0
        self._grouping_lengths = self._read_grouping_lengths(dataset)
        self._anchor_flags = self._read_anchor_flags(dataset)

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

    @staticmethod
    def _read_anchor_flags(dataset: Sized) -> tuple[bool, ...] | None:
        """Read an optional optimizer-step anchor classification.

        LeapBot exposes genuine episode-start (H0) samples as anchors.  When
        present, every distributed micro-batch contains the dataset-wide anchor
        proportion instead of placing all H0 examples into length-homogeneous
        batches.  With gradient accumulation this is stronger than the required
        optimizer-step guarantee because every constituent micro-batch is mixed.
        """

        getter = getattr(dataset, "sampler_anchor_flags", None)
        if getter is None:
            return None
        if not callable(getter):
            raise TypeError("dataset.sampler_anchor_flags must be callable")
        values = getter()
        if values is None:
            return None
        if not isinstance(values, Sequence):
            raise TypeError("sampler_anchor_flags() must return a sequence or None")
        if len(values) != len(dataset):
            raise ValueError(
                "sampler anchor flag count must match dataset length: "
                f"{len(values)} != {len(dataset)}"
            )
        normalized = tuple(bool(value) for value in values)
        anchor_count = sum(normalized)
        if anchor_count == 0 or anchor_count == len(normalized):
            raise ValueError(
                "sampler anchor flags must contain both anchor and non-anchor samples"
            )
        return normalized

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def set_epoch_offset(self, epoch_offset: int):
        self.epoch_offset = int(epoch_offset)

    def set_resume_batch_offset(self, batch_in_epoch: int):
        self.resume_batch_offset = int(batch_in_epoch)

    def clear_resume_batch_offset(self):
        self.resume_batch_offset = 0

    def anchor_batch_contract(self) -> dict[str, int | float] | None:
        """Return the deterministic H0/global-batch composition, if enabled."""

        if self._anchor_flags is None:
            return None
        global_batch_size = self.batch_size * self.num_processes
        anchor_count = sum(self._anchor_flags)
        anchor_per_global_batch = int(
            round(global_batch_size * anchor_count / len(self._anchor_flags))
        )
        anchor_per_global_batch = min(
            global_batch_size - 1,
            max(1, anchor_per_global_batch),
        )
        return {
            "dataset_anchor_count": anchor_count,
            "dataset_non_anchor_count": len(self._anchor_flags) - anchor_count,
            "anchor_per_global_micro_batch": anchor_per_global_batch,
            "non_anchor_per_global_micro_batch": (
                global_batch_size - anchor_per_global_batch
            ),
            "effective_anchor_fraction": (
                anchor_per_global_batch / global_batch_size
            ),
        }

    @staticmethod
    def _cyclic_slice(values: list[int], offset: int, count: int) -> list[int]:
        if not values:
            raise ValueError("cannot draw from an empty sampler pool")
        return [values[(offset + index) % len(values)] for index in range(count)]

    def _group_non_anchor_indices(
        self,
        indices: list[int],
        *,
        global_non_anchor_batch_size: int,
        generator: torch.Generator,
    ) -> list[int]:
        """Group only the expensive H>0 pool while preserving random batches."""

        if self._grouping_lengths is None:
            return indices
        mega_bucket_size = (
            global_non_anchor_batch_size * self.LENGTH_BUCKET_GLOBAL_BATCHES
        )
        chunks: list[list[int]] = []
        tail: list[int] = []
        for mega_offset in range(0, len(indices), mega_bucket_size):
            mega_bucket = indices[mega_offset : mega_offset + mega_bucket_size]
            mega_bucket.sort(key=self._grouping_lengths.__getitem__)
            complete_count = len(mega_bucket) // global_non_anchor_batch_size
            for offset in range(
                0,
                complete_count * global_non_anchor_batch_size,
                global_non_anchor_batch_size,
            ):
                chunks.append(
                    mega_bucket[offset : offset + global_non_anchor_batch_size]
                )
            remainder = mega_bucket[
                complete_count * global_non_anchor_batch_size :
            ]
            if remainder:
                if tail:
                    raise RuntimeError(
                        "only the final non-anchor length mega-bucket may have a tail"
                    )
                tail = remainder
        if tail:
            # Repeat only from the same sorted tail.  Letting a short tail wrap
            # into the first randomly shuffled chunk would create one accidental
            # H-small/H-large global batch and a severe distributed straggler.
            padded_tail = [
                tail[index % len(tail)]
                for index in range(global_non_anchor_batch_size)
            ]
            chunks.append(padded_tail)
            tail = []
        if chunks:
            chunk_order = torch.randperm(
                len(chunks), generator=generator
            ).tolist()
            ordered = [
                index
                for chunk_index in chunk_order
                for index in chunks[chunk_index]
            ]
        else:
            ordered = []
        ordered.extend(tail)
        return ordered

    def _anchored_epoch_indices(self, generator: torch.Generator) -> list[int]:
        """Build full distributed batches containing both H0 and H>0 samples."""

        if self._anchor_flags is None:
            raise RuntimeError("anchored epoch requested without anchor metadata")
        permutation = torch.randperm(
            len(self.dataset), generator=generator
        ).tolist()
        anchors = [index for index in permutation if self._anchor_flags[index]]
        non_anchors = [
            index for index in permutation if not self._anchor_flags[index]
        ]
        global_batch_size = self.batch_size * self.num_processes
        anchor_per_global_batch = int(
            round(global_batch_size * len(anchors) / len(permutation))
        )
        anchor_per_global_batch = min(
            global_batch_size - 1,
            max(1, anchor_per_global_batch),
        )
        non_anchor_per_global_batch = (
            global_batch_size - anchor_per_global_batch
        )
        non_anchors = self._group_non_anchor_indices(
            non_anchors,
            global_non_anchor_batch_size=non_anchor_per_global_batch,
            generator=generator,
        )
        num_global_batches = max(
            (len(anchors) + anchor_per_global_batch - 1)
            // anchor_per_global_batch,
            (len(non_anchors) + non_anchor_per_global_batch - 1)
            // non_anchor_per_global_batch,
        )
        accumulation_remainder = (
            num_global_batches % self.gradient_accumulation_steps
        )
        if accumulation_remainder:
            num_global_batches += (
                self.gradient_accumulation_steps - accumulation_remainder
            )

        ordered: list[int] = []
        anchor_offset = 0
        non_anchor_offset = 0
        for batch_index in range(num_global_batches):
            batch_anchors = self._cyclic_slice(
                anchors, anchor_offset, anchor_per_global_batch
            )
            batch_non_anchors = self._cyclic_slice(
                non_anchors,
                non_anchor_offset,
                non_anchor_per_global_batch,
            )
            anchor_offset += anchor_per_global_batch
            non_anchor_offset += non_anchor_per_global_batch

            base_anchor_count, extra_anchor_ranks = divmod(
                anchor_per_global_batch, self.num_processes
            )
            rank_anchor_counts = [base_anchor_count] * self.num_processes
            for extra_index in range(extra_anchor_ranks):
                rank_anchor_counts[
                    (batch_index + extra_index) % self.num_processes
                ] += 1
            rank_non_anchor_capacities = [
                self.batch_size - count for count in rank_anchor_counts
            ]
            if any(capacity <= 0 for capacity in rank_non_anchor_capacities):
                raise RuntimeError(
                    "anchor allocation left a rank without a non-anchor sample"
                )

            rank_anchor_bins: list[list[int]] = [
                [] for _ in range(self.num_processes)
            ]
            anchor_cursor = 0
            for rank, count in enumerate(rank_anchor_counts):
                rank_anchor_bins[rank].extend(
                    batch_anchors[anchor_cursor : anchor_cursor + count]
                )
                anchor_cursor += count

            if self._grouping_lengths is not None:
                batch_non_anchors.sort(
                    key=self._grouping_lengths.__getitem__, reverse=True
                )
            rank_non_anchor_bins: list[list[int]] = [
                [] for _ in range(self.num_processes)
            ]
            rank_cursor = batch_index % self.num_processes
            for index in batch_non_anchors:
                for _ in range(self.num_processes):
                    rank = rank_cursor % self.num_processes
                    rank_cursor += 1
                    if (
                        len(rank_non_anchor_bins[rank])
                        < rank_non_anchor_capacities[rank]
                    ):
                        rank_non_anchor_bins[rank].append(index)
                        break
                else:
                    raise RuntimeError("non-anchor rank allocation overflowed")

            for rank in range(self.num_processes):
                local_batch = [
                    *rank_anchor_bins[rank],
                    *rank_non_anchor_bins[rank],
                ]
                if len(local_batch) != self.batch_size:
                    raise RuntimeError(
                        "anchored sampler produced an incomplete local batch: "
                        f"rank={rank} size={len(local_batch)} expected={self.batch_size}"
                    )
                ordered.extend(local_batch)
        return ordered

    def _epoch_indices(self, generator: torch.Generator) -> list[int]:
        if self._anchor_flags is not None:
            return self._anchored_epoch_indices(generator)
        indices = torch.randperm(len(self.dataset), generator=generator).tolist()
        if self._grouping_lengths is None:
            return indices

        # Start from a full random permutation, then sort only within moderately
        # sized mega-buckets.  This preserves stochastic mixing while placing
        # similarly expensive prefixes in the same distributed micro-batch.
        # Complete global batches are shuffled once more to avoid a length
        # curriculum.  The only incomplete tail remains at the physical end,
        # where DataLoader/Accelerate expect it.
        global_micro_batch_size = self.batch_size * self.num_processes
        mega_bucket_size = (
            global_micro_batch_size * self.LENGTH_BUCKET_GLOBAL_BATCHES
        )
        complete_chunks: list[list[int]] = []
        tail: list[int] = []
        for mega_offset in range(0, len(indices), mega_bucket_size):
            mega_bucket = indices[mega_offset : mega_offset + mega_bucket_size]
            mega_bucket.sort(key=self._grouping_lengths.__getitem__)
            complete_count = len(mega_bucket) // global_micro_batch_size
            for offset in range(
                0,
                complete_count * global_micro_batch_size,
                global_micro_batch_size,
            ):
                chunk = mega_bucket[offset : offset + global_micro_batch_size]
                # Accelerate gives each rank one consecutive local batch.  Round
                # robin assignment keeps every rank's local maximum similar.
                chunk.sort(key=self._grouping_lengths.__getitem__, reverse=True)
                rank_bins = [[] for _ in range(self.num_processes)]
                for item_offset, index in enumerate(chunk):
                    rank_bins[item_offset % self.num_processes].append(index)
                complete_chunks.append(
                    [index for rank_bin in rank_bins for index in rank_bin]
                )
            remainder = mega_bucket[complete_count * global_micro_batch_size :]
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
        global_optimizer_batch_size = (
            global_micro_batch_size * self.gradient_accumulation_steps
        )
        remainder = len(ordered) % global_optimizer_batch_size
        if remainder:
            # Accelerate synchronizes gradients at the dataloader boundary even
            # when a full accumulation window has not completed. Padding the
            # absolute epoch order to an optimizer-batch boundary keeps every
            # update the declared size and makes resumed duplication exact.
            needed = global_optimizer_batch_size - remainder
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
        if self._anchor_flags is not None:
            global_batch_size = self.batch_size * self.num_processes
            anchor_count = sum(self._anchor_flags)
            anchor_per_global_batch = int(
                round(global_batch_size * anchor_count / len(self._anchor_flags))
            )
            anchor_per_global_batch = min(
                global_batch_size - 1,
                max(1, anchor_per_global_batch),
            )
            non_anchor_count = len(self._anchor_flags) - anchor_count
            non_anchor_per_global_batch = (
                global_batch_size - anchor_per_global_batch
            )
            num_global_batches = max(
                (anchor_count + anchor_per_global_batch - 1)
                // anchor_per_global_batch,
                (non_anchor_count + non_anchor_per_global_batch - 1)
                // non_anchor_per_global_batch,
            )
            accumulation_remainder = (
                num_global_batches % self.gradient_accumulation_steps
            )
            if accumulation_remainder:
                num_global_batches += (
                    self.gradient_accumulation_steps - accumulation_remainder
                )
            return num_global_batches * global_batch_size
        if self._grouping_lengths is None:
            return len(self.dataset)
        global_optimizer_batch_size = (
            self.batch_size
            * self.num_processes
            * self.gradient_accumulation_steps
        )
        return (
            (len(self.dataset) + global_optimizer_batch_size - 1)
            // global_optimizer_batch_size
        ) * global_optimizer_batch_size
