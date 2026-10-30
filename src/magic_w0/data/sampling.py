"""Weighted sample-level mixing with deterministic DDP sharding."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from itertools import islice

import torch
from torch.utils.data import BatchSampler, Sampler


class EpochBatchSampler(BatchSampler):
    """BatchSampler with deterministic epoch and one-shot batch-offset control.

    ``set_start_batch`` is used only when restoring a checkpoint.  It skips
    batches in the sampler's deterministic index stream, not batches from the
    DataLoader, so resuming never decodes/collates the samples before the saved
    cursor.  The offset is consumed by the next ``__iter__`` call and then reset;
    subsequent passes start normally at batch zero.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._start_batch = 0

    def set_epoch(self, epoch: int) -> None:
        self._start_batch = 0
        if hasattr(self.sampler, "set_epoch"):
            self.sampler.set_epoch(epoch)

    def set_start_batch(self, batch_offset: int) -> None:
        batch_offset = int(batch_offset)
        if batch_offset < 0 or batch_offset > len(self):
            raise ValueError(
                f"batch_offset must be in [0, {len(self)}], got {batch_offset}"
            )
        self._start_batch = batch_offset

    def resume_contract(self) -> dict:
        sampler = self.sampler
        return {
            "batch_size": self.batch_size,
            "drop_last": self.drop_last,
            "sampler_type": type(sampler).__qualname__,
            "num_replicas": getattr(sampler, "num_replicas", None),
            "rank": getattr(sampler, "rank", None),
            "seed": getattr(sampler, "seed", None),
            "drop_last_sampler": getattr(sampler, "drop_last", None),
        }

    def __iter__(self):
        start = self._start_batch
        self._start_batch = 0
        return islice(super().__iter__(), start, None)


class WeightedMixtureSampler(Sampler[int]):
    """Draw individual samples from concatenated sources according to weights.

    A shared global draw is generated on every DDP rank, then strided by rank.
    This keeps runs deterministic while giving each rank a different portion of
    the draw. Sampling uses replacement so source probability is independent of
    source size; an EgoDex/RoboDojo weight pair of ``[1, 1]`` therefore means a
    50/50 sample mix rather than a frame-count-proportional mix.
    """

    def __init__(
        self,
        source_lengths: Sequence[int],
        source_weights: Sequence[float],
        *,
        num_samples: int,
        num_replicas: int = 1,
        rank: int = 0,
        seed: int = 0,
    ) -> None:
        if not source_lengths or len(source_lengths) != len(source_weights):
            raise ValueError(
                "source_lengths and source_weights must be non-empty and aligned"
            )
        if any(int(length) <= 0 for length in source_lengths):
            raise ValueError("every mixed source must contain at least one sample")
        if any(float(weight) <= 0 for weight in source_weights):
            raise ValueError("every mixed source weight must be positive")
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        if num_replicas <= 0 or not 0 <= rank < num_replicas:
            raise ValueError("rank must satisfy 0 <= rank < num_replicas")

        self.source_lengths = tuple(int(length) for length in source_lengths)
        self.source_weights = torch.tensor(source_weights, dtype=torch.float64)
        self.source_weights /= self.source_weights.sum()
        self.num_samples = int(num_samples)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.epoch = 0
        offsets = [0]
        for length in self.source_lengths[:-1]:
            offsets.append(offsets[-1] + length)
        self.source_offsets = tuple(offsets)

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        total_size = self.num_samples * self.num_replicas
        sources = torch.multinomial(
            self.source_weights,
            num_samples=total_size,
            replacement=True,
            generator=generator,
        )
        indices = torch.empty(total_size, dtype=torch.int64)
        for source_index, (length, offset) in enumerate(
            zip(self.source_lengths, self.source_offsets, strict=True)
        ):
            positions = torch.nonzero(sources == source_index, as_tuple=False).flatten()
            if positions.numel() == 0:
                continue
            local_indices = torch.randint(
                length,
                (positions.numel(),),
                generator=generator,
                dtype=torch.int64,
            )
            indices[positions] = local_indices + offset
        yield from indices[self.rank : total_size : self.num_replicas].tolist()

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
