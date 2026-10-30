# Copyright 2026 MagicVLA authors.
#
# Licensed under the Apache License, Version 2.0 (the "License").
"""Cost-budgeted batching for sources whose per-sample cost varies.

A fixed ``batch_size`` assumes every sample costs the same. That holds for the
robotics sources (fixed camera count, fixed chunk) but not for multi-image VQA:
one sample may carry 1 image and the next 5, with chats an order of magnitude
apart in length.

Under DDP that spread lands on the *ranks*: every rank draws its own batch, the
caching allocator never returns a rank's peak, and each rank's reserved pool
ratchets up to its own worst batch. Job 5706 sat between 52% and 83% across its
8 GPUs, capping the whole run at whatever the worst rank could survive.

:class:`MultimodalBudgetBatchSampler` fixes the *cost* of a batch rather than its
sample count. The cost model has two terms because they price differently:

    batch_cost = sum(n_images) * image_token_cost + n_samples * max(sequence_tokens)

Vision-tower work is additive -- every image is resized to the same square before
the tower, so each contributes the same fixed token count. The language model sees
the vision placeholders inside each sample's token sequence, however, and pads the
*whole multimodal sequence*. It therefore pays once per sample for the longest full
sequence, not merely the longest text string.

The full-sequence term is the one that matters here. It includes both Qwen chat
formatting/text and the expanded image placeholders, so it captures the source of
the old underestimate: a row with four 64-token images is padded as a roughly
256-token-longer language sequence for every item in its batch.

Probe 5908 budgeted images only and left the sample cap effectively disabled.
Packing then put 20+ samples in a batch where a fixed size had put 4, so one long
chat priced the batch at 20 x 1228 -- five times the old worst case. One rank
jumped 12 GB at step 420 (97% of the card) and the run hung at step 500. Pricing
the padding is what prevents that: adding a long sample re-prices *every* sample
already in the batch, so it gets pushed to the next batch instead of blowing this
one up.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import torch
from torch.utils.data import Sampler


class MultimodalBudgetBatchSampler(Sampler[list[int]]):
    """Pack a rank's shard into batches bounded by modelled cost, not sample count.

    Args:
        costs: ``(n_images, sequence_tokens)`` per sample, indexed like the dataset
            (see ``VlmJsonlDataset.batch_costs``).
        budget: Maximum modelled cost per batch, in token-equivalents. A sample
            that alone exceeds the budget still gets its own batch -- it is never
            dropped, and never deadlocks the packer.
        image_token_cost: Vision tokens per image, calibrated to the processor.
            ``(image_size / vision_token_stride)^2`` reproduces it for Qwen3.5
            (64 for a 256px image). Too low and vision stops constraining the
            packer; too high and it packs one sample per batch.
        max_samples: Hard cap on samples per batch. Keep this real -- it is the
            backstop for anything the cost model does not capture. Probe 5908
            failed with it set high enough to never bind.
        num_replicas/rank: DDP shard, same semantics as ``DistributedSampler``:
            each rank gets a disjoint slice, and the tail that will not divide
            evenly is dropped so shards stay the same length.
        seed/shuffle: Deterministic per-epoch permutation. Call :meth:`set_epoch`
            between passes so the shards reshuffle.

    DDP note
    --------
    Ranks emit *different batch counts* per epoch, because packing depends on which
    samples the rank drew. That is safe here only because the interleaver wraps
    every source in an infinite iterator (see :class:`data.interleaved.InterleavedLoader`)
    and ``max_steps`` -- not epoch length -- ends training. Ranks simply drift to
    different offsets in their own shard. Do not use this sampler on a loader whose
    consumers expect ``len()`` to match across ranks.

    Ranks also end up with different sample counts per step, so a rank-mean loss
    weights each rank's samples differently. That skew already exists with a fixed
    batch size (supervised token counts vary 36..525 per step); packing widens it.
    Normalising the VQA loss by the globally reduced supervised token count is the
    proper fix and is tracked separately.
    """

    def __init__(
        self,
        costs: Sequence[tuple[int, int]],
        *,
        budget: int,
        image_token_cost: int,
        max_samples: int,
        num_replicas: int = 1,
        rank: int = 0,
        seed: int = 0,
        shuffle: bool = True,
    ) -> None:
        if budget <= 0:
            raise ValueError("budget must be positive")
        if image_token_cost <= 0:
            raise ValueError("image_token_cost must be positive")
        if max_samples <= 0:
            raise ValueError("max_samples must be positive")
        if num_replicas <= 0 or not (0 <= rank < num_replicas):
            raise ValueError(f"invalid shard: rank={rank} num_replicas={num_replicas}")

        self._images = []
        self._sequence = []
        for entry in costs:
            n_images, sequence_tokens = entry
            self._images.append(max(0, int(n_images)))
            self._sequence.append(max(1, int(sequence_tokens)))
        self._budget = int(budget)
        self._image_token_cost = int(image_token_cost)
        self._max_samples = int(max_samples)
        self._num_replicas = int(num_replicas)
        self._rank = int(rank)
        self._seed = int(seed)
        self._shuffle = bool(shuffle)
        self._epoch = 0
        self._start_batch = 0
        self._cached: tuple[int, list[list[int]]] | None = None

    def batch_cost(self, batch: Sequence[int]) -> int:
        """Modelled cost of ``batch``: additive vision + padded text."""
        if not batch:
            return 0
        images = sum(self._images[i] for i in batch)
        longest = max(self._sequence[i] for i in batch)
        return images * self._image_token_cost + len(batch) * longest

    def set_epoch(self, epoch: int) -> None:
        """Reshuffle for the next pass (called by the interleaver on exhaustion)."""
        self._epoch = int(epoch)
        self._start_batch = 0
        self._cached = None

    def set_start_batch(self, batch_offset: int) -> None:
        """Start the next iterator at a saved batch cursor without reading samples."""
        batch_offset = int(batch_offset)
        total = len(self._pack())
        if batch_offset < 0 or batch_offset > total:
            raise ValueError(
                f"batch_offset must be in [0, {total}], got {batch_offset}"
            )
        self._start_batch = batch_offset

    def resume_contract(self) -> dict:
        return {
            "budget": self._budget,
            "image_token_cost": self._image_token_cost,
            "max_samples": self._max_samples,
            "num_replicas": self._num_replicas,
            "rank": self._rank,
            "seed": self._seed,
            "shuffle": self._shuffle,
        }

    def _shard(self) -> list[int]:
        n = len(self._images)
        if self._shuffle:
            generator = torch.Generator()
            generator.manual_seed(self._seed + self._epoch)
            order = torch.randperm(n, generator=generator).tolist()
        else:
            order = list(range(n))
        # Drop the uneven tail so every rank's shard has the same length, matching
        # DistributedSampler(drop_last=True).
        usable = (n // self._num_replicas) * self._num_replicas
        return order[self._rank : usable : self._num_replicas]

    def _pack(self) -> list[list[int]]:
        if self._cached is not None and self._cached[0] == self._epoch:
            return self._cached[1]

        batches: list[list[int]] = []
        current: list[int] = []
        images = 0
        longest = 0
        for index in self._shard():
            # Price the batch *as it would be with this sample added*: a long chat
            # re-prices every sample already in the batch, which is exactly the
            # blowup a per-sample sum misses.
            projected_longest = max(longest, self._sequence[index])
            projected = (images + self._images[index]) * self._image_token_cost + (
                len(current) + 1
            ) * projected_longest
            if current and (
                projected > self._budget or len(current) >= self._max_samples
            ):
                batches.append(current)
                current, images, longest = [], 0, 0
                projected_longest = self._sequence[index]
            current.append(index)
            images += self._images[index]
            longest = projected_longest
        if current:
            batches.append(current)

        self._cached = (self._epoch, batches)
        return batches

    def __iter__(self) -> Iterator[list[int]]:
        start = self._start_batch
        self._start_batch = 0
        return iter(self._pack()[start:])

    def __len__(self) -> int:
        return len(self._pack())
