# Copyright 2026 MagicVLA authors. (Ported from MagicMix, Apache-2.0.)
#
# Licensed under the Apache License, Version 2.0 (the "License").
"""DataBackend -- turns one source config into a plain DataLoader.

A backend knows how to build a ``torch.utils.data.Dataset`` for a *single*
source; everything downstream (interleaving, modality tagging, loss routing) is
backend-agnostic. Only :meth:`make_dataset` is backend-specific;
:meth:`make_source_loader` is shared here, including the DDP sharding logic
(each rank gets a disjoint slice of every source via a ``DistributedSampler``).
"""

from __future__ import annotations

import abc

from torch.utils.data import DataLoader, DistributedSampler

from magic_w0.data.config import SubDatasetConfig
from magic_w0.data.sampling import EpochBatchSampler


class DataBackend(abc.ABC):
    version: str

    @abc.abstractmethod
    def make_dataset(self, sub: SubDatasetConfig):
        """Construct the backend-specific ``Dataset`` for one source."""

    def make_loader(
        self,
        dataset,
        *,
        batch_size: int,
        num_workers: int,
        pin_memory: bool = True,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 0,
        batch_sampler=None,
        collate_fn=None,
    ) -> DataLoader:
        """Wrap an already-built ``Dataset`` (or ``Subset``) in a DataLoader.

        ``collate_fn=None`` keeps torch's default collate. Passing one is how the
        model's CPU batch preparation is moved into the workers (see ``data.collate``).

        Under DDP (``world_size > 1``) the dataset is wrapped in a
        ``DistributedSampler`` so every rank reads a disjoint, evenly-sized shard
        -- this is what makes the data parallelism real rather than N ranks
        replaying the same samples. ``set_epoch`` is driven by the interleaver
        (see :class:`InterleavedLoader`) so the shards reshuffle every pass.

        Single-process runs keep plain ``shuffle=True`` and no sampler. The
        factory calls this once per train/eval subset (see
        :func:`data.factory.make_interleaved_loaders`).

        ``batch_sampler`` overrides all of the above: it already carries its own
        sharding and shuffling (see :class:`data.batching.ImageBudgetBatchSampler`),
        and DataLoader forbids combining it with batch_size/shuffle/sampler/drop_last.
        """
        if batch_sampler is not None:
            return DataLoader(
                dataset,
                batch_sampler=batch_sampler,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=num_workers > 0,
                collate_fn=collate_fn,
            )

        sampler = None
        shuffle = True
        if world_size > 1:
            sampler = DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                seed=seed,
                drop_last=True,
            )
            shuffle = False  # mutually exclusive with an explicit sampler
            # Use the same batches/order as DataLoader's implicit BatchSampler,
            # but expose an index-level resume cursor so a checkpoint can skip
            # preceding batches without decoding them.
            return DataLoader(
                dataset,
                batch_sampler=EpochBatchSampler(
                    sampler, batch_size=batch_size, drop_last=True
                ),
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=num_workers > 0,
                collate_fn=collate_fn,
            )

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,
            persistent_workers=num_workers > 0,
            collate_fn=collate_fn,
        )

    def make_source_loader(
        self,
        sub: SubDatasetConfig,
        *,
        batch_size: int,
        num_workers: int,
        pin_memory: bool = True,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 0,
        collate_fn=None,
    ) -> DataLoader:
        """Build a DataLoader over the *whole* source (no train/eval split)."""
        return self.make_loader(
            self.make_dataset(sub),
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            rank=rank,
            world_size=world_size,
            seed=seed,
            collate_fn=collate_fn,
        )
