# Copyright 2026 MagicVLA authors. (Ported from MagicMix, Apache-2.0.)
#
# Licensed under the Apache License, Version 2.0 (the "License").
"""Batch-level interleaving of multiple data sources (the RandomMix analog).

VLM image-text batches and robot action batches have different feature schemas,
so they cannot be concatenated into one collated batch. Instead we keep each
source homogeneous and interleave at the *batch* level: every training step we
draw one source according to its weight and yield that source's next batch,
tagged with its modality. This mirrors `webdataset.RandomMix(probs=...)`.

DDP note
--------
The source-selection RNG is seeded *identically* on every rank (pass a shared
``seed``, not ``base + rank``). That way all ranks pick the same source at each
step, so they run the same sub-graph and DDP's gradient all-reduce stays well
defined. Data parallelism comes from each source's ``DistributedSampler`` handing
every rank a disjoint shard (see :class:`DataBackend.make_source_loader`) -- not
from desynchronizing the source draw.
"""

from __future__ import annotations

from collections.abc import Iterator

import torch


class _InfiniteIterator:
    """Endlessly iterate a DataLoader, reshuffling its shard each pass.

    Replaces ``itertools.cycle`` (which would cache every batch in memory and
    freeze a ``DistributedSampler`` to a single shuffle): on each exhaustion we
    bump ``set_epoch`` so a ``DistributedSampler``/shuffling sampler re-draws,
    then rebuild the iterator.
    """

    def __init__(self, loader):
        self._loader = loader
        self._epoch = 0
        self._batches_consumed = 0
        # Keep construction lazy.  On resume the checkpoint is loaded after the
        # loaders are assembled; eagerly calling iter(loader) here would start
        # persistent workers and prefetch epoch-0 samples before the saved cursor
        # has been applied.
        self._it = None

    def _set_epoch(self) -> None:
        for attr in ("batch_sampler", "sampler"):
            sampler = getattr(self._loader, attr, None)
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(self._epoch)
                return

    def _start_iterator(self) -> None:
        self._set_epoch()
        batch_sampler = getattr(self._loader, "batch_sampler", None)
        if self._batches_consumed:
            set_start_batch = getattr(batch_sampler, "set_start_batch", None)
            if set_start_batch is None:
                raise RuntimeError(
                    "data-exact resume requires a batch sampler implementing "
                    "set_start_batch(); refusing to decode and discard "
                    f"{self._batches_consumed} batches"
                )
            set_start_batch(self._batches_consumed)
        self._it = iter(self._loader)

    def __next__(self):
        if self._it is None:
            self._start_iterator()
        try:
            batch = next(self._it)
        except StopIteration:
            self._epoch += 1
            self._batches_consumed = 0
            self._start_iterator()
            batch = next(self._it)
        self._batches_consumed += 1
        return batch

    def _contract(self) -> dict:
        batch_sampler = getattr(self._loader, "batch_sampler", None)
        sampler = getattr(batch_sampler, "sampler", None)
        if sampler is None:
            sampler = getattr(self._loader, "sampler", None)
        dataset = getattr(self._loader, "dataset", None)
        return {
            "dataset_length": len(dataset) if dataset is not None else None,
            "batches_per_epoch": len(self._loader),
            "batch_sampler_type": type(batch_sampler).__qualname__,
            "sampler_type": type(sampler).__qualname__,
            "num_replicas": getattr(sampler, "num_replicas", None),
            "rank": getattr(sampler, "rank", None),
            "seed": getattr(sampler, "seed", None),
            "batch_sampler": (
                batch_sampler.resume_contract()
                if hasattr(batch_sampler, "resume_contract")
                else None
            ),
        }

    def state_dict(self) -> dict:
        return {
            "epoch": self._epoch,
            "batches_consumed": self._batches_consumed,
            "contract": self._contract(),
        }

    def load_state_dict(self, state: dict) -> None:
        if self._it is not None:
            raise RuntimeError("data state must be restored before iteration starts")
        epoch = int(state.get("epoch", 0))
        consumed = int(state.get("batches_consumed", 0))
        if epoch < 0 or consumed < 0:
            raise ValueError(
                f"invalid saved data cursor: epoch={epoch}, batches_consumed={consumed}"
            )

        # Dynamic token-budget packing reshuffles before greedily assembling an
        # epoch. Different shuffle orders can legitimately produce different
        # batch counts, so the saved contract must be checked at the saved epoch,
        # not against a freshly constructed epoch-0 loader. This also primes the
        # sampler cache for set_start_batch() without decoding any samples.
        self._epoch = epoch
        self._set_epoch()
        expected = state.get("contract")
        current = self._contract()
        if expected != current:
            raise ValueError(
                "data loader contract changed across resume: "
                f"saved={expected}, current={current}"
            )
        batches_per_epoch = int(current["batches_per_epoch"])
        if consumed > batches_per_epoch:
            raise ValueError(
                "invalid saved data cursor: "
                f"epoch={epoch}, batches_consumed={consumed}, "
                f"batches_per_epoch={batches_per_epoch}"
            )
        self._batches_consumed = consumed

    def seek_batches(self, total_batches: int) -> None:
        """Reconstruct a legacy checkpoint cursor without reading any samples."""
        if self._it is not None:
            raise RuntimeError(
                "a data cursor can only be reconstructed before iteration"
            )
        remaining = int(total_batches)
        if remaining < 0:
            raise ValueError(f"total_batches must be non-negative, got {remaining}")
        self._epoch = 0
        self._batches_consumed = 0
        while True:
            self._set_epoch()
            batches_per_epoch = len(self._loader)
            if batches_per_epoch <= 0:
                raise RuntimeError("cannot seek an empty data loader")
            # Preserve the actual post-consumption representation at a boundary:
            # after consuming the last batch, _InfiniteIterator advances epoch only
            # when this source is selected once more and observes StopIteration.
            if remaining <= batches_per_epoch:
                self._batches_consumed = remaining
                return
            remaining -= batches_per_epoch
            self._epoch += 1


class InterleavedLoader:
    """Weighted round-robin over several infinite dataloaders.

    Args:
        loaders: One iterable per source (typically ``torch.utils.data.DataLoader``).
        modalities: Modality tag per source, e.g. ``["robotics", "vlm_sft"]``.
        weights: Sampling weight per source (need not sum to 1).
        seed: Base RNG seed for the source draw. In DDP, pass the *same* seed on
            every rank so the ranks pick the same source each step (required for
            DDP's gradient sync); per-rank sharding is handled by each source's
            ``DistributedSampler``, not by this seed.
        modality_key: Key under which the modality tag is injected into each batch.
    """

    def __init__(
        self,
        loaders,
        modalities: list[str],
        weights: list[float],
        seed: int = 0,
        modality_key: str = "modality",
    ):
        if not (len(loaders) == len(modalities) == len(weights)):
            raise ValueError("loaders, modalities and weights must have equal length")
        if len(loaders) == 0:
            raise ValueError("InterleavedLoader needs at least one source")

        self._iters = [_InfiniteIterator(loader) for loader in loaders]
        self._modalities = list(modalities)
        self._modality_key = modality_key

        w = torch.tensor(weights, dtype=torch.float64)
        if torch.any(w < 0) or w.sum() <= 0:
            raise ValueError("weights must be non-negative and sum to a positive value")
        self._probs = (w / w.sum()).float()

        self._generator = torch.Generator()
        self._generator.manual_seed(seed)

    def __iter__(self) -> Iterator[dict]:
        return self

    def __next__(self) -> dict:
        idx = int(
            torch.multinomial(
                self._probs, num_samples=1, generator=self._generator
            ).item()
        )
        batch = next(self._iters[idx])
        if isinstance(batch, dict):
            batch[self._modality_key] = self._modalities[idx]
        else:  # tolerate (inputs, targets)-style batches
            batch = {"data": batch, self._modality_key: self._modalities[idx]}
        return batch

    def state_dict(self) -> dict:
        """Return the small, sample-position-only state needed for exact resume."""
        return {
            "version": 1,
            "modalities": list(self._modalities),
            "probabilities": self._probs.tolist(),
            "generator_state": self._generator.get_state(),
            "sources": [iterator.state_dict() for iterator in self._iters],
        }

    def load_state_dict(self, state: dict) -> None:
        if int(state.get("version", -1)) != 1:
            raise ValueError(
                f"unsupported interleaved data state: {state.get('version')}"
            )
        if list(state.get("modalities", ())) != self._modalities:
            raise ValueError(
                "data modalities changed across resume: "
                f"saved={state.get('modalities')}, current={self._modalities}"
            )
        saved_probs = torch.tensor(
            state.get("probabilities", ()), dtype=self._probs.dtype
        )
        if not torch.equal(saved_probs, self._probs):
            raise ValueError(
                "data mixing probabilities changed across resume: "
                f"saved={saved_probs.tolist()}, current={self._probs.tolist()}"
            )
        sources = state.get("sources", ())
        if len(sources) != len(self._iters):
            raise ValueError(
                "data source count changed across resume: "
                f"saved={len(sources)}, current={len(self._iters)}"
            )
        for iterator, source_state in zip(self._iters, sources, strict=True):
            iterator.load_state_dict(source_state)
        self._generator.set_state(state["generator_state"])

    def seek_steps(self, global_step: int) -> None:
        """Rebuild train cursors for checkpoints created before data_state existed.

        This replays only the private source-selection RNG and sampler arithmetic;
        no DataLoader iterator is created, so no video/image is decoded or collated.
        """
        steps = int(global_step)
        if steps < 0:
            raise ValueError(f"global_step must be non-negative, got {steps}")
        if any(iterator._it is not None for iterator in self._iters):
            raise RuntimeError("data steps can only be reconstructed before iteration")
        counts = [0] * len(self._iters)
        for _ in range(steps):
            index = int(
                torch.multinomial(
                    self._probs, num_samples=1, generator=self._generator
                ).item()
            )
            counts[index] += 1
        for iterator, count in zip(self._iters, counts, strict=True):
            iterator.seek_batches(count)

    @property
    def modalities(self) -> list[str]:
        """Modality tags, in source order.

        Every rank builds this loader from the same config, so the list is
        identical across ranks -- which is what lets the trainer lay out a
        fixed-width reduction buffer for per-modality validation metrics without
        having to negotiate keys at eval time (a key set that differs by rank
        deadlocks the collective).
        """
        return list(self._modalities)

    @property
    def sampling_probs(self) -> dict[str, float]:
        return {m: float(p) for m, p in zip(self._modalities, self._probs, strict=True)}
