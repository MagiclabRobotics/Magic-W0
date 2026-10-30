# Copyright 2026 MagicVLA authors. (Ported from MagicMix, Apache-2.0.)
#
# Licensed under the Apache License, Version 2.0 (the "License").
"""Assemble weighted, interleaved co-training loaders (with a train/eval split)."""

from __future__ import annotations

import logging

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DistributedSampler, Subset

from magic_w0.data.backends import get_data_backend
from magic_w0.data.batching import MultimodalBudgetBatchSampler
from magic_w0.data.config import CoTrainDatasetConfig, DatasetGroupConfig
from magic_w0.data.interleaved import InterleavedLoader
from magic_w0.data.retry import RetryDataset
from magic_w0.data.sampling import EpochBatchSampler, WeightedMixtureSampler

logger = logging.getLogger(__name__)


def split_train_eval_indices(
    n: int, eval_fraction: float, seed: int
) -> tuple[list[int], list[int]]:
    """Deterministic (train_idx, eval_idx) partition of ``range(n)``.

    Seeded by ``seed`` so every DDP rank computes the *same* split -- train and
    eval therefore stay disjoint across ranks. The eval slice is the last
    ``round(n * eval_fraction)`` of a fixed random permutation.

    Cuts individual frames, so both halves of one trajectory can end up on
    opposite sides. Prefer :func:`split_train_eval_episodes` for any source that
    can report trajectories; this stays for the ones that cannot.
    """
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    n_eval = int(round(n * eval_fraction))
    return perm[n_eval:], perm[:n_eval]


def split_train_eval_episodes(
    ranges: list[tuple[int, int, int]], eval_fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic (train_idx, eval_idx) partition that never cuts an episode.

    ``ranges`` are ``(episode_index, start, stop)`` half-open spans of one
    dataset's index space (see ``sample_episode_ranges`` on the lerobot readers).
    Whole spans are drawn -- under the same seeded generator every rank uses for
    :func:`split_train_eval_indices` -- until the eval half reaches
    ``eval_fraction`` of the samples, so the realized fraction lands within one
    episode of the request rather than exactly on it.

    At least one episode always goes to eval when ``eval_fraction > 0``: a
    fraction that rounds to zero samples would otherwise disable validation
    silently instead of reporting an empty split.

    Returns int64 arrays rather than Python lists. ``Subset.indices`` is held in
    the parent process and copied into every worker on first touch; the same
    109.5M-element index cost 4.2 GB as Python ints and 856 MB as int64 in the
    dataloader-memory work, and this is the same array in the same place.
    """
    if not ranges:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

    total = sum(stop - start for _, start, stop in ranges)
    target = int(round(total * eval_fraction))
    g = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(ranges), generator=g).tolist()

    eval_positions: set[int] = set()
    taken = 0
    for position in order:
        if eval_positions and taken >= target:
            break
        _, start, stop = ranges[position]
        eval_positions.add(position)
        taken += stop - start

    def _collect(positions: list[int]) -> np.ndarray:
        spans = [
            np.arange(ranges[p][1], ranges[p][2], dtype=np.int64) for p in positions
        ]
        return np.concatenate(spans) if spans else np.empty(0, dtype=np.int64)

    train_positions = [p for p in range(len(ranges)) if p not in eval_positions]
    return _collect(train_positions), _collect(sorted(eval_positions))


def resolve_episode_ranges(view) -> list[tuple[int, int, int]] | None:
    """Episode spans for ``view``, looking *through* 1:1 dataset wrappers.

    Mirrors the walk in :func:`resolve_batch_costs`: the object the factory holds
    is the model's own wrapper around the reader, not the reader that knows where
    one trajectory ends.

    Returns ``None`` when nothing in the chain reports spans -- a source with no
    trajectories (a VQA jsonl) is the ordinary case, and the caller falls back to
    the per-sample split. Also ``None`` below a ``Subset``: the spans address the
    reader's index space, and a ``Subset`` renumbers it, so they would no longer
    describe the indices being partitioned.
    """
    node, seen = view, set()
    while node is not None and id(node) not in seen:
        seen.add(id(node))
        if isinstance(node, Subset):
            return None
        ranges_fn = getattr(node, "sample_episode_ranges", None)
        if callable(ranges_fn):
            return [
                (int(episode), int(start), int(stop))
                for episode, start, stop in ranges_fn()
            ]
        node = getattr(node, "dataset", None)
    return None


def split_source_indices(dataset, cfg: CoTrainDatasetConfig, label: str):
    """Split one source into (train_idx, eval_idx) per ``cfg.eval_split``.

    One place for the decision so the three mixing paths below cannot drift into
    holding out three different things.
    """
    if cfg.eval_split == "episode":
        ranges = resolve_episode_ranges(dataset)
        if ranges:
            return split_train_eval_episodes(ranges, cfg.eval_fraction, cfg.seed)
        logger.info(
            "%s: eval_split='episode' but no layer of %s reports episodes; "
            "holding out individual samples instead",
            label,
            type(dataset).__name__,
        )
    return split_train_eval_indices(len(dataset), cfg.eval_fraction, cfg.seed)


def resolve_batch_costs(view) -> list[int] | None:
    """Per-sample batching costs for ``view``, looking *through* dataset wrappers.

    The object the factory holds is rarely the backend that knows what a sample
    costs: the model injects its own 1:1 wrapper around every source (e.g.
    ``_MagicW0VqaTextDataset``), and a non-zero ``eval_fraction`` puts a ``Subset``
    on top of that. So walk the ``.dataset`` chain down to whichever layer reports
    ``batch_costs()``, composing any ``Subset`` index selection on the way back.

    Returns ``None`` when nothing in the chain reports costs -- the caller decides
    whether that is fatal.
    """
    node, index_map, seen = view, None, set()
    while node is not None and id(node) not in seen:
        seen.add(id(node))
        costs_fn = getattr(node, "batch_costs", None)
        if callable(costs_fn):
            costs = list(costs_fn())
            return [costs[i] for i in index_map] if index_map is not None else costs
        if isinstance(node, Subset):
            inner = list(node.indices)
            index_map = inner if index_map is None else [inner[i] for i in index_map]
        node = getattr(node, "dataset", None)
    return None


def _source_label(source) -> str:
    root = getattr(source, "root", None) or getattr(source, "repo_id", None)
    identity = str(root or getattr(source, "type", type(source).__name__))
    return f"{source.modality}:{identity}"


def _with_sample_retries(dataset, cfg: CoTrainDatasetConfig, *, seed: int, source):
    if cfg.sample_max_retries <= 0:
        return dataset
    return RetryDataset(
        dataset,
        max_retries=cfg.sample_max_retries,
        seed=seed,
        label=_source_label(source),
        log_limit=cfg.sample_retry_log_limit,
    )


def make_interleaved_loaders(
    cfg: CoTrainDatasetConfig,
    *,
    rank: int = 0,
    world_size: int = 1,
    pin_memory: bool = True,
    dataset_transform=None,
    collate_factory=None,
) -> tuple[InterleavedLoader, InterleavedLoader | None]:
    """Build the train and eval interleaved loaders.

    Each source is split ``(1 - eval_fraction) / eval_fraction`` into train/eval
    subsets, or ``eval_on_train`` builds an independently shuffled eval loader
    over each complete training source without holding samples out. Returns
    ``(train_loader, eval_loader)``; ``eval_loader`` is ``None`` when both modes
    are disabled.
    """
    if cfg.mixing_level == "sample":
        if any(isinstance(source, DatasetGroupConfig) for source in cfg.datasets):
            raise ValueError(
                "dataset_group sources require top-level mixing_level='batch'; "
                "the group already performs sample-level mixing"
            )
        return _make_sample_mixed_loaders(
            cfg,
            rank=rank,
            world_size=world_size,
            pin_memory=pin_memory,
            dataset_transform=dataset_transform,
            collate_factory=collate_factory,
        )

    # ``collate_factory(modality) -> collate_fn`` moves the model's CPU batch preparation
    # into the workers. It must be built PER SOURCE: InterleavedLoader tags the modality in
    # the main process after collation, so a worker-side preparer would otherwise never see
    # it and would route every batch to the model's native modality.
    want_eval = cfg.eval_fraction > 0.0 or cfg.eval_on_train
    train_loaders, eval_loaders, modalities, weights = [], [], [], []

    for sub in cfg.datasets:
        if isinstance(sub, DatasetGroupConfig):
            group_train, group_eval = _make_sample_mixed_group_loaders(
                cfg,
                sub,
                rank=rank,
                world_size=world_size,
                pin_memory=pin_memory,
                dataset_transform=dataset_transform,
                collate_factory=collate_factory,
            )
            train_loaders.append(group_train)
            if group_eval is not None:
                eval_loaders.append(group_eval)
            modalities.append(sub.modality)
            weights.append(sub.weight)
            continue

        bs = sub.batch_size or cfg.batch_size
        backend = get_data_backend(getattr(sub, "type", "tensor"))
        dataset = backend.make_dataset(sub)
        if dataset_transform is not None:
            dataset = dataset_transform(dataset, sub)

        if cfg.eval_fraction > 0.0:
            train_idx, eval_idx = split_source_indices(dataset, cfg, sub.modality)
            if len(train_idx) == 0 or len(eval_idx) == 0:
                raise ValueError(
                    f"eval_fraction={cfg.eval_fraction} (eval_split={cfg.eval_split!r}) leaves an "
                    f"empty split for source {sub.modality!r} (n={len(dataset)}); use more "
                    "samples or episodes, or a smaller fraction."
                )
            train_ds: object = Subset(dataset, train_idx)
            eval_ds: object | None = Subset(dataset, eval_idx)
        elif cfg.eval_on_train:
            train_ds, eval_ds = dataset, dataset
        else:
            train_ds, eval_ds = dataset, None

        shared_train_eval_view = eval_ds is train_ds
        train_ds = _with_sample_retries(train_ds, cfg, seed=cfg.seed, source=sub)
        if eval_ds is not None:
            eval_ds = (
                train_ds
                if shared_train_eval_view
                else _with_sample_retries(eval_ds, cfg, seed=cfg.seed + 1, source=sub)
            )

        loader_kw = dict(
            batch_size=bs,
            num_workers=cfg.num_workers,
            pin_memory=pin_memory,
            rank=rank,
            world_size=world_size,
            seed=cfg.seed,
            collate_fn=None
            if collate_factory is None
            else collate_factory(sub.modality),
        )
        # A source may replace the fixed batch_size with a cost budget: pack samples
        # until the batch's modelled cost would exceed it, so every rank's batch peaks
        # at the same place instead of wherever its own draw landed.
        budget = getattr(sub, "batch_token_budget", None)

        def _budget_sampler(view, _sub=sub, _budget=budget, *, seed=cfg.seed):
            if not _budget:
                return None
            costs = resolve_batch_costs(view)
            if costs is None:
                raise ValueError(
                    f"source {_sub.modality!r} sets batch_token_budget but no layer of "
                    f"{type(view).__name__} reports batch_costs()"
                )
            if len(costs) != len(view):
                raise ValueError(
                    f"source {_sub.modality!r} reported {len(costs)} costs for "
                    f"{len(view)} samples -- a dataset wrapper is not 1:1"
                )
            image_token_cost = getattr(_sub, "image_token_cost", None)
            if image_token_cost is None:
                size = getattr(_sub, "image_size", None) or 0
                stride = max(1, getattr(_sub, "vision_token_stride", 28))
                image_token_cost = max(1, size // stride) ** 2 if size else 1
            return MultimodalBudgetBatchSampler(
                costs,
                budget=_budget,
                image_token_cost=image_token_cost,
                max_samples=getattr(_sub, "max_batch_samples", 8),
                num_replicas=max(1, world_size),
                rank=rank,
                seed=seed,
            )

        train_loaders.append(
            backend.make_loader(
                train_ds, batch_sampler=_budget_sampler(train_ds), **loader_kw
            )
        )
        if eval_ds is not None:
            eval_loader_kw = {
                **loader_kw,
                "num_workers": (
                    cfg.num_workers
                    if cfg.eval_num_workers is None
                    else cfg.eval_num_workers
                ),
                "seed": cfg.seed + 1,
            }
            eval_loaders.append(
                backend.make_loader(
                    eval_ds,
                    batch_sampler=_budget_sampler(eval_ds, seed=cfg.seed + 1),
                    **eval_loader_kw,
                )
            )
        modalities.append(sub.modality)
        weights.append(sub.weight)

    def _wrap(loaders, seed):
        return InterleavedLoader(
            loaders,
            modalities=modalities,
            weights=weights,
            seed=seed,  # shared across ranks: synchronized source selection
            modality_key=cfg.modality_key,
        )

    train_loader = _wrap(train_loaders, cfg.seed)
    eval_loader = _wrap(eval_loaders, cfg.seed + 1) if want_eval else None
    return train_loader, eval_loader


def _make_sample_mixed_group_loaders(
    cfg: CoTrainDatasetConfig,
    group: DatasetGroupConfig,
    *,
    rank: int,
    world_size: int,
    pin_memory: bool,
    dataset_transform,
    collate_factory,
):
    """Build one virtual source by sample-mixing the group's leaf datasets.

    The returned objects are ordinary DataLoaders. The caller treats the virtual
    group as one source in the outer batch-level ``InterleavedLoader``.
    """
    batch_sizes = {
        source.batch_size or group.batch_size or cfg.batch_size
        for source in group.datasets
    }
    if len(batch_sizes) != 1:
        raise ValueError(
            f"dataset group {group.name!r} requires one shared batch size; "
            f"got {sorted(batch_sizes)}"
        )
    batch_size = next(iter(batch_sizes))
    if group.batch_size is not None:
        incompatible = [
            source.batch_size
            for source in group.datasets
            if source.batch_size is not None and source.batch_size != group.batch_size
        ]
        if incompatible:
            raise ValueError(
                f"dataset group {group.name!r} batch_size conflicts with child override: "
                f"expected {group.batch_size}, got {sorted(set(incompatible))}"
            )

    want_eval = cfg.eval_fraction > 0.0 or cfg.eval_on_train
    train_parts, eval_parts = [], []
    for source in group.datasets:
        backend = get_data_backend(getattr(source, "type", "tensor"))
        dataset = backend.make_dataset(source)
        if dataset_transform is not None:
            # Keep source-specific processing/statistics before virtual concatenation.
            dataset = dataset_transform(dataset, source)
        if cfg.eval_fraction > 0.0:
            train_idx, eval_idx = split_source_indices(
                dataset, cfg, f"group {group.name!r} child {source.modality!r}"
            )
            if len(train_idx) == 0 or len(eval_idx) == 0:
                raise ValueError(
                    f"eval_fraction={cfg.eval_fraction} (eval_split={cfg.eval_split!r}) leaves an "
                    f"empty split for group {group.name!r} child {source.modality!r} "
                    f"(n={len(dataset)})"
                )
            train_view, eval_view = (
                Subset(dataset, train_idx),
                Subset(dataset, eval_idx),
            )
        elif cfg.eval_on_train:
            train_view, eval_view = dataset, dataset
        else:
            train_view, eval_view = dataset, None

        shared_train_eval_view = eval_view is train_view
        train_view = _with_sample_retries(train_view, cfg, seed=cfg.seed, source=source)
        train_parts.append(train_view)
        if eval_view is not None:
            eval_parts.append(
                train_view
                if shared_train_eval_view
                else _with_sample_retries(
                    eval_view, cfg, seed=cfg.seed + 1, source=source
                )
            )

    collate_fn = None if collate_factory is None else collate_factory(group.modality)

    def _loader(parts, *, num_samples: int | None, seed: int, num_workers: int):
        mixed = ConcatDataset(parts)
        if group.sampling_mode == "concat_shuffle":
            # Every rank receives a disjoint shard of one shuffled physical
            # concatenation. The DistributedSampler + BatchSampler drop tails so
            # each rank has the same number of full batches for DDP.
            sampler = DistributedSampler(
                mixed,
                num_replicas=max(1, world_size),
                rank=rank,
                shuffle=True,
                seed=seed,
                drop_last=True,
            )
        else:
            if num_samples is None:  # guarded by DatasetGroupConfig validation
                raise RuntimeError("weighted_replacement requires num_samples")
            mixture_weights = [
                source.weight
                * (len(part) if group.weight_mode == "dataset_size" else 1.0)
                for source, part in zip(group.datasets, parts, strict=True)
            ]
            sampler = WeightedMixtureSampler(
                [len(part) for part in parts],
                mixture_weights,
                num_samples=num_samples,
                num_replicas=max(1, world_size),
                rank=rank,
                seed=seed,
            )
        batch_sampler = EpochBatchSampler(
            sampler, batch_size=batch_size, drop_last=True
        )
        # All registered backends inherit the same generic DataLoader builder.
        backend = get_data_backend(getattr(group.datasets[0], "type", "tensor"))
        return backend.make_loader(
            mixed,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            rank=rank,
            world_size=world_size,
            seed=seed,
            batch_sampler=batch_sampler,
            collate_fn=collate_fn,
        )

    default_samples = max(
        batch_size,
        (sum(len(part) for part in train_parts) + max(1, world_size) - 1)
        // max(1, world_size),
    )
    train_loader = _loader(
        train_parts,
        num_samples=(
            None
            if group.sampling_mode == "concat_shuffle"
            else group.samples_per_epoch or default_samples
        ),
        seed=cfg.seed,
        num_workers=cfg.num_workers,
    )
    eval_loader = None
    if want_eval:
        eval_samples = max(
            batch_size,
            (sum(len(part) for part in eval_parts) + max(1, world_size) - 1)
            // max(1, world_size),
        )
        eval_loader = _loader(
            eval_parts,
            num_samples=None
            if group.sampling_mode == "concat_shuffle"
            else eval_samples,
            seed=cfg.seed + 1,
            num_workers=(
                cfg.num_workers
                if cfg.eval_num_workers is None
                else cfg.eval_num_workers
            ),
        )
    return train_loader, eval_loader


def _make_sample_mixed_loaders(
    cfg: CoTrainDatasetConfig,
    *,
    rank: int,
    world_size: int,
    pin_memory: bool,
    dataset_transform,
    collate_factory,
) -> tuple[InterleavedLoader, InterleavedLoader | None]:
    """Build a loader whose individual batches contain weighted mixed sources."""
    modalities = {source.modality for source in cfg.datasets}
    if len(modalities) != 1:
        raise ValueError(
            "sample-level mixing requires every source to use the same modality; "
            f"got {sorted(modalities)}"
        )
    modality = next(iter(modalities))
    batch_sizes = {source.batch_size or cfg.batch_size for source in cfg.datasets}
    if len(batch_sizes) != 1:
        raise ValueError(
            "sample-level mixing requires one shared batch size; "
            f"got {sorted(batch_sizes)}"
        )
    batch_size = next(iter(batch_sizes))
    want_eval = cfg.eval_fraction > 0.0 or cfg.eval_on_train
    train_parts, eval_parts, weights, backends = [], [], [], []

    for source in cfg.datasets:
        backend = get_data_backend(getattr(source, "type", "tensor"))
        dataset = backend.make_dataset(source)
        if dataset_transform is not None:
            dataset = dataset_transform(dataset, source)
        if cfg.eval_fraction > 0.0:
            train_idx, eval_idx = split_source_indices(dataset, cfg, source.modality)
            if len(train_idx) == 0 or len(eval_idx) == 0:
                raise ValueError(
                    f"eval_fraction={cfg.eval_fraction} (eval_split={cfg.eval_split!r}) leaves an "
                    f"empty split for source {source.modality!r} (n={len(dataset)})"
                )
            train_view, eval_view = (
                Subset(dataset, train_idx),
                Subset(dataset, eval_idx),
            )
        elif cfg.eval_on_train:
            train_view, eval_view = dataset, dataset
        else:
            train_view, eval_view = dataset, None

        shared_train_eval_view = eval_view is train_view
        train_view = _with_sample_retries(train_view, cfg, seed=cfg.seed, source=source)
        train_parts.append(train_view)
        if eval_view is not None:
            eval_parts.append(
                train_view
                if shared_train_eval_view
                else _with_sample_retries(
                    eval_view, cfg, seed=cfg.seed + 1, source=source
                )
            )
        weights.append(source.weight)
        backends.append(backend)

    collate_fn = None if collate_factory is None else collate_factory(modality)

    def _loader(parts, *, num_samples: int, seed: int, num_workers: int):
        mixed = ConcatDataset(parts)
        sampler = WeightedMixtureSampler(
            [len(part) for part in parts],
            weights,
            num_samples=num_samples,
            num_replicas=max(1, world_size),
            rank=rank,
            seed=seed,
        )
        batch_sampler = EpochBatchSampler(
            sampler, batch_size=batch_size, drop_last=True
        )
        return backends[0].make_loader(
            mixed,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            rank=rank,
            world_size=world_size,
            seed=seed,
            batch_sampler=batch_sampler,
            collate_fn=collate_fn,
        )

    default_train_samples = max(
        batch_size,
        (sum(len(part) for part in train_parts) + max(1, world_size) - 1)
        // max(1, world_size),
    )
    train_loader = _loader(
        train_parts,
        num_samples=cfg.samples_per_epoch or default_train_samples,
        seed=cfg.seed,
        num_workers=cfg.num_workers,
    )
    eval_loader = None
    if want_eval:
        eval_samples = max(
            batch_size,
            (sum(len(part) for part in eval_parts) + max(1, world_size) - 1)
            // max(1, world_size),
        )
        eval_loader = _loader(
            eval_parts,
            num_samples=eval_samples,
            seed=cfg.seed + 1,
            num_workers=(
                cfg.num_workers
                if cfg.eval_num_workers is None
                else cfg.eval_num_workers
            ),
        )

    def _wrap(loader):
        return InterleavedLoader(
            [loader],
            modalities=[modality],
            weights=[1.0],
            seed=cfg.seed,
            modality_key=cfg.modality_key,
        )

    return _wrap(train_loader), _wrap(eval_loader) if eval_loader is not None else None


def make_interleaved_loader(
    cfg: CoTrainDatasetConfig,
    *,
    rank: int = 0,
    world_size: int = 1,
    pin_memory: bool = True,
    dataset_transform=None,
) -> InterleavedLoader:
    """Convenience: just the train loader (the ``1 - eval_fraction`` split)."""
    train_loader, _ = make_interleaved_loaders(
        cfg,
        rank=rank,
        world_size=world_size,
        pin_memory=pin_memory,
        dataset_transform=dataset_transform,
    )
    return train_loader
