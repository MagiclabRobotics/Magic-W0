# Copyright 2026 MagicVLA authors. (Apache-2.0.)
"""Collect the normalization contract a checkpoint needs to be usable alone.

A checkpoint carries weights, optimizer state and the model config. None of that
is enough to run the model: actions are trained in a normalized, partly-delta
space, and undoing that needs per-source statistics that live on the dataset, plus
``action_delta_indices`` which says which dimensions were made relative to the
anchor state in the first place. Both are data-side facts the model never sees.

Without them a checkpoint is resumable but not deployable -- inference would have
to rebuild the datasets and rescan them just to recover a few kilobytes of
numbers. This module walks a built loader and returns those kilobytes.

Nothing here interprets the statistics; it records what each source used, keyed by
its root, because a mixed-embodiment run has one set per source and there is no
defensible way to collapse them here. Choosing which set applies to a given robot
at inference is a decision for whoever deploys the model.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _unwrap(dataset: Any) -> Iterable[Any]:
    """Yield one normalization owner per wrapper/reader chain.

    A model-side wrapper may copy ``meta_stats`` from its reader without owning
    the rest of the normalization contract.  Exporting both that metadata-only
    wrapper and its child makes a single source look like two sources (for
    example ``<_MagicW0TrainingDataset>`` plus its LeRobot root).  A wrapper is an
    export boundary only when it also owns the state/action keys; otherwise we
    keep descending to the reader that owns the complete contract.
    """
    seen: set[int] = set()
    stack = [dataset]
    while stack:
        node = stack.pop()
        if node is None or id(node) in seen:
            continue
        seen.add(id(node))
        # torch's ConcatDataset, and the group wrapper, both expose `datasets`
        children = getattr(node, "datasets", None)
        if isinstance(children, (list, tuple)) and children:
            stack.extend(children)
            continue
        # Model-side wrappers keep the reader on `.dataset`.  A wrapper with a
        # complete contract is authoritative because it may have transformed
        # the child's statistics or keys.  A metadata-only wrapper is merely a
        # transparent view and must not become a synthetic normalization source.
        child = getattr(node, "dataset", None)
        if child is not None and child is not node:
            owns_contract = all(
                getattr(node, name, None) is not None
                for name in ("state_key", "action_key")
            )
            if getattr(node, "meta_stats", None) is not None and owns_contract:
                yield node
            else:
                stack.append(child)
            continue
        if getattr(node, "meta_stats", None) is not None:
            yield node


def collect_norm_metadata(loader: Any) -> dict[str, Any]:
    """Return ``{source_root: {stats, delta_indices, ...}}`` for a built loader.

    Returns an empty dict rather than raising when a loader exposes nothing
    recognisable: this is metadata for a checkpoint, and failing to collect it
    must not be able to kill a training run that is otherwise fine.
    """
    sources: dict[str, Any] = {}
    iterators = getattr(loader, "_iters", None)
    candidates = []
    if iterators is not None:
        for iterator in iterators:
            inner = getattr(iterator, "_loader", None) or getattr(
                iterator, "loader", None
            )
            if inner is not None:
                candidates.append(inner)
    else:
        candidates.append(loader)

    for candidate in candidates:
        dataset = getattr(candidate, "dataset", candidate)
        for reader in _unwrap(dataset):
            root = str(getattr(reader, "root", "") or f"<{type(reader).__name__}>")
            if root in sources:
                continue
            stats = getattr(reader, "meta_stats", None)
            if not stats:
                continue
            entry: dict[str, Any] = {
                "stats": {
                    key: {name: _jsonable(value) for name, value in feature.items()}
                    for key, feature in stats.items()
                }
            }
            # Inference denormalizes from this payload, and a per-horizon band
            # is a different array rank than the flat one it was written to
            # expect.  Naming the layout lets a serving stack detect the
            # difference instead of failing on a confusing dim mismatch.
            entry["action_stats_layout"] = (
                "per_horizon_dim"
                if getattr(reader, "action_stats_per_horizon", False)
                else "per_dim"
            )
            for name in (
                "state_key",
                "action_key",
                "state_indices",
                "action_indices",
                "state_target_indices",
                "action_target_indices",
                "preserve_selected_slots",
                "state_mask_key",
                "action_mask_key",
                "action_representation",
                "action_delta_indices",
                "action_supervision_indices",
                "action_rotation_groups",
                "action_relative_rotation_groups",
                "action_rotation_delta",
                "chunk_size",
                "target_state_dim",
                "target_action_dim",
                "state_active_indices",
                "action_active_indices",
                "fps",
            ):
                value = getattr(reader, name, None)
                if value is not None:
                    entry[name] = _jsonable(value)
            sources[root] = entry
    return sources
