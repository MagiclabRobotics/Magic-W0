# Copyright 2026 MagicVLA authors. (Apache-2.0.)
"""Standalone LeRobot **v2.1** dataset reader (no lerobot import).

The modern lerobot codebase (v3.0) refuses v2.1 datasets outright
(``BackwardCompatibilityError``), so mixing a v2.1 source into a co-training
run cannot go through ``LeRobotDataset``. This module reads the v2.1 layout
directly -- per-episode parquet under ``data/chunk-*/`` + per-episode mp4 under
``videos/chunk-*/<cam>/`` + jsonl metadata -- and yields samples with the SAME
schema as a v3 ``LeRobotDataset`` sample (uint8 CHW camera frames, float32
state, a ``[chunk, dim]`` action stack with an ``*_is_pad`` mask, and the
``task`` string), so the two versions can interleave transparently.

Only pyarrow + torchcodec are required (both ship in the training image).
The sibling v3.0 reader lives in :mod:`data.backends.lerobot_v30`; the
version dispatch is :class:`data.backends.lerobot.LeRobotBackend`.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from magic_w0.data.action_interpolation import (
    ActionInterpolationSpec,
    parse_action_interpolation_spec,
    semantic_upsample_actions,
)
from magic_w0.data.anchor_sampling import (
    AnchorSamplingSpec,
    episode_sample_anchors,
    load_keyframe_jsonl,
    parse_anchor_sampling_spec,
)
from magic_w0.data.embodiment import (
    UNKNOWN_EMBODIMENT,
    EmbodimentSpec,
    decode_embodiment_id,
    resolve_spec,
)
from magic_w0.data.lerobot_history import (
    episode_observation_indices,
    validate_observation_sampling,
)
from magic_w0.data.rotation6d import compose_relative_rot6d
from magic_w0.data.norm_stats import (
    apply_horizon_carveouts,
    collect_quantile_values,
    load_anchor_index,
    load_episode_index,
    load_norm_stats_artifact,
    load_sample_anchor_index_v2,
)
from magic_w0.data.video import make_video_decoder

logger = logging.getLogger(__name__)

_MIN_ESTIMATED_STD = 1e-6
_MIN_QUANTILE_SAMPLES = 200
_MIN_ESTIMATED_SPAN = 1e-6

_V21_INFO = "meta/info.json"
_V21_EPISODES = "meta/episodes.jsonl"
_V21_EPISODE_STATS = "meta/episodes_stats.jsonl"
_V21_TASKS = "meta/tasks.jsonl"

_ACTION_REPRESENTATION_ALIASES = {
    "abs": "abs",
    "absolute": "abs",
    "abs_action": "abs",
    "abs-action": "abs",
    "chunk_delta": "chunk_delta",
    "chunk-delta": "chunk_delta",
    "chunk_delta_action": "chunk_delta",
    "chunk-delta-action": "chunk_delta",
    "frame_delta": "frame_delta",
    "frame-delta": "frame_delta",
    "frame_delta_action": "frame_delta",
    "frame-delta-action": "frame_delta",
}


def aggregate_v21_stats(
    root: str | Path,
    episodes: Sequence[int] | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    """Combine per-episode stats (``meta/episodes_stats.jsonl``) into dataset
    stats shaped like ``LeRobotDatasetMetadata.stats`` (the normalizer input):
    weighted mean, pooled std via E[x^2], elementwise min/max, and the
    episode-count-weighted q01/q99 estimates carried by v2.1 metadata.
    """
    requested = {int(episode) for episode in episodes} if episodes is not None else None
    found: set[int] = set()
    accumulators: dict[str, dict[str, Any]] = {}

    # Full EgoDex metadata is several GiB. Aggregate line-by-line so startup
    # memory scales with feature width rather than number of episodes. For a
    # bounded debug subset the v2.1 index is ordered by episode_index, so we can
    # stop as soon as every requested episode has been seen.
    with open(Path(root) / _V21_EPISODE_STATS) as handle:
        for line in handle:
            if not line.strip():
                continue
            episode = json.loads(line)
            episode_index = int(episode["episode_index"])
            if requested is not None and episode_index not in requested:
                continue
            found.add(episode_index)
            for key, stats in episode["stats"].items():
                n = float(np.asarray(stats["count"]).reshape(-1)[0])
                mean = np.asarray(stats["mean"], dtype=np.float64)
                std = np.asarray(stats["std"], dtype=np.float64)
                acc = accumulators.setdefault(
                    key,
                    {
                        "count": 0.0,
                        "mean": np.zeros_like(mean, dtype=np.float64),
                        "square": np.zeros_like(mean, dtype=np.float64),
                        "min": np.asarray(stats["min"], dtype=np.float64),
                        "max": np.asarray(stats["max"], dtype=np.float64),
                        "quantiles": {},
                    },
                )
                acc["count"] += n
                acc["mean"] += n * mean
                acc["square"] += n * (std**2 + mean**2)
                acc["min"] = np.minimum(acc["min"], stats["min"])
                acc["max"] = np.maximum(acc["max"], stats["max"])
                for quantile in ("q01", "q99"):
                    if quantile in stats:
                        value = n * np.asarray(stats[quantile], dtype=np.float64)
                        quantiles = acc["quantiles"]
                        quantiles[quantile] = quantiles.get(quantile, 0.0) + value
            if requested is not None and found == requested:
                break

    if requested is not None and found != requested:
        raise ValueError(
            f"{root}: requested episode stats not found: {sorted(requested - found)}"
        )
    if not accumulators:
        raise FileNotFoundError(f"no episode stats under {root}")
    agg: dict[str, dict[str, np.ndarray]] = {}
    for key, acc in accumulators.items():
        n_tot = acc["count"]
        mean_tot = acc["mean"] / n_tot
        var = np.maximum(acc["square"] / n_tot - mean_tot**2, 0.0)
        agg[key] = {
            "mean": mean_tot.astype(np.float32),
            "std": np.sqrt(var).astype(np.float32),
            "min": acc["min"].astype(np.float32),
            "max": acc["max"].astype(np.float32),
            "count": np.full(mean_tot.shape, int(n_tot), dtype=np.int64),
        }
        for quantile, total in acc["quantiles"].items():
            agg[key][quantile] = (total / n_tot).astype(np.float32)
    return agg


def _read_episode_metadata(
    path: Path,
    episodes: Sequence[int] | None = None,
) -> list[tuple[int, int, list[str]]]:
    """Read only the episode fields needed by the dataset index.

    Retaining every raw JSON object from large v2.1 metadata files can require
    tens of GiB. This keeps the compact runtime index and exits early for a
    bounded episode subset.
    """
    requested = {int(episode) for episode in episodes} if episodes is not None else None
    found: set[int] = set()
    rows: list[tuple[int, int, list[str]]] = []
    with open(path) as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            episode_index = int(row["episode_index"])
            if requested is not None and episode_index not in requested:
                continue
            tasks = row.get("tasks") or ["unknown"]
            rows.append((episode_index, int(row["length"]), list(tasks)))
            found.add(episode_index)
            if requested is not None and found == requested:
                break
    if requested is not None and found != requested:
        raise ValueError(
            f"requested episodes not found in {path.parent.parent}: "
            f"{sorted(requested - found)}"
        )
    rows.sort(key=lambda row: row[0])
    return rows


def _resolve_task_filter(root: Path, task_filter: str | int | None) -> str | None:
    """Resolve a v2.1 task index or exact task text to canonical task text."""

    if task_filter is None:
        return None
    if isinstance(task_filter, bool):
        raise TypeError("task_filter must be a task index or task text, not boolean")

    task_index: int | None = None
    if isinstance(task_filter, int):
        task_index = task_filter
    elif isinstance(task_filter, str):
        value = task_filter.strip()
        if not value:
            raise ValueError("task_filter must not be empty")
        try:
            task_index = int(value)
        except ValueError:
            return value
    else:
        raise TypeError(
            "task_filter must be a task index or exact task text, got "
            f"{type(task_filter).__name__}"
        )

    tasks_path = root / _V21_TASKS
    if not tasks_path.exists():
        raise FileNotFoundError(
            f"{root}: cannot resolve task index {task_index}; missing {_V21_TASKS}"
        )
    available_indices: list[int] = []
    with tasks_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            index = int(row["task_index"])
            available_indices.append(index)
            if index == task_index:
                task = str(row.get("task", "")).strip()
                if not task:
                    raise ValueError(
                        f"{tasks_path}: task_index {task_index} has empty task text"
                    )
                return task
    raise ValueError(
        f"{root}: task_index {task_index} not found in {_V21_TASKS}; "
        f"available indices={available_indices}"
    )


def _filter_episode_rows_by_task(
    rows: Sequence[tuple[int, int, list[str]]],
    task: str | None,
    *,
    root: Path,
) -> list[tuple[int, int, list[str]]]:
    if task is None:
        return rows if isinstance(rows, list) else list(rows)
    selected = [row for row in rows if task in row[2]]
    if not selected:
        raise ValueError(
            f"{root}: no episodes match task_filter={task!r}; matching is exact "
            "against meta/episodes.jsonl tasks"
        )
    return selected


def _readonly_array(values: Any, *, dtype: np.dtype) -> np.ndarray:
    """Return one contiguous, immutable array suitable for sharing across fork."""

    array = np.fromiter(values, dtype=dtype)
    array.setflags(write=False)
    return array


def _encode_episode_labels(values: Sequence[str]) -> tuple[np.ndarray, tuple[str, ...]]:
    """Dictionary-encode per-episode strings without Python objects per row.

    DataLoader workers inherit dataset metadata through fork. A Python string
    (or a one-element list containing it) has a mutable reference count, so merely
    returning a randomly selected episode's label dirties the containing page and
    gradually copies the parent's metadata into every worker. Integer NumPy ids are
    read without touching Python object headers; only the small table of distinct
    labels remains made of Python strings.
    """

    table: list[str] = []
    label_to_id: dict[str, int] = {}
    ids = np.empty(len(values), dtype=np.int32)
    for position, raw_value in enumerate(values):
        value = str(raw_value)
        label_id = label_to_id.get(value)
        if label_id is None:
            label_id = len(table)
            label_to_id[value] = label_id
            table.append(value)
        ids[position] = label_id
    ids.setflags(write=False)
    return ids, tuple(table)


def _compact_episode_rows(
    rows: Sequence[tuple[int, int, list[str]]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
    """Encode the runtime episode table in COW-safe, behavior-preserving arrays."""

    indices = _readonly_array((row[0] for row in rows), dtype=np.int64)
    lengths = _readonly_array((row[1] for row in rows), dtype=np.int64)
    # The JSON metadata reader normalizes missing or empty tasks to unknown.
    # Keep that contract for sidecars too, including an older sidecar that may
    # contain an explicitly empty task list.
    task_ids, task_table = _encode_episode_labels(
        [row[2][0] if row[2] else "unknown" for row in rows]
    )
    return indices, lengths, task_ids, task_table


def _read_episode_subtasks(
    path: Path,
    episodes: Sequence[int] | None = None,
) -> dict[int, tuple[np.ndarray, np.ndarray, list[str]]]:
    """Read optional per-episode subtask segments from v2.1 metadata."""
    requested = {int(episode) for episode in episodes} if episodes is not None else None
    resolved: dict[int, tuple[np.ndarray, np.ndarray, list[str]]] = {}
    with open(path) as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            episode_index = int(row["episode_index"])
            if requested is not None and episode_index not in requested:
                continue
            names = row.get("subtask_names")
            starts = row.get("subtask_start_frames")
            ends = row.get("subtask_end_frames")
            if names and starts and ends:
                if not len(names) == len(starts) == len(ends):
                    raise ValueError(
                        f"{path}: episode {episode_index} has {len(names)} subtask names "
                        f"but {len(starts)} starts and {len(ends)} ends"
                    )
                order = np.argsort(np.asarray(starts, dtype=np.int64), kind="stable")
                resolved[episode_index] = (
                    np.asarray(starts, dtype=np.int64)[order],
                    np.asarray(ends, dtype=np.int64)[order],
                    [str(names[position]) for position in order],
                )
    return resolved


def _read_episode_embodiments(
    path: Path,
    spec: EmbodimentSpec,
    episodes: Sequence[int] | None = None,
) -> dict[int, str]:
    """Resolve one robot name per episode from ``meta/episodes.jsonl``.

    A second pass over the metadata rather than a fourth field in
    :func:`_read_episode_metadata`, so the hot path -- EgoDex's 9.3 GB file,
    parsed on every cold start -- keeps its three-field shape and its cost.  Only
    a source that actually configures ``episode_meta_path`` pays for this, and
    today that is RoboTwin2.0 alone.

    The path is a list of segments, because the field it targets lives under a
    key that itself contains dots.  ``[]`` suffixed on a segment selects the
    first list element.  A path that does not resolve yields
    ``UNKNOWN_EMBODIMENT`` for that episode instead of raising, because a
    partially-annotated dataset should degrade to "no claim" per episode rather
    than fail the whole run.
    """
    requested = {int(episode) for episode in episodes} if episodes is not None else None
    resolved: dict[int, str] = {}
    parts = list(spec.episode_meta_path or ())
    with open(path) as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            episode_index = int(row["episode_index"])
            if requested is not None and episode_index not in requested:
                continue
            node: Any = row
            for part in parts:
                if part.endswith("[]"):
                    node = node.get(part[:-2]) if isinstance(node, Mapping) else None
                    node = node[0] if isinstance(node, list) and node else None
                elif isinstance(node, Mapping):
                    node = node.get(part)
                else:
                    node = None
                if node is None:
                    break
            resolved[episode_index] = (
                spec.match_episode_meta(str(node))
                if isinstance(node, str)
                else UNKNOWN_EMBODIMENT
            )
            if requested is not None and len(resolved) == len(requested):
                break
    return resolved


def _validate_indices(
    indices: Sequence[int] | None,
    size: int,
    *,
    name: str,
) -> tuple[int, ...] | None:
    if indices is None:
        return None
    resolved = tuple(int(index) for index in indices)
    if not resolved:
        raise ValueError(f"{name} must not be empty")
    if len(set(resolved)) != len(resolved):
        raise ValueError(f"{name} must not contain duplicates")
    if min(resolved) < 0 or max(resolved) >= size:
        raise ValueError(f"{name}={resolved} is outside source dimension {size}")
    return resolved


def _validate_target_indices(
    indices: Sequence[int] | None,
    selected_size: int,
    *,
    name: str,
) -> tuple[int, ...] | None:
    """Validate destination slots for selected source dimensions."""
    if indices is None:
        return None
    resolved = tuple(int(index) for index in indices)
    if len(resolved) != selected_size:
        raise ValueError(
            f"{name} must contain one destination per selected source dimension; "
            f"expected {selected_size}, got {len(resolved)}"
        )
    if len(set(resolved)) != len(resolved):
        raise ValueError(f"{name} must not contain duplicate destination slots")
    if resolved and min(resolved) < 0:
        raise ValueError(f"{name} must contain only non-negative destination slots")
    return resolved


def _validate_rotation_groups(
    groups: Sequence[Sequence[int]] | None,
    size: int,
    *,
    name: str,
) -> tuple[tuple[int, ...], ...]:
    """Validate non-overlapping rot6d groups in the adapted action space."""
    if groups is None:
        return ()
    if isinstance(groups, (str, bytes)) or not isinstance(groups, Sequence):
        raise TypeError(f"{name} must be a sequence of six-index groups")
    resolved: list[tuple[int, ...]] = []
    used: set[int] = set()
    for position, raw_group in enumerate(groups):
        if isinstance(raw_group, (str, bytes)) or not isinstance(raw_group, Sequence):
            raise TypeError(f"{name}[{position}] must be a sequence of six indices")
        group = tuple(int(index) for index in raw_group)
        if len(group) != 6:
            raise ValueError(f"{name}[{position}] must contain exactly 6 indices")
        if len(set(group)) != 6:
            raise ValueError(f"{name}[{position}] must not contain duplicates")
        if min(group) < 0 or max(group) >= size:
            raise ValueError(
                f"{name}[{position}]={group} is outside shared dimension {size}"
            )
        overlap = used.intersection(group)
        if overlap:
            raise ValueError(f"{name} groups overlap at indices {sorted(overlap)}")
        used.update(group)
        resolved.append(group)
    return tuple(resolved)


def _validate_rotation_delta_partition(
    groups: Sequence[Sequence[int]], delta_indices: Sequence[int]
) -> None:
    """A rotation is either wholly relative or wholly absolute, never half of each."""
    delta_set = set(delta_indices)
    for group in groups:
        overlap = delta_set.intersection(group)
        if overlap and len(overlap) != len(group):
            raise ValueError(
                "action_delta_indices partially overlaps action_rotation_groups: "
                f"group={tuple(group)}, delta_members={sorted(overlap)}"
            )


def _validate_rotation_supervision_partition(
    groups: Sequence[Sequence[int]], supervision_indices: Sequence[int] | None
) -> None:
    """Require rot6d supervision to keep each six-value pose atomically."""
    if supervision_indices is None:
        return
    supervised = set(int(index) for index in supervision_indices)
    for group in groups:
        overlap = supervised.intersection(group)
        if overlap and len(overlap) != len(group):
            raise ValueError(
                "action_supervision_indices partially overlaps action_rotation_groups: "
                f"group={tuple(group)}, supervised_members={sorted(overlap)}"
            )


def _validate_selected_action_layout(
    *,
    selected_dim: int,
    target_dim: int,
    indices: tuple[int, ...] | None,
    preserve_selected_slots: bool,
    target_indices: tuple[int, ...] | None = None,
    delta_indices: Sequence[int],
    rotation_groups: Sequence[Sequence[int]],
) -> None:
    """Reject transform indices that point into right-padding.

    A selected action can either be compacted to ``selected_dim`` or retained
    in its source slots.  In the compact case, the target vector may still be
    wider because the model contract requires right-padding, but those padded
    dimensions are not real actions and must not participate in delta or
    rotation transforms.
    """
    if preserve_selected_slots or indices is None:
        return
    valid_slots = (
        set(target_indices) if target_indices is not None else set(range(selected_dim))
    )
    invalid = sorted(
        {
            int(index)
            for index in (
                *delta_indices,
                *(index for group in rotation_groups for index in group),
            )
            if int(index) not in valid_slots
        }
    )
    if invalid:
        raise ValueError(
            "action transform indices point into right-padding or unmapped/padded slots after "
            f"selection: invalid={invalid}, valid_slots={sorted(valid_slots)}, "
            f"target_dim={target_dim}; use target_indices or "
            "preserve_selected_slots=True for source-slot indices"
        )


def _parse_action_representation(value: str) -> str:
    normalized = str(value).strip().lower()
    try:
        return _ACTION_REPRESENTATION_ALIASES[normalized]
    except KeyError as exc:
        raise ValueError(
            f"unsupported action_representation {value!r}; expected one of "
            "['abs', 'chunk_delta', 'frame_delta']"
        ) from exc


def _adapt_last_dim(
    value: np.ndarray,
    *,
    indices: tuple[int, ...] | None,
    target_indices: tuple[int, ...] | None = None,
    target_dim: int,
    pad_value: float | bool,
    preserve_selected_slots: bool = False,
) -> np.ndarray:
    if preserve_selected_slots:
        if target_indices is not None:
            raise ValueError(
                "target_indices and preserve_selected_slots are mutually exclusive"
            )
        if indices is None:
            raise ValueError("preserve_selected_slots requires explicit indices")
        if indices and max(indices) >= target_dim:
            raise ValueError(
                f"selected index {max(indices)} exceeds target dimension {target_dim}"
            )
        adapted = np.full((*value.shape[:-1], target_dim), pad_value, dtype=value.dtype)
        adapted[..., list(indices)] = value[..., list(indices)]
        return adapted
    adapted = value[..., list(indices)] if indices is not None else value
    source_dim = int(adapted.shape[-1])
    if target_indices is not None:
        target_indices = _validate_target_indices(
            target_indices, source_dim, name="target_indices"
        )
        assert target_indices is not None
        if target_indices and max(target_indices) >= target_dim:
            raise ValueError(
                f"target index {max(target_indices)} exceeds target dimension {target_dim}"
            )
        projected = np.full(
            (*adapted.shape[:-1], target_dim), pad_value, dtype=adapted.dtype
        )
        projected[..., list(target_indices)] = adapted
        return projected
    if source_dim > target_dim:
        raise ValueError(
            f"selected dimension {source_dim} exceeds target dimension {target_dim}; "
            "set explicit indices to select source fields"
        )
    if source_dim == target_dim:
        return adapted
    pad_shape = (*adapted.shape[:-1], target_dim - source_dim)
    padding = np.full(pad_shape, pad_value, dtype=adapted.dtype)
    return np.concatenate((adapted, padding), axis=-1)


def _mask_inactive_dimensions(
    value: np.ndarray,
    dim_mask: np.ndarray,
    *,
    active_indices: tuple[int, ...] | None,
) -> None:
    """Zero inactive model-space slots without compacting or reordering them."""
    if active_indices is None:
        return
    if value.shape != dim_mask.shape:
        raise ValueError(
            "value and dimension mask must have identical shapes before applying "
            f"active indices: value={value.shape}, mask={dim_mask.shape}"
        )
    active = np.zeros(value.shape[-1], dtype=np.bool_)
    active[list(active_indices)] = True
    value[..., ~active] = 0
    dim_mask[..., ~active] = False


def _adapt_feature_stats(
    stats: Mapping[str, np.ndarray],
    *,
    indices: tuple[int, ...] | None,
    target_indices: tuple[int, ...] | None = None,
    target_dim: int,
    preserve_selected_slots: bool = False,
) -> dict[str, np.ndarray]:
    """Select/pad normalization stats without making padded zeros normalize to -1."""
    output: dict[str, np.ndarray] = {}
    for name, raw in stats.items():
        value = np.asarray(raw)
        if value.ndim == 0 or value.size == 1:
            output[name] = value.copy()
            continue
        if preserve_selected_slots:
            if target_indices is not None:
                raise ValueError(
                    "target_indices and preserve_selected_slots are mutually exclusive"
                )
            if indices is None:
                raise ValueError("preserve_selected_slots requires explicit indices")
            fill = (
                1.0
                if name in {"std", "max", "q99"}
                else -1.0
                if name in {"min", "q01"}
                else 0.0
            )
            selected = np.full((*value.shape[:-1], target_dim), fill, dtype=value.dtype)
            selected[..., list(indices)] = value[..., list(indices)]
            output[name] = selected
            continue
        selected = value[..., list(indices)] if indices is not None else value
        source_dim = int(selected.shape[-1])
        if target_indices is not None:
            target_indices = _validate_target_indices(
                target_indices, source_dim, name="target_indices"
            )
            assert target_indices is not None
            if target_indices and max(target_indices) >= target_dim:
                raise ValueError(
                    f"target index {max(target_indices)} exceeds target dimension {target_dim}"
                )
            fill = (
                1.0
                if name in {"std", "max", "q99"}
                else -1.0
                if name in {"min", "q01"}
                else 0.0
            )
            projected = np.full(
                (*selected.shape[:-1], target_dim), fill, dtype=selected.dtype
            )
            projected[..., list(target_indices)] = selected
            output[name] = projected
            continue
        if source_dim > target_dim:
            raise ValueError(
                f"selected stats dimension {source_dim} exceeds target dimension {target_dim}"
            )
        if source_dim < target_dim:
            fill = (
                1.0
                if name in {"std", "max", "q99"}
                else -1.0
                if name in {"min", "q01"}
                else 0.0
            )
            padding = np.full(
                (*selected.shape[:-1], target_dim - source_dim),
                fill,
                dtype=selected.dtype,
            )
            selected = np.concatenate((selected, padding), axis=-1)
        output[name] = selected
    return output


def _parse_quality_rule(
    rule: Mapping[str, Any] | None,
    *,
    feature_names: set[str],
    name: str,
) -> dict[str, tuple[str, ...]]:
    if rule is None:
        return {"require_true": (), "require_false": ()}
    if not isinstance(rule, Mapping):
        raise TypeError(f"quality.{name} must be a mapping")
    unknown = set(rule) - {"require_true", "require_false"}
    if unknown:
        raise ValueError(
            f"quality.{name} has unknown keys {sorted(unknown)}; "
            "allowed ['require_false', 'require_true']"
        )
    parsed = {
        key: tuple(str(field) for field in (rule.get(key) or ()))
        for key in ("require_true", "require_false")
    }
    missing = (
        set(parsed["require_true"]) | set(parsed["require_false"])
    ) - feature_names
    if missing:
        raise ValueError(
            f"quality.{name} references missing features: {sorted(missing)}"
        )
    overlap = set(parsed["require_true"]) & set(parsed["require_false"])
    if overlap:
        raise ValueError(
            f"quality.{name} requires fields both true and false: {sorted(overlap)}"
        )
    return parsed


def _quality_mask(
    table: Mapping[str, np.ndarray],
    rule: Mapping[str, tuple[str, ...]],
    length: int,
) -> np.ndarray:
    mask = np.ones(length, dtype=np.bool_)
    for field in rule["require_true"]:
        values = np.asarray(table[field])
        if values.ndim != 1:
            raise ValueError(
                f"quality predicate {field!r} must be scalar per frame, got shape {values.shape}"
            )
        mask &= values.astype(np.bool_)
    for field in rule["require_false"]:
        values = np.asarray(table[field])
        if values.ndim != 1:
            raise ValueError(
                f"quality predicate {field!r} must be scalar per frame, got shape {values.shape}"
            )
        mask &= ~values.astype(np.bool_)
    return mask


def _parse_camera_rotations(
    rotations: Mapping[str, int] | None,
    *,
    camera_keys: Sequence[str],
) -> dict[str, int]:
    """Validate per-camera clockwise rotations and normalize them to [0, 360)."""

    if rotations is None:
        return {}
    if not isinstance(rotations, Mapping):
        raise TypeError(
            "camera_rotations must map canonical camera keys to clockwise degrees"
        )

    selected = set(camera_keys)
    parsed: dict[str, int] = {}
    for raw_key, raw_degrees in rotations.items():
        key = str(raw_key)
        if key not in selected:
            raise ValueError(
                "camera_rotations references an unselected canonical camera: "
                f"{key!r}; selected={sorted(selected)}"
            )
        if isinstance(raw_degrees, bool) or not isinstance(
            raw_degrees, (int, np.integer)
        ):
            raise TypeError(
                f"camera_rotations[{key!r}] must be an integer number of degrees"
            )
        degrees = int(raw_degrees)
        if degrees % 90 != 0:
            raise ValueError(
                f"camera_rotations[{key!r}] must be a multiple of 90 degrees"
            )
        normalized = degrees % 360
        if normalized:
            parsed[key] = normalized
    return parsed


class LeRobotV21Dataset(Dataset):
    """Frame-indexed reader over one v2.1 dataset.

    Args:
        root: dataset directory (contains ``data/``, ``videos/``, ``meta/``).
        chunk_size: action-horizon length; actions ``[t, t+chunk)`` are stacked
            and right-padded with the last row (mask in ``<action_key>_is_pad``),
            mirroring lerobot ``delta_timestamps`` behaviour.
        action_key: which parquet column is the training action; ``None`` picks
            ``real_action`` when present, else ``action``.
        task_override: replace the (often placeholder) task string.
        embodiment: which robot produced this source, as a registry name, a
            registry index, or a mapping selecting one of the strategies in
            :class:`data.embodiment.EmbodimentSpec`. ``None`` reads
            ``observation.embodiment_id`` when the dataset has it and reports
            ``"unknown"`` otherwise, which is what half the current training
            sources need -- they predate the column.
        camera_keys: optional source-camera subset; unselected videos are never decoded.
        camera_key_map: optional ``{source_key: canonical_key}`` rename applied
            to decoded sample fields. ``camera_keys`` then exposes canonical keys
            so sources with equivalent cameras can share one model schema.
        camera_rotations: optional ``{canonical_camera_key: clockwise_degrees}``
            transform applied after video decode. Angles must be multiples of 90;
            negative values rotate counter-clockwise. The same transform is used
            for current/history observations and explicitly requested future frames.
        preserve_selected_slots: when state/action indices select a subset, keep
            each selected value at its original source index instead of packing
            the subset into leading dimensions. Unselected slots are padded and
            masked. This is required when action32 assigns fixed semantic slots.
        state_target_indices/action_target_indices: optional destination slot for
            every selected source dimension. This supports schema migrations that
            insert dimensions in the middle of a vector: unmapped target slots are
            zero-filled and receive a false dimension mask. Mutually exclusive
            with ``preserve_selected_slots``.
        target_state_dim/target_action_dim: right-pad source vectors into a common
            model space. Padded dimensions receive a false dimension mask.
        state_active_indices/action_active_indices: optional indices in the final
            model space which remain visible/supervised. Unlike ``state_indices``
            and ``action_indices``, these do not select-and-compact source fields:
            inactive slots keep their original positions but are zeroed and receive
            a false dimension mask.
        state_mask_key/action_mask_key: optional per-frame vector masks stored in
            parquet. They are selected/padded alongside state and action values.
        action_upsample_rate: integer temporal stretch applied only while building
            training action chunks. For example, rate 2 reads ceil(chunk/2)
            source frames and repeats each action twice to produce ``chunk`` rows.
        action_upsample_mode: ``"semantic"`` applies per-dimension interpolation
            configured by ``action_interpolation``. ``"repeat"`` is retained only
            for backward compatibility and performs zero-order hold.
        action_interpolation: semantic dimension groups. Joint positions use
            quintic Hermite interpolation; gripper and EEF position use linear
            interpolation; each rot6d group uses Gaussian-time-weighted SLERP.
            Its ``gripper_indices`` also identify dimensions which always remain
            absolute under either delta representation.
        action_representation: target representation returned by the reader:
            ``"abs"`` keeps stored absolute actions; ``"chunk_delta"`` subtracts
            the current state from every chunk row; ``"frame_delta"`` subtracts
            the preceding action (the first row subtracts the current state).
        action_delta_indices: optional dimensions to convert in either delta mode,
            after state/action selection and padding. Other dimensions stay absolute.
            Gripper dimensions declared in ``action_interpolation`` always stay
            absolute, including when listed here. The default converts every other
            dimension shared by state and action.
        action_supervision_indices: optional dimensions to keep in the emitted
            action dimension mask. Other action values remain in the 32-D model
            tensor but are excluded from flow/action loss and inference masking.
        action_rotation_groups: six-index rot6d groups in the adapted action/state
            space. Groups included in ``action_delta_indices`` use
            ``R_target @ R_reference.T`` instead of elementwise subtraction. A group
            may be wholly relative or wholly absolute; partial overlap is rejected.
            Kept separate from ``action_interpolation`` because rotation composition
            is required even for sources that are not temporally upsampled.
        action_stats_samples: maximum eligible anchors used to estimate action
            normalization statistics for a delta representation. No video is decoded.
        index_cache_path: where to look for the ``.anchors.npz`` /
            ``.episodes.npz`` sidecars when no ``norm_stats_path`` is given.
            The sidecars cache the two expensive parts of a cold start -- the
            quality scan over 280k EgoDex columns and parsing its 9.3 GB
            ``episodes.jsonl`` -- and neither depends on the action
            representation, so recomputing statistics can reuse them.  Without
            this, ``compute_lerobot_v21_norm.py`` has to drop
            ``norm_stats_path`` (the artifact it is about to write does not
            exist yet) and pays the full scan again.
        norm_stats_path: optional versioned JSON artifact produced by
            ``scripts/compute_norm_stats.py``. Its dataset/transform
            signature must exactly match this reader configuration.
        allow_norm_stats_dataset_mismatch: permit only the artifact's ``dataset``
            signature section to differ. State/action transforms and quality
            rules remain strict. Intended for explicit full-corpus statistics
            applied to a task subset; disabled by default.
        quality: optional ``current_frame`` and ``action_chunk`` boolean
            predicates. ``current_frame`` builds the dataset anchor index space;
            ``action_chunk`` independently masks future action timesteps.
        anchor_sampling: optional per-source episode-local sample selection.
            ``stride_frames`` retains ``offset + n * stride`` anchors within each
            episode and unions them with optional keyframe/context anchors. The
            episode's last frame is never added specially. All resulting anchors
            must still pass ``quality.current_frame`` when configured.
        obs_steps: number of observations, including the current one.
        obs_interval_seconds: requested spacing between observations; converted
            to frames using ``meta/info.json`` fps.
        supervise_episode_end_padding: whether repeated actions used to pad the
            end of an episode remain valid supervision targets.
    """

    def __init__(
        self,
        root: str | Path,
        chunk_size: int = 50,
        action_key: str | None = None,
        task_override: str | None = None,
        embodiment: Any = None,
        decoder_cache: int = 8,
        episodes: Sequence[int] | None = None,
        task_filter: str | int | None = None,
        obs_steps: int = 1,
        obs_interval_seconds: float = 0.1,
        video_backend: str = "auto",
        camera_keys: Sequence[str] | None = None,
        camera_key_map: Mapping[str, str] | None = None,
        camera_rotations: Mapping[str, int] | None = None,
        state_indices: Sequence[int] | None = None,
        action_indices: Sequence[int] | None = None,
        preserve_selected_slots: bool = False,
        state_target_indices: Sequence[int] | None = None,
        action_target_indices: Sequence[int] | None = None,
        target_state_dim: int | None = None,
        target_action_dim: int | None = None,
        state_active_indices: Sequence[int] | None = None,
        action_active_indices: Sequence[int] | None = None,
        state_mask_key: str | None = None,
        action_mask_key: str | None = None,
        action_upsample_rate: int = 1,
        action_upsample_mode: str = "semantic",
        action_interpolation: Mapping[str, Any] | None = None,
        action_representation: str = "abs",
        action_delta_indices: Sequence[int] | None = None,
        action_supervision_indices: Sequence[int] | None = None,
        action_rotation_groups: Sequence[Sequence[int]] | None = None,
        allow_norm_stats_dataset_mismatch: bool = False,
        action_stats_samples: int = 10_000,
        quality: Mapping[str, Any] | None = None,
        anchor_sampling: Mapping[str, Any] | None = None,
        norm_stats_path: str | Path | None = None,
        index_cache_path: str | Path | None = None,
        estimate_action_stats: bool = True,
        load_metadata_stats: bool = True,
        supervise_episode_end_padding: bool = False,
        action_stats_per_horizon: bool = False,
        load_subtasks: bool = False,
    ):
        self.root = Path(root)
        info_path = self.root / _V21_INFO
        info_bytes = info_path.read_bytes()
        info = json.loads(info_bytes)
        self._info_sha256 = hashlib.sha256(info_bytes).hexdigest()
        if not str(info.get("codebase_version", "")).startswith("v2"):
            raise ValueError(
                f"{root}: codebase_version={info.get('codebase_version')!r}, "
                "expected v2.x"
            )
        self.fps: float = float(info["fps"])
        self.obs_steps = int(obs_steps)
        self.obs_interval_seconds = float(obs_interval_seconds)
        self.obs_stride, self.observation_delta_indices = validate_observation_sampling(
            obs_steps=self.obs_steps,
            obs_interval_seconds=self.obs_interval_seconds,
            fps=self.fps,
        )
        self.chunk_size = int(chunk_size)
        self.supervise_episode_end_padding = bool(supervise_episode_end_padding)
        self.video_backend = video_backend
        self._data_path: str = info["data_path"]
        self._video_path: str = info["video_path"]
        self._chunks_size: int = int(info.get("chunks_size", 1000))
        feats = info["features"]
        feature_names = set(feats)
        self._feature_names = feature_names
        available_cameras = [k for k, v in feats.items() if v.get("dtype") == "video"]
        self._source_camera_keys = (
            list(camera_keys) if camera_keys is not None else available_cameras
        )
        missing_cameras = set(self._source_camera_keys) - set(available_cameras)
        if missing_cameras:
            raise ValueError(
                f"requested cameras are missing from {root}: {sorted(missing_cameras)}"
            )
        if camera_key_map is None:
            self.camera_key_map: dict[str, str] = {}
        elif not isinstance(camera_key_map, Mapping):
            raise TypeError(
                "camera_key_map must map source camera keys to canonical keys"
            )
        else:
            self.camera_key_map = {
                str(source): str(target) for source, target in camera_key_map.items()
            }
        unknown_mapped_cameras = set(self.camera_key_map) - set(available_cameras)
        if unknown_mapped_cameras:
            raise ValueError(
                f"camera_key_map references unknown source cameras in {root}: "
                f"{sorted(unknown_mapped_cameras)}"
            )
        unselected_mapped_cameras = set(self.camera_key_map) - set(
            self._source_camera_keys
        )
        if unselected_mapped_cameras:
            raise ValueError(
                "camera_key_map may only rename selected camera_keys; "
                f"unselected={sorted(unselected_mapped_cameras)}"
            )
        self.camera_keys = [
            self.camera_key_map.get(source_key, source_key)
            for source_key in self._source_camera_keys
        ]
        if not all(self.camera_keys):
            raise ValueError("camera_key_map canonical names must be non-empty")
        if len(set(self.camera_keys)) != len(self.camera_keys):
            raise ValueError(
                "camera_key_map produces duplicate canonical camera keys: "
                f"{self.camera_keys}"
            )
        self.camera_rotations = _parse_camera_rotations(
            camera_rotations, camera_keys=self.camera_keys
        )
        if action_key is None:
            action_key = "real_action" if "real_action" in feats else "action"
        self.action_key = action_key
        self.state_key = (
            "observation.qpos" if "observation.qpos" in feats else "observation.state"
        )
        if self.state_key not in feats or self.action_key not in feats:
            raise ValueError(
                f"{root}: missing state/action feature {self.state_key!r}/{self.action_key!r}"
            )
        source_state_dim = int(feats[self.state_key]["shape"][-1])
        source_action_dim = int(feats[self.action_key]["shape"][-1])
        self.source_state_dim = source_state_dim
        self.source_action_dim = source_action_dim
        self.state_indices = _validate_indices(
            state_indices, source_state_dim, name="state_indices"
        )
        self.action_indices = _validate_indices(
            action_indices, source_action_dim, name="action_indices"
        )
        self.preserve_selected_slots = bool(preserve_selected_slots)
        if self.preserve_selected_slots and (
            state_target_indices is not None or action_target_indices is not None
        ):
            raise ValueError(
                "state/action_target_indices and preserve_selected_slots are "
                "mutually exclusive"
            )
        if self.preserve_selected_slots and (
            self.state_indices is None or self.action_indices is None
        ):
            raise ValueError(
                "preserve_selected_slots requires state_indices and action_indices"
            )
        selected_state_dim = (
            len(self.state_indices)
            if self.state_indices is not None
            else source_state_dim
        )
        selected_action_dim = (
            len(self.action_indices)
            if self.action_indices is not None
            else source_action_dim
        )
        self.state_target_indices = _validate_target_indices(
            state_target_indices,
            selected_state_dim,
            name="state_target_indices",
        )
        self.action_target_indices = _validate_target_indices(
            action_target_indices,
            selected_action_dim,
            name="action_target_indices",
        )
        minimum_state_dim = (
            max(self.state_target_indices) + 1
            if self.state_target_indices
            else source_state_dim
            if self.preserve_selected_slots
            else selected_state_dim
        )
        minimum_action_dim = (
            max(self.action_target_indices) + 1
            if self.action_target_indices
            else source_action_dim
            if self.preserve_selected_slots
            else selected_action_dim
        )
        self.target_state_dim = int(
            minimum_state_dim if target_state_dim is None else target_state_dim
        )
        self.target_action_dim = int(
            minimum_action_dim if target_action_dim is None else target_action_dim
        )
        if self.target_state_dim <= 0 or self.target_action_dim <= 0:
            raise ValueError("target_state_dim and target_action_dim must be positive")
        if self.target_state_dim < minimum_state_dim:
            raise ValueError(
                f"target_state_dim={self.target_state_dim} cannot hold projected "
                f"state slots through {minimum_state_dim - 1}"
            )
        if self.target_action_dim < minimum_action_dim:
            raise ValueError(
                f"target_action_dim={self.target_action_dim} cannot hold projected "
                f"action slots through {minimum_action_dim - 1}"
            )
        active_state_indices = (
            self.state_target_indices
            if self.state_target_indices is not None
            else self.state_indices
            if self.preserve_selected_slots
            else tuple(range(selected_state_dim))
        )
        active_action_indices = (
            self.action_target_indices
            if self.action_target_indices is not None
            else self.action_indices
            if self.preserve_selected_slots
            else tuple(range(selected_action_dim))
        )
        self.state_active_indices = _validate_indices(
            state_active_indices,
            self.target_state_dim,
            name="state_active_indices",
        )
        self.action_active_indices = _validate_indices(
            action_active_indices,
            self.target_action_dim,
            name="action_active_indices",
        )
        if self.state_active_indices is not None:
            active_state_indices = self.state_active_indices
        if self.action_active_indices is not None:
            active_action_indices = self.action_active_indices
        self.action_representation = _parse_action_representation(action_representation)
        self.state_mask_key = state_mask_key
        self.action_mask_key = action_mask_key
        self.action_upsample_rate = int(action_upsample_rate)
        self.action_upsample_mode = str(action_upsample_mode)
        if self.action_upsample_rate <= 0:
            raise ValueError("action_upsample_rate must be a positive integer")
        if self.action_upsample_mode not in {"repeat", "semantic"}:
            raise ValueError(
                "action_upsample_mode must be 'semantic' or legacy 'repeat', got "
                f"{self.action_upsample_mode!r}"
            )
        self.action_interpolation_spec: ActionInterpolationSpec | None = None
        if action_interpolation is not None or (
            self.action_upsample_mode == "semantic" and self.action_upsample_rate > 1
        ):
            self.action_interpolation_spec = parse_action_interpolation_spec(
                action_interpolation,
                source_action_dim=selected_action_dim,
                target_action_dim=self.target_action_dim,
                active_action_indices=active_action_indices,
            )
        if action_delta_indices is None:
            delta_indices = tuple(
                sorted(set(active_state_indices).intersection(active_action_indices))
            )
        else:
            delta_indices = _validate_indices(
                action_delta_indices,
                min(self.target_state_dim, self.target_action_dim),
                name="action_delta_indices",
            )
        gripper_indices = (
            set(self.action_interpolation_spec.gripper_indices)
            if self.action_interpolation_spec is not None
            else set()
        )
        self.action_delta_indices = tuple(
            index for index in delta_indices if index not in gripper_indices
        )
        self.action_supervision_indices = (
            None
            if action_supervision_indices is None
            else _validate_indices(
                action_supervision_indices,
                self.target_action_dim,
                name="action_supervision_indices",
            )
        )
        shared_dim = min(self.target_state_dim, self.target_action_dim)
        configured_rotation_groups = _validate_rotation_groups(
            action_rotation_groups,
            shared_dim,
            name="action_rotation_groups",
        )
        interpolation_rotation_groups = (
            tuple(self.action_interpolation_spec.rotation_groups)
            if self.action_interpolation_spec is not None
            else ()
        )
        if configured_rotation_groups and interpolation_rotation_groups:
            if configured_rotation_groups != interpolation_rotation_groups:
                raise ValueError(
                    "action_rotation_groups must match "
                    "action_interpolation.rotation_groups when both are configured"
                )
        self.action_rotation_groups = (
            configured_rotation_groups or interpolation_rotation_groups
        )
        _validate_rotation_supervision_partition(
            self.action_rotation_groups, self.action_supervision_indices
        )
        _validate_selected_action_layout(
            selected_dim=selected_action_dim,
            target_dim=self.target_action_dim,
            indices=self.action_indices,
            preserve_selected_slots=self.preserve_selected_slots,
            target_indices=self.action_target_indices,
            delta_indices=self.action_delta_indices,
            rotation_groups=self.action_rotation_groups,
        )
        if self.action_representation != "abs":
            _validate_rotation_delta_partition(
                self.action_rotation_groups, self.action_delta_indices
            )
            delta_set = set(self.action_delta_indices)
            self.action_relative_rotation_groups = tuple(
                group
                for group in self.action_rotation_groups
                if delta_set.issuperset(group)
            )
        else:
            self.action_relative_rotation_groups = ()
        self.action_rotation_delta = (
            "compose_v1" if self.action_relative_rotation_groups else "none"
        )
        self.action_stats_samples = int(action_stats_samples)
        if self.action_stats_samples <= 0:
            raise ValueError("action_stats_samples must be a positive integer")
        for mask_name, mask_key in (
            ("state_mask_key", state_mask_key),
            ("action_mask_key", action_mask_key),
        ):
            if mask_key is not None and mask_key not in feats:
                raise ValueError(
                    f"{root}: {mask_name}={mask_key!r} is not a dataset feature"
                )

        quality = dict(quality or {})
        unknown_quality = set(quality) - {"current_frame", "action_chunk"}
        if unknown_quality:
            raise ValueError(
                f"quality has unknown keys {sorted(unknown_quality)}; "
                "allowed ['action_chunk', 'current_frame']"
            )
        self._current_frame_quality_rule = _parse_quality_rule(
            quality.get("current_frame"),
            feature_names=feature_names,
            name="current_frame",
        )
        self._action_chunk_quality_rule = _parse_quality_rule(
            quality.get("action_chunk"),
            feature_names=feature_names,
            name="action_chunk",
        )
        self._quality_columns = tuple(
            sorted(
                set(self._current_frame_quality_rule["require_true"])
                | set(self._current_frame_quality_rule["require_false"])
                | set(self._action_chunk_quality_rule["require_true"])
                | set(self._action_chunk_quality_rule["require_false"])
            )
        )
        self._current_frame_quality_columns = tuple(
            sorted(
                set(self._current_frame_quality_rule["require_true"])
                | set(self._current_frame_quality_rule["require_false"])
            )
        )
        self.anchor_sampling_spec: AnchorSamplingSpec | None = (
            parse_anchor_sampling_spec(
                anchor_sampling,
                root=self.root,
                feature_names=feature_names,
            )
        )

        # meta/episodes.jsonl is 9.3 GB on EgoDex and yields three fields per
        # episode, so parsing it is the first and largest cost of a cold start --
        # it runs before the statistics and the anchor index, and unlike those two
        # nothing used to cache it.  The sidecar is only consulted for the full
        # dataset: a bounded `episodes` subset is a debugging path, and reading the
        # jsonl for it is both cheap (the reader stops early) and less surprising
        # than filtering a cache.
        resolved_task_filter = _resolve_task_filter(self.root, task_filter)
        episode_rows = None
        index_cache = (
            index_cache_path if index_cache_path is not None else norm_stats_path
        )
        if index_cache is not None and episodes is None:
            episode_rows = load_episode_index(index_cache)
            if episode_rows is not None:
                logger.info(
                    "%s: loaded %d episodes from the index sidecar, skipping %s",
                    self.root.name,
                    len(episode_rows),
                    _V21_EPISODES,
                )
        if episode_rows is None:
            episode_rows = _read_episode_metadata(self.root / _V21_EPISODES, episodes)
        episode_rows = _filter_episode_rows_by_task(
            episode_rows,
            resolved_task_filter,
            root=self.root,
        )
        if not episode_rows:
            raise ValueError(f"{root}: no episodes selected")
        if resolved_task_filter is not None:
            logger.info(
                "%s: task_filter=%r selected %d episodes",
                self.root.name,
                resolved_task_filter,
                len(episode_rows),
            )
        (
            self._ep_indices,
            self._ep_lengths,
            self._ep_task_ids,
            self._task_table,
        ) = _compact_episode_rows(episode_rows)
        self._ep_subtasks: (
            list[tuple[np.ndarray, np.ndarray, list[str]] | None] | None
        ) = None
        if load_subtasks:
            by_episode = _read_episode_subtasks(
                self.root / _V21_EPISODES, self._ep_indices
            )
            if not by_episode:
                raise ValueError(
                    f"{root}: load_subtasks is set but meta/episodes.jsonl carries no "
                    "subtask_names/subtask_start_frames/subtask_end_frames columns"
                )
            self._ep_subtasks = [
                by_episode.get(int(episode_index)) for episode_index in self._ep_indices
            ]
        # Drop the per-episode Python list/int/string graph before DataLoader
        # workers fork. The compact arrays above are immutable, so random reads
        # do not trigger per-worker Copy-on-Write faults.
        del episode_rows
        self._cum = np.empty(len(self._ep_lengths) + 1, dtype=np.int64)
        self._cum[0] = 0
        np.cumsum(self._ep_lengths, out=self._cum[1:])
        self._cum.setflags(write=False)
        episode_layout = hashlib.sha256()
        for episode_index, length in zip(
            self._ep_indices, self._ep_lengths, strict=True
        ):
            episode_layout.update(f"{episode_index}:{length}\n".encode())
        self._episode_layout_sha256 = episode_layout.hexdigest()
        self._task_override = task_override
        self._embodiment_spec = resolve_spec(embodiment)
        self._embodiment_constant: str | None = None
        self._embodiment_column: str | None = None
        self._ep_embodiment_ids: np.ndarray | None = None
        self._embodiment_table: tuple[str, ...] = ()
        # Legacy fallback for tests/readers constructed without __init__. New
        # production datasets never retain this per-episode Python dictionary.
        self._embodiment_by_episode: dict[int, str] | None = None
        self._resolve_embodiment_strategy()
        self._all_episodes_selected = episodes is None and resolved_task_filter is None
        self.action_stats_per_horizon = bool(action_stats_per_horizon)
        if (
            self.action_stats_per_horizon
            and self.action_representation != "chunk_delta"
        ):
            raise ValueError(
                f"{root}: action_stats_per_horizon needs "
                f"action_representation='chunk_delta', got "
                f"{self.action_representation!r}"
            )
        self.norm_stats_path = (
            Path(norm_stats_path) if norm_stats_path is not None else None
        )
        self.allow_norm_stats_dataset_mismatch = bool(allow_norm_stats_dataset_mismatch)
        if self.allow_norm_stats_dataset_mismatch and self.norm_stats_path is None:
            raise ValueError(
                "allow_norm_stats_dataset_mismatch requires norm_stats_path"
            )
        self.index_cache_path = Path(index_cache) if index_cache is not None else None

        # Per-worker lazy caches. Delta-stat estimation below intentionally uses
        # this same action/state-only path and clears it before DataLoader workers
        # are created, so no decoded video or large table cache is retained.
        self._df_cache: OrderedDict[int, dict[str, np.ndarray]] = OrderedDict()
        self._decoders: OrderedDict[str, object] = OrderedDict()
        self._decoder_cache = int(decoder_cache)

        self.meta_stats: dict[str, dict[str, np.ndarray]] = {}
        if self.norm_stats_path is None and load_metadata_stats:
            self.meta_stats = aggregate_v21_stats(
                self.root,
                episodes=None if self._all_episodes_selected else self._ep_indices,
            )
            self.meta_stats[self.state_key] = _adapt_feature_stats(
                self.meta_stats[self.state_key],
                indices=self.state_indices,
                target_indices=self.state_target_indices,
                target_dim=self.target_state_dim,
                preserve_selected_slots=self.preserve_selected_slots,
            )
            self.meta_stats[self.action_key] = _adapt_feature_stats(
                self.meta_stats[self.action_key],
                indices=self.action_indices,
                target_indices=self.action_target_indices,
                target_dim=self.target_action_dim,
                preserve_selected_slots=self.preserve_selected_slots,
            )
        # Anchors come from the sidecar when one was precomputed, because the scan
        # that produces them opens every episode parquet: on EgoDex that is 287,055
        # files for three boolean columns, ~75 minutes single-threaded, paid again
        # on every start and every crash-recovery.  A missing sidecar simply falls
        # through to the scan, so this is an optimisation and never a requirement.
        #
        # Nothing is trusted about what is loaded.  norm_stats_signature() derives
        # eligible_anchors and eligible_indices_sha256 FROM this array, and
        # load_norm_stats_artifact compares that signature against the JSON written
        # beside the sidecar -- so a stale or foreign array fails there, loudly,
        # rather than quietly training on the wrong anchor set.
        self._sample_indices = None
        self.anchor_sampling_summary: dict[str, int] | None = None
        if self.anchor_sampling_spec is None:
            if self.index_cache_path is not None:
                self._sample_indices = load_anchor_index(self.index_cache_path)
                if self._sample_indices is not None:
                    # Say so. A cache that works silently cannot be told apart from a
                    # cache that is being ignored, and the scan it replaces produces no
                    # output either -- EIP 7326's log could not answer which had
                    # happened.
                    logger.info(
                        "%s: loaded %d precomputed quality anchors, skipping the scan",
                        self.root.name,
                        len(self._sample_indices),
                    )
            if self._sample_indices is None:
                self._sample_indices = self._build_current_frame_indices()
        else:
            # Never interpret the legacy .anchors.npz (quality candidates) as a
            # final stride/keyframe sample list. Only the versioned v2 sidecar has
            # that meaning.
            if self.index_cache_path is not None:
                self._sample_indices = load_sample_anchor_index_v2(
                    self.index_cache_path,
                    expected_cache_signature=self.anchor_sampling_cache_signature(),
                )
                if self._sample_indices is not None:
                    if self._sample_indices.size and int(
                        self._sample_indices[-1]
                    ) >= int(self._cum[-1]):
                        raise ValueError(
                            f"{self.index_cache_path}: sample anchors exceed the selected "
                            "dataset frame range"
                        )
                    logger.info(
                        "%s: loaded %d precomputed stride/keyframe sample anchors",
                        self.root.name,
                        len(self._sample_indices),
                    )
                    self.anchor_sampling_summary = {"final": len(self._sample_indices)}
            if self._sample_indices is None:
                self._sample_indices = self._build_sample_anchor_indices()
        needs_transformed_action_stats = self.action_representation != "abs" or (
            self.action_upsample_mode == "semantic" and self.action_upsample_rate > 1
        )
        if self.norm_stats_path is not None:
            if self.allow_norm_stats_dataset_mismatch:
                logger.warning(
                    "%s: applying normalization stats from %s with an explicitly "
                    "different dataset scope; state/action/quality transform "
                    "signatures remain strict",
                    self.root.name,
                    self.norm_stats_path,
                )
            loaded_stats = load_norm_stats_artifact(
                self.norm_stats_path,
                expected_signature=self.norm_stats_signature(),
                feature_dims={
                    self.state_key: self.target_state_dim,
                    self.action_key: (
                        (self.chunk_size, self.target_action_dim)
                        if self.action_stats_per_horizon
                        else self.target_action_dim
                    ),
                },
                allow_dataset_mismatch=self.allow_norm_stats_dataset_mismatch,
            )
            self.meta_stats.update(loaded_stats)
        elif estimate_action_stats and needs_transformed_action_stats:
            self.meta_stats[self.action_key] = (
                self._estimate_action_stats_per_horizon()
                if self.action_stats_per_horizon
                else self._estimate_action_stats()
            )
            self._df_cache.clear()
        if self.norm_stats_path is None and estimate_action_stats:
            estimated = self._estimate_missing_quantiles(self.state_key)
            if estimated is not None:
                self.meta_stats[self.state_key].update(estimated)
                self._df_cache.clear()
        raw_samples = int(self._cum[-1])
        accepted_samples = len(self)
        self.quality_summary = {
            "raw_samples": raw_samples,
            "accepted_samples": accepted_samples,
            "rejected_samples": raw_samples - accepted_samples,
            "accepted_fraction": accepted_samples / raw_samples,
        }

    def _resolve_embodiment_strategy(self) -> None:
        """Pick how this source answers "which robot", once, at construction.

        Resolved eagerly so a misconfiguration fails at startup rather than
        thousands of steps in, and logged so a run's provenance is readable from
        the training log alone -- the whole point of putting the robot into the
        prompt is defeated if nobody can tell afterwards what was in it.
        """
        spec = self._embodiment_spec
        default_column = "observation.embodiment_id"

        if spec is not None and spec.constant:
            self._embodiment_constant = spec.constant
            logger.info("%s: embodiment fixed to %r", self.root.name, spec.constant)
            return

        if spec is not None and spec.episode_meta_path:
            resolved = _read_episode_embodiments(
                self.root / _V21_EPISODES, spec, self._ep_indices
            )
            labels = [
                resolved.get(int(episode_index), UNKNOWN_EMBODIMENT)
                for episode_index in self._ep_indices
            ]
            (
                self._ep_embodiment_ids,
                self._embodiment_table,
            ) = _encode_episode_labels(labels)
            del labels, resolved
            counts: dict[str, int] = {}
            unique_ids, unique_counts = np.unique(
                self._ep_embodiment_ids, return_counts=True
            )
            for label_id, count in zip(unique_ids, unique_counts, strict=True):
                counts[self._embodiment_table[int(label_id)]] = int(count)
            logger.info(
                "%s: embodiment from %s -> %s",
                self.root.name,
                spec.episode_meta_path,
                ", ".join(f"{k} {v}" for k, v in sorted(counts.items())),
            )
            if counts.get(UNKNOWN_EMBODIMENT):
                logger.warning(
                    "%s: %d/%d episodes matched no embodiment pattern",
                    self.root.name,
                    counts[UNKNOWN_EMBODIMENT],
                    len(self._ep_indices),
                )
            return

        column = spec.column if spec is not None and spec.column else default_column
        if column in self._feature_names:
            self._embodiment_column = column
            logger.info("%s: embodiment read per frame from %s", self.root.name, column)
        elif spec is not None and spec.column:
            raise ValueError(
                f"{self.root}: embodiment column {column!r} is not in meta/info.json features"
            )
        else:
            self._embodiment_constant = UNKNOWN_EMBODIMENT
            logger.info(
                "%s: no %s and no configured embodiment; samples report %r",
                self.root.name,
                default_column,
                UNKNOWN_EMBODIMENT,
            )

    def _episode_position(self, episode_index: int) -> int:
        """Resolve an episode id without retaining a Python dict per episode."""

        episode = int(episode_index)
        position = int(np.searchsorted(self._ep_indices, episode, side="left"))
        if (
            position >= len(self._ep_indices)
            or int(self._ep_indices[position]) != episode
        ):
            raise KeyError(f"unknown v2.1 episode_index={episode}")
        return position

    def _episode_task(self, position: int) -> str:
        task_ids = getattr(self, "_ep_task_ids", None)
        task_table = getattr(self, "_task_table", None)
        if task_ids is not None and task_table is not None:
            return task_table[int(task_ids[position])]

        # Compatibility for unit tests and older pickled readers constructed
        # before the compact episode table was introduced.
        tasks = self._ep_tasks[position]
        return tasks[0] if tasks else "unknown"

    def _embodiment_for(
        self,
        episode_index: int,
        table: Mapping[str, Any],
        t: int,
        *,
        episode_position: int | None = None,
    ) -> str:
        # getattr rather than direct access: several tests build the reader via
        # __new__ to exercise __getitem__ against a synthetic table, so every
        # per-sample field has to tolerate an un-run __init__ (see the same
        # defence around the decoder cache below).
        constant = getattr(self, "_embodiment_constant", None)
        if constant is not None:
            return constant
        embodiment_ids = getattr(self, "_ep_embodiment_ids", None)
        embodiment_table = getattr(self, "_embodiment_table", None)
        if embodiment_ids is not None and embodiment_table:
            position = (
                self._episode_position(episode_index)
                if episode_position is None
                else int(episode_position)
            )
            return embodiment_table[int(embodiment_ids[position])]
        by_episode = getattr(self, "_embodiment_by_episode", None)
        if by_episode is not None:
            return by_episode.get(episode_index, UNKNOWN_EMBODIMENT)
        column_key = getattr(self, "_embodiment_column", None)
        if column_key is None:
            return UNKNOWN_EMBODIMENT
        column = table.get(column_key)
        if column is None:
            return UNKNOWN_EMBODIMENT
        return decode_embodiment_id(column[t])

    def __len__(self) -> int:
        sample_indices = getattr(self, "_sample_indices", None)
        return len(sample_indices) if sample_indices is not None else int(self._cum[-1])

    def sample_episode_ranges(self) -> list[tuple[int, int, int]]:
        """Return episode spans in this reader's filtered sample index space."""
        sample_indices = getattr(self, "_sample_indices", None)
        ranges: list[tuple[int, int, int]] = []
        for pos, episode_index in enumerate(self._ep_indices):
            raw_start = int(self._cum[pos])
            raw_stop = int(self._cum[pos + 1])
            if sample_indices is None:
                start, stop = raw_start, raw_stop
            else:
                start = int(np.searchsorted(sample_indices, raw_start, side="left"))
                stop = int(np.searchsorted(sample_indices, raw_stop, side="left"))
            if stop > start:
                ranges.append((int(episode_index), start, stop))
        return ranges

    def anchor_sampling_cache_signature(self) -> dict[str, Any]:
        """Contract embedded in the versioned final-anchor sidecar."""
        spec = self.anchor_sampling_spec
        if spec is None:
            raise RuntimeError(
                "anchor sampling cache signature requested without a spec"
            )
        return {
            "format": "magic_w0.sample_anchors.v2",
            "info_sha256": self._info_sha256,
            "episodes": "all"
            if self._all_episodes_selected
            else list(self._ep_indices),
            "episode_layout_sha256": self._episode_layout_sha256,
            "quality_current_frame": self._current_frame_quality_rule,
            "anchor_sampling": spec.signature(),
        }

    def norm_stats_signature(self) -> dict[str, Any]:
        """Return the exact data/transform contract attached to a norm artifact."""

        interpolation = (
            asdict(self.action_interpolation_spec)
            if self.action_interpolation_spec is not None
            else None
        )
        eligible_indices_sha256 = (
            "all_raw_frames"
            if self._sample_indices is None
            else hashlib.sha256(
                np.ascontiguousarray(self._sample_indices).view(np.uint8)
            ).hexdigest()
        )
        signature = {
            "backend": "lerobot_v21",
            "dataset": {
                "info_sha256": self._info_sha256,
                "episodes": (
                    "all"
                    if self._all_episodes_selected
                    else [int(episode) for episode in self._ep_indices]
                ),
                "episode_layout_sha256": self._episode_layout_sha256,
                "raw_frames": int(self._cum[-1]),
                "eligible_anchors": len(self),
                "eligible_indices_sha256": eligible_indices_sha256,
            },
            "state": {
                "key": self.state_key,
                "source_dim": self.source_state_dim,
                "indices": self.state_indices,
                **(
                    {"active_indices": self.state_active_indices}
                    if self.state_active_indices is not None
                    else {}
                ),
                "preserve_selected_slots": self.preserve_selected_slots,
                "target_dim": self.target_state_dim,
                "mask_key": self.state_mask_key,
                "obs_steps": self.obs_steps,
                "observation_delta_indices": self.observation_delta_indices,
            },
            "action": {
                "key": self.action_key,
                "source_dim": self.source_action_dim,
                "indices": self.action_indices,
                **(
                    {"active_indices": self.action_active_indices}
                    if self.action_active_indices is not None
                    else {}
                ),
                "preserve_selected_slots": self.preserve_selected_slots,
                "target_dim": self.target_action_dim,
                "mask_key": self.action_mask_key,
                "chunk_size": self.chunk_size,
                "representation": self.action_representation,
                "delta_indices": self.action_delta_indices,
                "rotation_delta": self.action_rotation_delta,
                "rotation_groups": self.action_rotation_groups,
                "relative_rotation_groups": self.action_relative_rotation_groups,
                "upsample_rate": self.action_upsample_rate,
                "upsample_mode": self.action_upsample_mode,
                "interpolation": interpolation,
                "supervision_indices": self.action_supervision_indices,
                # In the signature so a flat artifact cannot be loaded into a
                # per-horizon run, or the reverse: the two disagree about what
                # every number in the file means, and they broadcast against
                # each other without error.
                "stats_layout": (
                    "per_horizon_dim" if self.action_stats_per_horizon else "per_dim"
                ),
            },
            "quality": {
                "current_frame": self._current_frame_quality_rule,
                "action_chunk": self._action_chunk_quality_rule,
            },
        }
        if self.state_target_indices is not None:
            signature["state"]["target_indices"] = self.state_target_indices
        if self.action_target_indices is not None:
            signature["action"]["target_indices"] = self.action_target_indices
        if self.anchor_sampling_spec is not None:
            signature["anchor_sampling"] = self.anchor_sampling_spec.signature()
        return signature

    def _episode_path(self, ep: int) -> Path:
        rel = self._data_path.format(
            episode_chunk=ep // self._chunks_size, episode_index=ep
        )
        return self.root / rel

    def _build_current_frame_indices(self) -> np.ndarray | None:
        """Return raw frame indices eligible to serve as sample anchors.

        One parquet open per episode, reading only the quality columns. That is
        287,055 files and ~47 GB of reads for EgoDex, and it used to run on a
        single core while the rest of the machine idled -- so it is threaded here.
        Threads rather than processes because the work is dominated by pyarrow
        I/O and decompression, both of which release the GIL, and because the
        alternative would mean pickling this reader to every worker.

        Order is preserved: ``executor.map`` yields in submission order, and each
        episode's frame numbers are offset by its own ``_cum`` entry, so the
        result is identical to the serial version byte for byte -- which matters,
        because ``eligible_indices_sha256`` in the norm signature hashes it.
        """
        if not any(self._current_frame_quality_rule.values()):
            return None
        import os
        from concurrent.futures import ThreadPoolExecutor

        import pyarrow as pa
        import pyarrow.parquet as pq

        columns = list(self._current_frame_quality_columns)
        rule = self._current_frame_quality_rule
        total = len(self._ep_indices)
        episodes = zip(self._ep_indices, self._ep_lengths, strict=True)

        def scan(item: tuple[int, tuple[int, int]]) -> np.ndarray:
            pos, (ep, length) = item
            # ParquetFile rather than read_table for the leak documented in
            # _episode_table.  One-time here, but it runs once per episode and a
            ep, length = int(ep), int(length)
            # source can hold 287k of them, on up to 32 threads.  The retention
            # measured on _episode_table applies unchanged, and here it is paid in
            # the PARENT, before fork -- so whatever the scan leaves behind is
            # carried by every worker for the life of the run.
            table = pq.ParquetFile(self._episode_path(ep)).read(columns=columns)
            arrays = {
                name: table[name].to_numpy(zero_copy_only=False) for name in columns
            }
            mask = _quality_mask(arrays, rule, length)
            accepted = np.flatnonzero(mask).astype(np.int64) + int(self._cum[pos])
            del arrays, table
            pa.default_memory_pool().release_unused()
            return accepted

        workers = int(os.environ.get("MAGIC_W0_INDEX_WORKERS", 0)) or min(
            32, (os.cpu_count() or 8)
        )
        accepted: list[np.ndarray] = []
        # Report progress: on the largest source this runs for minutes with no
        # other output, which reads as a hang. The interval is coarse so that a
        # small dataset stays silent.
        step = max(1, total // 20)
        if total >= 5_000:
            logger.info(
                "%s: scanning %d episodes for quality anchors on %d threads",
                self.root.name,
                total,
                workers,
            )
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for done, part in enumerate(pool.map(scan, enumerate(episodes)), start=1):
                accepted.append(part)
                if total >= 5_000 and (done % step == 0 or done == total):
                    logger.info(
                        "%s: anchor scan %d/%d episodes (%.0f%%), %d anchors so far",
                        self.root.name,
                        done,
                        total,
                        100.0 * done / total,
                        sum(int(a.size) for a in accepted),
                    )
        indices = np.concatenate(accepted) if accepted else np.empty(0, dtype=np.int64)
        if indices.size == 0:
            raise ValueError(f"{self.root}: quality.current_frame rejected every frame")
        return indices

    def _build_sample_anchor_indices(self) -> np.ndarray:
        """Build final stride/keyframe anchors, independently within each episode.

        The current-frame quality columns and the optional parquet keyframe column
        are read in one pass. There is deliberately no "include episode end"
        branch: the final frame is retained only when reached naturally by the
        stride or selected by keyframe/context configuration.
        """
        import os
        from concurrent.futures import ThreadPoolExecutor

        import pyarrow.parquet as pq

        spec = self.anchor_sampling_spec
        if spec is None:
            raise RuntimeError("sample-anchor scan requested without anchor_sampling")
        keyframe_spec = spec.keyframes
        episode_lengths = dict(zip(self._ep_indices, self._ep_lengths, strict=True))
        jsonl_keyframes = (
            load_keyframe_jsonl(
                keyframe_spec,
                episode_lengths=episode_lengths,
                strict_unknown_episodes=self._all_episodes_selected,
            )
            if keyframe_spec is not None and keyframe_spec.source == "jsonl"
            else {}
        )
        parquet_keyframe_column = (
            keyframe_spec.column
            if keyframe_spec is not None and keyframe_spec.source == "parquet_column"
            else None
        )
        columns = list(self._current_frame_quality_columns)
        if parquet_keyframe_column is not None:
            columns.append(parquet_keyframe_column)
        columns = list(dict.fromkeys(columns))
        episodes = list(zip(self._ep_indices, self._ep_lengths, strict=True))

        def scan(
            item: tuple[int, tuple[int, int]],
        ) -> tuple[np.ndarray, dict[str, int]]:
            pos, (episode, length) = item
            arrays: dict[str, np.ndarray] = {}
            if columns:
                table = pq.ParquetFile(self._episode_path(episode)).read(
                    columns=columns
                )
                arrays = {
                    name: table[name].to_numpy(zero_copy_only=False) for name in columns
                }
            quality = _quality_mask(arrays, self._current_frame_quality_rule, length)
            if parquet_keyframe_column is not None:
                keyframe_values = np.asarray(arrays[parquet_keyframe_column])
                if keyframe_values.ndim != 1 or keyframe_values.shape[0] != length:
                    raise ValueError(
                        f"keyframe column {parquet_keyframe_column!r} for episode "
                        f"{episode} must be scalar per frame, got {keyframe_values.shape}"
                    )
                keyframes = np.flatnonzero(keyframe_values.astype(np.bool_))
            else:
                keyframes = jsonl_keyframes.get(episode, np.empty(0, dtype=np.int64))
            local, summary = episode_sample_anchors(
                length,
                spec,
                keyframe_indices=keyframes,
                quality_mask=quality,
            )
            return local + int(self._cum[pos]), summary

        workers = int(os.environ.get("MAGIC_W0_INDEX_WORKERS", 0)) or min(
            32, (os.cpu_count() or 8)
        )
        parts: list[np.ndarray] = []
        totals = {
            "raw_frames": int(self._cum[-1]),
            "regular": 0,
            "keyframe_context": 0,
            "overlap": 0,
            "quality_rejected": 0,
            "final": 0,
        }
        total_episodes = len(episodes)
        progress_step = max(1, total_episodes // 20)
        if total_episodes >= 5_000:
            logger.info(
                "%s: building stride/keyframe anchors for %d episodes on %d threads",
                self.root.name,
                total_episodes,
                workers,
            )
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for done, (part, summary) in enumerate(
                pool.map(scan, enumerate(episodes)), start=1
            ):
                if part.size:
                    parts.append(part)
                for key, value in summary.items():
                    totals[key] += int(value)
                if total_episodes >= 5_000 and (
                    done % progress_step == 0 or done == total_episodes
                ):
                    logger.info(
                        "%s: anchor build %d/%d episodes (%.0f%%), %d final anchors",
                        self.root.name,
                        done,
                        total_episodes,
                        100.0 * done / total_episodes,
                        totals["final"],
                    )
        indices = np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)
        if indices.size == 0:
            raise ValueError(
                f"{self.root}: anchor_sampling and quality rules rejected every frame"
            )
        if np.any(indices[1:] <= indices[:-1]):
            raise RuntimeError(
                "sample-anchor construction did not produce sorted unique indices"
            )
        self.anchor_sampling_summary = totals
        logger.info(
            "%s: anchor sampling stride=%d offset=%d regular=%d "
            "keyframe_context=%d overlap=%d quality_rejected=%d final=%d",
            self.root.name,
            spec.stride_frames,
            spec.offset,
            totals["regular"],
            totals["keyframe_context"],
            totals["overlap"],
            totals["quality_rejected"],
            totals["final"],
        )
        return indices

    # ---- episode-level caches ------------------------------------------------
    def _episode_table(self, ep: int) -> dict[str, np.ndarray]:
        cached = self._df_cache.get(ep)
        if cached is not None:
            return cached
        import pyarrow as pa
        import pyarrow.parquet as pq

        columns = [self.state_key, self.action_key, *self._quality_columns]
        if self.state_mask_key is not None:
            columns.append(self.state_mask_key)
        if self.action_mask_key is not None:
            columns.append(self.action_mask_key)
        if self._embodiment_column is not None:
            columns.append(self._embodiment_column)
        columns = list(dict.fromkeys(columns))
        # NOT pq.read_table: it leaks ~0.8-2.3 KB per call and this is the hottest
        # read in the pipeline.  read_table routes a single known path through
        # ParquetDataset -> ds.dataset() -> filesystem discovery, and something on
        # that path is never released -- not by gc.collect(), not by
        # malloc_trim(0), and not by Arrow's own pool.  Reading the same file 3000
        # times with the cache cleared each time grew RSS by 2.3 MB; the same 3000
        # reads through ParquetFile ran 2.7x faster.
        #
        # SWITCHING TO ParquetFile WAS NOT SUFFICIENT, and the claim that it grew
        # RSS "by 0.0 MB" was an artifact of measuring with reclaim in the window
        # -- the same methodological error this pipeline has now made twice.  Read
        # with nothing reclaiming, as a dataloader worker actually runs, 4000
        # random EgoDex episodes cost (two rounds each):
        #
        #   base                      +84.8 / +88.4 MB   21.7 / 22.6 KB per read
        #   malloc_trim(0) per read   +83.6 MB           21.4 KB   <- no effect
        #   release_unused() per read +32.4 / +36.3 MB    8.3 /  9.3 KB
        #   both                      +28.1 MB            7.2 KB
        #
        # The trajectories matter more than the totals: base climbs monotonically
        # (+35.0 -> +62.5 -> +71.8 -> +84.8 MB at 1k/2k/3k/4k reads) while
        # release_unused() PLATEAUS by read 2000 and then oscillates around 30 MB.
        # Unbounded against bounded is the whole difference.  malloc_trim doing
        # nothing places the retention in Arrow's pool (mimalloc), not glibc --
        # the opposite of what the block sizes suggest, since these episode files
        # are only 0.04-0.44 MB.  Same defect and same fix as
        # vlm_jsonl.py:_load_reference_images, which this path never received.
        #
        # This is the hottest read in the pipeline: _df_cache holds 4 episodes
        # while the sampler draws uniformly over 287k EgoDex episodes, so the hit
        # rate is ~0 and nearly every sample pays a fresh read.  At 72 samples per
        # batch, 4.5 s/step and 128 robotics workers per node that is ~3,240 reads
        # per worker per hour.  Only the workers read parquet, which is why rank
        # 0's main process sat flat at 17-19 GB for 64 hours (measured over both
        # 48-GPU runs) while the node ran out and EIP 7370 was OOMKilled.
        tbl = pq.ParquetFile(self._episode_path(ep)).read(columns=columns)
        arrs = {
            self.state_key: np.stack(
                tbl[self.state_key].to_numpy(zero_copy_only=False)
            ).astype(np.float32),
            self.action_key: np.stack(
                tbl[self.action_key].to_numpy(zero_copy_only=False)
            ).astype(np.float32),
        }
        arrs[self.state_key] = _adapt_last_dim(
            arrs[self.state_key],
            indices=self.state_indices,
            target_indices=self.state_target_indices,
            target_dim=self.target_state_dim,
            pad_value=0.0,
            preserve_selected_slots=self.preserve_selected_slots,
        )
        arrs[self.action_key] = _adapt_last_dim(
            arrs[self.action_key],
            indices=self.action_indices,
            target_indices=self.action_target_indices,
            target_dim=self.target_action_dim,
            pad_value=0.0,
            preserve_selected_slots=self.preserve_selected_slots,
        )
        if self.state_mask_key is None:
            raw_state_mask = np.ones(
                (
                    len(arrs[self.state_key]),
                    self.source_state_dim
                    if self.preserve_selected_slots
                    else len(self.state_indices)
                    if self.state_indices is not None
                    else self.source_state_dim,
                ),
                dtype=np.bool_,
            )
        else:
            raw_state_mask = np.stack(
                tbl[self.state_mask_key].to_numpy(zero_copy_only=False)
            ).astype(np.bool_)
        if self.action_mask_key is None:
            raw_action_mask = np.ones(
                (
                    len(arrs[self.action_key]),
                    self.source_action_dim
                    if self.preserve_selected_slots
                    else len(self.action_indices)
                    if self.action_indices is not None
                    else self.source_action_dim,
                ),
                dtype=np.bool_,
            )
        else:
            raw_action_mask = np.stack(
                tbl[self.action_mask_key].to_numpy(zero_copy_only=False)
            ).astype(np.bool_)
        arrs[f"{self.state_key}_dim_mask"] = _adapt_last_dim(
            raw_state_mask,
            indices=(
                self.state_indices
                if self.state_mask_key is not None or self.preserve_selected_slots
                else None
            ),
            target_indices=self.state_target_indices,
            target_dim=self.target_state_dim,
            pad_value=False,
            preserve_selected_slots=self.preserve_selected_slots,
        )
        arrs[f"{self.action_key}_dim_mask"] = _adapt_last_dim(
            raw_action_mask,
            indices=(
                self.action_indices
                if self.action_mask_key is not None or self.preserve_selected_slots
                else None
            ),
            target_indices=self.action_target_indices,
            target_dim=self.target_action_dim,
            pad_value=False,
            preserve_selected_slots=self.preserve_selected_slots,
        )
        _mask_inactive_dimensions(
            arrs[self.state_key],
            arrs[f"{self.state_key}_dim_mask"],
            active_indices=self.state_active_indices,
        )
        _mask_inactive_dimensions(
            arrs[self.action_key],
            arrs[f"{self.action_key}_dim_mask"],
            active_indices=self.action_active_indices,
        )
        for name in self._quality_columns:
            arrs[name] = tbl[name].to_numpy(zero_copy_only=False)
        arrs["_action_chunk_quality_mask"] = _quality_mask(
            arrs, self._action_chunk_quality_rule, len(arrs[self.action_key])
        )
        # Hand this read's buffers back to the OS before the next one allocates a
        # differently sized set.  release_unused() only returns memory the pool is
        # holding and not serving, so the zero-copy views some quality columns
        # keep into `tbl` are unaffected -- dropping the local name first simply
        # lets the columns nothing else references go back in the same call.
        del tbl
        pa.default_memory_pool().release_unused()
        self._df_cache[ep] = arrs
        if len(self._df_cache) > 4:
            self._df_cache.popitem(last=False)
        return arrs

    def _decode_frames(
        self, cam: str, ep: int, frames: tuple[int, ...]
    ) -> torch.Tensor:
        rel = self._video_path.format(
            episode_chunk=ep // self._chunks_size,
            video_key=cam,
            episode_index=ep,
        )
        path = str(self.root / rel)
        dec = self._decoders.get(path)
        if dec is None:
            dec = make_video_decoder(path, backend=self.video_backend, fps=self.fps)
            self._decoders[path] = dec
            if len(self._decoders) > self._decoder_cache:
                self._decoders.popitem(last=False)
        indices = [min(frame, dec.metadata.num_frames - 1) for frame in frames]
        return dec.get_frames_at(indices=indices).data  # uint8 [T,C,H,W]

    def _rotate_camera_frames(
        self, frames: torch.Tensor, canonical_key: str
    ) -> torch.Tensor:
        """Apply a source-configured camera mount correction after decode."""

        clockwise_degrees = getattr(self, "camera_rotations", {}).get(canonical_key, 0)
        if clockwise_degrees == 0:
            return frames
        return torch.rot90(
            frames,
            k=-(clockwise_degrees // 90),
            dims=(-2, -1),
        )

    def _build_action_chunk(
        self,
        table: Mapping[str, np.ndarray],
        *,
        t: int,
        length: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build actions and masks without touching video.

        Temporal upsampling happens in absolute-action space first. This matters
        for ``frame_delta``: repeating ``[a0, a1]`` twice must yield deltas
        ``[a0-s, 0, a1-a0, 0]`` rather than repeating each delta and doubling the
        integrated motion.
        """
        upsample_rate = int(getattr(self, "action_upsample_rate", 1))
        source_chunk_size = (self.chunk_size + upsample_rate - 1) // upsample_rate
        end = min(t + source_chunk_size, length)

        source_chunk = table[self.action_key][t:end]
        source_quality_valid = table.get(
            "_action_chunk_quality_mask", np.ones(length, dtype=np.bool_)
        )[t:end]
        source_value_dim_mask = table.get(
            f"{self.action_key}_dim_mask",
            np.ones_like(table[self.action_key], dtype=np.bool_),
        )[t:end]
        upsample_mode = getattr(self, "action_upsample_mode", "semantic")
        if upsample_rate == 1:
            chunk = source_chunk[: self.chunk_size].copy()
            quality_valid = source_quality_valid[: self.chunk_size].copy()
            value_dim_mask = source_value_dim_mask[: self.chunk_size].copy()
        elif upsample_mode == "semantic":
            spec = getattr(self, "action_interpolation_spec", None)
            if spec is None:
                raise ValueError(
                    "semantic action upsampling requires action_interpolation_spec"
                )
            chunk, quality_valid, value_dim_mask = semantic_upsample_actions(
                source_chunk,
                source_value_dim_mask,
                source_quality_valid,
                output_size=self.chunk_size,
                rate=upsample_rate,
                spec=spec,
            )
        elif upsample_mode == "repeat":
            chunk = np.repeat(source_chunk, upsample_rate, axis=0)[
                : self.chunk_size
            ].copy()
            quality_valid = np.repeat(source_quality_valid, upsample_rate)[
                : self.chunk_size
            ]
            value_dim_mask = np.repeat(source_value_dim_mask, upsample_rate, axis=0)[
                : self.chunk_size
            ].copy()
        else:
            raise ValueError(f"unsupported action_upsample_mode {upsample_mode!r}")

        produced_rows = chunk.shape[0]
        n_pad = self.chunk_size - produced_rows
        supervise_end_padding = bool(
            getattr(self, "supervise_episode_end_padding", False)
        )
        if n_pad > 0:
            chunk = np.concatenate(
                [chunk, np.repeat(chunk[-1:], n_pad, axis=0)], axis=0
            )
            pad_dim_mask = (
                np.repeat(value_dim_mask[-1:], n_pad, axis=0)
                if supervise_end_padding
                else np.zeros((n_pad, chunk.shape[-1]), dtype=np.bool_)
            )
            value_dim_mask = np.concatenate([value_dim_mask, pad_dim_mask], axis=0)

        row_valid = np.zeros(self.chunk_size, dtype=np.bool_)
        row_valid[:produced_rows] = quality_valid
        if supervise_end_padding and n_pad > 0:
            row_valid[produced_rows:] = bool(quality_valid[-1])
        is_pad = ~row_valid
        action_dim_mask = value_dim_mask.copy()
        action_dim_mask[~row_valid] = False

        spec = getattr(self, "action_interpolation_spec", None)
        rotation_groups = tuple(
            tuple(group)
            for group in getattr(
                self,
                "action_rotation_groups",
                getattr(spec, "rotation_groups", ()),
            )
        )
        # A rot6d value is one semantic value. If any component is unavailable,
        # supervising the other five would train on a frame that cannot represent
        # a rotation at all.
        for group in rotation_groups:
            columns = list(group)
            group_valid = action_dim_mask[:, columns].all(axis=1)
            action_dim_mask[:, columns] = group_valid[:, None]

        representation = getattr(self, "action_representation", "abs")
        if representation != "abs":
            current_state = np.asarray(table[self.state_key][t])
            state_dim_mask = table.get(
                f"{self.state_key}_dim_mask",
                np.ones_like(table[self.state_key], dtype=np.bool_),
            )[t]
            delta_indices = getattr(self, "action_delta_indices", None)
            if delta_indices is None:
                delta_indices = tuple(
                    range(min(chunk.shape[-1], current_state.shape[-1]))
                )
            gripper_indices = getattr(spec, "gripper_indices", ())
            delta_indices = tuple(
                index for index in delta_indices if index not in gripper_indices
            )

            # Rotations compose, they do not subtract. A group is either wholly
            # relative or wholly absolute; rejecting a partial group prevents
            # silently treating half a rotation as an ordinary vector.
            delta_set = set(delta_indices)
            _validate_rotation_delta_partition(rotation_groups, delta_indices)
            relative_rotation_groups = tuple(
                group for group in rotation_groups if delta_set.issuperset(group)
            )
            rotated = {index for group in relative_rotation_groups for index in group}
            vector_indices = tuple(i for i in delta_indices if i not in rotated)

            if representation == "chunk_delta":
                chunk[:, vector_indices] -= current_state[list(vector_indices)]
                for group in relative_rotation_groups:
                    columns = list(group)
                    chunk[:, columns] = compose_relative_rot6d(
                        chunk[:, columns], current_state[columns]
                    ).astype(chunk.dtype)
                action_dim_mask[:, delta_indices] &= state_dim_mask[list(delta_indices)]
                for group in relative_rotation_groups:
                    columns = list(group)
                    group_valid = action_dim_mask[:, columns].all(axis=1)
                    action_dim_mask[:, columns] = group_valid[:, None]
            elif representation == "frame_delta":
                reference = np.concatenate(
                    (current_state[None, :], chunk[:-1, :]), axis=0
                )
                reference_dim_mask = np.concatenate(
                    (
                        state_dim_mask[None, list(delta_indices)],
                        value_dim_mask[:-1, list(delta_indices)],
                    ),
                    axis=0,
                )
                chunk[:, vector_indices] -= reference[:, list(vector_indices)]
                for group in relative_rotation_groups:
                    columns = list(group)
                    chunk[:, columns] = compose_relative_rot6d(
                        chunk[:, columns], reference[:, columns]
                    ).astype(chunk.dtype)
                action_dim_mask[:, delta_indices] &= reference_dim_mask
                for group in relative_rotation_groups:
                    columns = list(group)
                    group_valid = action_dim_mask[:, columns].all(axis=1)
                    action_dim_mask[:, columns] = group_valid[:, None]
            else:  # Protect __new__-constructed tests and corrupted pickles too.
                raise ValueError(
                    f"unsupported action_representation {representation!r}"
                )

        supervision_indices = getattr(self, "action_supervision_indices", None)
        if supervision_indices is not None:
            _validate_rotation_supervision_partition(
                rotation_groups, supervision_indices
            )
            keep = np.zeros(action_dim_mask.shape[-1], dtype=np.bool_)
            keep[list(supervision_indices)] = True
            action_dim_mask &= keep[None, :]

        return chunk, is_pad, action_dim_mask

    def _estimate_missing_quantiles(self, key: str) -> dict[str, np.ndarray] | None:
        """Estimate state q01/q99 when v2.1 metadata only provides min/max."""
        stats = self.meta_stats.get(key)
        if stats is None or ("q01" in stats and "q99" in stats):
            return None
        if key != self.state_key:
            return None

        width = self.target_state_dim
        n_anchors = min(len(self), max(self.action_stats_samples, 1))
        if n_anchors <= 0:
            return None
        dataset_indices = np.linspace(0, len(self) - 1, n_anchors, dtype=np.int64)
        collector: list[list[np.ndarray]] = [[] for _ in range(width)]
        for index in dataset_indices:
            raw_index = (
                int(self._sample_indices[index])
                if self._sample_indices is not None
                else int(index)
            )
            pos = int(np.searchsorted(self._cum, raw_index, side="right") - 1)
            ep = int(self._ep_indices[pos])
            t = raw_index - int(self._cum[pos])
            table = self._episode_table(ep)
            rows = (
                [t]
                if self.obs_steps == 1
                else list(
                    episode_observation_indices(t, self.observation_delta_indices)
                )
            )
            values = np.asarray(table[self.state_key][rows], dtype=np.float64)
            mask = np.asarray(
                table.get(
                    f"{self.state_key}_dim_mask",
                    np.ones_like(table[self.state_key], dtype=np.bool_),
                )[rows],
                dtype=np.bool_,
            )
            collect_quantile_values(collector, values, mask)

        q01 = np.asarray(stats["min"], dtype=np.float32).copy()
        q99 = np.asarray(stats["max"], dtype=np.float32).copy()
        for dim, parts in enumerate(collector):
            if not parts:
                continue
            dim_values = np.concatenate(parts).astype(np.float64, copy=False)
            if dim_values.size < _MIN_QUANTILE_SAMPLES:
                continue
            low, high = np.quantile(dim_values, [0.01, 0.99])
            if high - low >= _MIN_ESTIMATED_SPAN:
                q01[dim], q99[dim] = low, high
        return {"q01": q01, "q99": q99}

    def _estimate_action_stats(self) -> dict[str, np.ndarray]:
        """Estimate stats in the configured action representation.

        Metadata stats describe stored absolute rows. They are invalid for either
        delta representation and do not capture newly interpolated values from
        semantic temporal upsampling. We deterministically sample eligible anchors,
        construct their chunks through the exact training path, and aggregate only
        supervised dimensions. Episode tables contain action/state/quality columns;
        this method never opens a video decoder.
        """
        n_anchors = min(len(self), self.action_stats_samples)
        dataset_indices = np.linspace(0, len(self) - 1, n_anchors, dtype=np.int64)
        values: list[list[np.ndarray]] = [[] for _ in range(self.target_action_dim)]

        for index in dataset_indices:
            raw_index = (
                int(self._sample_indices[index])
                if self._sample_indices is not None
                else int(index)
            )
            pos = int(np.searchsorted(self._cum, raw_index, side="right") - 1)
            ep = int(self._ep_indices[pos])
            t = raw_index - int(self._cum[pos])
            table = self._episode_table(ep)
            chunk, _, dim_mask = self._build_action_chunk(
                table, t=t, length=int(self._ep_lengths[pos])
            )
            for dim in range(self.target_action_dim):
                if dim_mask[:, dim].any():
                    values[dim].append(chunk[dim_mask[:, dim], dim])

        neutral = {
            "mean": 0.0,
            "std": 1.0,
            "min": -1.0,
            "max": 1.0,
            "q01": -1.0,
            "q99": 1.0,
        }
        stats = {
            name: np.full(self.target_action_dim, fill, dtype=np.float32)
            for name, fill in neutral.items()
        }
        counts = np.zeros(self.target_action_dim, dtype=np.int64)
        for dim, parts in enumerate(values):
            if not parts:
                continue
            dim_values = np.concatenate(parts).astype(np.float64, copy=False)
            counts[dim] = dim_values.size
            stats["mean"][dim] = dim_values.mean()
            stats["std"][dim] = max(float(dim_values.std()), _MIN_ESTIMATED_STD)
            stats["min"][dim] = dim_values.min()
            stats["max"][dim] = dim_values.max()
            stats["q01"][dim], stats["q99"][dim] = np.quantile(dim_values, [0.01, 0.99])
        stats["count"] = counts
        self.action_stats_summary = {
            "representation": self.action_representation,
            "upsample_mode": getattr(self, "action_upsample_mode", "semantic"),
            "upsample_rate": getattr(self, "action_upsample_rate", 1),
            "sampled_anchors": int(n_anchors),
            "valid_values_per_dim": counts,
        }
        return stats

    def _estimate_action_stats_per_horizon(self) -> dict[str, np.ndarray]:
        """Estimate one normalization band for each chunk position and dimension."""
        n_anchors = min(len(self), self.action_stats_samples)
        dataset_indices = np.linspace(0, len(self) - 1, n_anchors, dtype=np.int64)
        chunks = np.zeros(
            (n_anchors, self.chunk_size, self.target_action_dim), dtype=np.float32
        )
        masks = np.zeros_like(chunks, dtype=np.bool_)
        for slot, index in enumerate(dataset_indices):
            raw_index = (
                int(self._sample_indices[index])
                if self._sample_indices is not None
                else int(index)
            )
            pos = int(np.searchsorted(self._cum, raw_index, side="right") - 1)
            ep = int(self._ep_indices[pos])
            t = raw_index - int(self._cum[pos])
            table = self._episode_table(ep)
            chunk, _, dim_mask = self._build_action_chunk(
                table, t=t, length=int(self._ep_lengths[pos])
            )
            chunks[slot] = chunk
            masks[slot] = dim_mask

        neutral = {
            "mean": 0.0,
            "std": 1.0,
            "min": -1.0,
            "max": 1.0,
            "q01": -1.0,
            "q99": 1.0,
        }
        stats = {
            name: np.full(chunks.shape[1:], fill, dtype=np.float32)
            for name, fill in neutral.items()
        }
        counts = np.zeros(chunks.shape[1:], dtype=np.int64)
        for step in range(self.chunk_size):
            for dim in range(self.target_action_dim):
                selected = chunks[masks[:, step, dim], step, dim]
                if not selected.size:
                    continue
                selected = selected.astype(np.float64, copy=False)
                counts[step, dim] = selected.size
                stats["mean"][step, dim] = selected.mean()
                stats["std"][step, dim] = max(float(selected.std()), _MIN_ESTIMATED_STD)
                stats["min"][step, dim] = selected.min()
                stats["max"][step, dim] = selected.max()
                stats["q01"][step, dim], stats["q99"][step, dim] = np.quantile(
                    selected, [0.01, 0.99]
                )

        flat = {
            name: np.full(self.target_action_dim, fill, dtype=np.float32)
            for name, fill in neutral.items()
        }
        for dim in range(self.target_action_dim):
            selected = chunks[:, :, dim][masks[:, :, dim]]
            if not selected.size:
                continue
            selected = selected.astype(np.float64, copy=False)
            flat["mean"][dim] = selected.mean()
            flat["std"][dim] = max(float(selected.std()), _MIN_ESTIMATED_STD)
            flat["min"][dim] = selected.min()
            flat["max"][dim] = selected.max()
            flat["q01"][dim], flat["q99"][dim] = np.quantile(selected, [0.01, 0.99])
        carveouts = apply_horizon_carveouts(
            stats, flat, delta_dims=self.action_delta_indices or ()
        )
        stats["count"] = counts
        self.action_stats_summary = {
            "representation": self.action_representation,
            "upsample_mode": getattr(self, "action_upsample_mode", "semantic"),
            "upsample_rate": getattr(self, "action_upsample_rate", 1),
            "sampled_anchors": int(n_anchors),
            "valid_values_per_dim": counts.max(axis=0),
            **carveouts,
        }
        return stats

    def _subtask_for(self, pos: int, t: int, length: int) -> dict[str, str]:
        episode_subtasks = getattr(self, "_ep_subtasks", None)
        if episode_subtasks is None or episode_subtasks[pos] is None:
            return {}
        starts, ends, names = episode_subtasks[pos]
        index = int(np.searchsorted(starts, t, side="right")) - 1
        if index < 0 or t >= int(ends[index]):
            return {}
        return {"subtask": names[index]}

    def _subtask_field(self, pos: int, t: int, length: int) -> dict[str, str]:
        if getattr(self, "_ep_subtasks", None) is None:
            return {}
        return {"subtask": self._subtask_for(pos, t, length).get("subtask", "")}

    def get_episode_frame_images(
        self,
        episode_index: int,
        frame_index: int,
        camera_keys: tuple[str, ...] | list[str],
    ) -> dict:
        """Decode one frame by episode-local index using canonical camera keys."""
        ep = int(episode_index)
        frame = int(frame_index)
        position = self._episode_position(ep)
        length = int(self._ep_lengths[position])
        if frame < 0 or frame >= length:
            raise IndexError(
                f"future frame {frame} is outside v2.1 episode {ep} length {length}"
            )

        item: dict = {
            "episode_index": torch.tensor(ep, dtype=torch.int64),
            "frame_index": torch.tensor(frame, dtype=torch.int64),
            "timestamp": torch.tensor(frame / self.fps, dtype=torch.float32),
        }
        canonical_to_source = dict(
            zip(self.camera_keys, self._source_camera_keys, strict=True)
        )
        for canonical_key in camera_keys:
            source_key = canonical_to_source.get(canonical_key)
            if source_key is not None:
                frames = self._decode_frames(source_key, ep, (frame,))
                item[canonical_key] = self._rotate_camera_frames(frames, canonical_key)[
                    0
                ]
        return item

    def get_episode_frames(
        self,
        episode_index: int,
        frame_indices: tuple[int, ...] | list[int],
        camera_keys: tuple[str, ...] | list[str],
    ) -> dict:
        """Decode several frames of one episode, one decoder call per camera.

        Returns ``{canonical_key: uint8 [T,C,H,W]}`` for the requested cameras
        this source has.  Used for history clips, where a per-frame
        :meth:`get_episode_frame_images` loop would re-seek the video T times.
        """
        ep = int(episode_index)
        frames = tuple(int(frame) for frame in frame_indices)
        length = int(self._ep_lengths[self._episode_position(ep)])
        if not frames or min(frames) < 0 or max(frames) >= length:
            raise IndexError(
                f"history frames {frames} are outside v2.1 episode {ep} length {length}"
            )
        canonical_to_source = dict(
            zip(self.camera_keys, self._source_camera_keys, strict=True)
        )
        item: dict = {}
        for canonical_key in camera_keys:
            source_key = canonical_to_source.get(canonical_key)
            if source_key is not None:
                item[canonical_key] = self._rotate_camera_frames(
                    self._decode_frames(source_key, ep, frames), canonical_key
                )
        return item

    def get_episode_length(self, episode_index: int) -> int:
        """Return the number of frames in one selected episode."""
        return int(self._ep_lengths[self._episode_position(int(episode_index))])

    # ---- sample --------------------------------------------------------------
    def __getitem__(self, index: int) -> dict:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        sample_indices = getattr(self, "_sample_indices", None)
        raw_index = int(sample_indices[index]) if sample_indices is not None else index
        pos = int(np.searchsorted(self._cum, raw_index, side="right") - 1)
        ep = int(self._ep_indices[pos])
        t = raw_index - int(self._cum[pos])
        length = int(self._ep_lengths[pos])

        table = self._episode_table(ep)
        obs_indices = episode_observation_indices(t, self.observation_delta_indices)
        chunk, is_pad, action_dim_mask = self._build_action_chunk(
            table, t=t, length=length
        )
        state_dim_mask = table.get(
            f"{self.state_key}_dim_mask",
            np.ones_like(table[self.state_key], dtype=np.bool_),
        )

        item: dict = {
            self.state_key: torch.from_numpy(
                (
                    table[self.state_key][t]
                    if self.obs_steps == 1
                    else table[self.state_key][list(obs_indices)]
                ).copy()
            ),
            self.action_key: torch.from_numpy(chunk.copy()),
            f"{self.action_key}_is_pad": torch.from_numpy(is_pad.copy()),
            f"{self.action_key}_dim_mask": torch.from_numpy(action_dim_mask.copy()),
            f"{self.state_key}_dim_mask": torch.from_numpy(
                (
                    state_dim_mask[t]
                    if self.obs_steps == 1
                    else state_dim_mask[list(obs_indices)]
                ).copy()
            ),
            "task": self._task_override or self._episode_task(pos),
            **self._subtask_field(pos, t, length),
            "embodiment": self._embodiment_for(ep, table, t, episode_position=pos),
            "episode_index": torch.tensor(ep, dtype=torch.int64),
            "frame_index": torch.tensor(t, dtype=torch.int64),
            "index": torch.tensor(raw_index, dtype=torch.int64),
            "timestamp": torch.tensor(t / self.fps, dtype=torch.float32),
            # Stable references used by diagnostics and hard-example replay.
            # Images are intentionally not copied into diagnostic artifacts:
            # these fields are sufficient to decode them again from the source.
            "data_root": str(getattr(self, "root", "")),
            "source_id": getattr(getattr(self, "root", None), "name", ""),
            "observation_frame_indices": torch.tensor(obs_indices, dtype=torch.int64),
            "history_frame_indices": torch.tensor(obs_indices[:-1], dtype=torch.int64),
            "action_representation": getattr(self, "action_representation", "abs"),
        }
        source_cameras = getattr(self, "_source_camera_keys", self.camera_keys)
        for source_key, canonical_key in zip(
            source_cameras, self.camera_keys, strict=True
        ):
            frames = self._decode_frames(source_key, ep, obs_indices)
            frames = self._rotate_camera_frames(frames, canonical_key)
            item[canonical_key] = frames[0] if self.obs_steps == 1 else frames
        return item
