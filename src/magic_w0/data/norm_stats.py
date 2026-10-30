# Copyright 2026 MagicVLA authors. (Apache-2.0.)
"""Normalization-statistics accumulation and versioned artifact I/O."""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

NORM_STATS_FORMAT = "magic_w0.norm_stats.v1"
_STAT_NAMES = ("count", "mean", "std", "min", "max", "q01", "q99")

logger = logging.getLogger(__name__)


@dataclass
class MaskedStreamingStats:
    """Population statistics over a fixed cell shape, with a validity mask.

    ``dim`` is a width for the usual one-cell-per-dimension layout, or a full
    shape when the statistic carries an extra axis -- action bands gain a
    leading chunk-position axis under ``action_stats_per_horizon``.  Every
    update reduces over the leading sample axis, so the arithmetic is the same
    either way.
    """

    dim: int | tuple[int, ...]
    shape: tuple[int, ...] = field(init=False)
    count: np.ndarray = field(init=False)
    mean: np.ndarray = field(init=False)
    m2: np.ndarray = field(init=False)
    minimum: np.ndarray = field(init=False)
    maximum: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        shape = self.dim if isinstance(self.dim, tuple) else (self.dim,)
        if not shape or any(int(size) <= 0 for size in shape):
            raise ValueError("stats dimension must be positive")
        self.shape = tuple(int(size) for size in shape)
        self.count = np.zeros(self.shape, dtype=np.int64)
        self.mean = np.zeros(self.shape, dtype=np.float64)
        self.m2 = np.zeros(self.shape, dtype=np.float64)
        self.minimum = np.full(self.shape, np.inf, dtype=np.float64)
        self.maximum = np.full(self.shape, -np.inf, dtype=np.float64)

    def update(self, values: np.ndarray, valid: np.ndarray | None = None) -> None:
        values = np.asarray(values, dtype=np.float64).reshape(-1, *self.shape)
        if valid is None:
            valid_array = np.ones_like(values, dtype=np.bool_)
        else:
            valid_array = np.asarray(valid, dtype=np.bool_).reshape(-1, *self.shape)
            if valid_array.shape != values.shape:
                raise ValueError(
                    f"stats values/mask shape mismatch: {values.shape} vs {valid_array.shape}"
                )
        valid_array &= np.isfinite(values)
        batch_count = valid_array.sum(axis=0, dtype=np.int64)
        if not batch_count.any():
            return

        safe = np.where(valid_array, values, 0.0)
        batch_mean = np.divide(
            safe.sum(axis=0),
            batch_count,
            out=np.zeros(self.shape, dtype=np.float64),
            where=batch_count > 0,
        )
        batch_m2 = np.where(valid_array, (values - batch_mean) ** 2, 0.0).sum(axis=0)
        total = self.count + batch_count
        delta = batch_mean - self.mean
        self.mean += delta * batch_count / np.maximum(total, 1)
        self.m2 += batch_m2 + delta**2 * self.count * batch_count / np.maximum(total, 1)
        self.count = total
        self.minimum = np.minimum(
            self.minimum, np.where(valid_array, values, np.inf).min(axis=0)
        )
        self.maximum = np.maximum(
            self.maximum, np.where(valid_array, values, -np.inf).max(axis=0)
        )

    def finalize(
        self,
        *,
        quantile_values: list[list[np.ndarray]],
        min_std: float,
        quantiles: tuple[float, float] = (0.01, 0.99),
    ) -> dict[str, np.ndarray]:
        quantile_low, quantile_high = (float(value) for value in quantiles)
        if not 0.0 <= quantile_low < quantile_high <= 1.0:
            raise ValueError(
                f"quantiles must satisfy 0 <= low < high <= 1, got {quantiles}"
            )
        cells = int(np.prod(self.shape))
        if len(quantile_values) != cells:
            raise ValueError(
                f"quantile collector holds {len(quantile_values)} cells, stats need {cells}"
            )
        observed = self.count > 0
        variance = np.divide(
            self.m2,
            self.count,
            out=np.zeros_like(self.m2),
            where=observed,
        )
        std = np.maximum(np.sqrt(np.maximum(variance, 0.0)), min_std)
        mean = np.where(observed, self.mean, 0.0)
        minimum = np.where(observed, self.minimum, -1.0)
        maximum = np.where(observed, self.maximum, 1.0)
        # The collector is flat; cell ``i`` is the C-order position in ``shape``,
        # which is what ``collect_quantile_values`` produces when the caller
        # reshapes each sample to one row of ``prod(shape)`` values.
        q01 = np.full(cells, -1.0, dtype=np.float64)
        q99 = np.full(cells, 1.0, dtype=np.float64)
        for cell, parts in enumerate(quantile_values):
            if not parts:
                continue
            values = np.concatenate(parts).astype(np.float64, copy=False)
            values = values[np.isfinite(values)]
            if values.size:
                q01[cell], q99[cell] = np.quantile(
                    values, (quantile_low, quantile_high)
                )
        q01 = q01.reshape(self.shape)
        q99 = q99.reshape(self.shape)
        return {
            "count": self.count.copy(),
            "mean": mean.astype(np.float32),
            "std": np.where(observed, std, 1.0).astype(np.float32),
            "min": minimum.astype(np.float32),
            "max": maximum.astype(np.float32),
            "q01": q01.astype(np.float32),
            "q99": q99.astype(np.float32),
        }


def collect_quantile_values(
    collector: list[list[np.ndarray]],
    values: np.ndarray,
    valid: np.ndarray,
) -> None:
    """Append masked values to a small, preselected quantile sample."""
    dim = len(collector)
    values = np.asarray(values).reshape(-1, dim)
    valid = np.asarray(valid, dtype=np.bool_).reshape(-1, dim)
    if values.shape != valid.shape:
        raise ValueError("quantile values/mask shape mismatch")
    for index in range(dim):
        selected = values[valid[:, index], index]
        if selected.size:
            collector[index].append(np.asarray(selected, dtype=np.float32))


def apply_horizon_carveouts(
    per_horizon: MutableMapping[str, np.ndarray],
    flat: Mapping[str, np.ndarray],
    *,
    delta_dims: Sequence[int],
    min_span_fraction: float = 0.01,
) -> dict[str, Any]:
    """Make a per-(position, dimension) band table safe to divide by.

    Splitting an action band by chunk position is only sound where the value at
    that position actually varies across anchors.  Two cases where it does not,
    both of which produce a band narrow enough that the division amplifies noise
    without bound -- the failure openpi's norm_stats docs warn about, which
    slicing by position manufactures rather than merely risks:

    * **A dimension that is not a delta.**  Grippers pass through absolutely, so
      a position slice can catch one that never moves within that row across
      every anchor.  Those dimensions keep the full-horizon band, repeated down
      the column so the table keeps one shape.  The 2025 BEHAVIOR Challenge
      winner (arXiv:2512.06951 4.5.2) excludes grippers from its own
      per-timestep normalization for the same reason.
    * **A slice that is degenerate anyway.**  Early ``chunk_delta`` rows can be
      nearly constant when the immediate action target closely tracks the anchor
      state; later rows can also collapse on a dimension the robot happens not to
      use.  A slice narrower than ``min_span_fraction`` of that dimension's
      full-horizon band is replaced by it rather than trusted.

    ``per_horizon`` is edited in place.  Returns a summary naming what was
    changed, which belongs in the artifact and in the training log: silently
    widening a band is exactly the kind of thing that should not be silent.
    """
    span = per_horizon["q99"] - per_horizon["q01"]
    horizon, dims = span.shape
    delta = {int(index) for index in delta_dims}
    flat_span = np.asarray(flat["q99"], dtype=np.float64) - np.asarray(
        flat["q01"], dtype=np.float64
    )
    tiled: list[int] = []
    widened = 0
    for dim in range(dims):
        if dim not in delta:
            tiled.append(dim)
            for name in ("q01", "q99", "min", "max", "mean", "std"):
                if name in per_horizon and name in flat:
                    per_horizon[name][:, dim] = flat[name][dim]
            continue
        floor = max(1e-6, min_span_fraction * float(flat_span[dim]))
        narrow = span[:, dim] < floor
        if narrow.any():
            widened += int(narrow.sum())
            # Replace the whole normalization contract, not only quantiles.
            # Quantile training reads q01/q99, while a faithful implementation
            # of the source paper may choose mean/std; both must be protected
            # from a position slice whose spread is too small to trust.
            for name in ("q01", "q99", "min", "max", "mean", "std"):
                if name in per_horizon and name in flat:
                    per_horizon[name][narrow, dim] = flat[name][dim]
    widened_span = per_horizon["q99"] - per_horizon["q01"]
    delta_columns = sorted(index for index in delta if 0 <= index < dims)
    if delta_columns:
        first_to_last = float(
            np.median(
                widened_span[0, delta_columns]
                / np.maximum(widened_span[-1, delta_columns], 1e-9)
            )
        )
    else:
        first_to_last = 1.0
    return {
        "stats_layout": "per_horizon_dim",
        "horizon": int(horizon),
        "tiled_absolute_dims": tuple(tiled),
        "narrow_slices_widened": widened,
        "min_span_fraction": float(min_span_fraction),
        # How much narrower the first row's band is than the last: the whole
        # reason for the extra axis, and a number worth watching drift on.
        "span_ratio_first_to_last": first_to_last,
    }


def denormalize_values(
    value: np.ndarray,
    stats: Mapping[str, Any],
    mode: str,
) -> np.ndarray:
    """Invert flat ``[D]`` or per-horizon ``[H, D]`` normalization."""
    value = np.asarray(value, dtype=np.float32)
    if mode == "none":
        return value.copy()
    if mode == "mean_std":
        mean = np.asarray(stats["mean"], dtype=np.float32)
        std = np.asarray(stats["std"], dtype=np.float32)
        _validate_trailing_stats_shape(mean, value)
        _validate_trailing_stats_shape(std, value)
        return value * np.maximum(std, 1e-6) + mean
    if mode != "quantile":
        raise ValueError(f"unsupported normalization mode {mode!r}")
    low = np.asarray(stats["q01"] if "q01" in stats else stats["min"], dtype=np.float32)
    high = np.asarray(
        stats["q99"] if "q99" in stats else stats["max"], dtype=np.float32
    )
    _validate_trailing_stats_shape(low, value)
    _validate_trailing_stats_shape(high, value)
    return (value + 1.0) * 0.5 * np.maximum(high - low, 1e-6) + low


def _validate_trailing_stats_shape(stat: np.ndarray, value: np.ndarray) -> None:
    if stat.ndim == 0 or stat.ndim > value.ndim:
        raise ValueError(
            f"normalization stats shape {stat.shape} cannot apply to value shape {value.shape}"
        )
    if tuple(value.shape[-stat.ndim :]) != tuple(stat.shape):
        raise ValueError(
            f"normalization stats shape {stat.shape} does not match trailing value "
            f"shape {value.shape[-stat.ndim :]}"
        )


def save_norm_stats_artifact(
    path: str | Path,
    *,
    signature: Mapping[str, Any],
    stats: Mapping[str, Mapping[str, np.ndarray]],
    sampling: Mapping[str, Any],
) -> None:
    payload = {
        "format": NORM_STATS_FORMAT,
        "signature": _jsonable(signature),
        "sampling": _jsonable(sampling),
        "stats": {
            key: {name: np.asarray(value).tolist() for name, value in feature.items()}
            for key, feature in stats.items()
        },
    }
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def anchor_index_path(norm_stats_path: str | Path) -> Path:
    """Sidecar holding the eligible-anchor indices for a norm artifact.

    Kept beside the JSON rather than inside it because the array is large
    (EgoDex: ~73M int64 anchors) and because both are produced and consumed
    together -- two independently-pathed files would be two chances to pair a
    stale one with a fresh one.
    """
    return Path(norm_stats_path).with_suffix(".anchors.npz")


def save_anchor_index(
    norm_stats_path: str | Path, indices: np.ndarray | None
) -> Path | None:
    """Persist the anchors a quality filter accepted, next to the norm artifact.

    ``None`` means the source has no ``quality.current_frame`` rule and every raw
    frame is an anchor; there is nothing to cache, and no file is written, so a
    later load falls through to the ordinary path.
    """
    if indices is None:
        return None
    target = anchor_index_path(norm_stats_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, indices=np.ascontiguousarray(indices, dtype=np.int64))
    return target


def load_anchor_index(norm_stats_path: str | Path) -> np.ndarray | None:
    """Read the sidecar if it exists, else ``None`` so the caller rescans.

    Deliberately unvalidated here.  The caller feeds what it loads into
    ``norm_stats_signature()``, whose ``eligible_indices_sha256`` and
    ``eligible_anchors`` are computed FROM these indices; the existing signature
    comparison in :func:`load_norm_stats_artifact` therefore rejects a stale or
    foreign sidecar as a mismatch against the JSON written alongside it.  There is
    no path where a wrong index array is silently accepted.
    """
    source = anchor_index_path(norm_stats_path)
    if not source.exists():
        return None
    with np.load(source) as payload:
        return np.ascontiguousarray(payload["indices"], dtype=np.int64)


def sample_anchor_index_v2_path(norm_stats_path: str | Path) -> Path:
    """Versioned sidecar for stride/keyframe-selected final sample anchors."""
    return Path(norm_stats_path).with_suffix(".sample_anchors_v2.npz")


def _canonical_signature(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def save_sample_anchor_index_v2(
    norm_stats_path: str | Path,
    indices: np.ndarray,
    *,
    cache_signature: Mapping[str, Any],
) -> Path:
    """Persist final sampled anchors without reusing legacy quality sidecars.

    The old ``.anchors.npz`` means "frames accepted by current-frame quality".
    This sidecar means the final stride/keyframe union after quality filtering;
    the versioned name and embedded signature prevent the two from being mixed.
    """
    target = sample_anchor_index_v2_path(norm_stats_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        indices=np.ascontiguousarray(indices, dtype=np.int64),
        format_version=np.asarray(2, dtype=np.int64),
        anchor_kind=np.asarray("sample"),
        cache_signature=np.asarray(_canonical_signature(cache_signature)),
    )
    return target


def load_sample_anchor_index_v2(
    norm_stats_path: str | Path,
    *,
    expected_cache_signature: Mapping[str, Any],
) -> np.ndarray | None:
    """Load a matching final-anchor sidecar, or return ``None`` if absent."""
    source = sample_anchor_index_v2_path(norm_stats_path)
    if not source.exists():
        return None
    with np.load(source) as payload:
        version = int(np.asarray(payload["format_version"]).item())
        kind = str(np.asarray(payload["anchor_kind"]).item())
        actual = str(np.asarray(payload["cache_signature"]).item())
        if version != 2 or kind != "sample":
            raise ValueError(
                f"{source}: expected sample-anchor cache v2, got kind={kind!r}, "
                f"version={version}"
            )
        expected = _canonical_signature(expected_cache_signature)
        if actual != expected:
            raise ValueError(
                f"{source}: sample-anchor cache signature mismatch; regenerate the "
                "anchor sidecar and norm artifact for this source configuration"
            )
        indices = np.ascontiguousarray(payload["indices"], dtype=np.int64)
    if indices.ndim != 1:
        raise ValueError(f"{source}: anchor indices must be one-dimensional")
    if indices.size and (indices[0] < 0 or np.any(indices[1:] <= indices[:-1])):
        raise ValueError(
            f"{source}: anchor indices must be non-negative and strictly sorted"
        )
    return indices


def episode_index_path(norm_stats_path: str | Path) -> Path:
    """Sidecar holding the per-episode (index, length, tasks) table."""
    return Path(norm_stats_path).with_suffix(".episodes.npz")


def save_episode_index(
    norm_stats_path: str | Path,
    rows: Sequence[tuple[int, int, Sequence[str]]],
) -> Path | None:
    """Persist the episode table parsed out of ``meta/episodes.jsonl``.

    That file is 9.3 GB for EgoDex, and reading it means ``json.loads`` on 287,055
    lines to extract three fields per episode.  It happens before anything else in
    the reader's constructor and, unlike the action statistics and the anchor
    index, was not covered by any artifact -- so every start paid it in full.

    Tasks are stored as a flat string array plus offsets rather than an object
    array, so the file stays loadable without ``allow_pickle``.
    """
    if not rows:
        return None
    target = episode_index_path(norm_stats_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    flat: list[str] = []
    offsets = [0]
    for _, _, tasks in rows:
        flat.extend(str(task) for task in tasks)
        offsets.append(len(flat))
    np.savez_compressed(
        target,
        episode_index=np.asarray([row[0] for row in rows], dtype=np.int64),
        length=np.asarray([row[1] for row in rows], dtype=np.int64),
        task_offsets=np.asarray(offsets, dtype=np.int64),
        tasks=np.asarray(flat, dtype=object).astype("U")
        if flat
        else np.empty(0, dtype="U1"),
    )
    return target


def load_episode_index(
    norm_stats_path: str | Path,
) -> list[tuple[int, int, list[str]]] | None:
    """Read the sidecar if present, else ``None`` so the caller parses the jsonl.

    Unvalidated on purpose, exactly like :func:`load_anchor_index`: the caller
    hashes ``(episode_index, length)`` into ``episode_layout_sha256``, which
    :func:`load_norm_stats_artifact` then compares against the JSON written beside
    this file.  A sidecar from a different dataset or a re-exported one fails
    there.
    """
    source = episode_index_path(norm_stats_path)
    if not source.exists():
        return None
    with np.load(source, allow_pickle=False) as payload:
        indices = payload["episode_index"]
        lengths = payload["length"]
        offsets = payload["task_offsets"]
        tasks = payload["tasks"]
    return [
        (
            int(indices[position]),
            int(lengths[position]),
            [str(task) for task in tasks[offsets[position] : offsets[position + 1]]],
        )
        for position in range(len(indices))
    ]


def load_norm_stats_artifact(
    path: str | Path,
    *,
    expected_signature: Mapping[str, Any],
    feature_dims: Mapping[str, int | tuple[int, ...]],
    allow_dataset_mismatch: bool = False,
) -> dict[str, dict[str, np.ndarray]]:
    """Load a norm artifact, checking every array against its expected shape.

    ``feature_dims`` accepts either a width -- the historical one-row-per-
    dimension layout -- or a full shape tuple.  Action stats gained a leading
    chunk-position axis (see ``action_stats_per_horizon``), and an artifact
    written under one layout must never be silently reused under the other: the
    numbers would broadcast without error and quietly renormalize every row of
    the chunk against the wrong band.  The signature carries the layout too, so
    a mismatch is normally caught before this check; the shape check is the
    backstop for an artifact hand-edited or produced out of band.
    """
    artifact_path = Path(path)
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    if payload.get("format") != NORM_STATS_FORMAT:
        raise ValueError(
            f"{artifact_path}: unsupported norm format {payload.get('format')!r}; "
            f"expected {NORM_STATS_FORMAT!r}"
        )
    actual_signature = payload.get("signature")
    expected = _jsonable(expected_signature)
    if allow_dataset_mismatch and isinstance(actual_signature, Mapping):
        actual_signature = dict(actual_signature)
        actual_signature["dataset"] = expected.get("dataset")
    # ``stats_layout`` was added after the first flat [D] artifacts had already
    # been published.  Missing could only mean the historical per-dimension
    # layout; per-horizon [H,D] artifacts did not exist yet.  Keep that one-way
    # compatibility while retaining strict checks for every explicit value and
    # for all per-horizon consumers.  The shape validation below remains the
    # backstop: a purported legacy artifact with [H,D] arrays is still rejected.
    if isinstance(actual_signature, Mapping) and isinstance(
        expected.get("action"), Mapping
    ):
        actual_signature = dict(actual_signature)
        actual_action = dict(actual_signature.get("action", {}))
        expected_action = expected["action"]
        compatibility_notes: list[str] = []
        if (
            "stats_layout" not in actual_action
            and expected_action.get("stats_layout") == "per_dim"
        ):
            actual_action["stats_layout"] = "per_dim"
            compatibility_notes.append("action.stats_layout='per_dim'")
        for section in ("state", "action"):
            expected_section = expected.get(section)
            actual_section = actual_signature.get(section)
            if not isinstance(expected_section, Mapping) or not isinstance(
                actual_section, Mapping
            ):
                continue
            if (
                expected_section.get("preserve_selected_slots") is False
                and "preserve_selected_slots" not in actual_section
            ):
                updated = dict(actual_section)
                if section == "action":
                    updated.update(actual_action)
                updated["preserve_selected_slots"] = False
                actual_signature[section] = updated
                if section == "action":
                    actual_action = updated
                compatibility_notes.append(f"{section}.preserve_selected_slots=False")
        # Before this field existed, action statistics were always collected for
        # every dimension. A full-dimensional artifact is safe for a run that
        # supervises a subset; the reverse direction remains a strict mismatch.
        if (
            "supervision_indices" not in actual_action
            and "supervision_indices" in expected_action
        ):
            actual_action["supervision_indices"] = expected_action[
                "supervision_indices"
            ]
            compatibility_notes.append("action.supervision_indices=legacy_all_dims")
        if compatibility_notes:
            actual_signature["action"] = actual_action
            logger.warning(
                "%s: legacy norm artifact missing %s; applying historical defaults",
                artifact_path,
                ", ".join(compatibility_notes),
            )
    if actual_signature != expected:
        differences = _signature_differences(expected, actual_signature)
        preview = "; ".join(differences[:8])
        raise ValueError(f"{artifact_path}: norm signature mismatch: {preview}")
    raw_stats = payload.get("stats")
    if not isinstance(raw_stats, Mapping):
        raise TypeError(f"{artifact_path}: stats must be a mapping")

    loaded: dict[str, dict[str, np.ndarray]] = {}
    for key, dim in feature_dims.items():
        raw_feature = raw_stats.get(key)
        if raw_feature is None:
            raise ValueError(f"{artifact_path}: missing stats for feature {key!r}")
        if not isinstance(raw_feature, Mapping):
            raise TypeError(
                f"{artifact_path}: stats for feature {key!r} must be a mapping"
            )
        feature: dict[str, np.ndarray] = {}
        for name in _STAT_NAMES:
            if name not in raw_feature:
                raise ValueError(f"{artifact_path}: {key!r} is missing {name!r}")
            dtype = np.int64 if name == "count" else np.float32
            value = np.asarray(raw_feature[name], dtype=dtype)
            expected_shape = dim if isinstance(dim, tuple) else (dim,)
            if value.shape != expected_shape:
                raise ValueError(
                    f"{artifact_path}: {key}.{name} shape {value.shape}, "
                    f"expected {expected_shape}"
                )
            feature[name] = value
        _validate_feature_stats(artifact_path, key, feature)
        loaded[key] = feature
    return loaded


def _validate_feature_stats(
    path: Path, key: str, stats: Mapping[str, np.ndarray]
) -> None:
    for name in ("mean", "std", "min", "max", "q01", "q99"):
        if not np.isfinite(stats[name]).all():
            raise ValueError(f"{path}: {key}.{name} contains non-finite values")
    if (stats["count"] < 0).any():
        raise ValueError(f"{path}: {key}.count contains negative values")
    if (stats["std"] <= 0).any():
        raise ValueError(f"{path}: {key}.std must be positive")
    if (stats["min"] > stats["max"]).any():
        raise ValueError(f"{path}: {key}.min exceeds max")
    if (stats["q01"] > stats["q99"]).any():
        raise ValueError(f"{path}: {key}.q01 exceeds q99")


def _signature_differences(expected: Any, actual: Any, prefix: str = "") -> list[str]:
    if isinstance(expected, Mapping) and isinstance(actual, Mapping):
        differences: list[str] = []
        for key in sorted(set(expected) | set(actual)):
            child = f"{prefix}.{key}" if prefix else str(key)
            if key not in expected:
                differences.append(f"unexpected {child}")
            elif key not in actual:
                differences.append(f"missing {child}")
            else:
                differences.extend(
                    _signature_differences(expected[key], actual[key], child)
                )
        return differences
    if expected != actual:
        return [f"{prefix}: expected {expected!r}, got {actual!r}"]
    return []


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("norm metadata must not contain non-finite floats")
    return value
