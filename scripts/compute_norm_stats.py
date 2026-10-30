#!/usr/bin/env python3
"""Compute a training-compatible norm JSON for a LeRobot v2.1 source.

The script reuses :class:`data.backends.lerobot_v21.LeRobotV21Dataset` to ensure
that state/action selection, right-padding, dimension masks, quality filters,
observation history, action chunks, semantic upsampling, and
``abs``/``chunk_delta``/``frame_delta`` conversion exactly match training.
Video is never decoded.

For a transformed source, prefer config mode so every backend parameter comes
from the same YAML used for training::

    python3 scripts/compute_norm_stats.py \
      --config configs/train/robodojo_sft.yaml \
      --dataset-index 0 \
      --output .cache/magic_w0/norm/robodojo.json

For a plain ``observation.state`` + absolute ``action`` v2.1 table::

    python3 scripts/compute_norm_stats.py \
      --root /path/to/table_000_lerobot_v21_arx5 \
      --output outputs/norm/hy_umi_table_000.json

By default, mean/std/min/max use every eligible anchor. ``--sample-ratio``
selects a deterministic, uniformly spaced fraction instead; the legacy
``--max-anchors`` absolute cap remains available as a mutually exclusive
alternative. q01/q99 use a 10,000-anchor subset by default and accept
``--quantile-anchors all`` to retain the entire moment selection. The artifact
records all sampling choices. On Linux, ``--num-workers`` parallelizes the
anchor scan with worker processes while preserving the same global selection
and quantile sample. For exact full-data quantiles without retaining all values
in RAM, combine ``--quantile-anchors all`` with ``--quantile-storage memmap``
and point ``--quantile-temp-dir`` at a large local disk.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import multiprocessing
import shutil
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from magic_w0.data.backends.lerobot import LeRobotBackend
from magic_w0.data.backends.lerobot_v21 import LeRobotV21Dataset
from magic_w0.data.config import (
    DatasetGroupConfig,
    SubDatasetConfig,
    source_config_from_dict,
)
from magic_w0.data.lerobot_history import episode_observation_indices
from magic_w0.data.norm_stats import (
    MaskedStreamingStats,
    apply_horizon_carveouts,
    collect_quantile_values,
    save_anchor_index,
    save_episode_index,
    save_sample_anchor_index_v2,
    save_norm_stats_artifact,
)


def _positive_or_all(value: str) -> int | None:
    if value.strip().lower() == "all":
        return None
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer or 'all'")
    return parsed


def _ratio(value: str) -> float:
    parsed = float(value)
    if not 0.0 < parsed <= 1.0:
        raise argparse.ArgumentTypeError("expected a ratio in the interval (0, 1]")
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--config", type=Path, help="Training YAML containing the source."
    )
    source.add_argument("--root", type=Path, help="Plain local LeRobot v2.1 root.")
    parser.add_argument(
        "--source-index",
        type=int,
        default=None,
        help=(
            "Index within a dataset_group's sources. Required when "
            "--dataset-index selects a group unless --all-group-sources is used."
        ),
    )
    parser.add_argument(
        "--all-group-sources",
        action="store_true",
        help=(
            "Combine every leaf of the selected dataset_group into one norm "
            "domain. All non-dataset transform signatures must match exactly; "
            "training sources must set allow_norm_stats_dataset_mismatch=true."
        ),
    )
    parser.add_argument(
        "--dataset-index",
        type=int,
        default=0,
        help="Zero-based data.datasets index in --config mode (default: 0).",
    )
    parser.add_argument(
        "--reuse-configured-index-cache",
        action="store_true",
        help=(
            "Explicitly reuse .anchors.npz/.episodes.npz beside the source's "
            "configured norm_stats_path. Safe when only the action transform or "
            "sampling budget changed; omit it after changing data or quality rules."
        ),
    )
    parser.add_argument(
        "--params-json",
        type=Path,
        help="Optional backend params JSON merged in --root mode for advanced transforms.",
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Destination norm JSON."
    )
    anchor_sampling = parser.add_mutually_exclusive_group()
    anchor_sampling.add_argument(
        "--sample-ratio",
        type=_ratio,
        default=None,
        help=(
            "Uniform fraction of all eligible anchors used for moments; "
            "1 means all data and 0.2 means 20%% (default: 1)."
        ),
    )
    anchor_sampling.add_argument(
        "--max-anchors",
        type=_positive_or_all,
        default=None,
        help=(
            "Legacy absolute anchor cap for moments: positive integer or "
            "'all'. Mutually exclusive with --sample-ratio."
        ),
    )
    parser.add_argument(
        "--quantile-anchors",
        type=_positive_or_all,
        default=10_000,
        help=(
            "Selected anchors retained for q01/q99: positive integer or "
            "'all' (default: 10000)."
        ),
    )
    parser.add_argument(
        "--quantile-low",
        type=float,
        default=0.01,
        help=(
            "Lower normalization quantile as a probability (default: 0.01). "
            "Use 0.001 for q0.1. The artifact keeps the compatibility key q01."
        ),
    )
    parser.add_argument(
        "--quantile-high",
        type=float,
        default=0.99,
        help=(
            "Upper normalization quantile as a probability (default: 0.99). "
            "Use 0.999 for q99.9. The artifact keeps the compatibility key q99."
        ),
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help=(
            "CPU worker processes used by the anchor scan (default: 1). "
            "Workers require Linux fork and share the already-built dataset "
            "metadata copy-on-write. Quantile samples are merged exactly."
        ),
    )
    parser.add_argument(
        "--quantile-storage",
        choices=("memory", "memmap"),
        default="memory",
        help=(
            "Storage backend for q01/q99 values. 'memory' is fastest for bounded "
            "samples; 'memmap' writes dense worker shards to disk and performs "
            "exact external partition with bounded RAM (default: memory)."
        ),
    )
    parser.add_argument(
        "--quantile-temp-dir",
        type=Path,
        default=None,
        help=(
            "Existing local-disk directory under which memmap temporary files "
            "are created. Required with --quantile-storage memmap."
        ),
    )
    parser.add_argument(
        "--quantile-buffer-anchors",
        type=int,
        default=4096,
        help=(
            "Quantile anchors buffered per worker before a contiguous memmap "
            "write (default: 4096)."
        ),
    )
    parser.add_argument("--min-std", type=float, default=1e-6)
    parser.add_argument("--progress-interval", type=int, default=5_000)
    # Convenient direct-mode options. Complex quality/interpolation mappings belong
    # in --params-json or, preferably, the actual training YAML.
    parser.add_argument("--action-key", default="action")
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument(
        "--episodes",
        default=None,
        help="Comma-separated episode indices in --root mode (default: all).",
    )
    parser.add_argument(
        "--action-representation",
        choices=("abs", "chunk_delta", "frame_delta"),
        default="abs",
    )
    parser.add_argument(
        "--action-stats-per-horizon",
        action="store_true",
        help="One band per (chunk position, dimension) instead of per dimension. "
        "--config mode reads this from the source's params instead, so the "
        "artifact and the training run cannot disagree about the layout.",
    )
    parser.add_argument("--state-mask-key", default=None)
    parser.add_argument("--action-mask-key", default=None)
    parser.add_argument("--target-state-dim", type=int, default=None)
    parser.add_argument("--target-action-dim", type=int, default=None)
    return parser


def _sources_from_args(args: argparse.Namespace) -> list[SubDatasetConfig]:
    if args.config is not None:
        from magic_w0.training.config import load_recipe

        payload = load_recipe(args.config)
        try:
            sources = payload["data"]["datasets"]
        except (KeyError, TypeError) as exc:
            raise ValueError(f"{args.config}: expected data.datasets list") from exc
        if not isinstance(sources, list):
            raise TypeError(f"{args.config}: data.datasets must be a list")
        if args.dataset_index < 0 or args.dataset_index >= len(sources):
            raise IndexError(
                f"--dataset-index {args.dataset_index} outside [0, {len(sources)})"
            )
        parsed = source_config_from_dict(sources[args.dataset_index])
        if parsed is None:
            raise ValueError(f"dataset {args.dataset_index} is disabled")
        if isinstance(parsed, DatasetGroupConfig):
            if args.all_group_sources:
                if args.source_index is not None:
                    raise ValueError(
                        "pass either --all-group-sources or --source-index, not both"
                    )
                parsed_sources = list(parsed.datasets)
            elif args.source_index is None:
                raise TypeError(
                    "select a leaf source; dataset_group is not a single norm "
                    f"domain (it has {len(parsed.datasets)} sources -- pass "
                    "--source-index), or explicitly pass --all-group-sources "
                    "for one compatible shared domain"
                )
            else:
                if not 0 <= args.source_index < len(parsed.datasets):
                    raise IndexError(
                        f"--source-index {args.source_index} outside "
                        f"[0, {len(parsed.datasets)})"
                    )
                parsed_sources = [parsed.datasets[args.source_index]]
        else:
            if args.source_index is not None:
                raise ValueError(
                    "--source-index only applies when --dataset-index is a group"
                )
            if args.all_group_sources:
                raise ValueError(
                    "--all-group-sources requires --dataset-index to select a group"
                )
            parsed_sources = [parsed]
        for source_config in parsed_sources:
            if (
                not isinstance(source_config, SubDatasetConfig)
                or source_config.type != "lerobot"
            ):
                raise TypeError(
                    f"dataset {args.dataset_index} contains a non-LeRobot source"
                )
            if source_config.lerobot_version is None:
                source_config.lerobot_version = str(
                    payload.get("data", {}).get("lerobot_version", "v2.1")
                )
            if not source_config.lerobot_version.startswith("v2"):
                raise ValueError("compute_lerobot_v21_norm only supports v2.x sources")
    else:
        if args.dataset_index != 0:
            raise ValueError("--dataset-index is only meaningful with --config")
        if args.source_index is not None or args.all_group_sources:
            raise ValueError(
                "--source-index/--all-group-sources are only meaningful with --config"
            )
        params = {
            "action_key": args.action_key,
            "chunk_size": args.chunk_size,
            "action_stats_per_horizon": args.action_stats_per_horizon,
            "action_representation": args.action_representation,
            "state_mask_key": args.state_mask_key,
            "action_mask_key": args.action_mask_key,
            "target_state_dim": args.target_state_dim,
            "target_action_dim": args.target_action_dim,
        }
        params = {key: value for key, value in params.items() if value is not None}
        if args.params_json is not None:
            overrides = json.loads(args.params_json.read_text(encoding="utf-8"))
            if not isinstance(overrides, dict):
                raise TypeError("--params-json must contain a JSON mapping")
            params.update(overrides)
        parsed_sources = [
            SubDatasetConfig(
                type="lerobot",
                modality="robotics",
                root=str(args.root),
                episodes=(
                    [
                        int(value.strip())
                        for value in args.episodes.split(",")
                        if value.strip()
                    ]
                    if args.episodes is not None
                    else None
                ),
                lerobot_version="v2.1",
                params=params,
            )
        ]

    # Never consume an older artifact while producing a new one, and avoid the
    # reader's startup-only action estimate because this script replaces it below.
    # Sidecars are reusable across action-transform changes, but not necessarily
    # across data or quality-rule changes. Reuse is therefore explicit: silently
    # accepting the old cache here would re-sign its anchor/layout hashes into the
    # new artifact without ever comparing it with the underlying source.
    for parsed in parsed_sources:
        params = dict(parsed.params or {})
        previous_artifact = params.pop("norm_stats_path", None)
        # A shared-domain artifact deliberately differs in dataset identity. The
        # computation itself loads no artifact, so this reader option would be
        # invalid here and must not participate in the transform signature.
        params.pop("allow_norm_stats_dataset_mismatch", None)
        if getattr(args, "reuse_configured_index_cache", False):
            if previous_artifact is None:
                raise ValueError(
                    "--reuse-configured-index-cache requires a configured norm_stats_path"
                )
            params.setdefault("index_cache_path", previous_artifact)
        params["estimate_action_stats"] = False
        params["load_metadata_stats"] = False
        parsed.params = params
    return parsed_sources


def _selected_indices(length: int, maximum: int | None) -> range | np.ndarray:
    if length <= 0:
        raise ValueError("dataset contains no eligible anchors")
    if maximum is None or maximum >= length:
        return range(length)
    return np.unique(np.linspace(0, length - 1, maximum, dtype=np.int64))


def _moment_indices(
    length: int,
    *,
    maximum: int | None,
    sample_ratio: float | None,
) -> range | np.ndarray:
    """Resolve ratio/absolute moment sampling into one global selection."""
    if maximum is not None and sample_ratio is not None:
        raise ValueError("max_anchors and sample_ratio are mutually exclusive")
    if sample_ratio is not None:
        if not 0.0 < sample_ratio <= 1.0:
            raise ValueError("sample_ratio must be in the interval (0, 1]")
        maximum = min(length, max(1, int(np.ceil(length * sample_ratio))))
    # Neither option means the new ratio default, 1.0: all eligible anchors.
    return _selected_indices(length, maximum)


def _quantile_positions(selected_count: int, maximum: int) -> range | np.ndarray:
    """Positions in ``selected`` retained for quantiles without a huge set.

    In particular, full-data quantiles remain ``range(selected_count)`` rather
    than materialising tens of millions of Python integers.
    """
    return _selected_indices(selected_count, maximum)


def _raw_anchor(dataset: LeRobotV21Dataset, index: int) -> tuple[int, int, int]:
    raw = (
        int(dataset._sample_indices[index])
        if dataset._sample_indices is not None
        else int(index)
    )
    position = int(np.searchsorted(dataset._cum, raw, side="right") - 1)
    return position, dataset._ep_indices[position], raw - int(dataset._cum[position])


def _shared_norm_signature(
    datasets: list[LeRobotV21Dataset],
) -> dict[str, Any]:
    """Build a loadable shared signature after strict transform validation."""
    signatures = [dataset.norm_stats_signature() for dataset in datasets]
    reference = {key: value for key, value in signatures[0].items() if key != "dataset"}
    for index, signature in enumerate(signatures[1:], start=1):
        candidate = {key: value for key, value in signature.items() if key != "dataset"}
        if candidate != reference:
            raise ValueError(
                "dataset_group sources cannot share normalization because their "
                f"non-dataset transform signatures differ (source 0 vs {index})"
            )
    combined = dict(reference)
    combined["dataset"] = {
        "kind": "compatible_dataset_union",
        "source_count": len(datasets),
        "raw_frames": sum(int(item["dataset"]["raw_frames"]) for item in signatures),
        "eligible_anchors": sum(
            int(item["dataset"]["eligible_anchors"]) for item in signatures
        ),
        "sources": [item["dataset"] for item in signatures],
    }
    return combined


# The multiprocessing path deliberately uses fork. Building a LeRobot dataset
# parses metadata and loads large anchor sidecars; rebuilding all sources in
# every spawned worker would multiply both startup time and host memory. These
# globals are installed immediately before the pool is forked, so the immutable
# dataset metadata is shared copy-on-write. Episode parquet tables are opened
# only inside the worker that consumes them.
_WORKER_DATASETS: list[LeRobotV21Dataset] | None = None
_WORKER_CUMULATIVE: np.ndarray | None = None
_WORKER_SELECTED: range | np.ndarray | None = None
_WORKER_QUANTILE_POSITIONS: range | np.ndarray | None = None
_WORKER_QUANTILE_STORAGE = "memory"
_WORKER_QUANTILE_DIR: Path | None = None
_WORKER_QUANTILE_BUFFER_ANCHORS = 4096


def _local_quantile_selection(
    quantile_positions: range | np.ndarray,
    *,
    selected_count: int,
    position_start: int,
    position_stop: int,
) -> tuple[bool, set[int], int]:
    """Return the shard-local quantile membership test and exact anchor count."""
    if isinstance(quantile_positions, range):
        all_positions = (
            quantile_positions.start == 0
            and quantile_positions.stop == selected_count
            and quantile_positions.step == 1
        )
        if all_positions:
            return True, set(), position_stop - position_start
        values = range(
            max(position_start, quantile_positions.start),
            min(position_stop, quantile_positions.stop),
            quantile_positions.step,
        )
        local = set(values)
        return False, local, len(local)

    left = int(np.searchsorted(quantile_positions, position_start, side="left"))
    right = int(np.searchsorted(quantile_positions, position_stop, side="left"))
    local = {int(value) for value in quantile_positions[left:right]}
    return False, local, right - left


class _DiskQuantileWriter:
    """Buffer anchor tensors and append dense, transposed float32 disk shards.

    Invalid entries are represented by NaN. Rows are statistic cells and columns
    are samples, which lets the parent read one cell contiguously without ever
    mapping the full quantile population into resident memory.
    """

    def __init__(
        self,
        *,
        worker_id: int,
        quantile_anchors: int,
        dataset: LeRobotV21Dataset,
        directory: Path,
        buffer_anchors: int,
    ) -> None:
        self.worker_id = worker_id
        self.expected_anchors = int(quantile_anchors)
        self.obs_steps = int(dataset.obs_steps)
        self.state_dim = int(dataset.target_state_dim)
        self.chunk_size = int(dataset.chunk_size)
        self.action_dim = int(dataset.target_action_dim)
        self.per_horizon = bool(dataset.action_stats_per_horizon)
        self.buffer_anchors = int(buffer_anchors)
        self.written_anchors = 0
        self._states: list[np.ndarray] = []
        self._state_masks: list[np.ndarray] = []
        self._actions: list[np.ndarray] = []
        self._action_masks: list[np.ndarray] = []
        self.state_count = np.zeros(self.state_dim, dtype=np.int64)
        self.action_count = np.zeros(
            (self.chunk_size, self.action_dim) if self.per_horizon else self.action_dim,
            dtype=np.int64,
        )
        self.flat_action_count = (
            np.zeros(self.action_dim, dtype=np.int64) if self.per_horizon else None
        )

        self.state_shape = (
            self.state_dim,
            self.expected_anchors * self.obs_steps,
        )
        self.action_shape = (
            self.chunk_size * self.action_dim if self.per_horizon else self.action_dim,
            self.expected_anchors
            if self.per_horizon
            else self.expected_anchors * self.chunk_size,
        )
        self.state_path: Path | None = None
        self.action_path: Path | None = None
        self._state_map: np.memmap | None = None
        self._action_map: np.memmap | None = None
        if self.expected_anchors:
            self.state_path = directory / f"worker-{worker_id:04d}-state.f32"
            self.action_path = directory / f"worker-{worker_id:04d}-action.f32"
            self._state_map = np.memmap(
                self.state_path,
                dtype=np.float32,
                mode="w+",
                shape=self.state_shape,
            )
            self._action_map = np.memmap(
                self.action_path,
                dtype=np.float32,
                mode="w+",
                shape=self.action_shape,
            )

    def append(
        self,
        states: np.ndarray,
        state_mask: np.ndarray,
        actions: np.ndarray,
        action_mask: np.ndarray,
    ) -> None:
        states = np.asarray(states, dtype=np.float32)
        state_mask = np.asarray(state_mask, dtype=np.bool_)
        actions = np.asarray(actions, dtype=np.float32)
        action_mask = np.asarray(action_mask, dtype=np.bool_)
        valid_state = state_mask & np.isfinite(states)
        valid_action = action_mask & np.isfinite(actions)
        self.state_count += valid_state.reshape(-1, self.state_dim).sum(
            axis=0, dtype=np.int64
        )
        if self.per_horizon:
            self.action_count += valid_action.reshape(
                self.chunk_size, self.action_dim
            ).astype(np.int64)
            assert self.flat_action_count is not None
            self.flat_action_count += valid_action.reshape(-1, self.action_dim).sum(
                axis=0, dtype=np.int64
            )
        else:
            self.action_count += valid_action.reshape(-1, self.action_dim).sum(
                axis=0, dtype=np.int64
            )
        self._states.append(states.copy())
        self._state_masks.append(state_mask.copy())
        self._actions.append(actions.copy())
        self._action_masks.append(action_mask.copy())
        if len(self._states) >= self.buffer_anchors:
            self.flush()

    def flush(self) -> None:
        buffered = len(self._states)
        if not buffered:
            return
        if self._state_map is None or self._action_map is None:
            raise RuntimeError("received quantile data for an empty disk shard")

        states = np.stack(self._states).reshape(
            buffered, self.obs_steps, self.state_dim
        )
        state_mask = np.stack(self._state_masks).reshape(states.shape)
        states = np.where(state_mask & np.isfinite(states), states, np.nan)
        state_columns = states.transpose(2, 0, 1).reshape(self.state_dim, -1)
        state_start = self.written_anchors * self.obs_steps
        self._state_map[:, state_start : state_start + buffered * self.obs_steps] = (
            state_columns
        )

        actions = np.stack(self._actions).reshape(
            buffered, self.chunk_size, self.action_dim
        )
        action_mask = np.stack(self._action_masks).reshape(actions.shape)
        actions = np.where(action_mask & np.isfinite(actions), actions, np.nan)
        if self.per_horizon:
            action_columns = actions.reshape(buffered, -1).T
            action_start = self.written_anchors
            self._action_map[:, action_start : action_start + buffered] = action_columns
        else:
            action_columns = actions.transpose(2, 0, 1).reshape(self.action_dim, -1)
            action_start = self.written_anchors * self.chunk_size
            self._action_map[
                :, action_start : action_start + buffered * self.chunk_size
            ] = action_columns

        self.written_anchors += buffered
        self._states.clear()
        self._state_masks.clear()
        self._actions.clear()
        self._action_masks.clear()

    def close(self) -> dict[str, Any]:
        self.flush()
        if self.written_anchors != self.expected_anchors:
            raise RuntimeError(
                f"worker {self.worker_id} wrote {self.written_anchors} quantile "
                f"anchors, expected {self.expected_anchors}"
            )
        if self._state_map is not None:
            self._state_map.flush()
        if self._action_map is not None:
            self._action_map.flush()
        self._state_map = None
        self._action_map = None
        return {
            "state_path": str(self.state_path) if self.state_path else None,
            "state_shape": self.state_shape,
            "action_path": str(self.action_path) if self.action_path else None,
            "action_shape": self.action_shape,
            "state_count": self.state_count,
            "action_count": self.action_count,
            "flat_action_count": self.flat_action_count,
        }


def _merge_streaming_stats(
    target: MaskedStreamingStats, source: MaskedStreamingStats
) -> None:
    """Merge two Welford accumulators exactly up to floating-point ordering."""
    if target.shape != source.shape:
        raise ValueError(
            f"cannot merge stats with shapes {target.shape} and {source.shape}"
        )
    total = target.count + source.count
    delta = source.mean - target.mean
    target.mean += np.divide(
        delta * source.count,
        total,
        out=np.zeros_like(target.mean),
        where=total > 0,
    )
    target.m2 += source.m2 + np.divide(
        delta**2 * target.count * source.count,
        total,
        out=np.zeros_like(target.m2),
        where=total > 0,
    )
    target.count = total
    target.minimum = np.minimum(target.minimum, source.minimum)
    target.maximum = np.maximum(target.maximum, source.maximum)


def _compact_quantiles(
    collector: list[list[np.ndarray]],
) -> list[np.ndarray]:
    """Collapse per-anchor pieces before crossing the process boundary."""
    return [
        np.concatenate(parts).astype(np.float32, copy=False)
        if parts
        else np.empty(0, dtype=np.float32)
        for parts in collector
    ]


def _compute_stats_shard(
    task: tuple[int, int, int, int],
) -> dict[str, Any]:
    """Scan one contiguous slice of positions in the global selection."""
    worker_id, position_start, position_stop, progress_interval = task
    if (
        _WORKER_DATASETS is None
        or _WORKER_CUMULATIVE is None
        or _WORKER_SELECTED is None
        or _WORKER_QUANTILE_POSITIONS is None
    ):
        raise RuntimeError("norm worker context was not installed before fork")

    datasets = _WORKER_DATASETS
    cumulative = _WORKER_CUMULATIVE
    selected = _WORKER_SELECTED
    quantile_positions = _WORKER_QUANTILE_POSITIONS
    reference = datasets[0]
    action_dim = reference.target_action_dim
    per_horizon = bool(getattr(reference, "action_stats_per_horizon", False))
    state_stats = MaskedStreamingStats(reference.target_state_dim)
    action_stats = MaskedStreamingStats(
        (reference.chunk_size, action_dim) if per_horizon else action_dim
    )
    flat_action_stats = MaskedStreamingStats(action_dim) if per_horizon else None
    state_quantiles: list[list[np.ndarray]] = [
        [] for _ in range(reference.target_state_dim)
    ]
    action_quantiles: list[list[np.ndarray]] = [
        []
        for _ in range(reference.chunk_size * action_dim if per_horizon else action_dim)
    ]
    flat_action_quantiles: list[list[np.ndarray]] = (
        [[] for _ in range(action_dim)] if per_horizon else []
    )
    selected_per_source = np.zeros(len(datasets), dtype=np.int64)
    quantile_per_source = np.zeros(len(datasets), dtype=np.int64)

    (
        all_positions_are_quantiles,
        local_quantile_positions,
        local_quantile_count,
    ) = _local_quantile_selection(
        quantile_positions,
        selected_count=len(selected),
        position_start=position_start,
        position_stop=position_stop,
    )
    disk_writer: _DiskQuantileWriter | None = None
    if _WORKER_QUANTILE_STORAGE == "memmap":
        if _WORKER_QUANTILE_DIR is None:
            raise RuntimeError("memmap worker directory was not installed before fork")
        disk_writer = _DiskQuantileWriter(
            worker_id=worker_id,
            quantile_anchors=local_quantile_count,
            dataset=reference,
            directory=_WORKER_QUANTILE_DIR,
            buffer_anchors=_WORKER_QUANTILE_BUFFER_ANCHORS,
        )

    shard_size = position_stop - position_start
    for local_processed, selected_position in enumerate(
        range(position_start, position_stop), start=1
    ):
        global_index = int(selected[selected_position])
        source_index = int(np.searchsorted(cumulative, global_index, side="right"))
        source_start = 0 if source_index == 0 else int(cumulative[source_index - 1])
        index = global_index - source_start
        current = datasets[source_index]
        selected_per_source[source_index] += 1
        position, episode, frame = _raw_anchor(current, index)
        table = current._episode_table(episode)
        observation_frames = episode_observation_indices(
            frame, current.observation_delta_indices
        )
        state_rows = [frame] if current.obs_steps == 1 else list(observation_frames)
        states = table[current.state_key][state_rows]
        state_mask = table[f"{current.state_key}_dim_mask"][state_rows]
        actions, _, action_mask = current._build_action_chunk(
            table, t=frame, length=current._ep_lengths[position]
        )
        state_stats.update(states, state_mask)
        action_stats.update(actions, action_mask)
        if flat_action_stats is not None:
            flat_action_stats.update(actions, action_mask)
        if all_positions_are_quantiles or selected_position in local_quantile_positions:
            quantile_per_source[source_index] += 1
            if disk_writer is not None:
                disk_writer.append(states, state_mask, actions, action_mask)
            else:
                collect_quantile_values(state_quantiles, states, state_mask)
                collect_quantile_values(action_quantiles, actions, action_mask)
                if flat_action_stats is not None:
                    collect_quantile_values(flat_action_quantiles, actions, action_mask)
        if progress_interval and local_processed % progress_interval == 0:
            print(
                f"worker {worker_id}: processed "
                f"{local_processed:,}/{shard_size:,} anchors",
                flush=True,
            )

    disk_quantiles = disk_writer.close() if disk_writer is not None else None
    return {
        "worker_id": worker_id,
        "state_stats": state_stats,
        "action_stats": action_stats,
        "flat_action_stats": flat_action_stats,
        "state_quantiles": _compact_quantiles(state_quantiles),
        "action_quantiles": _compact_quantiles(action_quantiles),
        "flat_action_quantiles": _compact_quantiles(flat_action_quantiles),
        "disk_quantiles": disk_quantiles,
        "selected_per_source": selected_per_source,
        "quantile_per_source": quantile_per_source,
    }


def _finalize_with_external_quantiles(
    accumulator: MaskedStreamingStats,
    *,
    q01: np.ndarray,
    q99: np.ndarray,
    min_std: float,
    quantiles: tuple[float, float],
) -> dict[str, np.ndarray]:
    empty = [[] for _ in range(int(np.prod(accumulator.shape)))]
    finalized = accumulator.finalize(
        quantile_values=empty,
        min_std=min_std,
        quantiles=quantiles,
    )
    finalized["q01"] = np.asarray(q01, dtype=np.float32).reshape(accumulator.shape)
    finalized["q99"] = np.asarray(q99, dtype=np.float32).reshape(accumulator.shape)
    return finalized


def _linear_quantile_from_partitioned(
    values: np.memmap,
    *,
    count: int,
    quantile: float,
) -> float:
    rank = (count - 1) * quantile
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return float(values[lower])
    fraction = rank - lower
    return float(values[lower]) + fraction * (
        float(values[upper]) - float(values[lower])
    )


def _external_quantiles(
    *,
    descriptors: list[dict[str, Any]],
    kind: str,
    counts: np.ndarray,
    row_groups: list[tuple[int, ...]],
    work_dir: Path,
    quantiles: tuple[float, float],
    copy_chunk_values: int = 4_000_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute exact linear q01/q99 one stats cell at a time on disk."""
    flat_counts = np.asarray(counts, dtype=np.int64).reshape(-1)
    if len(row_groups) != flat_counts.size:
        raise ValueError("external quantile row/count layout mismatch")
    q01 = np.full(flat_counts.size, -1.0, dtype=np.float64)
    q99 = np.full(flat_counts.size, 1.0, dtype=np.float64)

    for cell, (count, rows) in enumerate(zip(flat_counts, row_groups, strict=True)):
        count = int(count)
        if count == 0:
            continue
        work_path = work_dir / f"partition-{kind}-{cell:04d}.f32"
        work = np.memmap(work_path, dtype=np.float32, mode="w+", shape=(count,))
        offset = 0
        for descriptor in descriptors:
            source_path = descriptor[f"{kind}_path"]
            if source_path is None:
                continue
            source_shape = tuple(int(value) for value in descriptor[f"{kind}_shape"])
            source = np.memmap(
                source_path,
                dtype=np.float32,
                mode="r",
                shape=source_shape,
            )
            for row in rows:
                width = source_shape[1]
                for start in range(0, width, copy_chunk_values):
                    chunk = np.asarray(
                        source[row, start : min(start + copy_chunk_values, width)]
                    )
                    finite = chunk[np.isfinite(chunk)]
                    if finite.size:
                        end = offset + finite.size
                        if end > count:
                            raise RuntimeError(
                                f"external quantile cell {cell} contains more valid "
                                f"values than its streaming count ({end} > {count})"
                            )
                        work[offset:end] = finite
                        offset = end
            del source
        if offset != count:
            raise RuntimeError(
                f"external quantile cell {cell} copied {offset} valid values, "
                f"but streaming stats counted {count}"
            )

        quantile_low, quantile_high = quantiles
        low_rank = (count - 1) * quantile_low
        high_rank = (count - 1) * quantile_high
        kth = sorted(
            {
                int(math.floor(low_rank)),
                int(math.ceil(low_rank)),
                int(math.floor(high_rank)),
                int(math.ceil(high_rank)),
            }
        )
        work.partition(kth)
        q01[cell] = _linear_quantile_from_partitioned(
            work, count=count, quantile=quantile_low
        )
        q99[cell] = _linear_quantile_from_partitioned(
            work, count=count, quantile=quantile_high
        )
        del work
        work_path.unlink()
        print(
            f"external quantiles {kind}: cell {cell + 1}/{flat_counts.size} "
            f"({count:,} values)",
            flush=True,
        )

    return q01.reshape(counts.shape), q99.reshape(counts.shape)


def _estimate_memmap_bytes(
    dataset: LeRobotV21Dataset,
    *,
    quantile_anchors: int,
) -> tuple[int, int]:
    dense = (
        quantile_anchors
        * (
            int(dataset.obs_steps) * int(dataset.target_state_dim)
            + int(dataset.chunk_size) * int(dataset.target_action_dim)
        )
        * np.dtype(np.float32).itemsize
    )
    largest_cell = (
        quantile_anchors
        * max(int(dataset.obs_steps), int(dataset.chunk_size))
        * np.dtype(np.float32).itemsize
    )
    return dense, largest_cell


def compute_stats(
    dataset: LeRobotV21Dataset,
    *,
    max_anchors: int | None,
    quantile_anchors: int | None,
    min_std: float,
    progress_interval: int,
    num_workers: int = 1,
    sample_ratio: float | None = None,
    quantile_storage: str = "memory",
    quantile_temp_dir: Path | None = None,
    quantile_buffer_anchors: int = 4096,
    quantile_low: float = 0.01,
    quantile_high: float = 0.99,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    return compute_combined_stats(
        [dataset],
        max_anchors=max_anchors,
        quantile_anchors=quantile_anchors,
        min_std=min_std,
        progress_interval=progress_interval,
        num_workers=num_workers,
        sample_ratio=sample_ratio,
        quantile_storage=quantile_storage,
        quantile_temp_dir=quantile_temp_dir,
        quantile_buffer_anchors=quantile_buffer_anchors,
        quantile_low=quantile_low,
        quantile_high=quantile_high,
    )


def _compute_combined_stats_impl(
    datasets: list[LeRobotV21Dataset],
    *,
    max_anchors: int | None,
    quantile_anchors: int | None,
    min_std: float,
    progress_interval: int,
    num_workers: int = 1,
    sample_ratio: float | None = None,
    quantile_storage: str = "memory",
    quantile_work_dir: Path | None = None,
    quantile_buffer_anchors: int = 4096,
    quantile_low: float = 0.01,
    quantile_high: float = 0.99,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    if not datasets:
        raise ValueError("at least one dataset is required")
    if quantile_anchors is not None and quantile_anchors <= 0:
        raise ValueError("quantile_anchors must be positive or None for all")
    quantiles = (float(quantile_low), float(quantile_high))
    if not 0.0 <= quantiles[0] < quantiles[1] <= 1.0:
        raise ValueError(
            f"quantiles must satisfy 0 <= low < high <= 1, got {quantiles}"
        )
    _shared_norm_signature(datasets)
    dataset = datasets[0]
    lengths = np.asarray([len(source) for source in datasets], dtype=np.int64)
    cumulative = np.cumsum(lengths)
    total_anchors = int(cumulative[-1])
    selected = _moment_indices(
        total_anchors,
        maximum=max_anchors,
        sample_ratio=sample_ratio,
    )
    quantile_limit = (
        len(selected)
        if quantile_anchors is None
        else min(quantile_anchors, len(selected))
    )
    quantile_positions = _quantile_positions(len(selected), quantile_limit)
    action_dim = dataset.target_action_dim
    # The flag lives on the dataset, so it reaches here from the training YAML in
    # --config mode and from the CLI otherwise: one source of truth, and the
    # signature this artifact is stamped with already records the choice.
    per_horizon = bool(getattr(dataset, "action_stats_per_horizon", False))
    if num_workers <= 0:
        raise ValueError("num_workers must be positive")
    worker_count = min(num_workers, len(selected))
    if quantile_storage == "memory" and len(quantile_positions) > 1_000_000:
        warnings.warn(
            "retaining more than 1,000,000 quantile anchors can require very "
            "large host memory; multiprocessing accelerates the scan but does "
            "not reduce the total quantile payload",
            RuntimeWarning,
            stacklevel=2,
        )

    global _WORKER_DATASETS, _WORKER_CUMULATIVE
    global _WORKER_SELECTED, _WORKER_QUANTILE_POSITIONS
    global _WORKER_QUANTILE_STORAGE, _WORKER_QUANTILE_DIR
    global _WORKER_QUANTILE_BUFFER_ANCHORS
    _WORKER_DATASETS = datasets
    _WORKER_CUMULATIVE = cumulative
    _WORKER_SELECTED = selected
    _WORKER_QUANTILE_POSITIONS = quantile_positions
    _WORKER_QUANTILE_STORAGE = quantile_storage
    _WORKER_QUANTILE_DIR = quantile_work_dir
    _WORKER_QUANTILE_BUFFER_ANCHORS = quantile_buffer_anchors
    edges = np.linspace(0, len(selected), worker_count + 1, dtype=np.int64)
    tasks = [
        (worker_id, int(edges[worker_id]), int(edges[worker_id + 1]), progress_interval)
        for worker_id in range(worker_count)
    ]
    if worker_count == 1:
        shard_results = [_compute_stats_shard(tasks[0])]
    else:
        try:
            fork_context = multiprocessing.get_context("fork")
        except ValueError as exc:
            raise RuntimeError(
                "--num-workers > 1 requires a platform with multiprocessing fork"
            ) from exc
        print(
            f"scanning {len(selected):,} anchors with {worker_count} workers",
            flush=True,
        )
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=fork_context,
        ) as executor:
            shard_results = list(executor.map(_compute_stats_shard, tasks))

    # executor.map preserves task order. Fixed-order merging makes repeated runs
    # deterministic apart from the unavoidable change in floating-point
    # association relative to a one-worker run.
    state_stats = MaskedStreamingStats(dataset.target_state_dim)
    action_stats = MaskedStreamingStats(
        (dataset.chunk_size, action_dim) if per_horizon else action_dim
    )
    flat_action_stats = MaskedStreamingStats(action_dim) if per_horizon else None
    state_quantiles: list[list[np.ndarray]] = [
        [] for _ in range(dataset.target_state_dim)
    ]
    action_quantiles: list[list[np.ndarray]] = [
        []
        for _ in range(dataset.chunk_size * action_dim if per_horizon else action_dim)
    ]
    flat_action_quantiles: list[list[np.ndarray]] = (
        [[] for _ in range(action_dim)] if per_horizon else []
    )
    disk_quantile_descriptors: list[dict[str, Any]] = []
    selected_per_source = np.zeros(len(datasets), dtype=np.int64)
    quantile_per_source = np.zeros(len(datasets), dtype=np.int64)
    for result in shard_results:
        _merge_streaming_stats(state_stats, result["state_stats"])
        _merge_streaming_stats(action_stats, result["action_stats"])
        if flat_action_stats is not None:
            _merge_streaming_stats(flat_action_stats, result["flat_action_stats"])
        if quantile_storage == "memmap":
            disk_quantile_descriptors.append(result["disk_quantiles"])
        else:
            for destination, values in zip(
                state_quantiles, result["state_quantiles"], strict=True
            ):
                if values.size:
                    destination.append(values)
            for destination, values in zip(
                action_quantiles, result["action_quantiles"], strict=True
            ):
                if values.size:
                    destination.append(values)
            for destination, values in zip(
                flat_action_quantiles,
                result["flat_action_quantiles"],
                strict=True,
            ):
                if values.size:
                    destination.append(values)
        selected_per_source += result["selected_per_source"]
        quantile_per_source += result["quantile_per_source"]

    if quantile_storage == "memmap":
        if quantile_work_dir is None:
            raise RuntimeError("memmap quantile work directory is unavailable")
        state_quantile_count = sum(
            (
                np.asarray(item["state_count"], dtype=np.int64)
                for item in disk_quantile_descriptors
            ),
            start=np.zeros(dataset.target_state_dim, dtype=np.int64),
        )
        action_quantile_shape = (
            (dataset.chunk_size, action_dim) if per_horizon else (action_dim,)
        )
        action_quantile_count = sum(
            (
                np.asarray(item["action_count"], dtype=np.int64)
                for item in disk_quantile_descriptors
            ),
            start=np.zeros(action_quantile_shape, dtype=np.int64),
        )
        state_q01, state_q99 = _external_quantiles(
            descriptors=disk_quantile_descriptors,
            kind="state",
            counts=state_quantile_count,
            row_groups=[(cell,) for cell in range(dataset.target_state_dim)],
            work_dir=quantile_work_dir,
            quantiles=quantiles,
        )
        action_q01, action_q99 = _external_quantiles(
            descriptors=disk_quantile_descriptors,
            kind="action",
            counts=action_quantile_count,
            row_groups=[(cell,) for cell in range(int(np.prod(action_quantile_shape)))],
            work_dir=quantile_work_dir,
            quantiles=quantiles,
        )
        state_finalized = _finalize_with_external_quantiles(
            state_stats,
            q01=state_q01,
            q99=state_q99,
            min_std=min_std,
            quantiles=quantiles,
        )
        action_finalized = _finalize_with_external_quantiles(
            action_stats,
            q01=action_q01,
            q99=action_q99,
            min_std=min_std,
            quantiles=quantiles,
        )
    else:
        state_finalized = state_stats.finalize(
            quantile_values=state_quantiles, min_std=min_std, quantiles=quantiles
        )
        action_finalized = action_stats.finalize(
            quantile_values=action_quantiles, min_std=min_std, quantiles=quantiles
        )
    carveouts: dict[str, Any] = {"stats_layout": "per_dim"}
    if flat_action_stats is not None:
        if quantile_storage == "memmap":
            assert quantile_work_dir is not None
            flat_action_quantile_count = sum(
                (
                    np.asarray(item["flat_action_count"], dtype=np.int64)
                    for item in disk_quantile_descriptors
                ),
                start=np.zeros(action_dim, dtype=np.int64),
            )
            flat_q01, flat_q99 = _external_quantiles(
                descriptors=disk_quantile_descriptors,
                kind="action",
                counts=flat_action_quantile_count,
                row_groups=[
                    tuple(step * action_dim + dim for step in range(dataset.chunk_size))
                    for dim in range(action_dim)
                ],
                work_dir=quantile_work_dir,
                quantiles=quantiles,
            )
            flat_action_finalized = _finalize_with_external_quantiles(
                flat_action_stats,
                q01=flat_q01,
                q99=flat_q99,
                min_std=min_std,
                quantiles=quantiles,
            )
        else:
            flat_action_finalized = flat_action_stats.finalize(
                quantile_values=flat_action_quantiles,
                min_std=min_std,
                quantiles=quantiles,
            )
        carveouts = apply_horizon_carveouts(
            action_finalized,
            flat_action_finalized,
            delta_dims=dataset.action_delta_indices or (),
        )
        print(
            f"per-horizon action bands {dataset.chunk_size}x{action_dim} -- "
            f"{len(carveouts['tiled_absolute_dims'])} absolute dims tiled, "
            f"{carveouts['narrow_slices_widened']} slices widened, "
            f"first/last row span ratio {carveouts['span_ratio_first_to_last']:.3f}",
            flush=True,
        )
    stats = {
        dataset.state_key: state_finalized,
        dataset.action_key: action_finalized,
    }
    sampling = {
        "eligible_anchors": total_anchors,
        "moment_anchors": len(selected),
        "moment_method": (
            "all_eligible_anchors"
            if len(selected) == total_anchors
            else (
                "uniformly_spaced_ratio_sample"
                if sample_ratio is not None
                else "uniformly_spaced_absolute_cap"
            )
        ),
        "requested_sample_ratio": (
            float(sample_ratio)
            if sample_ratio is not None
            else (1.0 if max_anchors is None else None)
        ),
        "effective_sample_ratio": len(selected) / total_anchors,
        "quantile_anchors": len(quantile_positions),
        "quantile_method": (
            "all_selected_anchors"
            if len(quantile_positions) == len(selected)
            else "uniformly_spaced_anchor_sample"
        ),
        "quantile_storage": quantile_storage,
        "quantiles": [quantiles[0], quantiles[1]],
        "min_std": min_std,
        "num_workers": worker_count,
        "action_stats": carveouts,
        "sources": [
            {
                "root": str(source.root),
                "eligible_anchors": int(length),
                "moment_anchors": int(selected_count),
                "quantile_anchors": int(quantile_count),
            }
            for source, length, selected_count, quantile_count in zip(
                datasets,
                lengths.tolist(),
                selected_per_source.tolist(),
                quantile_per_source.tolist(),
                strict=True,
            )
        ],
    }
    return stats, sampling


def compute_combined_stats(
    datasets: list[LeRobotV21Dataset],
    *,
    max_anchors: int | None,
    quantile_anchors: int | None,
    min_std: float,
    progress_interval: int,
    num_workers: int = 1,
    sample_ratio: float | None = None,
    quantile_storage: str = "memory",
    quantile_temp_dir: Path | None = None,
    quantile_buffer_anchors: int = 4096,
    quantile_low: float = 0.01,
    quantile_high: float = 0.99,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    """Compute stats, optionally using local-disk memmaps for exact quantiles."""
    if not datasets:
        raise ValueError("at least one dataset is required")
    if quantile_storage not in {"memory", "memmap"}:
        raise ValueError("quantile_storage must be 'memory' or 'memmap'")
    if quantile_buffer_anchors <= 0:
        raise ValueError("quantile_buffer_anchors must be positive")
    if not 0.0 <= quantile_low < quantile_high <= 1.0:
        raise ValueError(
            "quantiles must satisfy 0 <= low < high <= 1, got "
            f"({quantile_low}, {quantile_high})"
        )

    temporary: tempfile.TemporaryDirectory[str] | None = None
    work_dir: Path | None = None
    dense_bytes = 0
    workspace_bytes = 0
    if quantile_storage == "memmap":
        if quantile_temp_dir is None:
            raise ValueError(
                "quantile_temp_dir is required when quantile_storage='memmap'"
            )
        temp_parent = Path(quantile_temp_dir).expanduser().resolve()
        if not temp_parent.is_dir():
            raise ValueError(
                f"quantile_temp_dir must be an existing directory: {temp_parent}"
            )

        total_anchors = sum(len(source) for source in datasets)
        selected = _moment_indices(
            total_anchors,
            maximum=max_anchors,
            sample_ratio=sample_ratio,
        )
        quantile_count = (
            len(selected)
            if quantile_anchors is None
            else min(quantile_anchors, len(selected))
        )
        dense_bytes, workspace_bytes = _estimate_memmap_bytes(
            datasets[0], quantile_anchors=quantile_count
        )
        safety_bytes = max(
            64 << 20,
            int(math.ceil((dense_bytes + workspace_bytes) * 0.10)),
        )
        required_bytes = dense_bytes + workspace_bytes + safety_bytes
        disk = shutil.disk_usage(temp_parent)
        print(
            "memmap quantile disk estimate: "
            f"dense shards={dense_bytes / (1 << 30):.1f} GiB, "
            f"largest partition={workspace_bytes / (1 << 30):.1f} GiB, "
            f"required with safety={required_bytes / (1 << 30):.1f} GiB, "
            f"free={disk.free / (1 << 30):.1f} GiB at {temp_parent}",
            flush=True,
        )
        if disk.free < required_bytes:
            raise OSError(
                f"insufficient space for exact memmap quantiles under {temp_parent}: "
                f"need about {required_bytes / (1 << 30):.1f} GiB, "
                f"have {disk.free / (1 << 30):.1f} GiB free"
            )
        temporary = tempfile.TemporaryDirectory(
            prefix="magicw0-norm-quantiles-",
            dir=temp_parent,
        )
        work_dir = Path(temporary.name)
        print(f"memmap quantile workspace: {work_dir}", flush=True)

    try:
        stats, sampling = _compute_combined_stats_impl(
            datasets,
            max_anchors=max_anchors,
            quantile_anchors=quantile_anchors,
            min_std=min_std,
            progress_interval=progress_interval,
            num_workers=num_workers,
            sample_ratio=sample_ratio,
            quantile_storage=quantile_storage,
            quantile_work_dir=work_dir,
            quantile_buffer_anchors=quantile_buffer_anchors,
            quantile_low=quantile_low,
            quantile_high=quantile_high,
        )
        if quantile_storage == "memmap":
            sampling["quantile_memmap_dense_bytes"] = dense_bytes
            sampling["quantile_memmap_largest_partition_bytes"] = workspace_bytes
        return stats, sampling
    finally:
        if temporary is not None:
            path = temporary.name
            temporary.cleanup()
            print(f"removed memmap quantile workspace: {path}", flush=True)


def main() -> None:
    args = _build_parser().parse_args()
    if args.quantile_anchors is not None and args.quantile_anchors <= 0:
        raise ValueError("--quantile-anchors must be positive")
    if args.min_std <= 0:
        raise ValueError("--min-std must be positive")
    if args.progress_interval < 0:
        raise ValueError("--progress-interval must be non-negative")
    if args.num_workers <= 0:
        raise ValueError("--num-workers must be positive")
    if args.quantile_buffer_anchors <= 0:
        raise ValueError("--quantile-buffer-anchors must be positive")
    if not 0.0 <= args.quantile_low < args.quantile_high <= 1.0:
        raise ValueError(
            "--quantile-low/--quantile-high must satisfy 0 <= low < high <= 1"
        )
    if args.quantile_storage == "memmap" and args.quantile_temp_dir is None:
        raise ValueError(
            "--quantile-temp-dir is required with --quantile-storage memmap"
        )
    if args.quantile_storage == "memory" and args.quantile_temp_dir is not None:
        raise ValueError(
            "--quantile-temp-dir only applies with --quantile-storage memmap"
        )
    sources = _sources_from_args(args)
    datasets = [LeRobotBackend().make_dataset(source) for source in sources]
    if any(not isinstance(dataset, LeRobotV21Dataset) for dataset in datasets):
        raise TypeError("expected every source to build a LeRobotV21Dataset")

    stats, sampling = compute_combined_stats(
        datasets,
        max_anchors=args.max_anchors,
        quantile_anchors=args.quantile_anchors,
        min_std=args.min_std,
        progress_interval=args.progress_interval,
        num_workers=args.num_workers,
        sample_ratio=args.sample_ratio,
        quantile_storage=args.quantile_storage,
        quantile_temp_dir=args.quantile_temp_dir,
        quantile_buffer_anchors=args.quantile_buffer_anchors,
        quantile_low=args.quantile_low,
        quantile_high=args.quantile_high,
    )
    save_norm_stats_artifact(
        args.output,
        signature=(
            datasets[0].norm_stats_signature()
            if len(datasets) == 1
            else _shared_norm_signature(datasets)
        ),
        stats=stats,
        sampling=sampling,
    )
    # The anchors this dataset accepted, beside the stats.  Building them is the
    # other half of a cold start and the more expensive half for a quality-filtered
    # source: LeRobotV21Dataset._build_current_frame_indices opens every episode
    # parquet to read a few boolean columns, which for EgoDex is 287,055 files.
    # Writing it here costs nothing -- the array already exists, it was needed to
    # compute these very statistics -- and lets the reader skip that scan.
    # Sources without a quality.current_frame rule have no index to save; the
    # helper returns None and no file appears, so the reader falls through.
    sidecars = []
    for dataset in datasets:
        sidecar_base = (
            args.output
            if len(datasets) == 1
            else getattr(dataset, "index_cache_path", None)
        )
        if sidecar_base is None:
            sidecars.append(
                {"root": str(dataset.root), "anchor_index": None, "episode_index": None}
            )
            continue
        if dataset.anchor_sampling_spec is None:
            anchor_file = save_anchor_index(
                sidecar_base, getattr(dataset, "_sample_indices", None)
            )
        else:
            anchor_file = save_sample_anchor_index_v2(
                sidecar_base,
                getattr(dataset, "_sample_indices", None),
                cache_signature=dataset.anchor_sampling_cache_signature(),
            )
        # The episode table too. Parsing meta/episodes.jsonl can be the largest
        # single cost of a cold start, so each source keeps its own cache base.
        episode_file = save_episode_index(
            sidecar_base,
            [
                (
                    int(dataset._ep_indices[position]),
                    int(dataset._ep_lengths[position]),
                    [dataset._episode_task(position)],
                )
                for position in range(len(dataset._ep_indices))
            ],
        )
        sidecars.append(
            {
                "root": str(dataset.root),
                "anchor_index": str(anchor_file) if anchor_file else None,
                "episode_index": str(episode_file) if episode_file else None,
            }
        )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "source_sidecars": sidecars,
                "roots": [str(dataset.root) for dataset in datasets],
                "state_key": datasets[0].state_key,
                "action_key": datasets[0].action_key,
                "action_representation": datasets[0].action_representation,
                "action_delta_indices": list(datasets[0].action_delta_indices or ()),
                "anchor_sampling": datasets[0].anchor_sampling_summary,
                **sampling,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
