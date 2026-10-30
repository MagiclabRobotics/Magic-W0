# Copyright 2026 MagicVLA authors. (Apache-2.0.)
"""Deterministic episode-local sample-anchor selection for robot datasets."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class KeyframeSamplingSpec:
    """Where keyframes come from and which neighbouring frames to retain."""

    source: str
    column: str | None
    path: Path | None
    context_offsets: tuple[int, ...]
    content_sha256: str | None = None

    def signature(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "column": self.column,
            "path": str(self.path) if self.path is not None else None,
            "context_offsets": self.context_offsets,
            "content_sha256": self.content_sha256,
        }


@dataclass(frozen=True)
class AnchorSamplingSpec:
    """Per-source anchor sampling settings.

    ``stride_frames`` is applied to the frame index *within each episode*.  The
    last episode frame is never injected specially; it appears only when it is a
    natural stride position or a configured keyframe/context frame.
    """

    stride_frames: int = 1
    offset: int = 0
    keyframes: KeyframeSamplingSpec | None = None
    format_version: int = 2

    def signature(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "stride_frames": self.stride_frames,
            "offset": self.offset,
            "keyframes": self.keyframes.signature()
            if self.keyframes is not None
            else None,
        }


def parse_anchor_sampling_spec(
    value: Mapping[str, Any] | None,
    *,
    root: Path,
    feature_names: set[str],
) -> AnchorSamplingSpec | None:
    """Parse one source's ``params.anchor_sampling`` mapping.

    ``None`` means the legacy per-frame anchor behaviour. Explicit
    ``stride_frames: 1`` is accepted and remains useful when keyframes are
    configured or when a run wants the sampling contract recorded in its norm
    signature.
    """
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError("anchor_sampling must be a mapping")
    raw = dict(value)
    unknown = set(raw) - {"format_version", "stride_frames", "offset", "keyframes"}
    if unknown:
        raise ValueError(f"anchor_sampling has unknown keys {sorted(unknown)}")

    version = int(raw.get("format_version", 2))
    if version != 2:
        raise ValueError(f"anchor_sampling.format_version must be 2, got {version}")
    stride = int(raw.get("stride_frames", 1))
    offset = int(raw.get("offset", 0))
    if stride <= 0:
        raise ValueError(
            f"anchor_sampling.stride_frames must be positive, got {stride}"
        )
    if not 0 <= offset < stride:
        raise ValueError(
            "anchor_sampling.offset must satisfy 0 <= offset < stride_frames, "
            f"got offset={offset}, stride_frames={stride}"
        )

    keyframes = _parse_keyframe_spec(
        raw.get("keyframes"), root=root, feature_names=feature_names
    )
    return AnchorSamplingSpec(
        stride_frames=stride,
        offset=offset,
        keyframes=keyframes,
        format_version=version,
    )


def _parse_keyframe_spec(
    value: Mapping[str, Any] | None,
    *,
    root: Path,
    feature_names: set[str],
) -> KeyframeSamplingSpec | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError("anchor_sampling.keyframes must be a mapping")
    raw = dict(value)
    unknown = set(raw) - {"source", "column", "path", "context_offsets"}
    if unknown:
        raise ValueError(
            f"anchor_sampling.keyframes has unknown keys {sorted(unknown)}"
        )

    source = str(raw.get("source", "parquet_column"))
    if source not in {"parquet_column", "jsonl"}:
        raise ValueError(
            "anchor_sampling.keyframes.source must be 'parquet_column' or 'jsonl', "
            f"got {source!r}"
        )
    raw_offsets = raw.get("context_offsets", (0,))
    if isinstance(raw_offsets, (str, bytes)) or not isinstance(raw_offsets, Sequence):
        raise TypeError("anchor_sampling.keyframes.context_offsets must be a sequence")
    offsets = tuple(sorted({0, *(int(item) for item in raw_offsets)}))

    if source == "parquet_column":
        column = raw.get("column")
        if not isinstance(column, str) or not column:
            raise ValueError(
                "anchor_sampling.keyframes.column is required for parquet_column"
            )
        if column not in feature_names:
            raise ValueError(
                f"keyframe column {column!r} is not in meta/info.json features"
            )
        if raw.get("path") is not None:
            raise ValueError(
                "anchor_sampling.keyframes.path is only valid for source='jsonl'"
            )
        return KeyframeSamplingSpec(
            source=source,
            column=column,
            path=None,
            context_offsets=offsets,
        )

    path_value = raw.get("path")
    if not isinstance(path_value, (str, Path)) or not str(path_value):
        raise ValueError(
            "anchor_sampling.keyframes.path is required for source='jsonl'"
        )
    path = Path(path_value)
    if not path.is_absolute():
        path = root / path
    if not path.is_file():
        raise FileNotFoundError(f"keyframe sidecar does not exist: {path}")
    if raw.get("column") is not None:
        raise ValueError(
            "anchor_sampling.keyframes.column is only valid for source='parquet_column'"
        )
    return KeyframeSamplingSpec(
        source=source,
        column=None,
        path=path,
        context_offsets=offsets,
        content_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )


def load_keyframe_jsonl(
    spec: KeyframeSamplingSpec,
    *,
    episode_lengths: Mapping[int, int],
    strict_unknown_episodes: bool = True,
) -> dict[int, np.ndarray]:
    """Load ``{episode_index: sorted unique frame indices}`` from a sidecar."""
    if spec.source != "jsonl" or spec.path is None:
        return {}
    collected: dict[int, list[int]] = {}
    with spec.path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                episode = int(row["episode_index"])
                frames = row["frame_indices"]
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"{spec.path}:{line_number}: expected episode_index and frame_indices"
                ) from exc
            if episode not in episode_lengths and strict_unknown_episodes:
                raise ValueError(
                    f"{spec.path}:{line_number}: unknown episode_index {episode}"
                )
            if episode not in episode_lengths:
                continue
            if isinstance(frames, (str, bytes)) or not isinstance(frames, Sequence):
                raise TypeError(
                    f"{spec.path}:{line_number}: frame_indices must be a sequence"
                )
            length = int(episode_lengths[episode])
            parsed = [int(frame) for frame in frames]
            invalid = [frame for frame in parsed if frame < 0 or frame >= length]
            if invalid:
                raise ValueError(
                    f"{spec.path}:{line_number}: episode {episode} frame indices outside "
                    f"[0, {length}): {invalid[:8]}"
                )
            collected.setdefault(episode, []).extend(parsed)
    return {
        episode: np.asarray(sorted(set(frames)), dtype=np.int64)
        for episode, frames in collected.items()
    }


def episode_sample_anchors(
    length: int,
    spec: AnchorSamplingSpec,
    *,
    keyframe_indices: np.ndarray | Sequence[int] = (),
    quality_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, int]]:
    """Return sorted unique local-frame anchors and audit counters."""
    if length < 0:
        raise ValueError(f"episode length must be non-negative, got {length}")
    regular = np.arange(spec.offset, length, spec.stride_frames, dtype=np.int64)
    raw_keyframes = np.asarray(keyframe_indices, dtype=np.int64).reshape(-1)
    if raw_keyframes.size and (
        int(raw_keyframes.min()) < 0 or int(raw_keyframes.max()) >= length
    ):
        raise ValueError(f"keyframe indices must lie inside episode length {length}")

    expanded = np.empty(0, dtype=np.int64)
    if raw_keyframes.size:
        if spec.keyframes is None:
            raise ValueError("keyframe_indices were provided without a keyframe spec")
        offsets = np.asarray(spec.keyframes.context_offsets, dtype=np.int64)
        expanded = (raw_keyframes[:, None] + offsets[None, :]).reshape(-1)
        # Context is clipped by dropping out-of-range positions, never by mapping
        # several offsets onto frame 0/the last frame.
        expanded = np.unique(expanded[(expanded >= 0) & (expanded < length)])

    combined = np.union1d(regular, expanded)
    before_quality = combined.size
    if quality_mask is not None:
        quality = np.asarray(quality_mask, dtype=np.bool_)
        if quality.shape != (length,):
            raise ValueError(
                f"quality mask must have shape ({length},), got {quality.shape}"
            )
        combined = combined[quality[combined]]
    return combined.astype(np.int64, copy=False), {
        "regular": int(regular.size),
        "keyframe_context": int(expanded.size),
        "overlap": int(regular.size + expanded.size - before_quality),
        "quality_rejected": int(before_quality - combined.size),
        "final": int(combined.size),
    }
