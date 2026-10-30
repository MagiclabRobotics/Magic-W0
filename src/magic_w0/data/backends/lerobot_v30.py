# Copyright 2026 MagicVLA authors. (Apache-2.0.)
"""Standalone LeRobot **v3.0** dataset reader (no lerobot import).

Reads the v3 layout directly -- aggregated parquet under
``data/chunk-*/file-*.parquet`` (multiple episodes per file, rows addressed by
the global frame ``index``), per-camera concatenated mp4 under
``videos/<cam>/chunk-*/file-*.mp4`` (episodes located by
``from_timestamp``/``to_timestamp``), episode metadata under
``meta/episodes/chunk-*/file-*.parquet``, and dataset-level ``meta/stats.json``.

Yields the SAME sample schema as :class:`data.backends.lerobot_v21.LeRobotV21Dataset`
(and as a real ``LeRobotDataset`` with an action-chunk ``delta_timestamps``):
uint8 CHW camera frames, float32 state, a ``[chunk, dim]`` action stack with an
``<action_key>_is_pad`` mask, and the ``task`` string -- so v2.1 / v3.0 sources
interleave transparently with no lerobot dependency in the data layer.

Only pyarrow + torchcodec are required (both ship in the training image).
"""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from magic_w0.data.lerobot_history import (
    episode_observation_indices,
    validate_observation_sampling,
)
from magic_w0.data.video import make_video_decoder

_V30_INFO = "meta/info.json"
_V30_EPISODES_GLOB = "meta/episodes/chunk-*/file-*.parquet"


def _readonly_metadata(values, *, dtype: np.dtype) -> np.ndarray:
    """Return a contiguous, immutable metadata array safe to share after fork."""

    array = np.ascontiguousarray(values, dtype=dtype)
    array.setflags(write=False)
    return array


def _encode_episode_labels(
    values: list[str],
) -> tuple[np.ndarray, tuple[str, ...]]:
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


def load_v30_stats(root: str | Path) -> dict[str, dict[str, np.ndarray]]:
    """``meta/stats.json`` -> ``{feature: {stat: np.ndarray}}`` (the same shape
    as ``LeRobotDatasetMetadata.stats``, the normalizer input)."""
    raw = json.loads((Path(root) / "meta/stats.json").read_text())
    return {
        key: {stat: np.asarray(val, dtype=np.float32) for stat, val in stats.items()}
        for key, stats in raw.items()
    }


class LeRobotV30Dataset(Dataset):
    """Frame-indexed reader over one v3.0 dataset.

    Args:
        root: dataset directory (contains ``data/``, ``videos/``, ``meta/``).
        chunk_size: action-horizon length; actions ``[t, t+chunk)`` are stacked
            within the episode and right-padded with the last row (mask in
            ``<action_key>_is_pad``), mirroring lerobot ``delta_timestamps``.
        action_key: which parquet column is the training action; ``None`` picks
            ``real_action`` when present, else ``action``.
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
        decoder_cache: int = 8,
        obs_steps: int = 1,
        obs_interval_seconds: float = 0.1,
        video_backend: str = "auto",
        supervise_episode_end_padding: bool = False,
    ):
        self.root = Path(root)
        info = json.loads((self.root / _V30_INFO).read_text())
        if not str(info.get("codebase_version", "")).startswith("v3"):
            raise ValueError(
                f"{root}: codebase_version={info.get('codebase_version')!r}, expected v3.x"
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
        self.features: dict = info[
            "features"
        ]  # lerobot-format feature dict (for policy-feature mapping)
        self.camera_keys = [
            k for k, v in self.features.items() if v.get("dtype") == "video"
        ]
        if action_key is None:
            action_key = "real_action" if "real_action" in self.features else "action"
        self.action_key = action_key
        self.state_key = (
            "observation.qpos"
            if "observation.qpos" in self.features
            else "observation.state"
        )

        # ---- episode metadata (aggregated parquet, one row per episode) ------
        import pyarrow as pa
        import pyarrow.parquet as pq

        cols = [
            "episode_index",
            "tasks",
            "length",
            "dataset_from_index",
            "dataset_to_index",
            "data/chunk_index",
            "data/file_index",
        ]
        for cam in self.camera_keys:
            cols += [
                f"videos/{cam}/chunk_index",
                f"videos/{cam}/file_index",
                f"videos/{cam}/from_timestamp",
            ]
        tables = [
            pq.ParquetFile(p).read(
                columns=cols
            )  # not read_table; see lerobot_v21._episode_table
            for p in sorted(self.root.glob(_V30_EPISODES_GLOB))
        ]
        if not tables:
            raise FileNotFoundError(f"no episode metadata under {root}/meta/episodes")
        col = {
            c: np.concatenate([t[c].to_numpy(zero_copy_only=False) for t in tables])
            for c in cols
            if c != "tasks"
        }
        tasks = sum([t["tasks"].to_pylist() for t in tables], [])
        order = np.argsort(col["dataset_from_index"])
        self._ep_from = _readonly_metadata(
            col["dataset_from_index"][order], dtype=np.int64
        )
        self._ep_to = _readonly_metadata(col["dataset_to_index"][order], dtype=np.int64)
        self._ep_index = _readonly_metadata(col["episode_index"][order], dtype=np.int64)
        lookup_order = np.argsort(self._ep_index)
        self._ep_lookup_index = _readonly_metadata(
            self._ep_index[lookup_order], dtype=np.int64
        )
        self._ep_lookup_pos = _readonly_metadata(lookup_order, dtype=np.int64)
        self._ep_data_file = _readonly_metadata(
            np.column_stack(
                (
                    col["data/chunk_index"][order],
                    col["data/file_index"][order],
                )
            ),
            dtype=np.int64,
        )
        task_labels = [
            (tasks[int(source_position)] or ["unknown"])[0] for source_position in order
        ]
        self._ep_task_ids, self._task_table = _encode_episode_labels(task_labels)
        self._ep_video = {
            cam: {
                "file": _readonly_metadata(
                    np.column_stack(
                        (
                            col[f"videos/{cam}/chunk_index"][order],
                            col[f"videos/{cam}/file_index"][order],
                        )
                    ),
                    dtype=np.int64,
                ),
                "from_ts": _readonly_metadata(
                    col[f"videos/{cam}/from_timestamp"][order],
                    dtype=np.float64,
                ),
            }
            for cam in self.camera_keys
        }
        # global frame offset of each data file = min dataset_from_index over its episodes
        self._file_offset: dict[tuple[int, int], int] = {}
        for pos, file_row in enumerate(self._ep_data_file):
            key = (int(file_row[0]), int(file_row[1]))
            off = int(self._ep_from[pos])
            if key not in self._file_offset or off < self._file_offset[key]:
                self._file_offset[key] = off
        # Release the Arrow tables and per-episode Python task graph before
        # DataLoader workers fork. Runtime metadata now lives in read-only arrays
        # plus one small table per distinct task.
        del tables, col, tasks, task_labels, order, lookup_order
        pa.default_memory_pool().release_unused()

        self.meta_stats = load_v30_stats(self.root)
        # per-worker lazy caches
        self._table_cache: OrderedDict[tuple[int, int], dict[str, np.ndarray]] = (
            OrderedDict()
        )
        self._decoders: OrderedDict[str, object] = OrderedDict()
        self._decoder_cache = int(decoder_cache)

    def __len__(self) -> int:
        return int(self._ep_to[-1])

    def sample_episode_ranges(self) -> list[tuple[int, int, int]]:
        """``(episode_index, start, stop)`` spans of this reader's index space.

        Consumed by the factory's episode-level train/eval split. v3.0 metadata
        carries the spans directly as ``dataset_from_index``/``dataset_to_index``
        (already sorted by ``from`` in ``__init__``), so there is nothing to
        reconstruct.
        """
        return [
            (int(episode), int(start), int(stop))
            for episode, start, stop in zip(self._ep_index, self._ep_from, self._ep_to)
            if stop > start
        ]

    @property
    def num_frames(self) -> int:
        return len(self)

    # ---- file-level caches ----------------------------------------------------
    def _data_table(self, key: tuple[int, int]) -> dict[str, np.ndarray]:
        cached = self._table_cache.get(key)
        if cached is not None:
            return cached
        import pyarrow.parquet as pq

        rel = self._data_path.format(chunk_index=key[0], file_index=key[1])
        # ParquetFile rather than read_table: read_table leaks ~0.8-2.3 KB per
        # call through its ParquetDataset/filesystem-discovery path, and this is
        # the per-episode read.  See lerobot_v21._episode_table for the
        # measurement and for what it cost the 48-GPU pretrain.
        tbl = pq.ParquetFile(self.root / rel).read(
            columns=[self.state_key, self.action_key]
        )
        arrs = {
            self.state_key: np.stack(
                tbl[self.state_key].to_numpy(zero_copy_only=False)
            ).astype(np.float32),
            self.action_key: np.stack(
                tbl[self.action_key].to_numpy(zero_copy_only=False)
            ).astype(np.float32),
        }
        self._table_cache[key] = arrs
        if len(self._table_cache) > 2:  # v3 data files are ~100MB; keep the cache tight
            self._table_cache.popitem(last=False)
        return arrs

    def _decode_frames(
        self, cam: str, pos: int, frames: tuple[int, ...]
    ) -> torch.Tensor:
        vid = self._ep_video[cam]
        file_row = vid["file"][pos]
        chunk_idx, file_idx = int(file_row[0]), int(file_row[1])
        rel = self._video_path.format(
            video_key=cam, chunk_index=chunk_idx, file_index=file_idx
        )
        path = str(self.root / rel)
        dec = self._decoders.get(path)
        if dec is None:
            dec = make_video_decoder(path, backend=self.video_backend, fps=self.fps)
            self._decoders[path] = dec
            if len(self._decoders) > self._decoder_cache:
                self._decoders.popitem(last=False)
        # Episode frame t lives at from_timestamp + t/fps; sample mid-frame for robustness.
        end = dec.metadata.duration_seconds - 0.5 / self.fps
        timestamps = [
            min(float(vid["from_ts"][pos]) + (t + 0.5) / self.fps, end) for t in frames
        ]
        return dec.get_frames_played_at(seconds=timestamps).data  # uint8 [T,C,H,W]

    def _episode_position(self, episode_index: int) -> int:
        episode = int(episode_index)
        lookup_index = getattr(self, "_ep_lookup_index", None)
        lookup_pos = getattr(self, "_ep_lookup_pos", None)
        if lookup_index is not None and lookup_pos is not None:
            lookup = int(np.searchsorted(lookup_index, episode, side="left"))
            if lookup >= len(lookup_index) or int(lookup_index[lookup]) != episode:
                raise KeyError(f"unknown v3.0 episode_index={episode}")
            return int(lookup_pos[lookup])

        # Compatibility for older pickles and tests constructed via __new__.
        by_episode = self._ep_pos_by_index
        if episode not in by_episode:
            raise KeyError(f"unknown v3.0 episode_index={episode}")
        return int(by_episode[episode])

    def _episode_task(self, position: int) -> str:
        task_ids = getattr(self, "_ep_task_ids", None)
        task_table = getattr(self, "_task_table", None)
        if task_ids is not None and task_table is not None:
            return task_table[int(task_ids[position])]
        tasks = self._ep_tasks[position]
        return tasks[0] if tasks else "unknown"

    def _data_file_key(self, position: int) -> tuple[int, int]:
        file_row = self._ep_data_file[position]
        return int(file_row[0]), int(file_row[1])

    def get_episode_frame_images(
        self,
        episode_index: int,
        frame_index: int,
        camera_keys: tuple[str, ...] | list[str],
    ) -> dict:
        """Decode one frame from a v3.0 episode while preserving the v3 metadata interface."""
        ep = int(episode_index)
        frame = int(frame_index)
        pos = self._episode_position(ep)
        length = int(self._ep_to[pos] - self._ep_from[pos])
        if frame < 0 or frame >= length:
            raise IndexError(
                f"future frame {frame} is outside v3.0 episode {ep} length {length}"
            )
        item: dict = {
            "episode_index": torch.tensor(ep, dtype=torch.int64),
            "frame_index": torch.tensor(frame, dtype=torch.int64),
            "timestamp": torch.tensor(frame / self.fps, dtype=torch.float32),
        }
        for cam in camera_keys:
            if cam in self.camera_keys:
                item[cam] = self._decode_frames(cam, pos, (frame,))[0]
        return item

    def get_episode_frames(
        self,
        episode_index: int,
        frame_indices: tuple[int, ...] | list[int],
        camera_keys: tuple[str, ...] | list[str],
    ) -> dict:
        """Decode several frames of one v3.0 episode, one decoder call per camera."""
        ep = int(episode_index)
        frames = tuple(int(frame) for frame in frame_indices)
        pos = self._episode_position(ep)
        length = int(self._ep_to[pos] - self._ep_from[pos])
        if not frames or min(frames) < 0 or max(frames) >= length:
            raise IndexError(
                f"history frames {frames} are outside v3.0 episode {ep} length {length}"
            )
        return {
            cam: self._decode_frames(cam, pos, frames)
            for cam in camera_keys
            if cam in self.camera_keys
        }

    def get_episode_length(self, episode_index: int) -> int:
        """Return the number of frames in one selected episode."""
        position = self._episode_position(int(episode_index))
        return int(self._ep_to[position] - self._ep_from[position])

    # ---- sample ----------------------------------------------------------------
    def __getitem__(self, index: int) -> dict:
        pos = int(np.searchsorted(self._ep_from, index, side="right") - 1)
        t = index - int(self._ep_from[pos])
        ep_len = int(self._ep_to[pos] - self._ep_from[pos])

        file_key = self._data_file_key(pos)
        table = self._data_table(file_key)
        row = index - self._file_offset[file_key]
        obs_indices = episode_observation_indices(t, self.observation_delta_indices)
        obs_rows = [row - t + obs_t for obs_t in obs_indices]
        n_real = min(self.chunk_size, ep_len - t)
        chunk = table[self.action_key][row : row + n_real]
        n_pad = self.chunk_size - n_real
        if n_pad > 0:
            chunk = np.concatenate(
                [chunk, np.repeat(chunk[-1:], n_pad, axis=0)], axis=0
            )
        is_pad = torch.zeros(self.chunk_size, dtype=torch.bool)
        if n_pad > 0:
            is_pad[-n_pad:] = not self.supervise_episode_end_padding

        item: dict = {
            self.state_key: torch.from_numpy(
                (
                    table[self.state_key][row]
                    if self.obs_steps == 1
                    else table[self.state_key][obs_rows]
                ).copy()
            ),
            self.action_key: torch.from_numpy(chunk.copy()),
            f"{self.action_key}_is_pad": is_pad,
            "task": self._episode_task(pos),
            "episode_index": torch.tensor(int(self._ep_index[pos]), dtype=torch.int64),
            "frame_index": torch.tensor(t, dtype=torch.int64),
            "index": torch.tensor(index, dtype=torch.int64),
            "timestamp": torch.tensor(t / self.fps, dtype=torch.float32),
            "data_root": str(getattr(self, "root", "")),
            "source_id": getattr(getattr(self, "root", None), "name", ""),
            "observation_frame_indices": torch.tensor(obs_indices, dtype=torch.int64),
            "history_frame_indices": torch.tensor(obs_indices[:-1], dtype=torch.int64),
            "action_representation": "abs",
        }
        for cam in self.camera_keys:
            frames = self._decode_frames(cam, pos, obs_indices)
            item[cam] = frames[0] if self.obs_steps == 1 else frames
        return item
