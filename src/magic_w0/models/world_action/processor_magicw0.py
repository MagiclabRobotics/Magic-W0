"""Multi-source robotics and VLM preprocessing for ``magicw0``."""

from __future__ import annotations

from bisect import bisect_right
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from magic_w0.data.embodiment import UNKNOWN_EMBODIMENT
from magic_w0.models.action_backbone.processor_magicw0_base import (
    ACTION_DIM_MASK,
    ACTION_IS_PAD,
    ACTIONS,
    CAMERA_VALID,
    EMBODIMENT,
    STATE,
    STATE_DIM_MASK,
    TASK,
    MagicW0BaseDataProcessor,
    _normalize,
    _prepare_image,
)

from .configuration_magicw0 import MagicW0TrainingConfig
from .history_frames import (
    HISTORY_FRAMES,
    HISTORY_LEN,
    HISTORY_MODE,
    HISTORY_MODE_EXEMPT,
    HISTORY_MODE_FULL,
    HISTORY_MODE_KEEP,
    HISTORY_MODE_PARTIAL,
    history_frame_stride,
    past_frame_indices,
)

FUTURE_2D_IMAGES = "magic_w0.future_2d_images"
FUTURE_2D_PIXEL_VALUES = "magic_w0.future_2d_pixel_values"
VIEW_VALID_MASK = "magic_w0.view_valid_mask"
FUTURE_3D_IMAGES = "magic_w0.future_3d_images"
WORLD_3D_CURRENT_IMAGES = "magic_w0.world_3d_current_images"
WORLD_3D_VALID_MASK = "magic_w0.world_3d_valid_mask"
GEOMETRY_3D_TARGET = "magic_w0.geometry_3d_target"
MOTION_3D_TARGET = "magic_w0.motion_3d_target"


class MagicW0TrainingDataProcessor(MagicW0BaseDataProcessor):
    """Preserve the Base data contract and add source-aware future frames."""

    def __init__(self, config: MagicW0TrainingConfig) -> None:
        super().__init__(config)
        self.config = config

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, Any],
        *,
        model_config: MagicW0TrainingConfig,
    ) -> MagicW0TrainingDataProcessor:
        if payload:
            raise ValueError(
                f"MagicW0TrainingDataProcessor: unknown keys {sorted(payload)}"
            )
        return cls(model_config)

    def transform_dataset(self, dataset, source_config):
        if source_config.modality in self.vlm_modalities:
            return super().transform_dataset(dataset, source_config)
        if source_config.modality != self.config.cotrain_native_modality:
            return super().transform_dataset(dataset, source_config)
        return _MagicW0TrainingDataset(
            dataset, self.config, source_root=getattr(source_config, "root", None)
        )


class _MagicW0TrainingDataset(Dataset):
    """Normalize one raw sample once, then attach current/future world views."""

    def __init__(
        self,
        dataset,
        config: MagicW0TrainingConfig,
        source_root: str | None = None,
    ) -> None:
        required = ("meta_stats", "camera_keys", "state_key", "action_key")
        missing = [name for name in required if not hasattr(dataset, name)]
        if missing:
            raise TypeError(
                f"magicw0 source {type(dataset).__name__} is missing {missing}"
            )
        if (
            dataset.state_key != config.state_key
            or dataset.action_key != config.action_key
        ):
            raise ValueError(
                "magicw0 state/action keys do not match the source: "
                f"model=({config.state_key!r}, {config.action_key!r}), "
                f"source=({dataset.state_key!r}, {dataset.action_key!r})"
            )
        available = set(dataset.camera_keys)
        if not (set(config.camera_keys) & available):
            raise ValueError(
                "dataset has no configured robotics camera: "
                f"configured={sorted(config.camera_keys)}, available={sorted(available)}"
            )

        self.dataset = dataset
        self.config = config
        self.meta_stats = dataset.meta_stats
        self.offset_2d = (
            _resolve_future_offset(dataset, config.gen_2d_future_frame_offset)
            if config.enable_2d_world_expert
            else 0
        )
        self.offset_3d = (
            _resolve_future_offset(dataset, config.gen_3d_future_frame_offset)
            if config.enable_3d_world_expert
            else 0
        )
        # Default to the original MagicW0 contract: remove anchors whose world
        # horizon would cross the episode boundary. A source can opt into the
        # terminal-hold contract with ``supervise_episode_end_padding=True``;
        # then action padding is supervised by the backend and world targets
        # reuse the episode's final frame in ``_load_future_item``.
        keep_episode_end = bool(
            getattr(dataset, "supervise_episode_end_padding", False)
        )
        future_offsets = []
        if config.enable_2d_world_expert:
            future_offsets.append(self.offset_2d)
        if config.enable_3d_world_expert:
            future_offsets.append(self.offset_3d)
        max_future_offset = max(future_offsets, default=0)
        raw_indices = (
            None
            if keep_episode_end or max_future_offset <= 0
            else _valid_future_indices(dataset, max_future_offset)
        )
        self.indices = (
            None if raw_indices is None else np.asarray(raw_indices, dtype=np.int64)
        )
        self.history_stride = None
        if config.history_enabled:
            if not callable(getattr(dataset, "get_episode_frames", None)):
                raise TypeError(
                    f"visual history needs {type(dataset).__name__}.get_episode_frames"
                )
            self.history_stride = history_frame_stride(
                config.history_interval,
                int(getattr(dataset, "action_upsample_rate", 1) or 1),
            )
            root = str(source_root or getattr(dataset, "root", ""))
            self.history_dropout_exempt = any(
                pattern in root for pattern in config.history_dropout_exempt_sources
            )

    def __len__(self) -> int:
        return len(self.indices) if self.indices is not None else len(self.dataset)

    def _load_futures(self, item: dict[str, Any], source_index: int) -> dict[int, Any]:
        """Fetch each distinct future offset once, for the union of its cameras."""
        requests: dict[int, tuple[tuple[str, ...], list[str]]] = {}
        if self.config.enable_2d_world_expert:
            requests[self.offset_2d] = (tuple(self.config.gen_2d_camera_keys), ["2D"])
        if self.config.enable_3d_world_expert:
            keys, names = requests.get(self.offset_3d, ((), []))
            requests[self.offset_3d] = (
                tuple(dict.fromkeys((*keys, self.config.gen_3d_camera_key))),
                [*names, "3D"],
            )
        return {
            offset: _load_future_item(
                item,
                self.dataset,
                self.config,
                source_index,
                future_offset=offset,
                camera_keys=keys,
                target_name="/".join(names),
            )
            for offset, (keys, names) in requests.items()
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        source_index = int(self.indices[index]) if self.indices is not None else index
        item = self.dataset[source_index]
        state = _normalize(
            item[self.config.state_key],
            self.meta_stats[self.config.state_key],
            self.config.normalization,
        )
        actions = _normalize(
            item[self.config.action_key],
            self.meta_stats[self.config.action_key],
            self.config.normalization,
        )
        state_mask = item.get(f"{self.config.state_key}_dim_mask")
        action_mask = item.get(f"{self.config.action_key}_dim_mask")
        state_mask = (
            torch.ones_like(state, dtype=torch.bool)
            if state_mask is None
            else state_mask.bool()
        )
        action_mask = (
            torch.ones_like(actions, dtype=torch.bool)
            if action_mask is None
            else action_mask.bool()
        )
        if state.shape[-1] != self.config.state_dim:
            raise ValueError(
                f"state dim {state.shape[-1]} does not match configured {self.config.state_dim}"
            )
        if actions.shape[-1] != self.config.action_dim:
            raise ValueError(
                f"action dim {actions.shape[-1]} does not match configured {self.config.action_dim}"
            )
        if state_mask.shape != state.shape or action_mask.shape != actions.shape:
            raise ValueError(
                "state/action dimension-mask shapes must match their tensors"
            )

        output: dict[str, Any] = {
            STATE: state.float().masked_fill(~state_mask, 0.0),
            STATE_DIM_MASK: state_mask,
            ACTIONS: actions.float().masked_fill(~action_mask, 0.0),
            ACTION_DIM_MASK: action_mask,
            ACTION_IS_PAD: item[f"{self.config.action_key}_is_pad"].bool(),
            TASK: str(item["task"]),
            EMBODIMENT: str(item.get("embodiment", UNKNOWN_EMBODIMENT)),
        }

        world_camera_keys: list[str] = []
        if self.config.enable_2d_world_expert:
            world_camera_keys.extend(self.config.gen_2d_camera_keys)
        if self.config.enable_3d_world_expert:
            world_camera_keys.append(self.config.gen_3d_camera_key)
        all_current_keys = tuple(
            dict.fromkeys((*self.config.camera_keys, *world_camera_keys))
        )
        reference_key = next((key for key in all_current_keys if key in item), None)
        if reference_key is None:
            raise ValueError("sample contains no configured current image")
        reference = _prepare_image(item[reference_key], self.config.image_size)
        padded = torch.zeros_like(reference)
        for key in all_current_keys:
            output[key] = (
                _prepare_image(item[key], self.config.image_size)
                if key in item
                else padded.clone()
            )
        output[CAMERA_VALID] = torch.tensor(
            [key in item for key in self.config.camera_keys], dtype=torch.bool
        )

        # ONE FETCH PER DISTINCT FUTURE OFFSET, NOT ONE PER BRANCH.  The 2D and
        # 3D experts are configured onto the same camera by default
        # (gen_2d_camera_keys and gen_3d_camera_key are both
        # observation.images.cam_high) and neither *_future_frame_offset is
        # normally set, so both resolve through _resolve_future_offset to the
        # same number -- two calls that decoded the identical frame twice.  Each
        # of those is a fresh av.open plus a seek on a network filesystem, in 256
        # dataloader workers per node.  Fetching the union of the camera keys per
        # offset keeps both branches' semantics and drops one decode of the five
        # a sample now performs.
        futures = self._load_futures(item, source_index)

        if self.config.enable_2d_world_expert:
            future = futures[self.offset_2d]
            images: list[torch.Tensor] = []
            valid: list[bool] = []
            for key in self.config.gen_2d_camera_keys:
                is_valid = key in item and key in future
                valid.append(is_valid)
                images.append(
                    _prepare_image(future[key], self.config.image_size)
                    if is_valid
                    else padded.clone()
                )
            while len(images) < self.config.gen_2d_max_views:
                images.append(padded.clone())
                valid.append(False)
            output[FUTURE_2D_IMAGES] = torch.stack(
                images[: self.config.gen_2d_max_views]
            )
            output[VIEW_VALID_MASK] = torch.tensor(
                valid[: self.config.gen_2d_max_views], dtype=torch.bool
            )

        if self.config.enable_3d_world_expert:
            future = futures[self.offset_3d]
            key = self.config.gen_3d_camera_key
            is_valid = key in item and key in future
            output[WORLD_3D_CURRENT_IMAGES] = output[key]
            output[FUTURE_3D_IMAGES] = (
                _prepare_image(future[key], self.config.image_size)
                if is_valid
                else padded.clone()
            )
            output[WORLD_3D_VALID_MASK] = torch.tensor(is_valid, dtype=torch.bool)
        if self.history_stride is not None:
            output.update(self._history_clip(item, padded))
        return output

    def _history_clip(self, item: dict[str, Any], padded: torch.Tensor) -> dict:
        """Strictly past frames, left-padded to ``[history_frames,C,H,W]``.

        A fixed first dimension keeps default collation working; the prefix
        builder renders only the last ``length`` frames.
        """
        key = self.config.history_camera_key
        capacity = int(self.config.history_frames)
        clip = padded.new_zeros((capacity, *padded.shape))
        length = 0
        if key in item:
            episode = _item_scalar(item.get("episode_index"))
            frame = _item_scalar(item.get("frame_index"))
            if episode is None or frame is None:
                raise ValueError("history clips require episode/frame indices")
            indices = past_frame_indices(frame, capacity, self.history_stride)
            mode, indices = self._drop_history(indices)
            if indices:
                decoded = self.dataset.get_episode_frames(episode, indices, (key,))[key]
                frames = _prepare_image(decoded, self.config.image_size)
                length = int(frames.shape[0])
                clip[capacity - length :] = frames
        else:
            mode = HISTORY_MODE_KEEP
        return {
            HISTORY_FRAMES: clip,
            HISTORY_LEN: torch.tensor(length, dtype=torch.int64),
            HISTORY_MODE: torch.tensor(mode, dtype=torch.int64),
        }

    def _drop_history(self, indices: list[int]) -> tuple[int, list[int]]:
        """Per-sample keep / partial / full history dropout, before decoding.

        Partial keeps a uniformly drawn 1..n-1 most-recent frames; full keeps
        none.  Both are exactly an earlier point of some episode, so the model
        never sees a layout it could not meet at inference.  Exempt sources and
        windows too short to shorten keep everything.
        """
        if self.history_dropout_exempt:
            return HISTORY_MODE_EXEMPT, indices
        config = self.config
        if not indices or config.history_keep_prob >= 1.0:
            return HISTORY_MODE_KEEP, indices
        draw = float(torch.rand(()))
        if draw < config.history_full_drop_prob:
            return HISTORY_MODE_FULL, []
        if draw < config.history_full_drop_prob + config.history_partial_drop_prob:
            if len(indices) < 2:
                return HISTORY_MODE_KEEP, indices
            keep = int(torch.randint(1, len(indices), ()))
            return HISTORY_MODE_PARTIAL, indices[-keep:]
        return HISTORY_MODE_KEEP, indices


def _resolve_future_offset(dataset, configured: int | None) -> int:
    if configured is not None:
        return int(configured)
    chunk_size = int(getattr(dataset, "chunk_size", 0))
    if chunk_size <= 0:
        raise ValueError("source-aware world horizon requires dataset.chunk_size")
    rate = int(getattr(dataset, "action_upsample_rate", 1))
    if rate <= 0:
        raise ValueError("action_upsample_rate must be positive")
    return (chunk_size + rate - 1) // rate


def _item_scalar(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return int(value.detach().cpu().reshape(()).item())
    return int(value)


def _load_future_item(
    item: dict[str, Any],
    dataset,
    config: MagicW0TrainingConfig,
    source_index: int,
    *,
    future_offset: int,
    camera_keys: tuple[str, ...],
    target_name: str,
) -> dict[str, Any]:
    data_format = config.gen_world_data_format
    if data_format == "auto":
        data_format = (
            "lerobot_v21"
            if hasattr(dataset, "_cum")
            else "lerobot_v30"
            if hasattr(dataset, "_ep_from") and hasattr(dataset, "_ep_to")
            else "global_index"
        )
    if data_format in {"lerobot_v21", "lerobot_v30"}:
        loader = getattr(dataset, "get_episode_frame_images", None)
        if not callable(loader):
            raise TypeError(
                f"{data_format} source must expose get_episode_frame_images"
            )
        episode = _item_scalar(item.get("episode_index"))
        frame = _item_scalar(item.get("frame_index"))
        if episode is None or frame is None:
            raise ValueError(
                f"{data_format} future targets require episode/frame indices"
            )
        target_frame = frame + future_offset
        if bool(getattr(dataset, "supervise_episode_end_padding", False)):
            length_loader = getattr(dataset, "get_episode_length", None)
            if not callable(length_loader):
                raise TypeError(
                    f"{data_format} terminal world-target reuse requires "
                    "dataset.get_episode_length"
                )
            episode_length = int(length_loader(episode))
            if episode_length <= 0:
                raise ValueError(
                    f"{data_format} episode {episode} has non-positive length "
                    f"{episode_length}"
                )
            target_frame = min(target_frame, episode_length - 1)
        return loader(episode, target_frame, camera_keys)
    if data_format == "global_index":
        future = dataset[source_index + future_offset]
        if _item_scalar(item.get("episode_index")) != _item_scalar(
            future.get("episode_index")
        ):
            raise ValueError(f"future {target_name} target crosses an episode boundary")
        return future
    raise ValueError(f"unknown gen_world_data_format={config.gen_world_data_format!r}")


def _valid_future_indices(dataset, future_offset: int) -> list[int] | None:
    """Return source anchors whose future target stays inside the episode."""
    if hasattr(dataset, "_cum"):
        cum = [int(value) for value in dataset._cum]
        sample_indices = getattr(dataset, "_sample_indices", None)
        if sample_indices is not None:
            return [
                dataset_index
                for dataset_index, raw_index in enumerate(sample_indices)
                if int(raw_index) + future_offset
                < cum[bisect_right(cum, int(raw_index))]
            ]
        return [
            index
            for start, end in zip(cum[:-1], cum[1:], strict=True)
            for index in range(start, max(start, end - future_offset))
        ]
    if hasattr(dataset, "_ep_from") and hasattr(dataset, "_ep_to"):
        starts = [int(value) for value in dataset._ep_from]
        ends = [int(value) for value in dataset._ep_to]
        return [
            index
            for start, end in zip(starts, ends, strict=True)
            for index in range(start, max(start, end - future_offset))
        ]
    return None
