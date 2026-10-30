"""RoboDojo preprocessing owned by ``magicw0_base``."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from magic_w0.data.embodiment import UNKNOWN_EMBODIMENT
from magic_w0.models._normalization import normalize_tensor as _normalize

from .configuration_magicw0_base import MagicW0BaseConfig


STATE = "magic_w0.state"
ACTIONS = "magic_w0.actions"
ACTION_IS_PAD = "magic_w0.action_is_pad"
ACTION_DIM_MASK = "magic_w0.action_dim_mask"
CAMERA_VALID = "magic_w0.camera_valid"
TASK = "magic_w0.task"
SUBTASK = "magic_w0.subtask"
EMBODIMENT = "magic_w0.embodiment"
STATE_DIM_MASK = "magic_w0.state_dim_mask"
TEXT = "magic_w0.vlm.text"
VQA_IMAGES = "magic_w0.vlm.images"
VQA_N_IMAGES = "magic_w0.vlm.n_images"
VQA_CHAT = "magic_w0.vlm.chat"

# Keep the source aspect ratio by fitting the image inside the configured
# square canvas, then center-padding the remaining dimension with zeros.
# ``MagicW0BaseConfig.image_size`` is the single source of truth.


class MagicW0BaseDataProcessor:
    """Normalize robotics samples and pass VLM co-training samples through."""

    def __init__(self, config: MagicW0BaseConfig) -> None:
        self.config = config
        self.vlm_modalities = frozenset(config.cotrain_vlm_modalities)

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, Any],
        *,
        model_config: MagicW0BaseConfig,
    ) -> "MagicW0BaseDataProcessor":
        if payload:
            raise ValueError(
                f"MagicW0BaseDataProcessor: unknown config keys {sorted(payload)}"
            )
        return cls(model_config)

    def transform_dataset(self, dataset, source_config):
        if source_config.modality in self.vlm_modalities:
            return _MagicW0BaseVlmDataset(dataset)
        if source_config.modality != self.config.cotrain_native_modality:
            raise ValueError(
                "magicw0_base source modality must be the native robotics modality or "
                f"one of {sorted(self.vlm_modalities)}; got {source_config.modality!r}"
            )
        return _MagicW0BaseDataset(dataset, self.config)


class _MagicW0BaseVlmDataset(Dataset):
    """Keep VLM JSONL text or image-chat records in a collate-safe schema."""

    def __init__(self, dataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.dataset[index]
        if "vqa_images" in item:
            return {
                # VlmJsonlDataset applies resize+pad while decoding the original
                # bytes, before this model-owned wrapper sees the fixed stack.
                VQA_IMAGES: item["vqa_images"],
                VQA_N_IMAGES: item["vqa_n_images"],
                VQA_CHAT: item["vqa_chat"],
            }
        text = item.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(
                "magicw0_base VLM source must yield a non-empty 'text' string"
            )
        return {TEXT: text}


class _MagicW0BaseDataset(Dataset):
    def __init__(self, dataset, config: MagicW0BaseConfig) -> None:
        required = ("meta_stats", "camera_keys", "state_key", "action_key")
        missing = [name for name in required if not hasattr(dataset, name)]
        if missing:
            raise TypeError(
                f"magicw0_base source {type(dataset).__name__} is missing {missing}"
            )
        available_cameras = set(config.camera_keys) & set(dataset.camera_keys)
        if not available_cameras:
            raise ValueError(
                "dataset does not contain any configured camera; "
                f"configured={sorted(config.camera_keys)}, available={sorted(dataset.camera_keys)}"
            )
        if dataset.state_key != config.state_key:
            raise ValueError(
                f"configured state_key={config.state_key!r}, dataset uses {dataset.state_key!r}"
            )
        if dataset.action_key != config.action_key:
            raise ValueError(
                f"configured action_key={config.action_key!r}, dataset uses {dataset.action_key!r}"
            )

        self.dataset = dataset
        self.config = config
        self.meta_stats = dataset.meta_stats

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.dataset[index]
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
        state_dim_mask = item.get(f"{self.config.state_key}_dim_mask")
        if state_dim_mask is None:
            state_dim_mask = torch.ones_like(state, dtype=torch.bool)
        else:
            state_dim_mask = state_dim_mask.bool()
        action_dim_mask = item.get(f"{self.config.action_key}_dim_mask")
        if action_dim_mask is None:
            action_dim_mask = torch.ones_like(actions, dtype=torch.bool)
        else:
            action_dim_mask = action_dim_mask.bool()
        if state.shape[-1] != self.config.state_dim:
            raise ValueError(
                f"state dim {state.shape[-1]} does not match configured {self.config.state_dim}"
            )
        if actions.shape[-1] != self.config.action_dim:
            raise ValueError(
                f"action dim {actions.shape[-1]} does not match configured {self.config.action_dim}"
            )
        if state_dim_mask.shape != state.shape:
            raise ValueError(
                f"state dimension mask shape {tuple(state_dim_mask.shape)} does not match "
                f"state shape {tuple(state.shape)}"
            )
        if action_dim_mask.shape != actions.shape:
            raise ValueError(
                f"action dimension mask shape {tuple(action_dim_mask.shape)} does not match "
                f"action shape {tuple(actions.shape)}"
            )

        output: dict[str, Any] = {
            STATE: state.float().masked_fill(~state_dim_mask, 0.0),
            # Emitted, not just consumed.  Zeroing a masked dimension is only
            # safe if something downstream still knows it was masked: the state
            # is quantile-normalized to [-1, 1], so 0.0 is the MIDDLE of the
            # range, not "absent".  Measured on RoboTwin2.0, the right end
            # effector is valid in 41.3% of frames -- meaning 0.0 there means
            # "median pose" on some frames and "unknown" on others, with nothing
            # in the input to tell them apart.
            STATE_DIM_MASK: state_dim_mask,
            ACTIONS: actions.float().masked_fill(~action_dim_mask, 0.0),
            ACTION_IS_PAD: item[f"{self.config.action_key}_is_pad"].bool(),
            ACTION_DIM_MASK: action_dim_mask,
            TASK: str(item["task"]),
            # ``LeRobotV21Dataset`` normalizes this at the source now -- a
            # source that asked for annotations emits the key on every frame --
            # so this default is the second layer, for a reader that does not.
            # A batch is collated field-by-field, so one sample missing a key
            # its neighbour has takes down the whole batch, and the empty string
            # is the "no claim" value the prompt builder drops, which is why it
            # is safe here while 0.0 is not safe for the state.
            SUBTASK: str(item.get("subtask", "")),
            # Absent for sources that predate the reader change and for the VQA
            # branch; the prompt builder drops the line rather than asserting a
            # robot, so an un-annotated source degrades to today's behaviour.
            EMBODIMENT: str(item.get("embodiment", UNKNOWN_EMBODIMENT)),
            # Kept separate from model inputs so diagnostic hooks can identify
            # an anomalous frame without depending on dataset internals.
            "__sample_metadata__": {
                "data_root": item.get("data_root", ""),
                "source_id": item.get("source_id", ""),
                "episode_index": item.get("episode_index", -1),
                "frame_index": item.get("frame_index", -1),
                "observation_frame_indices": item.get(
                    "observation_frame_indices", torch.empty(0, dtype=torch.int64)
                ),
                "history_frame_indices": item.get(
                    "history_frame_indices", torch.empty(0, dtype=torch.int64)
                ),
                "index": item.get("index", -1),
                "timestamp": item.get("timestamp", 0.0),
                "task": str(item["task"]),
                "action_representation": item.get("action_representation", "abs"),
                "normalization": self.config.normalization,
            },
        }
        reference_image = item[
            next(key for key in self.config.camera_keys if key in item)
        ]
        reference_image = _prepare_image(reference_image, self.config.image_size)
        camera_valid = []
        for key in self.config.camera_keys:
            is_valid = key in item
            camera_valid.append(is_valid)
            output[key] = (
                _prepare_image(item[key], self.config.image_size)
                if is_valid
                else torch.zeros_like(reference_image)
            )
        output[CAMERA_VALID] = torch.tensor(camera_valid, dtype=torch.bool)
        return output


def _prepare_image(image: torch.Tensor, output_size: int = 256) -> torch.Tensor:
    """Fit an image inside the configured square and center zero-pad it.

    Supports channels-first ``CHW``/``TCHW`` tensors and preserves the input
    dtype. Integer images are rounded after interpolation; floating-point
    images are only resized (no implicit 0..255 rescaling).
    """
    if not isinstance(image, torch.Tensor) or image.ndim not in (3, 4):
        raise ValueError(
            "magicw0_base images must be CHW or TCHW tensors, "
            f"got {type(image).__name__} with shape {getattr(image, 'shape', None)}"
        )
    temporal = image.ndim == 4
    value = image if temporal else image.unsqueeze(0)
    if value.shape[1] not in (1, 3, 4):
        raise ValueError(
            "magicw0_base images must use 1, 3, or 4 channels in dimension 1, "
            f"got shape {tuple(image.shape)}"
        )
    height, width = int(value.shape[-2]), int(value.shape[-1])
    if height <= 0 or width <= 0:
        raise ValueError(
            "magicw0_base images must have positive spatial dimensions, "
            f"got {(height, width)}"
        )

    if output_size <= 0:
        raise ValueError(f"output_size must be positive, got {output_size}")
    scale = min(output_size / height, output_size / width)
    resized_height = max(1, int(height * scale))
    resized_width = max(1, int(width * scale))
    original_dtype = value.dtype
    resized = F.interpolate(
        value.to(dtype=torch.float32),
        size=(resized_height, resized_width),
        mode="bilinear",
        align_corners=False,
    )
    if original_dtype.is_floating_point:
        resized = resized.to(dtype=original_dtype)
    else:
        info = torch.iinfo(original_dtype)
        resized = resized.round().clamp(info.min, info.max).to(dtype=original_dtype)

    pad_h = output_size - resized_height
    pad_w = output_size - resized_width
    if pad_h < 0 or pad_w < 0:
        raise RuntimeError(
            "resize produced an image larger than the output canvas: "
            f"{(resized_height, resized_width)}"
        )
    padded = F.pad(
        resized,
        (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2),
        mode="constant",
        value=0,
    )
    return padded if temporal else padded[0]
