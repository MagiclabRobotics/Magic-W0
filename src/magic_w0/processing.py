"""Magic-W0 inference components; tensor names and computations are preserved."""

from __future__ import annotations
from typing import Any
import torch
from torch.nn import functional as F
from .config import HISTORY_PLACEHOLDER_TOKEN, _IMAGE_LABELS


def _render_robot_prompt_metadata(
    template: str,
    state_dim_mask: torch.Tensor | None,
    action_dim_mask: torch.Tensor | None = None,
) -> str:
    """Render the configured robot metadata line from the effective dimension mask.

    The 34-D target layout reserves slots ``0..6`` and ``8..14`` for the left
    and right joint groups.  Legacy 32-D sources leave slots 6 and 14 absent
    after remapping, which distinguishes 6-DoF arms from native 7-DoF data
    without adding source-specific metadata to the prompt configuration.
    """
    configured = str(template).strip()
    if not configured:
        return ""

    def reduce_mask(raw_mask: torch.Tensor | None) -> torch.Tensor | None:
        if raw_mask is None:
            return None
        mask = raw_mask if torch.is_tensor(raw_mask) else torch.as_tensor(raw_mask)
        if mask.ndim == 0 or mask.shape[-1] <= 14:
            return None
        return mask.bool().reshape(-1, mask.shape[-1]).any(dim=0)

    state_mask = reduce_mask(state_dim_mask)
    action_mask = reduce_mask(action_dim_mask)
    if state_mask is None:
        mask = action_mask
    elif action_mask is None or state_mask.shape != action_mask.shape:
        mask = state_mask
    else:
        mask = state_mask | action_mask
    if mask is None:
        return ""
    joint_slots = (*range(0, 7), *range(8, 15))
    has_joint = bool(mask[list(joint_slots)].any())
    action_space = "joint+EEF" if has_joint else "EEF"
    joint_dimensions = ""
    if has_joint:
        dof = 7 if bool(mask[6]) and bool(mask[14]) else 6
        joint_dimensions = f" Joint dimensions: {dof}-DoF per arm"
    return configured.replace("{action_space}", action_space).replace(
        "{joint_dimensions}", joint_dimensions
    )


def _prepare_image(image: torch.Tensor, output_size: int = 256) -> torch.Tensor:
    """Fit a CHW/TCHW image inside a square and center-pad with zeros."""
    if not isinstance(image, torch.Tensor) or image.ndim not in (3, 4):
        raise ValueError(
            f"Magic_W0 images must be CHW or TCHW tensors, got {type(image).__name__} with shape {getattr(image, 'shape', None)}"
        )
    temporal = image.ndim == 4
    value = image if temporal else image.unsqueeze(0)
    if value.shape[1] not in (1, 3, 4):
        raise ValueError(
            f"Magic_W0 images must use 1, 3, or 4 channels in dimension 1, got shape {tuple(image.shape)}"
        )
    height, width = (int(value.shape[-2]), int(value.shape[-1]))
    if height <= 0 or width <= 0:
        raise ValueError(
            f"Magic_W0 images must have positive spatial dimensions, got {(height, width)}"
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
            f"resize produced an image larger than the output canvas: {(resized_height, resized_width)}"
        )
    padded = F.pad(
        resized,
        (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2),
        mode="constant",
        value=0,
    )
    return padded if temporal else padded[0]


def _maybe_to(value, device):
    if device is None or not isinstance(value, torch.Tensor):
        return value
    return value.to(device, non_blocking=True)


def image_prompt_label(camera_key: str) -> str:
    """Render the current-view label as ``<label> image: ``."""
    tail = str(camera_key).split(".")[-1]
    return _IMAGE_LABELS.get(tail, tail.replace("_", " ").capitalize())


def history_block_text(valid_count: int, tokens_per_frame: int) -> str:
    """``History images: `` then one placeholder run per past frame."""
    if valid_count < 0 or tokens_per_frame < 1:
        raise ValueError(
            f"need valid_count >= 0 and tokens_per_frame >= 1, got {valid_count}, {tokens_per_frame}"
        )
    run = HISTORY_PLACEHOLDER_TOKEN * int(tokens_per_frame)
    return "History images: " + "\n".join([run] * int(valid_count)) + "\n"


def pool_frame_features(
    features: Any, grid_thw: torch.Tensor, *, merge: int, pool_size: int
) -> torch.Tensor:
    """Merged vision grid -> ``pool x pool`` by adaptive average pooling.

    On the 8x8 grid of a 256 canvas with pool 4 this is exactly 2x2 mean
    pooling.  Returns every frame's tokens concatenated in frame order.
    """
    pooled: list[torch.Tensor] = []
    for feature, (t, h, w) in zip(features, grid_thw.tolist(), strict=True):
        if int(t) != 1:
            raise ValueError(f"history frames must be single images, got t={t}")
        rows, cols = (int(h) // int(merge), int(w) // int(merge))
        if rows * cols != int(feature.shape[0]):
            raise ValueError(
                f"vision grid {rows}x{cols} does not match {int(feature.shape[0])} tokens"
            )
        if int(pool_size) > min(rows, cols):
            raise ValueError(f"pool_size={pool_size} exceeds the {rows}x{cols} grid")
        grid = feature.reshape(rows, cols, -1).permute(2, 0, 1).unsqueeze(0)
        grid = torch.nn.functional.adaptive_avg_pool2d(
            grid.float(), (int(pool_size), int(pool_size))
        )
        pooled.append(
            grid.squeeze(0)
            .permute(1, 2, 0)
            .reshape(int(pool_size) ** 2, -1)
            .to(feature.dtype)
        )
    return torch.cat(pooled, dim=0)
