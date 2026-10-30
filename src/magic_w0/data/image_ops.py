"""Shared image tensor preprocessing helpers."""

from __future__ import annotations

from typing import Any

import torch
from torch.nn import functional as F


def resize_with_pad_square(image: Any, *, size: int = 256) -> torch.Tensor:
    """Aspect-ratio preserving resize into a square black canvas.

    Accepts CHW/HWC images or batches with arbitrary leading dimensions. The
    returned tensor is float32, channel-first, RGB, in [0, 1], and has shape
    ``(..., 3, size, size)``.
    """

    if size <= 0:
        raise ValueError("size must be positive")
    tensor = torch.as_tensor(image)
    if tensor.ndim < 3:
        raise ValueError(
            f"expected image tensor with at least 3 dims, got {tuple(tensor.shape)}"
        )

    if tensor.shape[-3] in {1, 3, 4}:
        leading_shape = tensor.shape[:-3]
        channels, height, width = tensor.shape[-3:]
        chw = tensor.reshape(-1, channels, height, width)
    elif tensor.shape[-1] in {1, 3, 4}:
        leading_shape = tensor.shape[:-3]
        height, width, channels = tensor.shape[-3:]
        chw = tensor.reshape(-1, height, width, channels).permute(0, 3, 1, 2)
    else:
        raise ValueError(
            "expected channel-first or channel-last RGB-like image tensor, got "
            f"{tuple(tensor.shape)}"
        )

    if height <= 0 or width <= 0:
        raise ValueError(f"image height/width must be positive, got {(height, width)}")

    chw = chw.to(dtype=torch.float32)
    if chw.numel() and float(chw.detach().amax().cpu().item()) > 1.0:
        chw = chw / 255.0
    if channels == 1:
        chw = chw.repeat(1, 3, 1, 1)
    elif channels == 4:
        chw = chw[:, :3]
    elif channels != 3:
        raise ValueError(f"expected 1, 3, or 4 image channels, got {int(channels)}")

    scale = min(float(size) / float(height), float(size) / float(width))
    resized_h = max(1, int(round(float(height) * scale)))
    resized_w = max(1, int(round(float(width) * scale)))
    resized = F.interpolate(
        chw,
        size=(resized_h, resized_w),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )

    pad_h = size - resized_h
    pad_w = size - resized_w
    if pad_h < 0 or pad_w < 0:
        raise RuntimeError(
            f"letterbox resize overshot target: resized={(resized_h, resized_w)}, size={size}"
        )
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    padded = F.pad(resized, (left, right, top, bottom), value=0.0)
    return padded.reshape(*leading_shape, 3, size, size).clamp(0.0, 1.0)


__all__ = ["resize_with_pad_square"]
