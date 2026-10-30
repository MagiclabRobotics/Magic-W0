"""Shared tensor normalization for MagicW0 data processors."""

from __future__ import annotations

from collections.abc import Mapping

import torch


def normalize_tensor(
    value: torch.Tensor,
    stats: Mapping[str, object],
    mode: str,
) -> torch.Tensor:
    """Normalize one sample, accepting flat ``[D]`` or horizon ``[H, D]`` stats."""
    value = value.float()
    if mode == "none":
        return value
    if mode == "mean_std":
        mean = torch.as_tensor(stats["mean"], dtype=value.dtype)
        std = torch.as_tensor(stats["std"], dtype=value.dtype)
        _validate_trailing_shape(mean, value)
        _validate_trailing_shape(std, value)
        return (value - mean) / std.clamp_min(1e-6)
    if mode != "quantile":
        raise ValueError(f"unsupported normalization mode {mode!r}")
    low = torch.as_tensor(
        stats["q01"] if "q01" in stats else stats["min"], dtype=value.dtype
    )
    high = torch.as_tensor(
        stats["q99"] if "q99" in stats else stats["max"], dtype=value.dtype
    )
    _validate_trailing_shape(low, value)
    _validate_trailing_shape(high, value)
    scale = (high - low).clamp_min(1e-6)
    return ((value - low) / scale * 2.0 - 1.0).clamp(-1.0, 1.0)


def _validate_trailing_shape(stat: torch.Tensor, value: torch.Tensor) -> None:
    """Reject a horizon table built for a different action shape."""
    if stat.ndim <= 1:
        return
    if stat.ndim > value.ndim or tuple(value.shape[-stat.ndim :]) != tuple(stat.shape):
        raise ValueError(
            f"normalization band shape {tuple(stat.shape)} does not match value "
            f"shape {tuple(value.shape)}; a per-horizon band must be built for "
            "the same chunk size the reader emits"
        )
