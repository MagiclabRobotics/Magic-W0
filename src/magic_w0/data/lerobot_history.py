# Copyright 2026 MagicVLA authors. (Apache-2.0.)
"""Shared temporal-observation helpers for standalone LeRobot readers."""

from __future__ import annotations


def validate_observation_sampling(
    *, obs_steps: int, obs_interval_seconds: float, fps: float
) -> tuple[int, tuple[int, ...]]:
    """Validate settings and convert a wall-clock interval to frame offsets."""
    if obs_steps < 1:
        raise ValueError(f"obs_steps must be >= 1, got {obs_steps}")
    if fps <= 0:
        raise ValueError(f"dataset fps must be > 0, got {fps}")
    if obs_steps > 1 and obs_interval_seconds <= 0:
        raise ValueError(
            "obs_interval_seconds must be > 0 when obs_steps > 1, "
            f"got {obs_interval_seconds}"
        )

    stride = max(1, round(float(obs_interval_seconds) * float(fps)))
    offsets = tuple((i - obs_steps + 1) * stride for i in range(obs_steps))
    return stride, offsets


def episode_observation_indices(
    frame_index: int, offsets: tuple[int, ...]
) -> tuple[int, ...]:
    """Apply relative offsets without ever crossing the episode boundary."""
    return tuple(max(0, frame_index + offset) for offset in offsets)
