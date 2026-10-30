"""Step-based scheduler construction utilities."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class SchedulerConfig:
    """Configuration for optional LR scheduling.

    ``warmup_ratio`` expresses warmup as a fraction of ``total_steps`` so that
    changing the run length does not silently change the warmup *fraction* --
    the failure that made runs 7345/7346 uncomparable, where a 2000-step warmup
    inherited from a 200k recipe held a 400-step probe at 10% of nominal LR for
    its entire life.  It takes precedence over ``warmup_steps`` when set.

    ``min_lr_ratio`` is the floor as a fraction of the peak LR, reached at
    ``total_steps``.  The linear schedule decays to exactly zero, which wastes
    the tail of a long run; a cosine to 0.1 keeps the last steps productive.
    """

    name: str = "constant"
    warmup_steps: int = 0
    total_steps: int | None = None
    warmup_ratio: float | None = None
    min_lr_ratio: float = 0.0


def _resolve_warmup(config: SchedulerConfig, total_steps: int) -> int:
    if config.warmup_ratio is not None:
        if not 0.0 <= config.warmup_ratio < 1.0:
            raise ValueError(
                f"warmup_ratio must be in [0, 1), got {config.warmup_ratio}"
            )
        return int(round(config.warmup_ratio * total_steps))
    return max(config.warmup_steps, 0)


def build_scheduler(optimizer, config: SchedulerConfig | None = None):
    """Build a lightweight scheduler, or return None for constant LR."""
    if config is None or config.name == "constant":
        return None
    if config.name not in ("linear", "cosine"):
        raise ValueError(f"Unsupported scheduler: {config.name}")
    if config.total_steps is None or config.total_steps <= 0:
        raise ValueError(f"{config.name} scheduler requires a positive total_steps.")

    total_steps = max(config.total_steps, 1)
    warmup_steps = _resolve_warmup(config, total_steps)
    if warmup_steps >= total_steps:
        raise ValueError(
            f"warmup ({warmup_steps}) must be shorter than total_steps ({total_steps})"
        )
    floor = float(config.min_lr_ratio)
    if not 0.0 <= floor <= 1.0:
        raise ValueError(f"min_lr_ratio must be in [0, 1], got {floor}")
    decay_steps = max(total_steps - warmup_steps, 1)

    def lr_lambda(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = min(float(step - warmup_steps) / float(decay_steps), 1.0)
        if config.name == "linear":
            shape = 1.0 - progress
        else:
            shape = 0.5 * (1.0 + math.cos(math.pi * progress))
        return floor + (1.0 - floor) * shape

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
