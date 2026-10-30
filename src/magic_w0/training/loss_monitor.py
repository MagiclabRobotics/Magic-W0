"""Loss statistics for interleaved co-training."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class _LossStats:
    count: int = 0
    total: float = 0.0
    ema: float | None = None


class LossMonitor:
    """Track loss independently for each interleaved modality."""

    def __init__(self, ema_decay: float = 0.95):
        if not 0.0 <= ema_decay < 1.0:
            raise ValueError("ema_decay must be in [0, 1).")
        self.ema_decay = ema_decay
        self._stats: dict[str, _LossStats] = {}

    def update(self, modality: str, loss: float) -> dict[str, float | int]:
        stats = self._stats.setdefault(modality, _LossStats())
        stats.count += 1
        stats.total += loss
        stats.ema = (
            loss
            if stats.ema is None
            else self.ema_decay * stats.ema + (1 - self.ema_decay) * loss
        )
        return {
            "count": stats.count,
            "mean": stats.total / stats.count,
            "ema": stats.ema,
        }

    def summary(self) -> dict[str, dict[str, float | int]]:
        return {
            modality: {
                "count": stats.count,
                "mean": stats.total / stats.count,
                "ema": stats.ema,
            }
            for modality, stats in self._stats.items()
        }
