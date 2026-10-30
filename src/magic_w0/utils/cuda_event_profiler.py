"""Opt-in, low-overhead CUDA event timing for one training step."""

from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator

import torch


_ACTIVE_PROFILER: ContextVar[CudaEventProfiler | None] = ContextVar(
    "magicw0_cuda_event_profiler", default=None
)


class CudaEventProfiler:
    """Collect nested CUDA ranges and synchronize once when the step finishes."""

    def __init__(self, enabled: bool = False) -> None:
        self.enabled = bool(enabled and torch.cuda.is_available())
        self._ranges: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []
        self._token = None
        self._scope = ""

    def begin_step(self) -> None:
        self._ranges.clear()
        self._token = _ACTIVE_PROFILER.set(self if self.enabled else None)

    def end_step(self) -> dict[str, float]:
        try:
            if not self.enabled or not self._ranges:
                return {}
            torch.cuda.synchronize()
            totals_ms: defaultdict[str, float] = defaultdict(float)
            for name, start, end in self._ranges:
                totals_ms[name] += start.elapsed_time(end)
            return {
                f"profile_{name}_s": value / 1000.0 for name, value in totals_ms.items()
            }
        finally:
            if self._token is not None:
                _ACTIVE_PROFILER.reset(self._token)
                self._token = None

    @contextmanager
    def record(self, name: str) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        try:
            yield
        finally:
            end.record()
            self._ranges.append((name, start, end))

    @contextmanager
    def scope(self, name: str) -> Iterator[None]:
        previous = self._scope
        self._scope = f"{previous}_{name}" if previous else name
        try:
            yield
        finally:
            self._scope = previous

    def scoped_name(self, name: str) -> str:
        return f"{self._scope}_{name}" if self._scope else name


@contextmanager
def profile_cuda(name: str) -> Iterator[None]:
    """Record a range on the profiler active for the current training step."""
    profiler = _ACTIVE_PROFILER.get()
    if profiler is None:
        yield
        return
    with profiler.record(profiler.scoped_name(name)):
        yield
