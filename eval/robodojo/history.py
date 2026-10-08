"""Per-environment visual history for temporal Magic_W0 checkpoints.

The buffer follows the training schedule used by the reference adapter:
history contains strictly past frames at ``step - k * interval`` and is
left-padded with zero frames at the beginning of an episode.
"""

from __future__ import annotations

import numpy as np


def validate_history_schedule(history_interval: int, execute_steps: int) -> None:
    interval, execute = int(history_interval), int(execute_steps)
    if interval < 1 or execute < 1:
        raise ValueError(
            "history_interval and execute_steps must be >= 1, "
            f"got {interval}, {execute}"
        )
    if interval % execute:
        divisors = [d for d in range(1, interval + 1) if interval % d == 0]
        raise ValueError(
            f"history_interval={interval} must be divisible by "
            f"execute_steps={execute}; expected one of {divisors}"
        )


class HistoryBuffer:
    """Store replan observations independently for each environment."""

    def __init__(self, history_frames: int, history_interval: int, execute_steps: int):
        validate_history_schedule(history_interval, execute_steps)
        if int(history_frames) < 1:
            raise ValueError(f"history_frames must be >= 1, got {history_frames}")
        self.history_frames = int(history_frames)
        self.history_interval = int(history_interval)
        self.execute_steps = int(execute_steps)
        self._frames: dict[int, dict[int, np.ndarray]] = {}

    def step_of(self, replan_step: int) -> int:
        return int(replan_step) * self.execute_steps

    def past_steps(self, step: int) -> list[int]:
        return [
            step - back * self.history_interval
            for back in range(self.history_frames, 0, -1)
            if step - back * self.history_interval >= 0
        ]

    def clip(
        self, env_idx: int, step: int, current: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        stored = self._frames.get(int(env_idx), {})
        wanted = self.past_steps(int(step))
        missing = [value for value in wanted if value not in stored]
        if missing:
            raise RuntimeError(
                f"env {env_idx} is missing history frames at steps {missing}; "
                f"expected one observation per execute_steps={self.execute_steps} chunk"
            )
        current = np.asarray(current)
        clip = np.zeros((self.history_frames, *current.shape), dtype=current.dtype)
        if wanted:
            clip[self.history_frames - len(wanted) :] = np.stack(
                [stored[value] for value in wanted]
            )
        is_pad = np.arange(self.history_frames) < self.history_frames - len(wanted)
        return clip, is_pad

    def record(self, env_idx: int, step: int, frame: np.ndarray) -> None:
        frames = self._frames.setdefault(int(env_idx), {})
        frames[int(step)] = np.array(frame, copy=True)
        horizon = int(step) - self.history_frames * self.history_interval
        for old_step in [value for value in frames if value < horizon]:
            del frames[old_step]

    def reset(self) -> None:
        self._frames.clear()
