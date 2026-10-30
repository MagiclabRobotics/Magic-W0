"""Checkpoint IO helpers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch


def save_checkpoint(path: str | Path, payload: dict[str, Any]) -> None:
    """Save a checkpoint payload atomically, creating parent directories when needed.

    Writing to a staging file first means a run killed mid-save leaves the previous
    checkpoint intact instead of a truncated one.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.with_suffix(target.suffix + ".tmp")
    with staging.open("wb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
        if hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_DONTNEED"):
            try:
                os.posix_fadvise(
                    handle.fileno(),
                    0,
                    0,
                    os.POSIX_FADV_DONTNEED,
                )
            except OSError:
                pass
    staging.replace(target)


def load_checkpoint(
    path: str | Path, *, map_location: str | None = None, weights_only: bool = True
) -> dict[str, Any]:
    """Load a checkpoint payload.

    ``weights_only`` defaults to torch's safe mode; resuming passes False because our
    own payload carries the optimizer and scheduler state alongside the tensors.
    """
    return torch.load(Path(path), map_location=map_location, weights_only=weights_only)


def resolve_checkpoint(path: str | Path) -> Path:
    """Accept either a checkpoint file or a run directory and return the file to load.

    Pointing at the output directory is the common case ("resume this run"), and there
    the right file is the one with the highest step -- NOT the newest mtime, because a
    resumed run rewrites earlier steps' files and NOT lexicographic order, because
    ``step-9500`` sorts after ``step-10000``.
    """
    target = Path(path)
    if target.is_file():
        return target
    if not target.is_dir():
        raise FileNotFoundError(f"resume_from path does not exist: {target}")

    candidates = [
        candidate
        for candidate in target.glob("checkpoint-*.pt")
        if not candidate.name.endswith(".optimizer.pt")
    ]
    if not candidates:
        raise FileNotFoundError(f"no checkpoint-*.pt found under {target}")

    def step_of(candidate: Path) -> int:
        tail = candidate.stem.rsplit("-", 1)[-1]
        return int(tail) if tail.isdigit() else -1

    return max(candidates, key=step_of)
