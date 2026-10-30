"""Diagnostics and reproducibility helpers."""

from __future__ import annotations

import random


def seed_all(seed: int) -> None:
    """Seed common random number generators used by trainers."""
    random.seed(seed)
    try:
        import numpy as np
    except ImportError:
        np = None
    if np is not None:
        np.random.seed(seed)

    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
