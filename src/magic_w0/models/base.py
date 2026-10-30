# Copyright 2026 MagicVLA authors.
"""Thin policy base replacing lerobot's ``PreTrainedPolicy`` for training.

Provides only what a co-training policy needs from the loop: it stores
``self.config`` and returns ``dict(self.named_parameters())`` from
``get_optim_params()`` (the named-dict contract Muon's matrix/AdamW split
relies on). The HF-hub / PEFT / from_pretrained / safetensors machinery of
lerobot's base is intentionally omitted (deferred with inference/deploy).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, TypeVar

from torch import nn

# Kept for API parity: modeling_pi05 imports the ``T`` TypeVar from its base.
T = TypeVar("T", bound="StandalonePolicy")


@dataclass(frozen=True, slots=True)
class ModelForwardOutput:
    """Uniform result consumed by every trainer implementation."""

    loss: Any
    metrics: Mapping[str, Any]
    loss_name: str


class StandalonePolicy(nn.Module):
    """Minimal nn.Module policy base (no lerobot, no HF-hub)."""

    config_class = None
    name = None

    def __init__(self, config, *inputs, **kwargs):
        super().__init__()
        self.config = config

    def get_optim_params(self) -> dict:
        """Named parameter dict for the optimizer (Muon needs names to split)."""
        return dict(self.named_parameters())
