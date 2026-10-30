"""Temporal weighting for action-chunk regression losses.

The model may predict a longer action chunk than the controller consumes before
replanning.  This module keeps the default objective uniform, while allowing a
configuration such as::

    action_loss_weighting:
      type: piecewise
      execution_horizon: 16
      weights: [1.0, 0.5, 0.25]

For ``piecewise``, the three weights apply to ``[0, H)``, ``[H, 2H)``, and the
remaining tail respectively.  The returned weights are used in both the loss
numerator and denominator, so changing them does not change the overall loss
scale merely because the weights have a different mean.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any

import torch


def validate_action_loss_weighting(
    spec: Mapping[str, Any] | None, *, chunk_size: int
) -> None:
    """Validate a temporal action-loss weighting specification."""

    if spec is None:
        return
    if not isinstance(spec, Mapping):
        raise ValueError("action_loss_weighting must be a mapping or null")

    kind = spec.get("type", "uniform")
    if kind == "uniform":
        unknown = set(spec) - {"type"}
        if unknown:
            raise ValueError(
                f"uniform action_loss_weighting has unknown keys: {sorted(unknown)}"
            )
        return

    if kind == "piecewise":
        unknown = set(spec) - {"type", "execution_horizon", "weights"}
        if unknown:
            raise ValueError(
                f"piecewise action_loss_weighting has unknown keys: {sorted(unknown)}"
            )
        horizon = spec.get("execution_horizon")
        if isinstance(horizon, bool) or not isinstance(horizon, int):
            raise ValueError(
                "action_loss_weighting.execution_horizon must be an integer"
            )
        if not 1 <= horizon <= chunk_size:
            raise ValueError(
                "action_loss_weighting.execution_horizon must be in [1, chunk_size]"
            )
        weights = spec.get("weights")
        if isinstance(weights, (str, bytes)) or not isinstance(weights, Sequence):
            raise ValueError(
                "piecewise action_loss_weighting.weights must be a 3-element sequence"
            )
        if len(weights) != 3:
            raise ValueError(
                "piecewise action_loss_weighting.weights must contain exactly 3 values"
            )
        values = [float(value) for value in weights]
        if any(not math.isfinite(value) or value < 0.0 for value in values):
            raise ValueError(
                "piecewise action_loss_weighting.weights must be finite and non-negative"
            )
        if not any(value > 0.0 for value in values):
            raise ValueError(
                "piecewise action_loss_weighting.weights must contain a positive value"
            )
        return

    if kind == "exponential":
        unknown = set(spec) - {"type", "execution_horizon", "tail_weight"}
        if unknown:
            raise ValueError(
                f"exponential action_loss_weighting has unknown keys: {sorted(unknown)}"
            )
        tau = spec.get("execution_horizon")
        if isinstance(tau, bool) or not isinstance(tau, (int, float)):
            raise ValueError(
                "exponential action_loss_weighting.execution_horizon must be numeric"
            )
        if not math.isfinite(float(tau)) or float(tau) <= 0.0:
            raise ValueError(
                "exponential action_loss_weighting.execution_horizon must be positive"
            )
        floor = float(spec.get("tail_weight", 0.0))
        if not math.isfinite(floor) or not 0.0 <= floor <= 1.0:
            raise ValueError(
                "exponential action_loss_weighting.tail_weight must be in [0, 1]"
            )
        return

    raise ValueError(
        "action_loss_weighting.type must be one of: uniform, piecewise, exponential"
    )


def action_loss_time_weights(
    spec: Mapping[str, Any] | None,
    length: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return positive/non-negative temporal weights with shape ``[length]``."""

    if length <= 0:
        raise ValueError("action loss weighting length must be positive")
    weights = torch.ones(length, device=device, dtype=dtype)
    if not spec:
        return weights

    kind = spec.get("type", "uniform")
    if kind == "uniform":
        return weights
    if kind == "piecewise":
        horizon = int(spec["execution_horizon"])
        values = torch.as_tensor(spec["weights"], device=device, dtype=dtype)
        positions = torch.arange(length, device=device)
        bucket = torch.div(positions, horizon, rounding_mode="floor").clamp_max(2)
        return values[bucket]
    if kind == "exponential":
        tau = float(spec["execution_horizon"])
        floor = float(spec.get("tail_weight", 0.0))
        positions = torch.arange(length, device=device, dtype=dtype)
        return torch.exp(-positions / tau).clamp_min(floor)
    raise ValueError(f"unsupported action_loss_weighting.type: {kind!r}")
