"""Norm metrics for model diagnostics."""

from __future__ import annotations

import torch


def parameter_l2_norm(model) -> float:
    """Compute the L2 norm of all model parameters."""
    total = torch.zeros((), device=next(model.parameters()).device)
    for parameter in model.parameters():
        total = total + parameter.detach().float().pow(2).sum()
    return float(total.sqrt().cpu().item())


def grad_l2_norm(model) -> float:
    """Compute the L2 norm of all available gradients."""
    total = torch.zeros((), device=next(model.parameters()).device)
    for parameter in model.parameters():
        if parameter.grad is not None:
            total = total + parameter.grad.detach().float().pow(2).sum()
    return float(total.sqrt().cpu().item())
