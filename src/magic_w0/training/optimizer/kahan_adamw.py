# Copyright 2026 MagicVLA authors. (Apache-2.0.)
"""AdamW that can actually move a bf16 parameter.

The model is loaded in bf16 and `torch.optim.AdamW` builds its state with
``zeros_like(param)``, so both moments and the parameter update itself live in
bf16. bf16 carries 8 mantissa bits: ``torch.finfo(torch.bfloat16).eps`` is
0.0078125, so a parameter of magnitude ``w`` cannot register any change smaller
than ``w * 0.0078``. Adam normalises, making each step about ``lr`` regardless of
gradient scale, so the condition for a step to land is simply

    lr >= |w| * 0.0078

At the recipe's lr of 2e-5 that means only parameters below ~0.0026 move at all.
Measured with the repo's own build_adamw, 10 steps, consistent gradient sign:

    weight scale   dtype      elements changed   mean |delta|   expected
       1.0         bf16          446 / 65536       1.67e-06      2.00e-04
       0.1         bf16         4133 / 65536       1.51e-05      2.00e-04
       0.02        bf16        19876 / 65536       7.24e-05      2.00e-04
       0.005       bf16        57677 / 65536       1.98e-04      2.00e-04
       any         fp32        65536 / 65536       2.00e-04      2.00e-04

The loss still falls, because small parameters -- norms, biases -- keep learning
and they matter. But the large weights are frozen, and the selection is
systematic rather than random: it is always the big ones that stop.

The textbook fix is an fp32 master copy, which costs 4 B/param for the copy plus
4 B for each fp32 moment: 16 B/param against the current 8, i.e. 43 GB instead of
22 for a 2.7 G-parameter model. Kahan compensated summation gets the same
arithmetic effect for one extra bf16 buffer -- 10 B/param -- by keeping the bits
each addition throws away and feeding them back into the next one. That is
exactly the problem it was invented for: accumulating many small increments into
a large running value.

Moments stay bf16 here. They are exponential averages with beta 0.9/0.95, so they
are dominated by recent gradients rather than by a long accumulation, and a
relative error of 0.4% in the second moment moves the step size by 0.2% after the
square root -- far below the effect being fixed. If that turns out to matter it is
a separate change with its own measurement.
"""

from __future__ import annotations

import torch
from torch.optim import Optimizer


class KahanAdamW(Optimizer):
    """AdamW whose parameter update survives low-precision storage.

    Identical to :class:`torch.optim.AdamW` in update rule -- decoupled weight
    decay, bias-corrected moments -- and identical in result for fp32 parameters,
    where the compensation term stays zero and no buffer is allocated. The
    compensation exists only for parameters that cannot represent their own
    update.
    """

    def __init__(
        self,
        params,
        *,
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-4,
    ) -> None:
        if lr <= 0:
            raise ValueError(f"lr must be positive, got {lr}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"betas must lie in [0, 1), got {betas}")
        if eps < 0:
            raise ValueError(f"eps must be non-negative, got {eps}")
        if weight_decay < 0:
            raise ValueError(f"weight_decay must be non-negative, got {weight_decay}")
        super().__init__(
            params,
            {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay},
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                if parameter.grad.is_sparse:
                    raise RuntimeError("KahanAdamW does not support sparse gradients")

                state = self.state[parameter]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(parameter)
                    state["exp_avg_sq"] = torch.zeros_like(parameter)
                    # Only low-precision parameters need somewhere to put the bits
                    # an addition discards. fp32 and fp64 keep them already.
                    if parameter.dtype in (torch.bfloat16, torch.float16):
                        state["compensation"] = torch.zeros_like(parameter)

                state["step"] += 1
                step = state["step"]
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                gradient = parameter.grad

                exp_avg.mul_(beta1).add_(gradient, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(gradient, gradient, value=1 - beta2)

                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step

                # Build the whole update in fp32 before it touches the parameter.
                # Computing it in bf16 would round the update itself, which is a
                # second, independent way to lose it.
                denominator = (
                    exp_avg_sq.float().sqrt() / (bias_correction2**0.5)
                ).add_(eps)
                update = exp_avg.float().div(denominator).mul_(-lr / bias_correction1)

                if weight_decay:
                    # Decoupled, as in AdamW: applied to the parameter, not the
                    # gradient. Folded into the same update so it goes through the
                    # same compensated addition -- applied separately it would be
                    # rounded away for exactly the same reason.
                    update.add_(parameter.float(), alpha=-lr * weight_decay)

                compensation = state.get("compensation")
                if compensation is None:
                    parameter.add_(update)
                    continue

                # Kahan: carry forward what the last addition could not represent,
                # then measure what this one drops.
                #   corrected = update + carried
                #   new       = p + corrected          (rounds)
                #   carried   = corrected - (new - p)  (the part that vanished)
                corrected = update.add_(compensation.float())
                previous = parameter.float()
                parameter.add_(corrected.to(parameter.dtype))
                compensation.copy_(
                    (corrected - (parameter.float() - previous)).to(compensation.dtype)
                )

        return loss


__all__ = ["KahanAdamW"]
