# Copyright 2026 MagicVLA authors. (Apache-2.0.)
"""AdamW with fp32 master weights over bf16 parameters.

The model is loaded in bf16 and ``torch.optim.AdamW`` builds its state with
``zeros_like(param)``, so the moments and the parameter update all happen in
bf16. bf16 has 8 mantissa bits -- ``torch.finfo(torch.bfloat16).eps`` is
0.0078125 -- so a parameter of magnitude ``w`` cannot register any change below
``w * 0.0078``. Adam normalises, making each step roughly ``lr`` whatever the
gradient scale, so a step lands only when

    lr >= |w| * 0.0078

At lr 2e-5 that is |w| <= 0.0026. Measured through the repository's own
build_adamw, 10 steps, consistent gradient sign:

    weight scale   elements changed   mean |delta|   expected
       1.0            446 / 65536       1.67e-06      2.00e-04
       0.1           4133 / 65536       1.51e-05      2.00e-04
       0.02         19876 / 65536       7.24e-05      2.00e-04
       0.005        57677 / 65536       1.98e-04      2.00e-04
       (fp32)       65536 / 65536       2.00e-04      2.00e-04

Loss still falls, because norms and biases are small and keep learning. But the
large weights are frozen, and it is always the large ones -- a systematic
selection nobody chose.

This is the standard mixed-precision remedy, not a novel one: forward and
backward stay in bf16, while the optimizer keeps an fp32 copy of each parameter,
applies the update there, and casts the result back. AMP, DeepSpeed and Megatron
all do exactly this. Compensated (Kahan) summation reaches a similar place for
2 B/param instead of 8, but it makes updates intermittent rather than smooth for
large weights, and whether that is equivalent for Adam's dynamics is a question
this does not have to ask.

Cost, for a 2.7 G-parameter model: 8 B/param becomes 16 -- bf16 weights 2, bf16
grads 2, fp32 master 4, and two fp32 moments at 4 each -- so about 21.6 GB more.
The master copy is kept in optimizer state, so a checkpoint round-trip preserves
it; rebuilding it from the bf16 parameter on resume would throw away precisely
the bits it exists to hold.
"""

from __future__ import annotations

import torch
from torch.optim import Optimizer


class MasterWeightAdamW(Optimizer):
    """AdamW that steps an fp32 copy and writes the result back.

    The update rule is torch's AdamW exactly -- decoupled weight decay,
    bias-corrected moments. For parameters already in fp32 or fp64 no copy is
    made and this reduces to the ordinary optimizer.
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
                    raise RuntimeError(
                        "MasterWeightAdamW does not support sparse gradients"
                    )

                low_precision = parameter.dtype in (torch.bfloat16, torch.float16)
                state = self.state[parameter]
                if not state:
                    state["step"] = 0
                    # Moments live at master precision. Keeping them in bf16 while
                    # the master is fp32 would put the rounding back one level up.
                    state["exp_avg"] = torch.zeros_like(parameter, dtype=torch.float32)
                    state["exp_avg_sq"] = torch.zeros_like(
                        parameter, dtype=torch.float32
                    )
                    if low_precision:
                        state["master"] = parameter.detach().float().clone()

                state["step"] += 1
                step = state["step"]
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                master = state["master"] if low_precision else parameter
                gradient = parameter.grad.float()

                exp_avg.mul_(beta1).add_(gradient, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(gradient, gradient, value=1 - beta2)

                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                denominator = exp_avg_sq.sqrt().div_(bias_correction2**0.5).add_(eps)

                if weight_decay:
                    master.mul_(1 - lr * weight_decay)
                master.addcdiv_(exp_avg, denominator, value=-(lr / bias_correction1))

                if low_precision:
                    parameter.copy_(master)

        return loss

    def load_state_dict(self, state_dict) -> None:
        """Restore, then undo torch's cast of the state to the parameter dtype.

        ``Optimizer.load_state_dict`` runs every non-``step`` floating tensor
        through ``value.to(dtype=param.dtype)`` -- its docstring says
        "floating-point types are assumed to always match the type of params",
        which is exactly the assumption this class exists to break. So the fp32
        master and the fp32 moments come back as bf16, and because the state is
        then non-empty the ``if not state`` guard never rebuilds them: a resumed
        run silently returns to rounding every update away.

        Measured before this override, bf16 model, 300 steps either side of a
        resume: 996 of 4096 elements moved uninterrupted against 5 of 4096 after
        resuming, a ~500x collapse, with loss and grad_norm both looking normal.
        """
        super().load_state_dict(state_dict)
        for group in self.param_groups:
            for parameter in group["params"]:
                state = self.state.get(parameter)
                if not state:
                    continue
                for key in ("master", "exp_avg", "exp_avg_sq"):
                    tensor = state.get(key)
                    if tensor is not None and tensor.dtype != torch.float32:
                        state[key] = tensor.float()


__all__ = ["MasterWeightAdamW"]
