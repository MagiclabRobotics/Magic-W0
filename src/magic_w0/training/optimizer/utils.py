"""Optimizer construction utilities."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class AdamWConfig:
    """Configuration for the default AdamW optimizer."""

    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    # How a bf16 parameter is kept updatable. Without either of these,
    # torch.optim.AdamW builds its state with zeros_like(param) and applies the
    # update straight to the bf16 tensor -- and bf16 has 8 mantissa bits, so a
    # parameter of magnitude w cannot register a change below w * 0.0078. Adam's
    # step is about lr whatever the gradient scale, so at lr 2e-5 only parameters
    # under ~0.0026 move at all. Measured through this factory at weight scale
    # 1.0: 446 of 65536 elements changed over 10 steps, mean displacement 1/120 of
    # what it should have been -- and it is always the LARGE weights that freeze.
    #
    # master_weights is the default because it is the ordinary mixed-precision
    # remedy -- AMP, DeepSpeed and Megatron all keep an fp32 copy -- so it needs no
    # argument beyond "this is how it is normally done". It costs 8 B/param on top
    # of the 8 already spent (fp32 master 4, two fp32 moments instead of two bf16),
    # about +21.6 GB for 2.7 G parameters.
    #
    # kahan reaches a similar place for 2 B/param by keeping the bits each addition
    # discards and feeding them back. Measured at weight scale 0.02 -- the
    # Transformer range -- it restores 99.8% of elements against 30% unfixed. But
    # for larger weights the compensation has to accumulate past a whole
    # quantisation step before it releases, so updates become intermittent rather
    # than smooth, and whether that is equivalent for Adam's dynamics is not
    # something the numerics settle. Use it when the memory matters more.
    #
    # Setting both false reproduces a run from before this was found.
    master_weights: bool = True
    kahan: bool = False


def trainable_parameters(parameters: Iterable) -> list:
    """Return parameters that should be passed to an optimizer."""
    return [param for param in parameters if getattr(param, "requires_grad", False)]


class FP32AdamW(torch.optim.AdamW):
    """AdamW with FP32 first- and second-moment state.

    The model parameters are intentionally *not* promoted.  This is the narrow
    precision fix for BF16-weight training: gradients and all Adam statistics
    are accumulated in FP32, while the final update is explicitly rounded back
    to the parameter dtype.  ``step()`` remains custom because calling the
    parent implementation would initialize BF16 moments and lose this fix.
    """

    def __init__(
        self,
        params,
        *,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        super().__init__(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )

    @staticmethod
    def _state_tensor(
        state: dict,
        name: str,
        parameter: torch.Tensor,
    ) -> torch.Tensor:
        """Return a newly initialized/current FP32 state tensor."""
        value = state.get(name)
        if value is None:
            value = torch.zeros_like(parameter, dtype=torch.float32)
        elif value.dtype != torch.float32:
            raise RuntimeError(f"{name} must be torch.float32, got {value.dtype}")
        elif value.device != parameter.device:
            value = value.to(device=parameter.device)
        state[name] = value
        return value

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Gradients can be absent for unused parameters, so collect only the
        # parameters that participate in this step.  Grouping by device and
        # step keeps foreach kernels valid even when different parameters have
        # progressed at different rates.
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            records: dict[tuple[torch.device, int], list[tuple]] = {}
            for parameter in group["params"]:
                gradient = parameter.grad
                if gradient is None:
                    continue
                if gradient.is_sparse:
                    raise RuntimeError("FP32AdamW does not support sparse gradients")

                # The source gradient may be BF16 because the parameter is BF16;
                # all statistics are deliberately updated from an FP32 copy.
                gradient = gradient.detach().to(dtype=torch.float32)
                state = self.state[parameter]
                exp_avg = self._state_tensor(state, "exp_avg", parameter)
                exp_avg_sq = self._state_tensor(state, "exp_avg_sq", parameter)
                step = int(state.get("step", 0)) + 1
                state["step"] = step
                records.setdefault((parameter.device, step), []).append(
                    (parameter, gradient, exp_avg, exp_avg_sq)
                )

            for (_, step), entries in records.items():
                parameters, gradients, exp_avgs, exp_avg_sqs = zip(
                    *entries, strict=True
                )
                parameters = list(parameters)
                gradients = list(gradients)
                exp_avgs = list(exp_avgs)
                exp_avg_sqs = list(exp_avg_sqs)

                torch._foreach_mul_(exp_avgs, beta1)
                torch._foreach_add_(exp_avgs, gradients, alpha=1.0 - beta1)
                torch._foreach_mul_(exp_avg_sqs, beta2)
                torch._foreach_addcmul_(
                    exp_avg_sqs,
                    gradients,
                    gradients,
                    value=1.0 - beta2,
                )

                bias_correction1 = 1.0 - beta1**step
                bias_correction2 = 1.0 - beta2**step
                denominators = list(torch._foreach_sqrt(exp_avg_sqs))
                torch._foreach_div_(denominators, bias_correction2**0.5)
                torch._foreach_add_(denominators, group["eps"])

                if group["weight_decay"]:
                    torch._foreach_mul_(
                        parameters,
                        1.0 - group["lr"] * group["weight_decay"],
                    )

                # Foreach addcdiv performs the update in FP32 and converts only
                # at the destination: BF16 parameters stay BF16, while FP32
                # parameters stay FP32.
                torch._foreach_addcdiv_(
                    parameters,
                    exp_avgs,
                    denominators,
                    value=-(group["lr"] / bias_correction1),
                )

        return loss


def build_adamw(parameters: Iterable, config: AdamWConfig):
    """Build AdamW, choosing the variant the parameter dtype actually needs."""
    params = trainable_parameters(parameters)
    if not params:
        raise ValueError("No trainable parameters were found for the optimizer.")
    if config.master_weights and config.kahan:
        raise ValueError(
            "master_weights and kahan both fix the same thing; enable one, not both"
        )
    low_precision = any(
        param.dtype in (torch.bfloat16, torch.float16) for param in params
    )
    # FP32AdamW (origin/main) keeps the moments in fp32 but writes the update
    # straight into the bf16 parameter, so the *statistics* are exact and the
    # *recording* still is not: at lr 2e-5 a weight of magnitude 0.1 has a bf16
    # spacing of 7.8e-4, and the step rounds away.  Master weights keep an fp32
    # copy of the parameter itself, which is the part that has to change.  For
    # fp32 parameters the two are equivalent, and FP32AdamW's foreach kernels
    # are the better default there.
    if low_precision and config.master_weights:
        from .master_weight_adamw import MasterWeightAdamW

        return MasterWeightAdamW(
            params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=config.betas,
            eps=config.eps,
        )
    if low_precision and config.kahan:
        from .kahan_adamw import KahanAdamW

        return KahanAdamW(
            params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=config.betas,
            eps=config.eps,
        )
    return FP32AdamW(
        params,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=config.betas,
        eps=config.eps,
    )
