# Copyright 2026 MagicVLA authors. (Apache-2.0.)
"""Hybrid Muon optimizer with an AdamW fallback path."""

from __future__ import annotations

from collections import defaultdict

import torch
import torch.distributed as dist
from torch.optim import Optimizer
from magic_w0.utils.cuda_event_profiler import profile_cuda

_NS_COEFFS = (3.4445, -4.7750, 2.0315)


def newton_schulz_cost(parameter: torch.Tensor) -> int:
    """Rough FLOP proxy for one Newton-Schulz call, used to balance the shard plan.

    The iteration runs on the matrix with rows <= cols, so with ``r = min(dim)`` and
    ``c = max(dim)`` each step costs ``value @ value.T`` (r^2 c) + ``square @ square``
    (r^3) + one more r^2 c -- i.e. r^2 (2c + r). Only relative magnitudes matter; the
    constant and the step count cancel out of the assignment.
    """
    rows, cols = parameter.size(0), parameter.size(1)
    small, large = min(rows, cols), max(rows, cols)
    return small * small * (2 * large + small)


def plan_newton_schulz_shards(
    parameters: list[torch.Tensor], world_size: int
) -> list[int]:
    """Assign each matrix to the rank that will orthogonalize it (longest job first).

    Round-robin is badly unbalanced in practice -- a handful of wide MLP matrices carry
    most of the cost -- so matrices are sorted by descending cost and handed to whichever
    rank is currently least loaded. The plan is a pure function of the parameter order,
    which DDP guarantees is identical on every rank, so no communication is needed to
    agree on it.
    """
    loads = [0] * world_size
    owners = [0] * len(parameters)
    for index in sorted(
        range(len(parameters)), key=lambda i: -newton_schulz_cost(parameters[i])
    ):
        rank = min(range(world_size), key=lambda candidate: loads[candidate])
        owners[index] = rank
        loads[rank] += newton_schulz_cost(parameters[index])
    return owners


@torch.no_grad()
def zeropower_via_newtonschulz5(
    gradient: torch.Tensor,
    *,
    steps: int = 5,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Approximate the orthogonal factor of a two-dimensional gradient."""
    if gradient.ndim != 2:
        raise ValueError("Newton-Schulz orthogonalization requires a 2D tensor")
    a, b, c = _NS_COEFFS
    # Run the iteration in bf16 so the matmuls hit A800 tensor cores (~16x the
    # fp32 no-TF32 path). The Newton-Schulz iteration is self-correcting, so bf16
    # is safe here -- this matches the reference Muon. Scoped to the optimizer, so
    # it never touches the precision-sensitive fp32 flow-matching head.
    value = gradient.bfloat16()
    transpose = value.size(0) > value.size(1)
    if transpose:
        value = value.T
    value = value / (value.norm() + eps)
    for _ in range(steps):
        square = value @ value.T
        value = a * value + (b * square + c * (square @ square)) @ value
    value = value.T if transpose else value
    return value.to(gradient.dtype)


@torch.no_grad()
def zeropower_via_newtonschulz5_batched(
    gradients: list[torch.Tensor],
    *,
    steps: int = 5,
    batch_size: int = 1,
) -> list[torch.Tensor]:
    """Orthogonalize equal canonical shapes together with batched matmuls."""
    if batch_size <= 1:
        return [zeropower_via_newtonschulz5(value, steps=steps) for value in gradients]

    groups: dict[tuple[torch.device, torch.dtype, int, int], list[int]] = defaultdict(
        list
    )
    for index, gradient in enumerate(gradients):
        if gradient.ndim != 2:
            raise ValueError("Newton-Schulz orthogonalization requires 2D tensors")
        rows, cols = gradient.shape
        groups[
            (gradient.device, gradient.dtype, min(rows, cols), max(rows, cols))
        ].append(index)

    results: list[torch.Tensor | None] = [None] * len(gradients)
    a, b, c = _NS_COEFFS
    for indices in groups.values():
        for start in range(0, len(indices), batch_size):
            chunk = indices[start : start + batch_size]
            if len(chunk) == 1:
                index = chunk[0]
                results[index] = zeropower_via_newtonschulz5(
                    gradients[index], steps=steps
                )
                continue
            values = [
                gradients[index].T
                if gradients[index].size(0) > gradients[index].size(1)
                else gradients[index]
                for index in chunk
            ]
            value = torch.stack(values).bfloat16()
            norm = torch.linalg.vector_norm(value, dim=(-2, -1), keepdim=True)
            value = value / (norm + 1e-7)
            for _ in range(steps):
                square = value @ value.transpose(-2, -1)
                value = a * value + (b * square + c * (square @ square)) @ value
            for offset, index in enumerate(chunk):
                update = value[offset]
                if gradients[index].size(0) > gradients[index].size(1):
                    update = update.T
                results[index] = update.to(gradients[index].dtype)
    return [value for value in results if value is not None]


class Muon(Optimizer):
    """Apply Muon to matrix groups and AdamW to fallback groups."""

    def __init__(
        self,
        param_groups,
        *,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        adamw_betas: tuple[float, float] = (0.9, 0.95),
        adamw_eps: float = 1e-8,
        weight_decay: float = 0.0,
        update_rms: float = 0.0,
        shard_ns: bool = False,
        ns_batch_size: int = 1,
        shard_overlap: int = 1,
        shard_bucket_mib: int = 256,
    ) -> None:
        super().__init__(
            param_groups,
            defaults={
                "lr": lr,
                "momentum": momentum,
                "nesterov": nesterov,
                "ns_steps": ns_steps,
                "betas": adamw_betas,
                "eps": adamw_eps,
                "weight_decay": weight_decay,
                "update_rms": update_rms,
                "shard_ns": shard_ns,
                "ns_batch_size": ns_batch_size,
                "shard_overlap": shard_overlap,
                "shard_bucket_mib": shard_bucket_mib,
            },
        )
        for group in self.param_groups:
            if "use_muon" not in group:
                raise ValueError("every Muon parameter group must declare use_muon")

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            if group["use_muon"]:
                with profile_cuda("optimizer_muon"):
                    self._step_muon(group)
            else:
                with profile_cuda("optimizer_adamw"):
                    self._step_adamw(group)
        return loss

    @staticmethod
    def _update_scale(parameter: torch.Tensor, update_rms: float) -> float:
        """Factor applied to the orthogonalized update before the learning rate."""
        if update_rms > 0.0:
            # Update-RMS matching (Moonlight, "Muon is Scalable for LLM Training"). The
            # Newton-Schulz output has unit singular values, so ||O||_F = sqrt(min(m, n))
            # and its element RMS is 1/sqrt(max(m, n)). Scaling by
            # `update_rms * sqrt(max(m, n))` pins EVERY matrix's update RMS to the target
            # regardless of shape. Set it to AdamW's observed update RMS (~0.2) and both
            # groups take same-sized steps at the SAME learning rate, which is what lets an
            # AdamW-tuned lr / weight decay carry over to the Muon group (ratio 1:1).
            return update_rms * max(parameter.size(0), parameter.size(1)) ** 0.5
        # Default (Keller Jordan): corrects for aspect ratio only, leaving the update RMS at
        # 1/sqrt(fan_in) -- shape dependent, so no single Muon:AdamW lr ratio matches every
        # matrix at once. Kept as the default so existing runs stay reproducible.
        return max(1.0, parameter.size(0) / parameter.size(1)) ** 0.5

    def _shard_world_size(self, group) -> int:
        """Ranks to spread Newton-Schulz over; 1 disables sharding."""
        if not group.get("shard_ns", False):
            return 1
        if not (dist.is_available() and dist.is_initialized()):
            return 1
        return dist.get_world_size()

    def _step_muon(self, group) -> None:
        parameters, updates = [], []
        with profile_cuda("muon_prepare"):
            for parameter in group["params"]:
                gradient = parameter.grad
                if gradient is None:
                    continue
                if gradient.ndim != 2:
                    raise ValueError(
                        f"Muon group received shape {tuple(gradient.shape)}; expected a matrix"
                    )
                state = self.state[parameter]
                momentum_buffer = state.setdefault(
                    "momentum_buffer", torch.zeros_like(gradient)
                )
                momentum_buffer.mul_(group["momentum"]).add_(gradient)
                parameters.append(parameter)
                updates.append(
                    gradient.add(momentum_buffer, alpha=group["momentum"])
                    if group["nesterov"]
                    else momentum_buffer
                )
        if not parameters:
            return

        if self._shard_world_size(group) > 1:
            self._step_muon_sharded(group, parameters, updates)
            return

        with profile_cuda("muon_ns"):
            orthogonalized = zeropower_via_newtonschulz5_batched(
                updates,
                steps=group["ns_steps"],
                batch_size=group["ns_batch_size"],
            )
        for parameter, update in zip(parameters, orthogonalized, strict=True):
            with profile_cuda("muon_apply"):
                update.mul_(self._update_scale(parameter, group["update_rms"]))
                self._apply(parameter, update, group)

    def _master(self, parameter: torch.Tensor) -> torch.Tensor:
        """fp32 copy of a low-precision parameter, or the parameter itself.

        bf16 has 8 mantissa bits, so a parameter of magnitude ``w`` cannot record
        a change below ``w * 0.0078``.  Muon's update is scaled to a target RMS and
        then multiplied by ``lr``, so like Adam's it is roughly ``lr`` in size --
        which at lr 2e-5 is below that threshold for every weight above ~0.0026,
        i.e. essentially every matrix in the model.  Writing the update into the
        bf16 tensor discards it.

        The copy is kept in optimizer state so it survives a checkpoint; rebuilding
        it from the bf16 parameter on resume would throw away exactly the bits it
        exists to hold.
        """
        if parameter.dtype not in (torch.bfloat16, torch.float16):
            return parameter
        state = self.state[parameter]
        master = state.get("master")
        if master is None:
            master = parameter.detach().float().clone()
            state["master"] = master
        return master

    def _apply(self, parameter: torch.Tensor, update: torch.Tensor, group) -> None:
        # The single write-back for the matrix path -- both the plain and the
        # sharded branch route through here.
        master = self._master(parameter)
        if group["weight_decay"]:
            master.mul_(1 - group["lr"] * group["weight_decay"])
        master.add_(update.float(), alpha=-group["lr"])
        if master is not parameter:
            parameter.copy_(master)

    def _step_muon_sharded(self, group, parameters: list, momenta: list) -> None:
        """Orthogonalize ~1/W of the matrices per rank and share the results.

        Under DDP every rank holds the same all-reduced gradient, so the unsharded path
        runs the SAME Newton-Schulz over EVERY matrix on EVERY rank -- W-fold redundant
        compute, which dominates the Muon step (measured on a 2.8B-parameter matrix group:
        ~0.65 s of a 1.88 s/step optimizer step on 8xA800). Here rank r computes only the
        matrices assigned to it and a SUM all-reduce shares them: every slot of the flat
        buffer is written by exactly one rank and left zero on the others, so the sum
        reconstructs bit-identical updates (adding zero is exact). Buckets bound the extra
        memory, and ``shard_overlap`` keeps that many all-reduces in flight so they run on
        NCCL's stream while the next bucket's Newton-Schulz runs on the compute stream.

        Momentum buffers stay replicated on every rank, so optimizer state, checkpointing
        and resume are untouched by sharding.
        """
        world_size, rank = dist.get_world_size(), dist.get_rank()
        owners = plan_newton_schulz_shards(parameters, world_size)
        bucket_bytes = max(1, int(group["shard_bucket_mib"])) * 2**20
        depth = max(0, int(group["shard_overlap"]))

        def flush(pending) -> None:
            work, buffer, start, stop = pending
            work.wait()
            offset = 0
            for index in range(start, stop):
                parameter = parameters[index]
                count = parameter.numel()
                with profile_cuda("muon_apply"):
                    update = buffer[offset : offset + count].view(parameter.shape)
                    update = update.to(parameter.dtype) * self._update_scale(
                        parameter, group["update_rms"]
                    )
                    self._apply(parameter, update, group)
                offset += count

        in_flight: list = []
        start = 0
        while start < len(parameters):
            dtype = parameters[start].dtype
            stop, count = start, 0
            while stop < len(parameters) and parameters[stop].dtype == dtype:
                width = parameters[stop].element_size()
                if (
                    stop > start
                    and (count + parameters[stop].numel()) * width > bucket_bytes
                ):
                    break
                count += parameters[stop].numel()
                stop += 1

            buffer = torch.zeros(count, device=parameters[start].device, dtype=dtype)
            owned_indices = [
                index for index in range(start, stop) if owners[index] == rank
            ]
            with profile_cuda("muon_ns"):
                owned_updates = zeropower_via_newtonschulz5_batched(
                    [momenta[index] for index in owned_indices],
                    steps=group["ns_steps"],
                    batch_size=group["ns_batch_size"],
                )
            updates_by_index = dict(zip(owned_indices, owned_updates, strict=True))
            offset = 0
            for index in range(start, stop):
                numel = parameters[index].numel()
                if owners[index] == rank:
                    buffer[offset : offset + numel].copy_(
                        updates_by_index[index].reshape(-1)
                    )
                offset += numel
            in_flight.append(
                (dist.all_reduce(buffer, async_op=True), buffer, start, stop)
            )
            while len(in_flight) > depth:
                flush(in_flight.pop(0))
            start = stop

        while in_flight:
            flush(in_flight.pop(0))

    def _step_adamw(self, group) -> None:
        beta1, beta2 = group["betas"]
        for parameter in group["params"]:
            if parameter.grad is None:
                continue
            gradient = parameter.grad.float()
            state = self.state[parameter]
            if not state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(parameter, dtype=torch.float32)
                state["exp_avg_sq"] = torch.zeros_like(parameter, dtype=torch.float32)
            state["step"] += 1
            exp_avg = state["exp_avg"]
            exp_avg_sq = state["exp_avg_sq"]
            exp_avg.mul_(beta1).add_(gradient, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(gradient, gradient, value=1 - beta2)
            bias_correction1 = 1 - beta1 ** state["step"]
            bias_correction2 = 1 - beta2 ** state["step"]
            denominator = (
                exp_avg_sq.sqrt().div_(bias_correction2**0.5).add_(group["eps"])
            )
            # Same reason as the matrix path: an lr-sized step vanishes when written
            # into bf16. The moments were already fp32 here; the parameter was not.
            master = self._master(parameter)
            if group["weight_decay"]:
                master.mul_(1 - group["lr"] * group["weight_decay"])
            master.addcdiv_(
                exp_avg,
                denominator,
                value=-(group["lr"] / bias_correction1),
            )
            if master is not parameter:
                parameter.copy_(master)

    def load_state_dict(self, state_dict) -> None:
        """Restore, then undo torch's cast of the state to the parameter dtype.

        ``Optimizer.load_state_dict`` runs every non-``step`` floating tensor
        through ``value.to(dtype=param.dtype)`` -- "floating-point types are
        assumed to always match the type of params" -- which is precisely the
        assumption the fp32 master breaks. Without this, `master` and the AdamW
        branch's moments come back as bf16, `_master` sees a non-None entry and
        returns it, and every subsequent update is rounded away exactly as before
        the fix.

        Measured before this override on the production Muon settings
        (update_rms 0.2, lr 2e-5), 300 steps either side of a resume: 996 of 4096
        elements moved uninterrupted against 5 of 4096 after resuming.
        """
        super().load_state_dict(state_dict)
        for saved_group, group in zip(
            state_dict["param_groups"], self.param_groups, strict=True
        ):
            for parameter_id, parameter in zip(
                saved_group["params"], group["params"], strict=True
            ):
                saved_state = state_dict["state"].get(parameter_id)
                if not saved_state:
                    continue
                state = self.state[parameter]
                for key in ("master", "exp_avg", "exp_avg_sq"):
                    tensor = saved_state.get(key)
                    if tensor is not None:
                        state[key] = tensor.to(
                            device=parameter.device, dtype=torch.float32
                        )


__all__ = [
    "Muon",
    "zeropower_via_newtonschulz5",
    "zeropower_via_newtonschulz5_batched",
]
