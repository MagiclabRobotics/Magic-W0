"""Base trainer primitives for distributed training."""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import torch

from magic_w0.utils.cuda_event_profiler import profile_cuda

from magic_w0.training.optimizer import (
    AdamWConfig,
    OptimizerConfig,
    build_optimizer,
)
from magic_w0.training.scheduler import SchedulerConfig, build_scheduler
from magic_w0.training.utils import seed_all

from .distribution_strategy import (
    DistributionStrategyConfig,
    barrier,
    get_rank,
    get_world_size,
    is_main_process,
    setup_distributed,
)


@dataclass(slots=True)
class TrainerState:
    """Mutable progress state shared by distributed trainers."""

    global_step: int = 0
    epoch: int = 0
    micro_step: int = 0


class _Zero2GradientOverlap:
    """Launch deterministic owner reductions as gradient buckets become ready."""

    def __init__(self, optimizer, *, rank: int, world_size: int) -> None:
        self.optimizer = optimizer
        self.rank = rank
        self.world_size = world_size
        self.parameters = [
            parameter
            for group in optimizer.param_groups
            for parameter in group["params"]
        ]
        self.owners = optimizer._param_to_rank
        self.bucket_bytes = (
            max(1, int(optimizer.optim.defaults.get("shard_bucket_mib", 256))) * 2**20
        )
        self._buckets_by_key: dict[str, tuple[tuple[torch.Tensor, ...], ...]] = {}
        self._handles = [
            parameter.register_post_accumulate_grad_hook(self._mark_ready)
            for parameter in self.parameters
        ]
        self._key: str | None = None
        self._buckets: tuple[tuple[torch.Tensor, ...], ...] | None = None
        self._bucket_by_parameter: dict[torch.Tensor, int] = {}
        self._ready_counts: list[int] = []
        self._ready: list[bool] = []
        self._discovery_ready_order: list[torch.Tensor] = []
        self._seen: set[torch.Tensor] = set()
        self._unexpected: set[torch.Tensor] = set()
        self._next_bucket = 0
        self._pending: list[
            tuple[
                Any,
                torch.Tensor,
                tuple[torch.Tensor, ...],
                int,
                torch.Tensor | None,
            ]
        ] = []

    def prepare(self, key: str) -> None:
        if self._key is not None:
            raise RuntimeError("zero2 gradient overlap step was not finished")
        self._key = key
        self._buckets = self._buckets_by_key.get(key)
        if self._buckets is None:
            return
        self._bucket_by_parameter = {
            parameter: bucket_index
            for bucket_index, bucket in enumerate(self._buckets)
            for parameter in bucket
        }
        self._ready_counts = [0] * len(self._buckets)
        self._ready = [False] * len(self._buckets)

    def _mark_ready(self, parameter: torch.Tensor) -> None:
        if self._key is None:
            return
        if self._buckets is None:
            self._discovery_ready_order.append(parameter)
            return
        bucket_index = self._bucket_by_parameter.get(parameter)
        if bucket_index is None:
            self._unexpected.add(parameter)
            return
        if parameter in self._seen:
            raise RuntimeError("zero2 parameter gradient became ready more than once")
        self._seen.add(parameter)
        self._ready_counts[bucket_index] += 1
        if self._ready_counts[bucket_index] == len(self._buckets[bucket_index]):
            self._ready[bucket_index] = True
        self._launch_ready_buckets()

    def _launch_ready_buckets(self) -> None:
        import torch.distributed as dist

        while (
            self._buckets is not None
            and self._next_bucket < len(self._buckets)
            and self._ready[self._next_bucket]
        ):
            bucket = self._buckets[self._next_bucket]
            owner = self.owners[bucket[0]]
            bucket_size = sum(
                parameter.numel() * parameter.element_size() for parameter in bucket
            )
            if len(bucket) == 1 and bucket_size > self.bucket_bytes:
                parameter = bucket[0]
                gradient = parameter.grad
                chunk_elements = max(1, self.bucket_bytes // gradient.element_size())
                for chunk in gradient.reshape(-1).split(chunk_elements):
                    work = dist.reduce(
                        chunk, dst=owner, op=dist.ReduceOp.SUM, async_op=True
                    )
                    self._pending.append((work, chunk, bucket, owner, gradient))
            else:
                flat = torch.cat([parameter.grad.reshape(-1) for parameter in bucket])
                work = dist.reduce(flat, dst=owner, op=dist.ReduceOp.SUM, async_op=True)
                self._pending.append((work, flat, bucket, owner, None))
            for parameter in bucket:
                parameter.grad = None
            self._next_bucket += 1

    def _build_buckets(
        self, ready_order: tuple[torch.Tensor, ...]
    ) -> tuple[tuple[torch.Tensor, ...], ...]:
        ready_index = {parameter: index for index, parameter in enumerate(ready_order)}
        buckets: list[tuple[int, tuple[torch.Tensor, ...]]] = []
        for owner in range(self.world_size):
            owned = [
                parameter
                for parameter in ready_order
                if self.owners[parameter] == owner
            ]
            bucket: list[torch.Tensor] = []
            bucket_size = 0
            bucket_dtype = None
            for parameter in owned:
                parameter_bytes = parameter.numel() * parameter.element_size()
                if bucket and (
                    parameter.dtype != bucket_dtype
                    or bucket_size + parameter_bytes > self.bucket_bytes
                ):
                    buckets.append((ready_index[bucket[-1]], tuple(bucket)))
                    bucket = []
                    bucket_size = 0
                bucket.append(parameter)
                bucket_size += parameter_bytes
                bucket_dtype = parameter.dtype
            if bucket:
                buckets.append((ready_index[bucket[-1]], tuple(bucket)))
        buckets.sort(key=lambda entry: entry[0])
        return tuple(bucket for _, bucket in buckets)

    def _synchronize_ready_order(self) -> tuple[torch.Tensor, ...]:
        import torch.distributed as dist

        parameter_index = {
            parameter: index for index, parameter in enumerate(self.parameters)
        }
        ready_positions = torch.full(
            (len(self.parameters),),
            -1,
            dtype=torch.int64,
            device=self.parameters[0].device,
        )
        for position, parameter in enumerate(self._discovery_ready_order):
            ready_positions[parameter_index[parameter]] = position
        if self.world_size > 1:
            dist.all_reduce(ready_positions, op=dist.ReduceOp.MAX)
        positions = ready_positions.cpu().tolist()
        return tuple(
            self.parameters[index]
            for index in sorted(
                (index for index, position in enumerate(positions) if position >= 0),
                key=lambda index: (positions[index], index),
            )
        )

    def finish(self) -> bool:
        if self._key is None:
            raise RuntimeError("zero2 gradient overlap step was not prepared")
        if self._buckets is None:
            active_parameters = tuple(
                parameter.grad is not None for parameter in self.parameters
            )
            active_count = sum(active_parameters)
            if len(self._discovery_ready_order) != active_count:
                raise RuntimeError(
                    "zero2 could not observe every active parameter during graph discovery"
                )
            ready_order = self._synchronize_ready_order()
            self._buckets_by_key[self._key] = self._build_buckets(ready_order)
            self._reset_step()
            return False

        self._launch_ready_buckets()
        if self._unexpected or self._next_bucket != len(self._buckets):
            raise RuntimeError(
                f"zero2 active parameter set changed for modality {self._key!r}"
            )
        chunked_gradients: dict[torch.Tensor, torch.Tensor] = {}
        for work, flat, bucket, owner, chunked_gradient in self._pending:
            work.wait()
            if self.rank != owner:
                continue
            flat.div_(self.world_size)
            if chunked_gradient is not None:
                chunked_gradients[bucket[0]] = chunked_gradient
                continue
            offset = 0
            for parameter in bucket:
                count = parameter.numel()
                parameter.grad = flat[offset : offset + count].view_as(parameter)
                offset += count
        for parameter, gradient in chunked_gradients.items():
            parameter.grad = gradient
        self._reset_step()
        return True

    def _reset_step(self) -> None:
        self._key = None
        self._buckets = None
        self._bucket_by_parameter = {}
        self._ready_counts = []
        self._ready = []
        self._discovery_ready_order = []
        self._seen = set()
        self._unexpected = set()
        self._next_bucket = 0
        self._pending = []


class BaseDistributedTrainer(ABC):
    """Abstract trainer contract for distributed execution."""

    @abstractmethod
    def fit(self) -> None:
        """Run the full training lifecycle."""

    @abstractmethod
    def train_loop(self, epoch: int) -> None:
        """Run one training epoch."""

    @abstractmethod
    def val_loop(self) -> dict[str, Any] | None:
        """Run validation and return metrics on the main process."""

    @abstractmethod
    def save_checkpoint(self, epoch: int, step: int = 0) -> None:
        """Save trainer state."""

    @abstractmethod
    def load_model(self) -> None:
        """Load or create the model."""

    @abstractmethod
    def load_dataset(self) -> None:
        """Load datasets and dataloaders."""

    @abstractmethod
    def backward(self, loss) -> None:
        """Backpropagate a scalar loss."""

    @abstractmethod
    def clip_grad_norm(self, max_norm: float):
        """Clip gradient norm and return the resulting norm."""


class DistributedTrainerBase(BaseDistributedTrainer):
    """Common state and helpers for concrete distributed trainers."""

    def __init__(
        self,
        *,
        strategy_config: DistributionStrategyConfig | None = None,
        seed: int = 0,
    ) -> None:
        self.strategy_config = strategy_config or DistributionStrategyConfig()
        self.local_rank = setup_distributed(self.strategy_config)
        self.rank = get_rank()
        self.world_size = get_world_size()
        self.device = torch.device(f"cuda:{self.local_rank}")
        self.state = TrainerState()
        self.seed = seed
        # Offset by rank. The global RNG's only consumers here are the flow
        # matching timestep and its noise (modeling_magicw0_base.py:1310, :1337),
        # and dropout is 0.0 in every recipe -- so seeding every rank identically
        # made all W ranks integrate the flow objective at exactly the same
        # (t, epsilon) on every step, forever: nothing else touches the philox
        # stream, so the ranks never drift apart. A global batch of W*B samples
        # then carried only B distinct timesteps, and because epsilon was COMMON to
        # the ranks, DDP's average reinforced it coherently instead of suppressing
        # it by 1/sqrt(W). Scaling from 8 GPUs to 64 added no timestep coverage at
        # all.
        #
        # This does not disturb the invariant interleaved.py:12-19 warns about --
        # that all ranks must draw the same data source each step, or DDP's
        # gradient all-reduce is not well defined. That draw uses its own private
        # torch.Generator (interleaved.py:95), as does every sampler in the data
        # path (batching.py:140, sampling.py:64, factory.py:25). None of them reads
        # the global RNG, which is why the warning could sit next to this bug for
        # so long: it protects an invariant the global seed does not carry.
        self.seed_offset = self.rank
        seed_all(seed + self.rank)

        self.model = None
        self.optimizer = None
        self.lr_scheduler = None
        self.train_dataloader = None
        self.val_dataloader = None

        self.use_amp = self.strategy_config.use_amp
        self.amp_dtype = torch.bfloat16 if self.strategy_config.bf16 else torch.float16
        self.grad_scaler = (
            torch.amp.GradScaler("cuda")
            if self.use_amp and self.amp_dtype == torch.float16
            else None
        )

    def fit(self) -> None:
        """Run load hooks and then train for `num_epochs`."""
        self.load_model()
        self.load_dataset()
        barrier()
        for epoch in range(self.num_epochs):
            self.state.epoch = epoch
            self.train_loop(epoch)
            self.val_loop()

    @property
    def num_epochs(self) -> int:
        return 1

    def configure_optimizer(
        self,
        *,
        learning_rate: float,
        weight_decay: float,
        named_parameters,
        optimizer_config: OptimizerConfig | None = None,
        scheduler_config: SchedulerConfig | None = None,
    ) -> None:
        if self.model is None:
            raise ValueError("Cannot configure optimizer before load_model().")
        resolved_config = optimizer_config or AdamWConfig(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )
        self.optimizer = build_optimizer(named_parameters, resolved_config)
        self.lr_scheduler = build_scheduler(self.optimizer, scheduler_config)

    def autocast_context(self):
        if not self.use_amp:
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=self.amp_dtype)

    def backward(self, loss) -> None:
        if self.grad_scaler is not None:
            self.grad_scaler.scale(loss).backward()
            return
        loss.backward()

    def prepare_zero2_gradient_overlap(self, key: str) -> None:
        overlap = getattr(self, "_zero2_gradient_overlap", None)
        if overlap is not None:
            overlap.prepare(key)

    def reduce_zero2_gradients(self) -> None:
        """Keep averaged gradients only on each ZeroRedundancyOptimizer owner.

        The overlap path launches owner reductions during backward. Its first
        step for each modality discovers the active graph and uses this manual
        path as a correctness-preserving fallback.
        """
        import torch.distributed as dist

        optimizer = self.optimizer
        if optimizer is None or not hasattr(optimizer, "_param_to_rank"):
            raise ValueError("zero2 requires ZeroRedundancyOptimizer")
        overlap = getattr(self, "_zero2_gradient_overlap", None)
        if overlap is not None and overlap.finish():
            return
        parameters = [
            parameter
            for group in optimizer.param_groups
            for parameter in group["params"]
        ]
        active_parameters = [parameter.grad is not None for parameter in parameters]
        owners = optimizer._param_to_rank

        active = torch.tensor(
            active_parameters,
            dtype=torch.int32,
            device=self.device,
        )
        dist.all_reduce(active, op=dist.ReduceOp.SUM)
        inconsistent = (active != 0) & (active != self.world_size)
        if bool(inconsistent.any()):
            raise RuntimeError(
                "zero2 requires the same used-parameter set on every rank"
            )

        local_optimizer = optimizer.optim
        bucket_bytes = (
            max(1, int(local_optimizer.defaults.get("shard_bucket_mib", 256))) * 2**20
        )
        parameters_by_owner = [[] for _ in range(self.world_size)]
        for parameter, is_active in zip(parameters, active_parameters, strict=True):
            if is_active:
                parameters_by_owner[owners[parameter]].append(parameter)

        def finish_bucket(pending) -> None:
            work, flat, bucket, owner = pending
            work.wait()
            if self.rank == owner:
                flat.div_(self.world_size)
                offset = 0
                for parameter in bucket:
                    count = parameter.numel()
                    parameter.grad = flat[offset : offset + count].view_as(parameter)
                    offset += count

        overlap = max(0, int(local_optimizer.defaults.get("shard_overlap", 1)))
        in_flight: list = []

        def reduce_bucket(bucket: list[torch.Tensor], owner: int) -> None:
            if not bucket:
                return
            flat = torch.cat([parameter.grad.reshape(-1) for parameter in bucket])
            work = dist.reduce(flat, dst=owner, op=dist.ReduceOp.SUM, async_op=True)
            for parameter in bucket:
                parameter.grad = None
            in_flight.append((work, flat, bucket, owner))
            if len(in_flight) > overlap:
                finish_bucket(in_flight.pop(0))

        for owner, owned_parameters in enumerate(parameters_by_owner):
            bucket: list[torch.Tensor] = []
            bucket_size = 0
            bucket_dtype = None
            for parameter in owned_parameters:
                parameter_bytes = parameter.grad.numel() * parameter.grad.element_size()
                if bucket and (
                    parameter.grad.dtype != bucket_dtype
                    or bucket_size + parameter_bytes > bucket_bytes
                ):
                    reduce_bucket(bucket, owner)
                    bucket = []
                    bucket_size = 0
                bucket.append(parameter)
                bucket_size += parameter_bytes
                bucket_dtype = parameter.grad.dtype
            reduce_bucket(bucket, owner)
        for pending in in_flight:
            finish_bucket(pending)

    def optimizer_step(self) -> None:
        if self.optimizer is None:
            raise ValueError("Optimizer has not been configured.")
        if self.grad_scaler is not None:
            with profile_cuda("optimizer_update"):
                self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            with profile_cuda("optimizer_update"):
                self.optimizer.step()
        if self.lr_scheduler is not None:
            with profile_cuda("optimizer_scheduler"):
                self.lr_scheduler.step()

    def zero_grad(self) -> None:
        if self.optimizer is None:
            raise ValueError("Optimizer has not been configured.")
        self.optimizer.zero_grad(set_to_none=True)

    def clip_grad_norm(self, max_norm: float):
        if self.model is None:
            raise ValueError("Cannot clip gradients before load_model().")
        strategy = getattr(getattr(self, "strategy_config", None), "strategy", "ddp")
        if strategy == "zero2":
            import torch.distributed as dist

            squared_norm = torch.zeros((), dtype=torch.float32, device=self.device)
            for group in self.optimizer.param_groups:
                for parameter in group["params"]:
                    if parameter.grad is not None:
                        squared_norm.add_(parameter.grad.float().pow(2).sum())
            dist.all_reduce(squared_norm, op=dist.ReduceOp.SUM)
            total_norm = squared_norm.sqrt()
            coefficient = (max_norm / (total_norm + 1e-6)).clamp(max=1.0)
            for group in self.optimizer.param_groups:
                for parameter in group["params"]:
                    if parameter.grad is not None:
                        parameter.grad.mul_(coefficient.to(parameter.grad.dtype))
            return total_norm
        return torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)

    def log(self, message: str) -> None:
        if is_main_process():
            print(message, flush=True)
