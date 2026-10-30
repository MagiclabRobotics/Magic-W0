"""Distribution strategy descriptors and process helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
import os


@dataclass(frozen=True, slots=True)
class DistributionStrategyConfig:
    """Runtime distribution strategy."""

    strategy: str = "fsdp"
    backend: str = "nccl"
    use_amp: bool = False
    bf16: bool = True
    gradient_as_bucket_view: bool = False
    bucket_cap_mb: float | None = None
    zero2_overlap_backward: bool = False

    def __post_init__(self) -> None:
        if self.strategy not in {"ddp", "fsdp", "zero2"}:
            raise ValueError("strategy must be one of 'ddp', 'fsdp', or 'zero2'.")
        if self.bucket_cap_mb is not None and self.bucket_cap_mb <= 0:
            raise ValueError("bucket_cap_mb must be positive when provided.")


def setup_distributed(config: DistributionStrategyConfig) -> int:
    """Initialize torch.distributed and return the local rank."""
    import torch
    import torch.distributed as dist

    local_rank = get_local_rank()
    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available in this build.")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    launched_with_torchrun = "RANK" in os.environ and "MASTER_ADDR" in os.environ
    torch.cuda.set_device(local_rank)
    if world_size <= 1 and not launched_with_torchrun:
        return local_rank
    if not dist.is_initialized():
        timeout_minutes = int(os.environ.get("MAGIC_W0_DIST_TIMEOUT_MINUTES", "30"))
        init_kwargs = dict(
            backend=config.backend,
            timeout=timedelta(minutes=timeout_minutes),
        )
        if config.backend == "nccl":
            init_kwargs["device_id"] = torch.device("cuda", local_rank)
        dist.init_process_group(**init_kwargs)
    return local_rank


def cleanup_distributed() -> None:
    """Destroy the default process group if initialized."""
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def get_local_rank() -> int:
    """Return the local device rank assigned by torchrun."""
    return int(os.environ.get("LOCAL_RANK", 0))


def get_rank() -> int:
    """Return the current distributed rank."""
    try:
        import torch.distributed as dist
    except ImportError:
        return 0
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return int(os.environ.get("RANK", 0))


def get_world_size() -> int:
    """Return the distributed world size."""
    try:
        import torch.distributed as dist
    except ImportError:
        return 1
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return int(os.environ.get("WORLD_SIZE", 1))


def is_main_process() -> bool:
    """Return whether this process should emit user-facing logs."""
    return get_rank() == 0


def barrier() -> None:
    """Synchronize all ranks."""
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        if dist.get_backend() == "nccl":
            dist.barrier(device_ids=[get_local_rank()])
        else:
            dist.barrier()


def all_reduce(tensor, op=None):
    """All-reduce a tensor across all ranks."""
    import torch.distributed as dist

    if op is None:
        op = dist.ReduceOp.SUM
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor, op=op)
    return tensor
