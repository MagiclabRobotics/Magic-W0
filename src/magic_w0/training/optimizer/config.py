"""Configuration and parameter grouping for trainer optimizers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import torch

from .muon import Muon
from .utils import AdamWConfig, build_adamw

# 2D weights whose name contains any of these use AdamW, not Muon. Beyond the usual
# input/output layers (embeddings, lm_head, readout heads) and gains (norm), this also
# excludes DiT-style AdaLN conditioning (`modulation`) and the flow-matching action head
# (`output_projection`): these are zero-initialized gain/head projections, not the hidden
# feature-mixing matrices Muon's orthogonalized update is meant for. Sending them through
# Muon both fights the zero-init warm-up and puts them on the Muon LR (~200x the AdamW
# fallback LR). Attention out-projections are named `out_proj` and are NOT matched here.
_ADAMW_NAME_HINTS = (
    "embed",
    "embedding",
    "lm_head",
    "action_head",
    "readout",
    "norm",
    "wte",
    "wpe",
    "modulation",
    "output_projection",
)


@dataclass(frozen=True, slots=True)
class MuonConfig:
    """Muon for hidden matrices and AdamW for all remaining parameters."""

    lr: float = 0.02
    adamw_lr: float = 3e-4
    momentum: float = 0.95
    nesterov: bool = True
    ns_steps: int = 5
    ns_batch_size: int = 1
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    weight_decay: float = 0.0
    vlm_lr_scale: float = 1.0
    vlm_name_hint: str = "paligemma"
    # 0 = off (default, unchanged behaviour). Set to AdamW's observed update RMS (~0.2) to
    # pin every matrix's update RMS to that value instead of the shape-dependent
    # 1/sqrt(fan_in) Keller Jordan scaling leaves. Only with matching on is `lr == adamw_lr`
    # (1:1) the principled setting -- unmatched, the RMS-equivalent ratio varies several-fold
    # across matrix widths, so no single ratio suits every layer.
    update_rms: float = 0.0
    # Spread Newton-Schulz across DDP ranks instead of recomputing it identically on all of
    # them. Bit-identical updates, at the cost of one SUM all-reduce of the matrix group per
    # step. A no-op outside a >1-rank distributed job. Requires DDP (not FSDP), which the
    # trainer already enforces for Muon.
    shard_ns: bool = False
    shard_overlap: int = 1
    shard_bucket_mib: int = 256
    # Partition optimizer state across DDP ranks while keeping parameters and gradients
    # replicated. This preserves Muon's matrix shapes and is the compatibility step toward
    # ZeRO-2; full gradient partitioning requires a Muon-aware reduction path as well.
    shard_optimizer_state: bool = False

    def __post_init__(self) -> None:
        if self.lr <= 0 or self.adamw_lr <= 0:
            raise ValueError("Muon learning rates must be positive")
        if self.ns_steps <= 0:
            raise ValueError("Muon ns_steps must be positive")
        if self.ns_batch_size <= 0:
            raise ValueError("Muon ns_batch_size must be positive")
        if self.vlm_lr_scale <= 0:
            raise ValueError("Muon vlm_lr_scale must be positive")
        if self.update_rms < 0:
            raise ValueError("Muon update_rms must be >= 0 (0 disables RMS matching)")
        if self.shard_overlap < 0:
            raise ValueError("Muon shard_overlap must be >= 0")
        if self.shard_bucket_mib <= 0:
            raise ValueError("Muon shard_bucket_mib must be positive")


OptimizerConfig = AdamWConfig | MuonConfig


def optimizer_config_from_dict(
    payload: dict[str, Any] | None,
    *,
    default_learning_rate: float,
    default_weight_decay: float,
) -> OptimizerConfig:
    """Build an optimizer config from the YAML ``training.optimizer`` block."""
    raw = dict(payload or {})
    optimizer_type = str(raw.pop("type", "adamw")).lower()
    if "betas" in raw:
        raw["betas"] = tuple(raw["betas"])
    if optimizer_type == "adamw":
        if "lr" in raw:
            raw["learning_rate"] = raw.pop("lr")
        raw.setdefault("learning_rate", default_learning_rate)
        raw.setdefault("weight_decay", default_weight_decay)
        return AdamWConfig(**raw)
    if optimizer_type == "muon":
        if "learning_rate" in raw:
            raw["lr"] = raw.pop("learning_rate")
        raw.setdefault("lr", default_learning_rate)
        raw.setdefault("weight_decay", default_weight_decay)
        return MuonConfig(**raw)
    raise ValueError("training.optimizer.type must be 'adamw' or 'muon'")


def _expected_muon_state_bytes(parameter, *, use_muon: bool) -> int:
    """Estimate the persistent owner-local state created by Muon."""
    numel = parameter.numel()
    master_bytes = (
        4 * numel if parameter.dtype in (torch.bfloat16, torch.float16) else 0
    )
    update_state_bytes = parameter.element_size() * numel if use_muon else 2 * 4 * numel
    return master_bytes + update_state_bytes


def _plan_muon_state_shards(param_groups, world_size: int):
    """Greedily balance whole parameters by expected optimizer-state bytes."""
    if world_size < 1:
        raise ValueError("world_size must be positive")
    entries = []
    for group_index, group in enumerate(param_groups):
        use_muon = bool(group.get("use_muon", False))
        for parameter_index, parameter in enumerate(group["params"]):
            entries.append(
                (
                    _expected_muon_state_bytes(parameter, use_muon=use_muon),
                    group_index,
                    parameter_index,
                    parameter,
                )
            )
    entries.sort(key=lambda entry: (-entry[0], entry[1], entry[2]))
    loads = [0] * world_size
    owners = {}
    for state_bytes, _group_index, _parameter_index, parameter in entries:
        owner = min(range(world_size), key=lambda rank: (loads[rank], rank))
        owners[parameter] = owner
        loads[owner] += state_bytes
    return owners


def build_optimizer(
    named_parameters: Mapping[str, Any] | Iterable[tuple[str, Any]],
    config: OptimizerConfig,
):
    """Build the configured optimizer from stable, unwrapped parameter names."""
    entries = list(
        named_parameters.items()
        if isinstance(named_parameters, Mapping)
        else named_parameters
    )
    entries = [
        (name, parameter) for name, parameter in entries if parameter.requires_grad
    ]
    if not entries:
        raise ValueError("No trainable parameters were found for the optimizer.")
    if isinstance(config, AdamWConfig):
        return build_adamw((parameter for _, parameter in entries), config)

    buckets: dict[tuple[bool, float], list] = {}
    for name, parameter in entries:
        lower_name = name.lower()
        use_muon = parameter.ndim == 2 and not any(
            hint in lower_name for hint in _ADAMW_NAME_HINTS
        )
        scale = (
            config.vlm_lr_scale if config.vlm_name_hint.lower() in lower_name else 1.0
        )
        buckets.setdefault((use_muon, scale), []).append(parameter)

    groups = []
    for (use_muon, scale), parameters in sorted(buckets.items()):
        base_lr = config.lr if use_muon else config.adamw_lr
        groups.append(
            {
                "params": parameters,
                "use_muon": use_muon,
                "lr": base_lr * scale,
                "weight_decay": config.weight_decay,
            }
        )
    muon_defaults = {
        "lr": config.lr,
        "momentum": config.momentum,
        "nesterov": config.nesterov,
        "ns_steps": config.ns_steps,
        "ns_batch_size": config.ns_batch_size,
        "adamw_betas": config.betas,
        "adamw_eps": config.eps,
        "weight_decay": config.weight_decay,
        "update_rms": config.update_rms,
        "shard_ns": config.shard_ns,
        "shard_overlap": config.shard_overlap,
        "shard_bucket_mib": config.shard_bucket_mib,
    }
    if config.shard_optimizer_state:
        import torch.distributed as dist
        from torch.distributed.optim import ZeroRedundancyOptimizer

        if not (dist.is_available() and dist.is_initialized()):
            raise ValueError(
                "shard_optimizer_state requires an initialized process group"
            )
        if config.shard_ns:
            raise ValueError(
                "shard_optimizer_state and shard_ns cannot be enabled together"
            )

        class StateAwareZeroRedundancyOptimizer(ZeroRedundancyOptimizer):
            def _partition_parameters(self, params_per_rank=None):
                if params_per_rank is not None:
                    return super()._partition_parameters(params_per_rank)
                if not self._partition_parameters_cache:
                    owners = _plan_muon_state_shards(self.param_groups, self.world_size)
                    self._partition_parameters_cache = [
                        [] for _ in range(self.world_size)
                    ]
                    for group in self.param_groups:
                        params_by_rank = [[] for _ in range(self.world_size)]
                        for parameter in group["params"]:
                            params_by_rank[owners[parameter]].append(parameter)
                        for rank, parameters in enumerate(params_by_rank):
                            rank_group = dict(group)
                            rank_group["params"] = parameters
                            self._partition_parameters_cache[rank].append(rank_group)
                return self._partition_parameters_cache

        return StateAwareZeroRedundancyOptimizer(
            groups,
            optimizer_class=Muon,
            parameters_as_bucket_view=True,
            **muon_defaults,
        )
    return Muon(groups, **muon_defaults)


__all__ = [
    "MuonConfig",
    "OptimizerConfig",
    "_expected_muon_state_bytes",
    "_plan_muon_state_shards",
    "build_optimizer",
    "optimizer_config_from_dict",
]
