"""DDP/FSDP trainer implementation."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from magic_w0.data.retry import (
    DATA_RETRY_ATTEMPTS,
    DATA_RETRY_REPLACED,
    preserve_retry_metadata,
)
from magic_w0.models.base import ModelForwardOutput
from magic_w0.training import host_memory
from magic_w0.training.hooks import EvalHook, Hook, HookContext, HookRunner
from magic_w0.training.loss_monitor import LossMonitor
from magic_w0.training.optimizer import MuonConfig, OptimizerConfig
from magic_w0.training.scheduler import SchedulerConfig
from magic_w0.training.utils import move_to_device

from .base_trainer import DistributedTrainerBase
from .checkpoint_io import save_checkpoint
from .distribution_strategy import (
    DistributionStrategyConfig,
    all_reduce,
    barrier,
    get_rank,
    get_world_size,
    is_main_process,
)
from .from_noise_plot import render_per_horizon_png
from .metrics_logger import MetricsLogger, WandbConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _CheckpointTensorRef:
    index: int
    shape: tuple[int, ...]
    dtype: Any


# Loss components broken out per modality during validation. A constant, not
# whatever a rank's batches happened to report: the reduction buffer's width is
# part of a collective, and a width that varies by rank hangs the job.
_VAL_METRIC_KEYS: tuple[str, ...] = (
    "loss",
    "flow_loss",
    "fast_loss",
    "subtask_ce",
    "vlm_ce",
    "gen_2d_loss",
    "geometry_3d_loss",
    "motion_3d_loss",
    "world_3d_loss",
)

_ACTION_WORLD_STREAM_MASK_LOG_RECORD = "_action_world_stream_mask"
_ACTION_WORLD_MASK_MODES: tuple[str, ...] = (
    "keep_both",
    "mask_2d",
    "mask_3d",
)
_ACTION_WORLD_MASK_LOSS_KEYS: tuple[str, ...] = (
    "flow_loss_keep_both",
    "flow_loss_mask_2d",
    "flow_loss_mask_3d",
)
_ACTION_WORLD_MASK_RATIO_KEYS: tuple[str, ...] = (
    "world_mask_keep_ratio",
    "world_mask_2d_ratio",
    "world_mask_3d_ratio",
)
_W0_ACTION_WORLD_MASK_MODES: tuple[str, ...] = (
    "keep_both",
    "mask_geometry",
    "mask_motion",
    "mask_dino",
)
_W0_ACTION_WORLD_MASK_LOSS_KEYS: tuple[str, ...] = (
    "flow_loss_keep_both",
    "flow_loss_mask_geometry",
    "flow_loss_mask_motion",
    "flow_loss_mask_dino",
)
_W0_ACTION_WORLD_MASK_RATIO_KEYS: tuple[str, ...] = (
    "world_mask_keep_ratio",
    "world_mask_geometry_ratio",
    "world_mask_motion_ratio",
    "world_mask_dino_ratio",
)

# A branch-gradient snapshot is taken after backward and before clipping.  Keep
# both the legacy MagicW0 names and W0's current Geometry/Motion/DINO names here;
# checking only the old names silently disabled all four W0 branch metrics.
_BRANCH_GRADIENT_TRIGGER_KEYS: frozenset[str] = frozenset(
    {
        "gen_2d_loss",
        "world_3d_loss",
        "geometry_loss",
        "motion_flow_loss",
        "dino_flow_loss",
    }
)


def _should_measure_branch_gradients(model_metrics: Mapping[str, object]) -> bool:
    return bool(_BRANCH_GRADIENT_TRIGGER_KEYS.intersection(model_metrics))


# W0 teacher/data-quality counts are accumulated for every optimizer step and
# persisted as one cumulative sidecar instead of becoming per-step W&B curves.
_DATA_QUALITY_COUNT_KEYS: tuple[str, ...] = (
    "teacher_retry_rows",
    "geometry_raw_nonfinite_elements",
    "motion_raw_nonfinite_elements",
    "dino_raw_nonfinite_elements",
    "geometry_recovered_rows",
    "motion_recovered_rows",
    "geometry_dropped_rows",
    "motion_dropped_rows",
    "dino_invalid_tokens",
)


def _accumulate_action_world_mask_window(
    totals: list[float],
    model_metrics: Mapping[str, object],
    *,
    modes: tuple[str, ...] = _ACTION_WORLD_MASK_MODES,
) -> None:
    """Add one local robotics step to its mask-mode logging bucket."""
    record = model_metrics.get(_ACTION_WORLD_STREAM_MASK_LOG_RECORD)
    if not isinstance(record, Mapping):
        return
    mode = record.get("mode")
    if mode not in modes:
        raise ValueError(f"unknown action/world stream mask log mode {mode!r}")
    sample_count = int(record.get("sample_count", 0))
    if sample_count < 0:
        raise ValueError("action/world stream mask sample_count must be non-negative")
    if sample_count == 0:
        return
    flow_loss = model_metrics.get("flow_loss")
    if not isinstance(flow_loss, (int, float)) or isinstance(flow_loss, bool):
        raise TypeError("action/world stream mask logging requires scalar flow_loss")
    index = modes.index(str(mode))
    totals[index] += float(flow_loss) * sample_count
    totals[len(modes) + index] += sample_count


def _action_world_mask_metrics_from_totals(
    totals: Sequence[float],
    *,
    modes: tuple[str, ...] = _ACTION_WORLD_MASK_MODES,
    loss_keys: tuple[str, ...] = _ACTION_WORLD_MASK_LOSS_KEYS,
    ratio_keys: tuple[str, ...] = _ACTION_WORLD_MASK_RATIO_KEYS,
) -> dict[str, float]:
    """Build global window metrics; loss for an empty bucket stays absent."""
    size = len(modes)
    if len(loss_keys) != size or len(ratio_keys) != size:
        raise ValueError("action/world mask metric layout sizes differ")
    if len(totals) != size * 2:
        raise ValueError(f"expected {size * 2} action/world mask totals")
    counts = [float(value) for value in totals[size:]]
    total_count = sum(counts)
    if total_count <= 0:
        return {}
    metrics = {
        key: round(count / total_count, 6)
        for key, count in zip(ratio_keys, counts, strict=True)
    }
    for key, loss_sum, count in zip(
        loss_keys,
        totals[:size],
        counts,
        strict=True,
    ):
        if count > 0:
            metrics[key] = round(float(loss_sum) / count, 6)
    return metrics


def _action_world_mask_metric_layout(model_config):
    """Select the legacy three-mode or W0 four-mode logging contract."""
    mask_config = getattr(model_config, "action_world_stream_mask", None)
    is_w0 = all(
        hasattr(mask_config, name)
        for name in (
            "mask_geometry_prob",
            "mask_motion_prob",
            "mask_dino_prob",
        )
    )
    if is_w0:
        return (
            _W0_ACTION_WORLD_MASK_MODES,
            _W0_ACTION_WORLD_MASK_LOSS_KEYS,
            _W0_ACTION_WORLD_MASK_RATIO_KEYS,
        )
    return (
        _ACTION_WORLD_MASK_MODES,
        _ACTION_WORLD_MASK_LOSS_KEYS,
        _ACTION_WORLD_MASK_RATIO_KEYS,
    )


def _accumulate_data_quality_window(
    totals: list[int],
    nonzero_rank_steps: list[int],
    model_metrics: Mapping[str, object],
) -> bool:
    """Accumulate all local W0 quality events between distributed flushes."""
    if len(totals) != len(_DATA_QUALITY_COUNT_KEYS) or len(nonzero_rank_steps) != len(
        _DATA_QUALITY_COUNT_KEYS
    ):
        raise ValueError("data-quality accumulator has an invalid size")
    seen = False
    for index, key in enumerate(_DATA_QUALITY_COUNT_KEYS):
        value = model_metrics.get(key)
        if value is None:
            continue
        seen = True
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise TypeError(f"data-quality metric {key} must be numeric")
        count = int(value)
        if count < 0 or float(count) != float(value):
            raise ValueError(
                f"data-quality metric {key} must be a non-negative integer"
            )
        totals[index] += count
        nonzero_rank_steps[index] += int(count > 0)
    return seen


def _data_config_fingerprint(run_config: dict | None) -> str | None:
    data_config = (run_config or {}).get("data")
    if data_config is None:
        return None
    # ``eval_on_train`` only creates/removes an independently shuffled evaluation
    # loader over the same training view. It does not alter the train sampler,
    # source mixture, rank shards, or sample order, so toggling it must not make an
    # otherwise data-exact training resume fail. Validation cursor compatibility is
    # handled separately while loading the rank-local state below.
    data_config = dict(data_config)
    data_config.pop("eval_on_train", None)
    encoded = json.dumps(
        data_config, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _resolve_num_epochs(num_epochs: int | None, max_steps: int | None) -> int | None:
    if num_epochs is not None and num_epochs <= 0:
        raise ValueError("num_epochs must be positive when set")
    if max_steps is not None and max_steps <= 0:
        raise ValueError("max_steps must be positive when set")
    if num_epochs is not None:
        return num_epochs
    return None if max_steps is not None else 1


class DistributedTrainer(DistributedTrainerBase):
    """Model-driven trainer for torchrun-launched distributed execution."""

    def __init__(
        self,
        model,
        *,
        strategy_config: DistributionStrategyConfig | None = None,
        train_dataloader: Iterable | None = None,
        val_dataloader: Iterable | None = None,
        output_dir: str | Path | None = None,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        grad_clip_norm: float = 1.0,
        scheduler_config: SchedulerConfig | None = None,
        optimizer_config: OptimizerConfig | None = None,
        num_epochs: int | None = None,
        max_steps: int | None = None,
        log_interval: int = 10,
        save_every_epochs: int | None = 1,
        save_every_steps: int | None = None,
        keep_last_n_checkpoints: int | None = None,
        save_milestone_every_steps: int | None = None,
        find_unused_parameters: bool = True,
        hooks: Sequence[Hook] | None = None,
        eval_every_steps: int = 0,
        eval_steps: int = 20,
        validation_steps: int = 20,
        validate_on_checkpoint: bool = True,
        from_noise_batches: int = 0,
        from_noise_steps: int | None = None,
        loss_ema_decay: float = 0.95,
        low_overhead_profile: bool = False,
        seed: int = 0,
        resume_from: str | Path | None = None,
        resume_allow_data_reset: bool = False,
        resume_allow_optimizer_reset: bool = False,
        wandb_config: WandbConfig | None = None,
        run_config: dict | None = None,
    ) -> None:
        self.initial_model = model
        self.resume_from = resume_from
        # Opt-in escape hatch for changing the data mixture or distributed topology
        # between stages. Compatible checkpoints still restore their exact cursor.
        self.resume_allow_data_reset = bool(resume_allow_data_reset)
        # A staged fine-tune may deliberately freeze or unfreeze parameters at a
        # checkpoint boundary. That changes optimizer parameter groups, so the
        # old optimizer state is structurally incompatible even though model
        # weights, global step, data cursor, and LR progress remain reusable.
        self.resume_allow_optimizer_reset = bool(resume_allow_optimizer_reset)
        self._resume_path: Path | None = None
        self.initial_train_dataloader = train_dataloader
        self.initial_val_dataloader = val_dataloader
        self._data_config_fingerprint = _data_config_fingerprint(run_config)
        self._norm_metadata: dict | None = None
        # Python-object collectives on an NCCL group stage their serialized
        # tensors on CUDA.  Checkpointing can be the first gather in the job,
        # which then lazily allocates NCCL P2P/channel buffers exactly when a
        # large model is already near the device limit. Keep a dedicated CPU
        # group for optimizer shards and rank-local loader cursors, and reuse it.
        self._checkpoint_gloo_group = None
        self._data_retry_attempts_run_total = 0
        self._data_replaced_samples_run_total = 0
        (
            self._action_world_mask_modes,
            self._action_world_mask_loss_keys,
            self._action_world_mask_ratio_keys,
        ) = _action_world_mask_metric_layout(getattr(model, "config", None))
        self._action_world_mask_log_window = [0.0] * (
            2 * len(self._action_world_mask_modes)
        )
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.grad_clip_norm = grad_clip_norm
        self.scheduler_config = scheduler_config
        self.optimizer_config = optimizer_config
        self._num_epochs = _resolve_num_epochs(num_epochs, max_steps)
        self.max_steps = max_steps
        self.log_interval = log_interval
        self.save_every_epochs = save_every_epochs
        if save_every_steps is not None and save_every_steps <= 0:
            raise ValueError("save_every_steps must be positive when set")
        self.save_every_steps = save_every_steps
        if keep_last_n_checkpoints is not None and keep_last_n_checkpoints <= 0:
            raise ValueError("keep_last_n_checkpoints must be positive when set")
        self.keep_last_n_checkpoints = keep_last_n_checkpoints
        if save_milestone_every_steps is not None and save_milestone_every_steps <= 0:
            raise ValueError("save_milestone_every_steps must be positive when set")
        self.save_milestone_every_steps = save_milestone_every_steps
        self.find_unused_parameters = find_unused_parameters
        if validation_steps <= 0:
            raise ValueError("validation_steps must be positive")
        self.validation_steps = validation_steps
        # Validation rides the checkpoint cadence: an eval number is only useful
        # next to the weights that produced it, and pairing them means one knob
        # instead of two that silently drift apart.
        self.validate_on_checkpoint = bool(validate_on_checkpoint)
        if from_noise_batches < 0:
            raise ValueError("from_noise_batches must be non-negative")
        if from_noise_steps is not None and from_noise_steps <= 0:
            raise ValueError("from_noise_steps must be positive when set")
        self.from_noise_batches = int(from_noise_batches)
        self.from_noise_steps = from_noise_steps
        # (global_step, per-horizon MSE array) on rank 0, for the end-of-run figure.
        self._from_noise_history: list = []
        resolved_hooks = list(hooks or ())
        if eval_every_steps > 0:
            resolved_hooks.append(
                EvalHook(every_steps=eval_every_steps, eval_steps=eval_steps)
            )
        self.hook_runner = HookRunner(
            resolved_hooks,
            sync_decision=self._sync_hook_decision,
        )
        self.loss_monitor = LossMonitor(ema_decay=loss_ema_decay)
        self.low_overhead_profile = bool(low_overhead_profile)
        self.metrics_logger = MetricsLogger(
            output_dir=output_dir,
            wandb_config=wandb_config,
            run_config=run_config,
        )
        super().__init__(
            strategy_config=strategy_config or DistributionStrategyConfig(),
            seed=seed,
        )

    @property
    def num_epochs(self) -> int | None:
        return self._num_epochs

    def fit(self) -> None:
        """Run the full distributed training lifecycle."""
        exit_code = 0
        hooks_started = False
        try:
            self.load_model()
            self.load_dataset()
            barrier()
            self.hook_runner.train_start(self._hook_context())
            hooks_started = True
            epoch = 0
            while self.num_epochs is None or epoch < self.num_epochs:
                self.state.epoch = epoch
                epoch_start_step = self.state.global_step
                self.train_loop(epoch)
                self._maybe_run_epoch_validation()
                if (
                    self.save_every_epochs is not None
                    and self.save_every_epochs > 0
                    and (epoch + 1) % self.save_every_epochs == 0
                ):
                    self.save_checkpoint(epoch=epoch, step=self.state.global_step)
                barrier()
                if self._reached_max_steps():
                    break
                if (
                    self.num_epochs is None
                    and self.state.global_step == epoch_start_step
                ):
                    raise RuntimeError(
                        "training epoch produced no optimizer steps; cannot reach max_steps"
                    )
                epoch += 1
        except BaseException:
            exit_code = 1
            raise
        finally:
            try:
                if hooks_started:
                    self.hook_runner.train_end(self._hook_context())
            finally:
                # Written even on the failure path: a run that died at step 40k
                # still measured 39k steps' worth of open-loop error, and that is
                # exactly when someone wants to look at the curve.
                if is_main_process():
                    self._write_from_noise_figure()
                self.metrics_logger.close(exit_code=exit_code)

    def load_model(self) -> None:
        """Create, move, wrap, and optimize the model."""
        import torch
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.nn.parallel import DistributedDataParallel as DDP

        from .base_trainer import _Zero2GradientOverlap

        model = self.initial_model
        if model is None:
            raise ValueError("DistributedTrainer requires a model instance.")
        if not isinstance(model, torch.nn.Module):
            raise TypeError("DistributedTrainer model must be a torch.nn.Module.")
        model.to(self.device)
        zero2 = self.strategy_config.strategy == "zero2"
        self._zero2_gradient_overlap = None
        if zero2:
            if not isinstance(self.optimizer_config, MuonConfig):
                raise ValueError("zero2 currently requires MuonConfig")
            if not self.optimizer_config.shard_optimizer_state:
                raise ValueError("zero2 requires optimizer.shard_optimizer_state: true")

        # Restore weights before DDP construction or the explicit ZeRO-2 broadcast;
        # optimizer state follows configure_optimizer(), once parameter groups exist.
        resume_payload = self._read_resume_payload()
        if resume_payload is not None:
            self._restore_model_state(model, resume_payload)

        if get_world_size() <= 1:
            self.model = model
        elif self.strategy_config.strategy == "fsdp":
            if isinstance(self.optimizer_config, MuonConfig):
                raise ValueError(
                    "Muon currently requires DDP because FSDP flattens the named matrix parameters"
                )
            self.model = FSDP(model, device_id=self.device)
        elif self.strategy_config.strategy == "ddp":
            ddp_kwargs = {}
            if self.strategy_config.bucket_cap_mb is not None:
                ddp_kwargs["bucket_cap_mb"] = self.strategy_config.bucket_cap_mb
            self.model = DDP(
                model,
                device_ids=[self.local_rank],
                find_unused_parameters=self.find_unused_parameters,
                gradient_as_bucket_view=self.strategy_config.gradient_as_bucket_view,
                **ddp_kwargs,
            )
        else:
            self.model = model
            with torch.no_grad():
                for parameter in model.parameters():
                    torch.distributed.broadcast(parameter, src=0)
        self.configure_optimizer(
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            named_parameters=model.named_parameters(),
            optimizer_config=self.optimizer_config,
            scheduler_config=self.scheduler_config,
        )
        if (
            zero2
            and get_world_size() > 1
            and self.strategy_config.zero2_overlap_backward
        ):
            self._zero2_gradient_overlap = _Zero2GradientOverlap(
                self.optimizer,
                rank=self.rank,
                world_size=self.world_size,
            )
        if resume_payload is not None:
            self._restore_training_state(resume_payload)

    def _read_resume_payload(self) -> dict | None:
        """Read the checkpoint named by ``resume_from``, or None when not resuming.

        Every rank reads the same file rather than having rank 0 broadcast it: the
        file is on shared storage, and a rank-local read keeps this independent of
        whether the process group is up yet.
        """
        if self.resume_from is None:
            return None
        from .checkpoint_io import load_checkpoint, resolve_checkpoint

        self._resume_path = resolve_checkpoint(self.resume_from)
        return load_checkpoint(
            self._resume_path, map_location="cpu", weights_only=False
        )

    def _restore_model_state(self, model, payload: dict) -> None:
        from magic_w0.models.world_action.configuration_magicw0 import (
            MagicW0TrainingConfig,
        )

        if isinstance(getattr(model, "config", None), MagicW0TrainingConfig):
            from magic_w0.training.resume import validate_resume_model_config

            validate_resume_model_config(payload.get("config"), model.config)
        state = payload.get("model")
        if state is None:
            raise ValueError(
                f"{self._resume_path} carries no 'model' state to resume from"
            )
        # strict on purpose: a key mismatch means the checkpoint was written by a
        # different architecture, and silently starting half-random is far worse than
        # failing at step 0.
        model.load_state_dict(state)

    def _restore_training_state(self, payload: dict) -> None:
        """Restore optimizer, LR schedule and step counter from a resume payload."""
        optimizer_state = payload.get("optimizer")
        optimizer_file = payload.get("optimizer_file")
        optimizer_restored = False
        optimizer_reset = False
        if optimizer_file is not None and self.optimizer is not None:
            if self._resume_path is None:
                raise ValueError("external optimizer resume requires a checkpoint path")
            from .checkpoint_io import load_checkpoint

            optimizer_state = load_checkpoint(
                self._resume_path.parent / optimizer_file,
                map_location="cpu",
                weights_only=False,
            )
        if optimizer_state is not None and self.optimizer is not None:
            saved_group_sizes = tuple(
                len(group["params"]) for group in optimizer_state["param_groups"]
            )
            current_group_sizes = tuple(
                len(group["params"]) for group in self.optimizer.param_groups
            )
            groups_compatible = saved_group_sizes == current_group_sizes
            if not groups_compatible and self.resume_allow_optimizer_reset:
                # Freezing/unfreezing at a stage boundary legitimately changes
                # optimizer groups. Keep the freshly constructed optimizer and
                # replay the LR schedule below from the restored global step.
                optimizer_reset = True
            else:
                # Raises on incompatible groups unless the explicit fallback above
                # was enabled. Other optimizer incompatibilities must stay fatal.
                self.optimizer.load_state_dict(optimizer_state)
                optimizer_restored = True

        step = int(payload.get("step", 0))
        scheduler_state = payload.get("scheduler")
        if self.lr_scheduler is not None:
            if scheduler_state is not None and not optimizer_reset:
                self.lr_scheduler.load_state_dict(scheduler_state)
            elif step > 0:
                # Checkpoints written before scheduler state was included: replay the
                # schedule so the LR resumes where it left off instead of restarting
                # warmup at ~0 and re-shocking a converged model. Pure arithmetic --
                # no forward pass, no optimizer step, which is exactly why torch's
                # "step() called before optimizer.step()" warning does not apply here.
                import warnings

                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore", message=r".*before `optimizer\.step\(\)`.*"
                    )
                    for _ in range(step):
                        self.lr_scheduler.step()

        self.state.global_step = step
        self._resume_data_state_reset = False
        data_state_restored = self._restore_data_state(payload)

        saved_world = (payload.get("extra_state") or {}).get("world_size")
        self.metrics_logger.log(
            {
                "event": "resume",
                "path": str(self._resume_path),
                "global_step": step,
                "optimizer_restored": optimizer_restored,
                "optimizer_reset": optimizer_reset,
                # None means the schedule was replayed rather than loaded.
                "scheduler_restored": (
                    scheduler_state is not None and not optimizer_reset
                ),
                "saved_world_size": saved_world,
                "world_size": get_world_size(),
                "data_state_restored": data_state_restored,
                "data_state_reset": bool(self._resume_data_state_reset),
            }
        )

    def _optimizer_parameter_names(self, optimizer) -> list[list[str]]:
        model = self.initial_model
        if model is None:
            raise ValueError("optimizer state restore requires the initial model")
        name_by_parameter = {
            id(parameter): name for name, parameter in model.named_parameters()
        }
        try:
            return [
                [name_by_parameter[id(parameter)] for parameter in group["params"]]
                for group in optimizer.param_groups
            ]
        except KeyError as exc:
            raise ValueError(
                "optimizer contains a parameter that is not present in the model"
            ) from exc

    def _local_data_state(self) -> dict | None:
        """Capture only consumed sample cursors; prefetched batches are excluded."""
        loaders = {
            "train": getattr(self, "initial_train_dataloader", None),
            "validation": getattr(self, "initial_val_dataloader", None),
        }
        state: dict[str, object] = {"rank": get_rank()}
        for name, loader in loaders.items():
            state_dict = getattr(loader, "state_dict", None)
            if state_dict is not None:
                state[name] = state_dict()
        return state if len(state) > 1 else None

    def _checkpoint_cpu_group(self):
        """Return one cached all-rank Gloo group for checkpoint object traffic."""
        import torch.distributed as dist

        if not (dist.is_available() and dist.is_initialized()):
            return None
        checkpoint_group = getattr(self, "_checkpoint_gloo_group", None)
        if checkpoint_group is None:
            if not dist.is_gloo_available():
                raise RuntimeError(
                    "distributed checkpoint gathering requires the Gloo backend "
                    "so optimizer and DataLoader state never stage on CUDA"
                )
            checkpoint_group = dist.new_group(backend="gloo")
            self._checkpoint_gloo_group = checkpoint_group
            if is_main_process():
                logger.info(
                    "checkpoint state will gather over a dedicated Gloo/CPU "
                    "process group"
                )
        return checkpoint_group

    @staticmethod
    def _move_checkpoint_state_to_cpu(value):
        import torch

        if torch.is_tensor(value):
            return value.detach().cpu()
        if isinstance(value, dict):
            return {
                key: DistributedTrainer._move_checkpoint_state_to_cpu(item)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [
                DistributedTrainer._move_checkpoint_state_to_cpu(item) for item in value
            ]
        if isinstance(value, tuple):
            return tuple(
                DistributedTrainer._move_checkpoint_state_to_cpu(item) for item in value
            )
        return value

    @staticmethod
    def _extract_checkpoint_tensors(value) -> tuple[Any, list]:
        """Replace tensors with lightweight metadata for object transport."""
        import torch

        tensors = []

        def extract(item):
            if torch.is_tensor(item):
                tensor = item.detach().cpu().contiguous()
                reference = _CheckpointTensorRef(
                    index=len(tensors),
                    shape=tuple(tensor.shape),
                    dtype=tensor.dtype,
                )
                tensors.append(tensor)
                return reference
            if isinstance(item, dict):
                return {key: extract(child) for key, child in item.items()}
            if isinstance(item, list):
                return [extract(child) for child in item]
            if isinstance(item, tuple):
                return tuple(extract(child) for child in item)
            return item

        return extract(value), tensors

    @staticmethod
    def _restore_checkpoint_tensors(value, tensors: list):
        if isinstance(value, _CheckpointTensorRef):
            return tensors[value.index]
        if isinstance(value, dict):
            return {
                key: DistributedTrainer._restore_checkpoint_tensors(child, tensors)
                for key, child in value.items()
            }
        if isinstance(value, list):
            return [
                DistributedTrainer._restore_checkpoint_tensors(child, tensors)
                for child in value
            ]
        if isinstance(value, tuple):
            return tuple(
                DistributedTrainer._restore_checkpoint_tensors(child, tensors)
                for child in value
            )
        return value

    @staticmethod
    def _checkpoint_tensor_references(value) -> list[_CheckpointTensorRef]:
        references = []

        def collect(item) -> None:
            if isinstance(item, _CheckpointTensorRef):
                references.append(item)
            elif isinstance(item, dict):
                for child in item.values():
                    collect(child)
            elif isinstance(item, (list, tuple)):
                for child in item:
                    collect(child)

        collect(value)
        references.sort(key=lambda reference: reference.index)
        if [reference.index for reference in references] != list(
            range(len(references))
        ):
            raise ValueError("checkpoint tensor metadata has invalid indices")
        return references

    @staticmethod
    def _transfer_checkpoint_tensors(
        tensors: list,
        *,
        peer: int,
        group,
        sending: bool,
    ) -> None:
        import torch.distributed as dist

        operation = dist.isend if sending else dist.irecv
        for start in range(0, len(tensors), 256):
            operations = [
                dist.P2POp(
                    operation,
                    tensor,
                    peer=peer,
                    group=group,
                    tag=index,
                )
                for index, tensor in enumerate(
                    tensors[start : start + 256], start=start
                )
            ]
            for request in dist.batch_isend_irecv(operations):
                request.wait()

    @staticmethod
    def _release_checkpoint_host_memory() -> None:
        """Return checkpoint-only CPU allocations to the host when possible."""
        import gc

        gc.collect()
        try:
            import ctypes

            malloc_trim = ctypes.CDLL(None).malloc_trim
            malloc_trim.argtypes = [ctypes.c_size_t]
            malloc_trim.restype = ctypes.c_int
            malloc_trim(0)
        except (AttributeError, OSError):
            pass

    @staticmethod
    def _log_checkpoint_host_memory(step: int, phase: str) -> None:
        if not host_memory.probe_enabled():
            return
        snapshot = host_memory.sample()
        fields = (
            "host_rss_main_gb",
            "host_cgroup_used_gb",
            "host_cgroup_anon_gb",
            "host_cgroup_file_gb",
            "host_cgroup_headroom_gb",
            "host_cgroup_events_oom_kill_total",
        )
        details = " ".join(
            f"{name.removeprefix('host_')}={float(snapshot[name]):.3f}"
            for name in fields
            if snapshot.get(name) is not None
        )
        logger.warning(
            "checkpoint-host-memory step=%d rank=%d phase=%s %s",
            step,
            get_rank(),
            phase,
            details,
        )

    def _consolidate_zero2_optimizer_state_cpu(self) -> dict | None:
        """Stream rank-local CPU state to rank 0 and rebuild native full state."""
        import torch
        import torch.distributed as dist

        local_optimizer = self.optimizer.optim

        def build_local_payload() -> dict:
            return {
                "optimizer": self._move_checkpoint_state_to_cpu(
                    local_optimizer.state_dict()
                ),
                "parameter_names": self._optimizer_parameter_names(local_optimizer),
            }

        full_names = self._optimizer_parameter_names(self.optimizer)
        name_to_id: dict[str, int] = {}
        full_groups = []
        next_id = 0
        for names, group in zip(full_names, self.optimizer.param_groups, strict=True):
            parameter_ids = list(range(next_id, next_id + len(names)))
            next_id += len(names)
            name_to_id.update(zip(names, parameter_ids, strict=True))
            full_group = {
                key: self._move_checkpoint_state_to_cpu(value)
                for key, value in group.items()
                if key != "params"
            }
            full_group["params"] = parameter_ids
            full_groups.append(full_group)

        full_state = {}
        covered: set[str] = set()

        def merge_rank_payload(rank: int, rank_payload: dict) -> None:
            optimizer_state = rank_payload["optimizer"]
            parameter_names = rank_payload["parameter_names"]
            groups = optimizer_state["param_groups"]
            if len(parameter_names) != len(groups):
                raise ValueError(
                    f"optimizer rank {rank} has inconsistent parameter groups"
                )
            for names, group in zip(parameter_names, groups, strict=True):
                parameter_ids = group["params"]
                if len(names) != len(parameter_ids):
                    raise ValueError(
                        f"optimizer rank {rank} has inconsistent parameter names"
                    )
                for name, local_id in zip(names, parameter_ids, strict=True):
                    if name not in name_to_id:
                        raise ValueError(
                            f"optimizer rank {rank} contains unknown parameter {name}"
                        )
                    if name in covered:
                        raise ValueError(
                            f"optimizer parameter {name} is owned by multiple ranks"
                        )
                    covered.add(name)
                    if local_id in optimizer_state["state"]:
                        full_state[name_to_id[name]] = optimizer_state["state"][
                            local_id
                        ]

        world_size = get_world_size()
        rank = get_rank()
        if dist.is_available() and dist.is_initialized() and world_size > 1:
            checkpoint_group = self._checkpoint_cpu_group()
            local_payload = build_local_payload()
            if rank == 0:
                merge_rank_payload(0, local_payload)
                del local_payload
                self._release_checkpoint_host_memory()
            for source_rank in range(1, world_size):
                if rank == source_rank:
                    metadata, tensors = self._extract_checkpoint_tensors(local_payload)
                    dist.send_object_list(
                        [metadata],
                        dst=0,
                        group=checkpoint_group,
                    )
                    self._transfer_checkpoint_tensors(
                        tensors,
                        peer=0,
                        group=checkpoint_group,
                        sending=True,
                    )
                    del metadata, tensors
                    del local_payload
                    self._release_checkpoint_host_memory()
                elif rank == 0:
                    received_metadata = [None]
                    dist.recv_object_list(
                        received_metadata,
                        src=source_rank,
                        group=checkpoint_group,
                    )
                    references = self._checkpoint_tensor_references(
                        received_metadata[0]
                    )
                    received_tensors = [
                        torch.empty(
                            reference.shape,
                            dtype=reference.dtype,
                            device="cpu",
                        )
                        for reference in references
                    ]
                    self._transfer_checkpoint_tensors(
                        received_tensors,
                        peer=source_rank,
                        group=checkpoint_group,
                        sending=False,
                    )
                    received_payload = self._restore_checkpoint_tensors(
                        received_metadata[0], received_tensors
                    )
                    merge_rank_payload(source_rank, received_payload)
                    del received_metadata, references, received_tensors
                    del received_payload
                    self._release_checkpoint_host_memory()
                # Keep later ranks from serializing while the current payload is
                # still resident on the sender and rank 0.
                dist.barrier(group=checkpoint_group)
            if rank != 0:
                return None
        else:
            local_payload = build_local_payload()
            merge_rank_payload(0, local_payload)

        missing = set(name_to_id) - covered
        if missing:
            raise ValueError(
                f"optimizer ranks do not cover all parameters: {sorted(missing)[:8]}"
            )
        return {"state": full_state, "param_groups": full_groups}

    def _gather_data_state(self) -> dict | None:
        """Gather each rank's small loader cursor onto rank 0 for checkpointing."""
        local_state = self._local_data_state()

        import torch.distributed as dist

        world_size = get_world_size()
        if dist.is_available() and dist.is_initialized() and world_size > 1:
            gathered = [None] * world_size if is_main_process() else None
            dist.gather_object(
                local_state,
                gathered,
                dst=0,
                group=self._checkpoint_cpu_group(),
            )
            if not is_main_process():
                return None
            rank_states = gathered
        else:
            rank_states = [local_state]
        if all(state is None for state in rank_states):
            return None
        return {
            "version": 1,
            "world_size": world_size,
            "data_config_fingerprint": getattr(self, "_data_config_fingerprint", None),
            "ranks": rank_states,
        }

    def _restore_data_state(self, payload: dict) -> bool:
        data_state = payload.get("data_state")
        if data_state is None:
            step = int(payload.get("step", 0))
            loader = getattr(self, "initial_train_dataloader", None)
            seek_steps = getattr(loader, "seek_steps", None)
            saved_world_size = (payload.get("extra_state") or {}).get("world_size")
            if seek_steps is not None and step > 0:
                if (
                    saved_world_size is not None
                    and int(saved_world_size) != get_world_size()
                ):
                    raise ValueError(
                        "legacy data cursor reconstruction requires an unchanged "
                        f"world size: saved={saved_world_size}, "
                        f"current={get_world_size()}"
                    )
                seek_steps(step)
                if is_main_process():
                    logger.warning(
                        "resume checkpoint %s predates data_state; reconstructed the "
                        "training sample cursor from global_step=%d without decoding "
                        "skipped samples",
                        self._resume_path,
                        step,
                    )
                return True
            if self._resume_path is not None and is_main_process():
                logger.warning(
                    "resume checkpoint %s has no data_state; training samples will "
                    "restart from the beginning of their deterministic streams",
                    self._resume_path,
                )
            return False
        if int(data_state.get("version", -1)) != 1:
            raise ValueError(
                f"unsupported checkpoint data_state version: {data_state.get('version')}"
            )
        saved_world_size = int(data_state.get("world_size", -1))
        current_world_size = get_world_size()
        if saved_world_size != current_world_size:
            if getattr(self, "resume_allow_data_reset", False):
                self._resume_data_state_reset = True
                step = int(payload.get("step", 0))
                loader = getattr(self, "initial_train_dataloader", None)
                seek_steps = getattr(loader, "seek_steps", None)
                if seek_steps is not None and step > 0:
                    seek_steps(step)
                    restored = True
                else:
                    restored = False
                if self._resume_path is not None and is_main_process():
                    logger.warning(
                        "resume_allow_data_reset=True: restored model, optimizer, "
                        "scheduler, and global step from %s, but world size changed "
                        "(%d -> %d); rebuilt data streams for the new topology%s",
                        self._resume_path,
                        saved_world_size,
                        current_world_size,
                        f" at global_step={step}" if restored else " from their seeds",
                    )
                return restored
            raise ValueError(
                "data-exact resume requires an unchanged world size: "
                f"saved={saved_world_size}, current={current_world_size}"
            )
        saved_fingerprint = data_state.get("data_config_fingerprint")
        current_fingerprint = getattr(self, "_data_config_fingerprint", None)
        if (
            saved_fingerprint is not None
            and current_fingerprint is not None
            and saved_fingerprint != current_fingerprint
        ):
            if getattr(self, "resume_allow_data_reset", False):
                self._resume_data_state_reset = True
                if self._resume_path is not None and is_main_process():
                    logger.warning(
                        "resume_allow_data_reset=True: restored model, optimizer, "
                        "scheduler, and global step from %s, but the data config "
                        "changed (%s -> %s); resetting the train/validation data "
                        "streams to their configured seeds",
                        self._resume_path,
                        saved_fingerprint,
                        current_fingerprint,
                    )
                return False
            raise ValueError(
                "data-exact resume requires an unchanged data configuration: "
                f"saved_sha256={saved_fingerprint}, "
                f"current_sha256={current_fingerprint}"
            )
        rank_states = data_state.get("ranks", ())
        rank = get_rank()
        if len(rank_states) != current_world_size or rank >= len(rank_states):
            raise ValueError(
                "checkpoint data_state has incomplete rank coverage: "
                f"saved={len(rank_states)}, expected={current_world_size}"
            )
        local_state = rank_states[rank]
        if (
            not isinstance(local_state, dict)
            or int(local_state.get("rank", -1)) != rank
        ):
            raise ValueError(
                f"checkpoint data_state rank mismatch for current rank {rank}: "
                f"{local_state}"
            )
        loader_by_name = {
            "train": getattr(self, "initial_train_dataloader", None),
            "validation": getattr(self, "initial_val_dataloader", None),
        }
        for name in ("train", "validation"):
            if name not in local_state:
                continue
            loader = loader_by_name[name]
            load_state_dict = getattr(loader, "load_state_dict", None)
            if load_state_dict is None:
                if name == "validation" and loader is None:
                    # Disabling eval_on_train must not invalidate the exact train
                    # cursor. Its independent validation cursor can be discarded.
                    continue
                raise ValueError(
                    f"checkpoint contains {name} data state but the current loader "
                    "does not support load_state_dict()"
                )
            load_state_dict(local_state[name])
        return True

    def load_dataset(self) -> None:
        """Attach dataloaders supplied by the training entrypoint."""
        if self.initial_train_dataloader is None:
            raise ValueError("DistributedTrainer requires a train_dataloader.")
        self.train_dataloader = self.initial_train_dataloader
        self.val_dataloader = self.initial_val_dataloader

    def train_loop(self, epoch: int) -> None:
        """Run one distributed training epoch."""
        if self.model is None or self.train_dataloader is None:
            raise ValueError(
                "load_model() and load_dataset() must run before train_loop()."
            )
        self.model.train()
        import torch

        # MAGIC_W0_PROFILE=1 inserts CUDA syncs on sampled steps so the per-phase
        # times are real GPU times (not just async kernel-launch times). Off by
        # default -- the syncs add a little overhead.
        _profile = os.environ.get("MAGIC_W0_PROFILE") == "1"
        _profile_interval = max(
            1, int(os.environ.get("MAGIC_W0_PROFILE_INTERVAL", self.log_interval or 1))
        )
        _vlm_diagnostics = os.environ.get("MAGIC_W0_VLM_DIAGNOSTICS") == "1"
        _memory_diagnostics = (
            os.environ.get("MAGIC_W0_MEMORY_DIAGNOSTICS") == "1"
            and torch.cuda.is_available()
        )
        _strategy = getattr(getattr(self, "strategy_config", None), "strategy", "ddp")

        def _sync(enabled: bool) -> None:
            if enabled and torch.cuda.is_available():
                torch.cuda.synchronize()

        def _print_cuda_memory(label: str, details: str = "") -> None:
            if not _memory_diagnostics:
                return
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            details_suffix = f" {details}" if details else ""
            print(
                f"[cuda-memory][rank={get_rank()}][step={self.state.global_step + 1}] "
                f"{label}{details_suffix} "
                f"allocated={torch.cuda.memory_allocated() / 2**30:.3f}GiB "
                f"reserved={torch.cuda.memory_reserved() / 2**30:.3f}GiB "
                f"peak_allocated={torch.cuda.max_memory_allocated() / 2**30:.3f}GiB "
                f"peak_reserved={torch.cuda.max_memory_reserved() / 2**30:.3f}GiB "
                f"device_free={free_bytes / 2**30:.3f}GiB "
                f"device_total={total_bytes / 2**30:.3f}GiB",
                flush=True,
            )

        _iter_start = time.perf_counter()
        # These timers are intentionally reduced only with the normal log cadence.
        # A collective every optimizer step would distort the throughput we are
        # trying to measure; at a log point we still report the slowest rank, which
        # is the pace that determines real multi-node wall-clock progress.
        _log_window_start = _iter_start
        _log_window_start_step = self.state.global_step
        if torch.cuda.is_available() and self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        _log_window_t_data_max = 0.0
        _log_window_t_prep_max = 0.0
        _log_window_retry_attempts = 0
        _log_window_replaced_samples = 0
        _log_window_has_retry_metadata = False
        _data_quality_totals = [0] * len(_DATA_QUALITY_COUNT_KEYS)
        _data_quality_nonzero_rank_steps = [0] * len(_DATA_QUALITY_COUNT_KEYS)
        _data_quality_window_steps = 0
        model_config = getattr(self.initial_model, "config", None)
        if not hasattr(self, "_action_world_mask_modes"):
            (
                self._action_world_mask_modes,
                self._action_world_mask_loss_keys,
                self._action_world_mask_ratio_keys,
            ) = _action_world_mask_metric_layout(model_config)
        _action_world_mask_log_window = getattr(
            self,
            "_action_world_mask_log_window",
            [0.0] * (2 * len(self._action_world_mask_modes)),
        )
        if len(_action_world_mask_log_window) != 2 * len(self._action_world_mask_modes):
            _action_world_mask_log_window = [0.0] * (
                2 * len(self._action_world_mask_modes)
            )
        self._action_world_mask_log_window = _action_world_mask_log_window
        mask_config = getattr(model_config, "action_world_stream_mask", None)
        _action_world_mask_logging_enabled = bool(
            getattr(mask_config, "enabled", False)
            and (
                self._action_world_mask_modes == _W0_ACTION_WORLD_MASK_MODES
                or (
                    getattr(model_config, "enable_2d_world_expert", False)
                    and getattr(model_config, "enable_3d_world_expert", False)
                )
            )
        )

        def _slowest_rank_seconds(value: float) -> float:
            import torch.distributed as dist

            probe = torch.tensor([value], dtype=torch.float64, device=self.device)
            return float(all_reduce(probe, op=dist.ReduceOp.MAX)[0].item())

        def _host_reduce_many(
            local_values: dict[str, float | None],
            keys: tuple[str, ...],
            op_name: str,
        ) -> dict[str, float]:
            """Reduce a fixed host-metric vector without rank-divergent collectives.

            Optional cgroup files can differ between nodes. Missing values use an
            infinity sentinel, so every rank enters the same two collectives and a
            metric is omitted only when no rank can read it.
            """
            import torch.distributed as dist

            missing = float("-inf") if op_name == "max" else float("inf")
            probe = torch.tensor(
                [
                    missing
                    if local_values.get(key) is None
                    else float(local_values[key])
                    for key in keys
                ],
                dtype=torch.float64,
                device=self.device,
            )
            op = dist.ReduceOp.MAX if op_name == "max" else dist.ReduceOp.MIN
            reduced = all_reduce(probe, op=op)
            return {
                key: round(float(value), 3)
                for key, value in zip(keys, reduced.tolist())
                if float(value) != missing
            }

        def _flush_data_quality_window() -> None:
            """Reduce the pending quality counts once and update rank 0's sidecar."""
            nonlocal _data_quality_window_steps
            if _data_quality_window_steps <= 0:
                return
            local = torch.tensor(
                [*_data_quality_totals, *_data_quality_nonzero_rank_steps],
                dtype=torch.int64,
                device=self.device,
            )
            global_counts = all_reduce(local).tolist()
            size = len(_DATA_QUALITY_COUNT_KEYS)
            update_data_quality = getattr(
                self.metrics_logger, "update_data_quality", None
            )
            if is_main_process() and update_data_quality is not None:
                update_data_quality(
                    dict(
                        zip(
                            _DATA_QUALITY_COUNT_KEYS,
                            (int(value) for value in global_counts[:size]),
                            strict=True,
                        )
                    ),
                    dict(
                        zip(
                            _DATA_QUALITY_COUNT_KEYS,
                            (int(value) for value in global_counts[size:]),
                            strict=True,
                        )
                    ),
                    global_step=self.state.global_step,
                )
            _data_quality_totals[:] = [0] * size
            _data_quality_nonzero_rank_steps[:] = [0] * size
            _data_quality_window_steps = 0

        for batch_index, batch in enumerate(self.train_dataloader, start=1):
            # Wall time spent waiting for the dataloader to hand over this batch
            # (video decode + collate). Compared against the full-iteration time
            # below, t_data/t_iter shows whether a step is data- or compute-bound.
            t_data = time.perf_counter() - _iter_start
            if self._reached_max_steps():
                break
            profile_this_step = (
                _profile
                and torch.cuda.is_available()
                and (self.state.global_step + 1) % _profile_interval == 0
            )
            self.zero_grad()
            if _memory_diagnostics:
                torch.cuda.reset_peak_memory_stats()
                _print_cuda_memory("after_zero_grad")
            _t = time.perf_counter()
            batch = self._prepare_batch(
                batch
            )  # CPU: apply_chat_template, FAST encode, H2D
            if isinstance(batch, dict) and DATA_RETRY_ATTEMPTS in batch:
                _log_window_has_retry_metadata = True
                _log_window_retry_attempts += int(batch[DATA_RETRY_ATTEMPTS])
                _log_window_replaced_samples += int(batch.get(DATA_RETRY_REPLACED, 0))
            _sync(profile_this_step)
            t_prep = time.perf_counter() - _t
            _log_window_t_data_max = max(_log_window_t_data_max, t_data)
            _log_window_t_prep_max = max(_log_window_t_prep_max, t_prep)
            modality_key = getattr(model_config, "cotrain_modality_key", "modality")
            vlm_modalities = getattr(model_config, "vlm_modalities", frozenset())
            prepared_modality = str(batch.get(modality_key, "robotics"))
            diagnose_vlm_step = (
                _vlm_diagnostics
                and torch.cuda.is_available()
                and prepared_modality in vlm_modalities
            )
            diagnostic_events = None
            if profile_this_step or diagnose_vlm_step:
                diagnostic_events = {
                    name: torch.cuda.Event(enable_timing=True)
                    for name in (
                        "fwd_start",
                        "fwd_stop",
                        "bwd_start",
                        "bwd_stop",
                        "clip_start",
                        "clip_stop",
                        "opt_start",
                        "opt_stop",
                    )
                }
                diagnostic_events["fwd_start"].record()
            self._set_model_training_progress()
            _t = time.perf_counter()
            with self.autocast_context():
                outputs = self._forward_model(batch)
            has_data_quality = _accumulate_data_quality_window(
                _data_quality_totals,
                _data_quality_nonzero_rank_steps,
                outputs.metrics,
            )
            _data_quality_window_steps += int(has_data_quality)
            if _action_world_mask_logging_enabled:
                _accumulate_action_world_mask_window(
                    _action_world_mask_log_window,
                    outputs.metrics,
                    modes=self._action_world_mask_modes,
                )
            if diagnostic_events is not None:
                diagnostic_events["fwd_stop"].record()
            _sync(profile_this_step)
            t_fwd = time.perf_counter() - _t
            self._ensure_finite_loss(outputs.loss)
            if diagnostic_events is not None:
                diagnostic_events["bwd_start"].record()
            _t = time.perf_counter()
            if _strategy == "zero2":
                self.prepare_zero2_gradient_overlap(
                    str(outputs.metrics.get("modality", "default"))
                )
            try:
                self.backward(outputs.loss)
            except torch.OutOfMemoryError:
                _print_cuda_memory("backward_oom")
                if _memory_diagnostics:
                    print(torch.cuda.memory_summary(abbreviated=True), flush=True)
                raise
            _print_cuda_memory("after_backward")
            if _strategy == "zero2":
                self.reduce_zero2_gradients()
            if diagnostic_events is not None:
                diagnostic_events["bwd_stop"].record()
            _sync(profile_this_step)
            t_bwd = time.perf_counter() - _t
            next_step = self.state.global_step + 1
            will_log = self.log_interval > 0 and next_step % self.log_interval == 0
            branch_grad_metrics = (
                self._branch_gradient_metrics()
                if will_log and _should_measure_branch_gradients(outputs.metrics)
                else {}
            )
            if diagnostic_events is not None:
                diagnostic_events["clip_start"].record()
            grad_norm = self.clip_grad_norm(self.grad_clip_norm)
            if diagnostic_events is not None:
                diagnostic_events["clip_stop"].record()
                diagnostic_events["opt_start"].record()
            _t = time.perf_counter()
            self.optimizer_step()  # Muon Newton-Schulz + AdamW
            _print_cuda_memory("after_optimizer")
            if diagnostic_events is not None:
                diagnostic_events["opt_stop"].record()
            _sync(profile_this_step)
            t_opt = time.perf_counter() - _t
            self.state.global_step += 1
            self.state.micro_step += 1
            loss_value = float(outputs.loss.detach().cpu().item())
            modality = self._resolve_modality(batch, outputs)
            stats = self.loss_monitor.update(modality, loss_value)
            is_log_step = (
                (
                    self.log_interval > 0
                    and self.state.global_step % self.log_interval == 0
                )
                # A production base recipe normally logs every 10 steps. Ensure
                # every VLM batch is visible in an explicitly requested profile,
                # even when its random interleave position is not a normal log step.
                or diagnose_vlm_step
                or profile_this_step
            )
            vlm_diagnostic_metrics: dict[str, float | int] = {}
            distributed_profile_metrics: dict[str, float | int] = {}
            data_retry_metrics: dict[str, int] = {}
            action_world_mask_metrics: dict[str, float] = {}
            if is_log_step:
                # The per-step quality fields do not become time-series metrics.
                # Flush all steps since the last log into one cumulative sidecar.
                _flush_data_quality_window()
                # Keep the single-step latency and rolling throughput based on the
                # slowest rank.  All ranks enter this branch at the same global step.
                t_iter_local = time.perf_counter() - _iter_start
                if profile_this_step and diagnostic_events is not None:
                    import torch.distributed as dist

                    # Waiting here exposes rank skew after forward/backward/step.
                    # It is intentionally sampled: a barrier every production step
                    # would perturb the overlap and throughput being measured.
                    _sync(profile_this_step)
                    barrier_started = time.perf_counter()
                    barrier()
                    _sync(profile_this_step)
                    barrier_wait_ms = 1000.0 * (time.perf_counter() - barrier_started)
                    phase_pairs = {
                        "fwd": ("fwd_start", "fwd_stop"),
                        "bwd": ("bwd_start", "bwd_stop"),
                        "clip": ("clip_start", "clip_stop"),
                        "opt": ("opt_start", "opt_stop"),
                    }
                    local_ms = torch.tensor(
                        [
                            diagnostic_events[start].elapsed_time(
                                diagnostic_events[stop]
                            )
                            for start, stop in phase_pairs.values()
                        ]
                        + [barrier_wait_ms],
                        dtype=torch.float64,
                        device=self.device,
                    )
                    minimum_ms = all_reduce(local_ms.clone(), op=dist.ReduceOp.MIN)
                    maximum_ms = all_reduce(local_ms.clone(), op=dist.ReduceOp.MAX)
                    for index, name in enumerate((*phase_pairs.keys(), "rank_wait")):
                        distributed_profile_metrics[
                            f"dist_profile_{name}_rank_min_ms"
                        ] = round(float(minimum_ms[index].item()), 3)
                        distributed_profile_metrics[
                            f"dist_profile_{name}_rank_max_ms"
                        ] = round(float(maximum_ms[index].item()), 3)
                    distributed_profile_metrics["dist_profile_world_size"] = (
                        get_world_size()
                    )
                t_window_local = time.perf_counter() - _log_window_start
                t_window_steps = self.state.global_step - _log_window_start_step
                t_iter_max = _slowest_rank_seconds(t_iter_local)
                t_window_s_per_step = _slowest_rank_seconds(t_window_local) / max(
                    t_window_steps, 1
                )
                # Max over both ranks and every step since the previous log. A
                # rising DataLoader/page-fault stall is otherwise easy to miss
                # because rank 0's final local step may be fast.
                t_data_window_rank_max = _slowest_rank_seconds(_log_window_t_data_max)
                t_prep_window_rank_max = _slowest_rank_seconds(_log_window_t_prep_max)
                _log_window_start = time.perf_counter()
                _log_window_start_step = self.state.global_step
                _log_window_t_data_max = 0.0
                _log_window_t_prep_max = 0.0

                if will_log and _action_world_mask_logging_enabled:
                    global_mask_totals = all_reduce(
                        torch.tensor(
                            _action_world_mask_log_window,
                            dtype=torch.float64,
                            device=self.device,
                        )
                    )
                    action_world_mask_metrics = _action_world_mask_metrics_from_totals(
                        global_mask_totals.tolist(),
                        modes=self._action_world_mask_modes,
                        loss_keys=self._action_world_mask_loss_keys,
                        ratio_keys=self._action_world_mask_ratio_keys,
                    )
                    _action_world_mask_log_window[:] = [0.0] * len(
                        _action_world_mask_log_window
                    )

                if _log_window_has_retry_metadata:
                    import torch.distributed as dist

                    local_retry_counts = torch.tensor(
                        [
                            _log_window_retry_attempts,
                            _log_window_replaced_samples,
                        ],
                        dtype=torch.int64,
                        device=self.device,
                    )
                    global_retry_counts = all_reduce(
                        local_retry_counts, op=dist.ReduceOp.SUM
                    )
                    retry_attempts = int(global_retry_counts[0].item())
                    replaced_samples = int(global_retry_counts[1].item())
                    self._data_retry_attempts_run_total += retry_attempts
                    self._data_replaced_samples_run_total += replaced_samples
                    data_retry_metrics = {
                        "data_retry_attempts_window": retry_attempts,
                        "data_replaced_samples_window": replaced_samples,
                        "data_retry_attempts_run_total": (
                            self._data_retry_attempts_run_total
                        ),
                        "data_replaced_samples_run_total": (
                            self._data_replaced_samples_run_total
                        ),
                    }
                    _log_window_retry_attempts = 0
                    _log_window_replaced_samples = 0
                    _log_window_has_retry_metadata = False

                # GPU memory: per-process peak reserved (torch allocator stats) +
                # device used/total via NVML. Each probe is guarded because some
                # nodes restrict NVML access (e.g. MIG). W0 filters peak reserved
                # memory from its records; legacy models retain their old payload.
                #
                # The probe runs on EVERY rank and reduces across them, because the
                # binding constraint is the worst GPU, not rank 0 -- in the eo cotrain
                # run rank 0 sat ~30 points below the peak rank, so a rank-0-only
                # number read as "half the memory is free" when one GPU was at 83%.
                # `mem_used_min_gb` is kept so the per-rank spread is visible in the
                # log itself. Reaching the collectives is safe: global_step is
                # synchronized, so this branch is rank-invariant.
                mem_allocated_gb = mem_allocated_min_gb = 0.0
                mem_peak_allocated_gb = mem_peak_allocated_min_gb = 0.0
                mem_reserved_current_gb = mem_reserved_current_min_gb = 0.0
                mem_reserved_gb = mem_used_gb = mem_used_min_gb = mem_total_gb = 0.0
                if torch.cuda.is_available():
                    local_allocated = local_peak_allocated = 0.0
                    local_reserved_current = 0.0
                    local_reserved = local_used = 0.0
                    try:
                        local_allocated = torch.cuda.memory_allocated() / 1e9
                        local_peak_allocated = torch.cuda.max_memory_allocated() / 1e9
                        local_reserved_current = torch.cuda.memory_reserved() / 1e9
                        local_reserved = torch.cuda.max_memory_reserved() / 1e9
                    except Exception as exc:  # noqa: BLE001 -- CUDA probe is optional
                        logger.debug("CUDA max_memory_reserved unavailable: %s", exc)
                    try:
                        _free, _total = torch.cuda.mem_get_info()
                        local_used = (_total - _free) / 1e9
                        mem_total_gb = round(_total / 1e9, 1)
                    except Exception as exc:  # noqa: BLE001 -- NVML may be restricted
                        logger.debug("CUDA mem_get_info unavailable: %s", exc)

                    def _reduce(value: float, op_name: str) -> float:
                        import torch.distributed as dist

                        probe = torch.tensor([value], dtype=torch.float64)
                        if (
                            torch.cuda.is_available()
                            and dist.is_available()
                            and dist.is_initialized()
                        ):
                            probe = probe.to(torch.cuda.current_device())
                        op = (
                            dist.ReduceOp.MAX if op_name == "max" else dist.ReduceOp.MIN
                        )
                        return round(float(all_reduce(probe, op)[0].item()), 1)

                    mem_allocated_gb = _reduce(local_allocated, "max")
                    mem_allocated_min_gb = _reduce(local_allocated, "min")
                    mem_peak_allocated_gb = _reduce(local_peak_allocated, "max")
                    mem_peak_allocated_min_gb = _reduce(local_peak_allocated, "min")
                    mem_reserved_current_gb = _reduce(local_reserved_current, "max")
                    mem_reserved_current_min_gb = _reduce(local_reserved_current, "min")
                    mem_reserved_gb = _reduce(local_reserved, "max")
                    mem_used_gb = _reduce(local_used, "max")
                    mem_used_min_gb = _reduce(local_used, "min")
                    torch.cuda.reset_peak_memory_stats()

                # Host memory, opt-in. The GPU numbers above have never been the
                # thing that kills these runs; the host has, twice. See
                # trainer/host_memory.py for why two W&B numbers cannot attribute a
                # node-level climb across 8 ranks and 256 workers. Reduced the same
                # way and for the same reason: the binding constraint is the worst
                # rank, and the node nearest the OOM killer is not necessarily the
                # one rank 0 sits on. This branch is rank-invariant -- global_step
                # is synchronized and the env flag comes from the job spec -- so
                # reaching the collectives here is safe.
                host_mem: dict[str, float] = {}
                if host_memory.probe_enabled():
                    local_host = host_memory.sample()
                    # Gauges/counters where the largest rank or node is the risk.
                    host_max_keys = (
                        "host_rss_main_gb",
                        "host_rss_workers_gb",
                        "host_rss_worker_max_gb",
                        "host_cgroup_used_gb",
                        "host_cgroup_anon_gb",
                        "host_cgroup_file_gb",
                        "host_cgroup_inactive_file_gb",
                        "host_cgroup_active_file_gb",
                        "host_cgroup_shmem_gb",
                        "host_cgroup_pgmajfault_total",
                        "host_cgroup_workingset_refault_file_total",
                        "host_cgroup_pgscan_total",
                        "host_cgroup_pgsteal_total",
                        "host_cgroup_events_high_total",
                        "host_cgroup_events_max_total",
                        "host_cgroup_events_oom_total",
                        "host_cgroup_events_oom_kill_total",
                        "host_cgroup_psi_some_avg10",
                        "host_cgroup_psi_some_avg60",
                        "host_cgroup_psi_full_avg10",
                        "host_cgroup_psi_full_avg60",
                        "host_dev_shm_used_gb",
                    )
                    # Gauges where the smallest value is the binding node.
                    host_min_keys = (
                        "host_worker_count",
                        "host_avail_gb",
                        "host_cgroup_max_gb",
                        "host_cgroup_high_gb",
                        "host_cgroup_headroom_gb",
                        "host_cgroup_high_headroom_gb",
                        "host_dev_shm_free_gb",
                    )
                    host_mem.update(_host_reduce_many(local_host, host_max_keys, "max"))
                    host_mem.update(_host_reduce_many(local_host, host_min_keys, "min"))

                if diagnose_vlm_step and diagnostic_events is not None:
                    import torch.distributed as dist

                    phase_pairs = {
                        "fwd": ("fwd_start", "fwd_stop"),
                        "bwd": ("bwd_start", "bwd_stop"),
                        "clip": ("clip_start", "clip_stop"),
                        "opt": ("opt_start", "opt_stop"),
                    }
                    event_ms = [
                        diagnostic_events[start].elapsed_time(diagnostic_events[stop])
                        for start, stop in phase_pairs.values()
                    ]
                    model_stage_names = ("hidden", "lm_head", "ce")
                    model_stage_ms = [
                        float(outputs.metrics[f"vlm_diag_{name}_ms"])
                        for name in model_stage_names
                    ]
                    timing_names = (*phase_pairs.keys(), *model_stage_names)
                    local_ms = torch.tensor(
                        [*event_ms, *model_stage_ms],
                        dtype=torch.float64,
                        device=self.device,
                    )
                    minimum_ms = all_reduce(local_ms.clone(), op=dist.ReduceOp.MIN)
                    maximum_ms = all_reduce(local_ms.clone(), op=dist.ReduceOp.MAX)
                    for index, name in enumerate(timing_names):
                        vlm_diagnostic_metrics[f"vlm_diag_{name}_rank_min_ms"] = round(
                            float(minimum_ms[index].item()), 3
                        )
                        vlm_diagnostic_metrics[f"vlm_diag_{name}_rank_max_ms"] = round(
                            float(maximum_ms[index].item()), 3
                        )

                    shape_names = (
                        "batch_samples",
                        "sequence_tokens",
                        "valid_input_tokens",
                        "image_grid_rows",
                    )
                    local_shape = torch.tensor(
                        [
                            int(outputs.metrics[f"vlm_{name}"])
                            if name != "image_grid_rows"
                            else int(outputs.metrics["vlm_diag_image_grid_rows"])
                            for name in shape_names
                        ],
                        dtype=torch.int64,
                        device=self.device,
                    )
                    minimum_shape = all_reduce(
                        local_shape.clone(), op=dist.ReduceOp.MIN
                    )
                    maximum_shape = all_reduce(
                        local_shape.clone(), op=dist.ReduceOp.MAX
                    )
                    for index, name in enumerate(shape_names):
                        vlm_diagnostic_metrics[f"vlm_diag_{name}_rank_min"] = int(
                            minimum_shape[index].item()
                        )
                        vlm_diagnostic_metrics[f"vlm_diag_{name}_rank_max"] = int(
                            maximum_shape[index].item()
                        )

            if is_log_step and is_main_process():
                step_log = {
                    "event": "train_step",
                    "epoch": epoch,
                    "batch": batch_index,
                    "global_step": self.state.global_step,
                    "max_steps": self._estimated_max_steps(),
                    "loss_name": outputs.loss_name,
                    "modality": modality,
                    "loss": round(loss_value, 6),
                    "loss_ema": round(float(stats["ema"]), 6),
                    "loss_mean": round(float(stats["mean"]), 6),
                    "grad_norm": round(float(grad_norm), 6),
                    "mem_allocated_gb": mem_allocated_gb,
                    "mem_allocated_min_gb": mem_allocated_min_gb,
                    "mem_peak_allocated_gb": mem_peak_allocated_gb,
                    "mem_peak_allocated_min_gb": mem_peak_allocated_min_gb,
                    "mem_reserved_current_gb": mem_reserved_current_gb,
                    "mem_reserved_current_min_gb": mem_reserved_current_min_gb,
                    "mem_reserved_gb": mem_reserved_gb,  # W0 filters this field
                    "mem_used_gb": mem_used_gb,  # max across ranks -- the binding GPU
                    "mem_used_min_gb": mem_used_min_gb,  # min across ranks -- spread
                    "mem_total_gb": mem_total_gb,
                    "t_data": round(t_data, 4),
                    "t_prep": round(t_prep, 4),
                    "t_data_window_rank_max": round(t_data_window_rank_max, 4),
                    "t_prep_window_rank_max": round(t_prep_window_rank_max, 4),
                    "t_fwd": round(t_fwd, 4),
                    "t_bwd": round(t_bwd, 4),
                    "t_opt": round(t_opt, 4),
                    # t_iter remains the local rank-0 value for backward-compatible
                    # JSON consumers.  The terminal's last/avg/ETA uses the two
                    # max-reduced fields below instead.
                    "t_iter": round(t_iter_local, 4),
                    "t_iter_max": round(t_iter_max, 4),
                    "t_window_s_per_step": round(t_window_s_per_step, 4),
                    "t_window_steps": t_window_steps,
                    "learning_rates": [
                        round(float(group["lr"]), 10)
                        for group in self.optimizer.param_groups
                    ],
                }
                step_log.update(host_mem)  # empty unless the host probe is enabled
                step_log.update(data_retry_metrics)
                # Pass through the model's per-term component metrics (e.g. KI's
                # flow_loss / fast_loss, or a VQA batch's supervised_tokens) so each
                # loss term is visible in metrics.jsonl and W&B, not just the total.
                for key, value in outputs.metrics.items():
                    if (
                        key in step_log
                        or not isinstance(value, (int, float))
                        or isinstance(value, bool)
                    ):
                        continue
                    step_log[key] = (
                        round(float(value), 6) if isinstance(value, float) else value
                    )
                step_log.update(vlm_diagnostic_metrics)
                step_log.update(distributed_profile_metrics)
                step_log.update(branch_grad_metrics)
                step_log.update(action_world_mask_metrics)
                self.metrics_logger.log(step_log)
            self.hook_runner.step_end(
                self._hook_context(
                    epoch=epoch,
                    batch_index=batch_index,
                    modality=modality,
                    loss=loss_value,
                    loss_name=outputs.loss_name,
                    metrics=dict(outputs.metrics),
                    stats=stats,
                    grad_norm=float(grad_norm),
                    batch=batch,
                )
            )
            self._maybe_run_step_validation(epoch)
            self._maybe_save_step_checkpoint(epoch)
            _iter_start = time.perf_counter()  # reset for the next iteration's t_data

        # Preserve a final partial window when max_steps/dataloader length is not a
        # multiple of log_interval. All ranks leave the synchronized loop together.
        _flush_data_quality_window()

    def _set_model_training_progress(self) -> None:
        """Give opt-in models deterministic step progress before each forward.

        The callback is intentionally optional so every existing model keeps its
        current contract. It runs on every rank from the shared optimizer step;
        schedules implemented as pure functions therefore need no collectives or
        checkpoint-local cursor state.
        """
        model = self.model.module if hasattr(self.model, "module") else self.model
        setter = getattr(model, "set_training_progress", None)
        if setter is None:
            return
        setter(
            global_step=self.state.global_step,
            max_steps=getattr(self, "max_steps", None),
        )

    def _branch_gradient_metrics(self) -> dict[str, float | int]:
        """Report whether trainable world branches receive optimizer gradients."""
        import torch
        import torch.distributed as dist

        optimizer_ids = {
            id(parameter)
            for group in self.optimizer.param_groups
            for parameter in group["params"]
        }
        zero2 = (
            getattr(getattr(self, "strategy_config", None), "strategy", "ddp")
            == "zero2"
        )
        result: dict[str, float | int] = {}
        named_parameters = tuple(self.initial_model.named_parameters())
        branch_prefixes = getattr(
            self.initial_model,
            "gradient_branch_prefixes",
            {"world_2d": "world_2d_expert.", "world_3d": "world_3d_expert."},
        )
        for label, prefix in branch_prefixes.items():
            parameters = [
                parameter
                for name, parameter in named_parameters
                if name.startswith(prefix) and parameter.requires_grad
            ]
            if not parameters:
                continue
            gradients = [
                parameter.grad for parameter in parameters if parameter.grad is not None
            ]
            squared_norm = None
            for gradient in gradients:
                term = gradient.detach().float().square().sum()
                squared_norm = term if squared_norm is None else squared_norm + term
            gradient_count = len(gradients)
            if zero2:
                reduced = torch.tensor(
                    [
                        float(squared_norm) if squared_norm is not None else 0.0,
                        float(gradient_count),
                    ],
                    dtype=torch.float64,
                    device=self.device,
                )
                dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
                squared_norm = reduced[0]
                gradient_count = int(reduced[1].item())
            result[f"{label}_grad_norm"] = round(
                float(squared_norm.sqrt().cpu()) if squared_norm is not None else 0.0,
                6,
            )
            result[f"{label}_params_with_grad"] = gradient_count
            result[f"{label}_trainable_params"] = len(parameters)
            result[f"{label}_optimizer_params"] = sum(
                id(parameter) in optimizer_ids for parameter in parameters
            )
        return result

    def val_loop(self, max_batches: int | None = None) -> dict[str, float] | None:
        """Run a validation pass if a validation dataloader exists.

        Reports the total loss the way it always has, plus each loss component
        broken out per modality.  A co-training run interleaves VQA cross-entropy
        with robot flow loss, and averaging those two into one number produces a
        curve that moves when the *mixture* moves rather than when the model does.

        Every cross-rank buffer here has a layout fixed before the loop starts
        (modalities from the loader, metric names from a constant).  Reducing a
        layout derived from what a rank happened to see would deadlock the
        moment two ranks drew different sources.
        """
        if self.model is None or self.val_dataloader is None:
            return None
        import torch

        was_training = self.model.training
        self.model.eval()
        try:
            modalities = self._val_modalities()
            # Optional model-declared keys, fixed before any rank reads batches.
            extra_metric_keys = getattr(
                getattr(self, "initial_model", None), "validation_metric_keys", ()
            )
            metric_keys = tuple(dict.fromkeys((*_VAL_METRIC_KEYS, *extra_metric_keys)))
            # [modality, metric + batch count], plus a final row that counts every
            # batch whatever its tag.  The headline number must not depend on the
            # modality list being complete: routing it through the per-modality
            # rows meant an unlisted tag reported val_loss=0, which reads as a
            # healthy loss rather than a missing measurement.
            all_row = len(modalities)
            totals = torch.zeros(
                (len(modalities) + 1, len(metric_keys) + 1),
                dtype=torch.float64,
                device=self.device,
            )
            modality_row = {name: index for index, name in enumerate(modalities)}
            unlisted: set[str] = set()
            from_noise_error = None
            from_noise_count = None
            from_noise_batches = 0
            from_noise_budget = getattr(self, "from_noise_batches", 0)
            num_batches = 0

            with torch.no_grad():
                for batch in self.val_dataloader:
                    batch = self._prepare_batch(batch)
                    with self.autocast_context():
                        outputs = self._forward_model(batch)
                    modality = self._resolve_modality(batch, outputs)
                    row = modality_row.get(modality)
                    if row is None:
                        unlisted.add(modality)
                    rows = [all_row] if row is None else [all_row, row]
                    for column, key in enumerate(metric_keys):
                        value = (
                            float(outputs.loss.detach().cpu().item())
                            if key == "loss"
                            else outputs.metrics.get(key)
                        )
                        if isinstance(value, (int, float)) and not isinstance(
                            value, bool
                        ):
                            for target in rows:
                                totals[target, column] += float(value)
                    for target in rows:
                        totals[target, -1] += 1.0
                    num_batches += 1

                    if from_noise_batches < from_noise_budget:
                        # Same precision as the training step above, so the
                        # measurement describes the numerics the run actually
                        # uses rather than an fp32 model nobody serves.
                        with self.autocast_context():
                            sampled = self._sample_from_noise(batch)
                        if sampled is not None:
                            error, count = sampled
                            if from_noise_error is None:
                                from_noise_error = torch.zeros_like(error)
                                from_noise_count = torch.zeros_like(count)
                            from_noise_error += error
                            from_noise_count += count
                            from_noise_batches += 1

                    if max_batches is not None and num_batches >= max_batches:
                        break

            all_reduce(totals)
            counts = totals[:, -1]
            metrics: dict[str, float] = {}
            for row, modality in enumerate(modalities):
                batches = float(counts[row].cpu().item())
                if batches <= 0:
                    continue
                for column, key in enumerate(metric_keys):
                    value = float(totals[row, column].cpu().item())
                    metrics[f"val_{modality}_{key}"] = round(value / batches, 6)
                metrics[f"val_{modality}_batches"] = int(batches)
                metrics[f"{modality}/loss"] = metrics[f"val_{modality}_loss"]
                metrics[f"{modality}/batches"] = int(batches)
            grand_batches = float(counts[all_row].cpu().item())
            grand_loss = float(totals[all_row, metric_keys.index("loss")].cpu().item())
            metrics["val_loss"] = (
                round(grand_loss / grand_batches, 6) if grand_batches else 0.0
            )
            metrics["val_batches"] = int(grand_batches)
            if grand_batches:
                for column, key in enumerate(metric_keys):
                    if key == "loss":
                        continue
                    metrics[key] = round(
                        float(totals[all_row, column].cpu().item()) / grand_batches,
                        6,
                    )
            if len(unlisted) == 1:
                name = next(iter(unlisted))
                metrics[f"{name}/loss"] = metrics["val_loss"]
                metrics[f"{name}/batches"] = metrics["val_batches"]
            if unlisted and is_main_process():
                logger.warning(
                    "validation saw modalities %s that the loader did not advertise; "
                    "they are in val_loss but have no per-modality breakdown",
                    sorted(unlisted),
                )

            per_horizon = self._reduce_from_noise(from_noise_error, from_noise_count)
            if per_horizon is not None:
                metrics["val_from_noise_mse"] = round(float(per_horizon.mean()), 6)
                metrics["val_from_noise_mse_h0"] = round(float(per_horizon[0]), 6)
                metrics["val_from_noise_mse_hlast"] = round(float(per_horizon[-1]), 6)
                if is_main_process():
                    history = getattr(self, "_from_noise_history", None)
                    if history is None:
                        history = self._from_noise_history = []
                    history.append((self.state.global_step, per_horizon))

            if is_main_process():
                self.metrics_logger.log(
                    {
                        "event": "validation",
                        "epoch": self.state.epoch,
                        "global_step": self.state.global_step,
                        **metrics,
                        **(
                            {"val_from_noise_per_horizon": per_horizon.tolist()}
                            if per_horizon is not None
                            else {}
                        ),
                    }
                )
                self._write_from_noise_figure()
            self._last_validated_step = self.state.global_step
            return metrics
        finally:
            # fit() runs this mid-epoch on the checkpoint cadence, so leaving the
            # model in eval mode would silently switch off dropout for the rest
            # of training.
            self.model.train(was_training)

    def _val_modalities(self) -> list[str]:
        """Modality tags the validation loader can produce, identical on all ranks."""
        modalities = getattr(self.val_dataloader, "modalities", None)
        if modalities:
            model = self.model.module if hasattr(self.model, "module") else self.model
            config = getattr(model, "config", None)
            vlm_modalities = getattr(config, "vlm_modalities", ())
            canonical = (
                "vlm" if modality in vlm_modalities else str(modality)
                for modality in modalities
            )
            return list(dict.fromkeys(canonical))
        model = self.model.module if hasattr(self.model, "module") else self.model
        config = getattr(model, "config", None)
        native = getattr(config, "cotrain_native_modality", None)
        return [str(native)] if native else ["unknown"]

    def _sample_from_noise(self, batch):
        """Per-horizon squared error and element count for one batch, or ``None``.

        Optional on the model, the way ``prepare_batch`` is: a policy that has no
        reverse-flow sampler simply never contributes the metric.

        Skipped under FSDP.  Sampling calls the *inner* module directly, which is
        what keeps DDP's gradient machinery out of a no-grad rollout, but FSDP
        gathers sharded parameters in its own forward -- reaching past the wrapper
        would read shards rather than weights, and report a number that looks
        like an answer.  Use DDP for a run that needs this metric.
        """
        strategy_config = getattr(self, "strategy_config", None)
        if (
            getattr(strategy_config, "strategy", None) == "fsdp"
            and get_world_size() > 1
        ):
            if not getattr(self, "_from_noise_fsdp_warned", False):
                logger.warning(
                    "from-noise evaluation is skipped under FSDP: the sampler calls "
                    "the unwrapped module, whose parameters are sharded. Run with "
                    "strategy='ddp' to measure open-loop error."
                )
                self._from_noise_fsdp_warned = True
            return None
        model = self.model.module if hasattr(self.model, "module") else self.model
        evaluate = getattr(model, "evaluate_from_noise", None)
        if evaluate is None:
            return None
        result = evaluate(batch, num_steps=getattr(self, "from_noise_steps", None))
        error = result.get("from_noise_per_horizon_error")
        count = result.get("from_noise_per_horizon_count")
        if error is None or count is None:
            return None
        return error.double(), count.double()

    def _reduce_from_noise(self, error, count):
        """Mean per-horizon error across ranks, or ``None`` if nobody measured it.

        The horizon length has to be agreed *before* the reduction: a rank whose
        validation shard ran out early holds no tensor at all, and a collective
        whose operands differ in length by rank hangs rather than errors.
        """
        import torch
        import torch.distributed as dist

        local_length = 0 if error is None else int(error.numel())
        agreed = torch.tensor([local_length], dtype=torch.int64, device=self.device)
        all_reduce(agreed, op=dist.ReduceOp.MAX)
        length = int(agreed[0].cpu().item())
        if length == 0:
            return None
        if error is None or int(error.numel()) != length:
            error = torch.zeros(length, dtype=torch.float64, device=self.device)
            count = torch.zeros(length, dtype=torch.float64, device=self.device)
        stacked = torch.stack([error, count])
        all_reduce(stacked)
        summed_error, summed_count = stacked[0], stacked[1]
        return (summed_error / summed_count.clamp_min(1.0)).cpu().numpy()

    def _write_from_noise_figure(self) -> None:
        """Refresh the PNG on rank 0 and hand the current one to W&B."""
        if not getattr(self, "_from_noise_history", None) or self.output_dir is None:
            return
        target = self.output_dir / "from_noise" / "from_noise_open_loop.png"
        written = render_per_horizon_png(
            target,
            self._from_noise_history,
            title="from-noise open-loop error (normalized)",
        )
        if written is not None:
            self.metrics_logger.log_image(
                "validation/from_noise/per_horizon",
                written,
                global_step=self.state.global_step,
            )

    def save_checkpoint(self, epoch: int, step: int = 0) -> None:
        """Collect rank-local data cursors and save state from the main process.

        Rolling ZeRO-2 saves use a model checkpoint plus one consolidated optimizer
        file. Milestones embed that same full optimizer state in the model checkpoint,
        matching the native DDP payload and avoiding a duplicate companion file.
        """
        if self.output_dir is None:
            return
        if self.model is None or self.optimizer is None:
            raise ValueError("Cannot save before load_model().")
        self._log_checkpoint_host_memory(step, "before")
        try:
            self._save_checkpoint_impl(epoch, step)
        finally:
            self._release_checkpoint_host_memory()
            self._log_checkpoint_host_memory(step, "after_release")

    def _save_checkpoint_impl(self, epoch: int, step: int) -> None:
        # Complete queued validation/training kernels first, then return unused
        # allocator blocks to CUDA before checkpoint communication.  This is a
        # secondary safety margin: data-state objects use Gloo below and therefore
        # no longer depend on having spare GPU memory.
        import torch

        target = self.output_dir / f"checkpoint-epoch-{epoch}-step-{step}.pt"
        zero2_sharded = (
            getattr(getattr(self, "strategy_config", None), "strategy", "ddp")
            == "zero2"
            and getattr(self.optimizer, "optim", None) is not None
        )
        logger.warning(
            "checkpoint step=%d rank=%d: starting save preparation",
            step,
            get_rank(),
        )
        if torch.cuda.is_available() and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            torch.cuda.empty_cache()
        logger.warning(
            "checkpoint step=%d rank=%d: gathering DataLoader state",
            step,
            get_rank(),
        )
        data_state = self._gather_data_state()
        logger.warning(
            "checkpoint step=%d rank=%d: DataLoader state gathered",
            step,
            get_rank(),
        )
        zero2_optimizer_state = None
        if zero2_sharded:
            logger.warning(
                "checkpoint step=%d rank=%d: gathering optimizer state to rank 0 "
                "over CPU/Gloo",
                step,
                get_rank(),
            )
            zero2_optimizer_state = self._consolidate_zero2_optimizer_state_cpu()
            logger.warning(
                "checkpoint step=%d rank=%d: CPU optimizer state gather complete",
                step,
                get_rank(),
            )
        else:
            consolidate = getattr(self.optimizer, "consolidate_state_dict", None)
            if consolidate is not None:
                logger.warning(
                    "checkpoint step=%d rank=%d: consolidating optimizer state to rank 0",
                    step,
                    get_rank(),
                )
                consolidate(to=0)
                logger.warning(
                    "checkpoint step=%d rank=%d: optimizer state consolidated",
                    step,
                    get_rank(),
                )
        if not is_main_process():
            return
        milestone = self._is_milestone_step(step)
        checkpoint_kind = "milestone" if milestone else "rolling"
        wrapped_model = (
            self.model.module if hasattr(self.model, "module") else self.model
        )
        payload = self._checkpoint_payload(
            wrapped_model,
            extra_state={
                "epoch": epoch,
                "requested_step": step,
                "rank": get_rank(),
                "world_size": get_world_size(),
            },
            include_optimizer=not zero2_sharded,
        )
        if zero2_sharded and milestone:
            payload["optimizer"] = zero2_optimizer_state
        if zero2_sharded and not milestone:
            optimizer_path = target.with_name(f"{target.stem}.optimizer.pt")
            # A two-file checkpoint cannot be atomically replaced as a pair. Remove
            # the old commit point before replacing its optimizer, so interruption
            # falls back to an earlier checkpoint instead of mixing two save attempts.
            target.unlink(missing_ok=True)
            logger.warning(
                "checkpoint step=%d rank=0: writing %s", step, optimizer_path
            )
            save_checkpoint(optimizer_path, zero2_optimizer_state)
            payload["optimizer_file"] = optimizer_path.name
        if data_state is not None:
            payload["data_state"] = data_state
        logger.warning("checkpoint step=%d rank=0: writing %s", step, target)
        save_checkpoint(target, payload)
        logger.warning("checkpoint step=%d rank=0: write complete", step)
        legacy_shard_dir = target.parent / f"{target.stem}.optimizer"
        if legacy_shard_dir.is_dir():
            import shutil

            shutil.rmtree(legacy_shard_dir)
        if milestone:
            target.with_name(f"{target.stem}.optimizer.pt").unlink(missing_ok=True)
        self._prune_checkpoints()
        self.metrics_logger.log(
            {
                "event": "checkpoint",
                "epoch": epoch,
                "global_step": self.state.global_step,
                "path": str(target),
                "checkpoint_kind": checkpoint_kind,
            }
        )

    def _is_checkpoint_step(self, step: int) -> bool:
        """Whether ``step`` is one the run persists weights at."""
        every = getattr(self, "save_every_steps", None)
        if self._is_milestone_step(step):
            return True
        return bool(every) and step % every == 0

    def _maybe_run_step_validation(self, epoch: int) -> None:
        """Validate on the checkpoint cadence.

        Pairing the two means every saved checkpoint has an eval number measured
        at exactly its weights -- which is what makes "pick the best checkpoint"
        answerable at all -- and it is one knob rather than two that drift.

        Runs before the save so a preemption between them costs the checkpoint,
        which the next cadence step rewrites, rather than the measurement.

        Every rank takes this branch on the same global step, so the collectives
        inside ``val_loop`` stay matched.
        """
        # getattr for the same reason the checkpoint knobs below use it: tests
        # build a trainer through __new__ and set only the fields they exercise.
        if not getattr(self, "validate_on_checkpoint", False):
            return
        if self.val_dataloader is None:
            return
        if not self._is_checkpoint_step(self.state.global_step):
            return
        if self.state.global_step == getattr(self, "_last_validated_step", None):
            return
        self.val_loop(max_batches=getattr(self, "validation_steps", 20))

    def _maybe_run_epoch_validation(self) -> None:
        """Close the epoch with a validation pass, unless this step just had one.

        The epoch boundary usually *is* a checkpoint step, so without this guard
        the last step of a run is measured twice at one ``global_step``: two
        points on the same x in W&B, a duplicate curve in the PNG, and a second
        round of reverse-flow sampling, which is the expensive half.
        """
        if self.state.global_step == getattr(self, "_last_validated_step", None):
            return
        self.val_loop(max_batches=self.validation_steps)

    def _maybe_save_step_checkpoint(self, epoch: int) -> None:
        """Checkpoint mid-epoch so a preempted long run does not lose its progress."""
        step = self.state.global_step
        if not self._is_checkpoint_step(step):
            return
        self.save_checkpoint(epoch=epoch, step=step)
        # Rank 0 writes several GB to shared storage; hold the other ranks here so
        # they do not sit in a collective waiting on it.
        barrier()

    def _prune_checkpoints(self) -> None:
        """Keep recent recovery snapshots while permanently excluding milestones."""
        keep = getattr(self, "keep_last_n_checkpoints", None)
        if self.output_dir is None or not keep:
            return
        recovery_checkpoints = sorted(
            (
                path
                for path in self.output_dir.glob("checkpoint-epoch-*-step-*.pt")
                if not path.name.endswith(".optimizer.pt")
                if not self._is_milestone_step(self._checkpoint_step(path))
            ),
            key=self._checkpoint_sort_key,
        )
        for stale in recovery_checkpoints[:-keep]:
            stale.unlink(missing_ok=True)
            optimizer_file = stale.with_name(f"{stale.stem}.optimizer.pt")
            optimizer_file.unlink(missing_ok=True)
            shard_dir = stale.parent / f"{stale.stem}.optimizer"
            if shard_dir.is_dir():
                import shutil

                shutil.rmtree(shard_dir)

    @staticmethod
    def _checkpoint_sort_key(path: Path) -> tuple[int, int]:
        """Order ``checkpoint-epoch-{e}-step-{s}.pt`` by (epoch, step)."""
        parts = path.stem.split("-")
        try:
            return int(parts[2]), int(parts[4])
        except (IndexError, ValueError):
            return -1, -1

    @staticmethod
    def _checkpoint_step(path: Path) -> int:
        try:
            return int(path.stem.rsplit("-", 1)[-1])
        except ValueError:
            return -1

    def _is_milestone_step(self, step: int) -> bool:
        every = getattr(self, "save_milestone_every_steps", None)
        return bool(every and step > 0 and step % every == 0)

    def _forward_model(self, batch) -> ModelForwardOutput:
        outputs = self.model(batch)
        if not isinstance(outputs, ModelForwardOutput):
            raise TypeError(
                f"{type(self.model).__name__}.forward must return ModelForwardOutput, "
                f"got {type(outputs).__name__}."
            )
        if getattr(outputs.loss, "ndim", None) != 0:
            raise ValueError("ModelForwardOutput.loss must be a scalar tensor.")
        return outputs

    def _prepare_batch(self, batch):
        """Allow policies to preprocess CPU media before the generic GPU move.

        When the loaders were built with a ``collate_factory`` (see ``data.collate``) the
        workers already ran the model's CPU preparation, so all that is left here is the
        host-to-device move -- that is the point of the offload, and it is what turns
        ``t_prep`` from ~2 s of main-process work into worker time overlapped with compute.
        """
        if isinstance(batch, dict) and "vlm_inputs" in batch:
            return move_to_device(batch, self.device)
        model = self.model.module if hasattr(self.model, "module") else self.model
        prepare_batch = getattr(model, "prepare_batch", None)
        prepared = (
            prepare_batch(batch, self.device)
            if prepare_batch is not None
            else move_to_device(batch, self.device)
        )
        if isinstance(batch, dict) and isinstance(prepared, dict):
            preserve_retry_metadata(batch, prepared)
        return prepared

    def _hook_context(self, **overrides) -> HookContext:
        model = self.model.module if hasattr(self.model, "module") else self.model
        context = {
            "step": self.state.global_step,
            "max_steps": self._estimated_max_steps(),
            "epoch": self.state.epoch,
            "model": model,
            "optimizer": self.optimizer,
            "eval_fn": self._run_hook_eval if self.val_dataloader is not None else None,
            "main_process": is_main_process(),
        }
        context.update(overrides)
        return HookContext(**context)

    def _run_hook_eval(self, max_batches: int) -> dict[str, float]:
        # val_loop restores whatever training mode it was called in.
        return self.val_loop(max_batches=max_batches) or {
            "val_loss": 0.0,
            "val_batches": 0,
        }

    def _sync_hook_decision(self, fired: bool) -> bool:
        import torch
        import torch.distributed as dist

        flag = torch.tensor(int(fired), dtype=torch.int32, device=self.device)
        all_reduce(flag, op=dist.ReduceOp.MAX)
        return bool(flag.item())

    def _estimated_max_steps(self) -> int:
        max_steps = getattr(self, "max_steps", None)
        if max_steps is not None:
            return max_steps
        try:
            return self.num_epochs * len(self.initial_train_dataloader)
        except (TypeError, AttributeError):
            return 0

    def _reached_max_steps(self) -> bool:
        max_steps = getattr(self, "max_steps", None)
        return max_steps is not None and self.state.global_step >= max_steps

    @staticmethod
    def _resolve_modality(batch, outputs: ModelForwardOutput) -> str:
        modality = outputs.metrics.get("modality")
        if modality is None and isinstance(batch, dict):
            modality = batch.get("modality")
        return str(modality or outputs.loss_name)

    @staticmethod
    def _ensure_finite_loss(loss) -> None:
        import torch

        if not bool(torch.isfinite(loss).item()):
            raise FloatingPointError(f"non-finite loss at training step: {loss.item()}")

    def _checkpoint_payload(
        self, model, *, extra_state: dict, include_optimizer: bool = True
    ) -> dict:
        from dataclasses import asdict, is_dataclass

        payload = {
            "step": self.state.global_step,
            "model": model.state_dict(),
            "extra_state": extra_state,
        }
        if include_optimizer:
            payload["optimizer"] = self.optimizer.state_dict()
        # Without this a resume can only replay the schedule step by step; with it the
        # LR is restored exactly, including schedules that are not a pure function of
        # the step count. getattr because a partially built trainer (no
        # configure_optimizer yet) still has to be able to write a payload.
        scheduler = getattr(self, "lr_scheduler", None)
        if scheduler is not None:
            payload["scheduler"] = scheduler.state_dict()
        config = getattr(model, "config", None)
        if config is not None:
            saved_config = asdict(config) if is_dataclass(config) else config
            if isinstance(saved_config, dict):
                # Execution backend, not model identity. Keeping this out of the
                # checkpoint lets the same weights resume under the baseline or
                # Transformers FLA implementation selected by the current YAML.
                saved_config = dict(saved_config)
                saved_config.pop("expert_deltanet_backend", None)
                saved_config.pop("expert_full_attention_backend", None)
            payload["config"] = saved_config
        # The normalization contract, so the checkpoint can be used without
        # rebuilding and rescanning the datasets just to recover a few kilobytes
        # of statistics.  Actions are trained normalized and partly delta'd
        # against the anchor state; without the per-source stats and
        # action_delta_indices, a prediction cannot be turned back into a command.
        # Collected once and cached -- it is fixed for the life of the run.
        # getattr for the same reason as lr_scheduler above: a partially built
        # trainer must still be able to write a payload.
        if getattr(self, "_norm_metadata", None) is None:
            from magic_w0.data.norm_export import collect_norm_metadata

            try:
                self._norm_metadata = collect_norm_metadata(
                    getattr(self, "initial_train_dataloader", None)
                )
            except Exception:  # noqa: BLE001 - generic models may omit this metadata
                if getattr(
                    getattr(self, "initial_model", None),
                    "requires_normalization_metadata",
                    False,
                ):
                    raise RuntimeError(
                        "cannot save this model checkpoint without normalization "
                        "metadata; fix dataset metadata collection before continuing"
                    ) from None
                logger.warning(
                    "could not collect normalization metadata", exc_info=True
                )
                self._norm_metadata = {}
        if not self._norm_metadata and getattr(
            getattr(self, "initial_model", None),
            "requires_normalization_metadata",
            False,
        ):
            raise RuntimeError(
                "cannot save this model checkpoint without normalization metadata; "
                "the training dataloader did not expose any dataset statistics"
            )
        if self._norm_metadata:
            payload["normalization"] = self._norm_metadata
        return payload
