# Copyright 2026 MagicVLA authors. (Apache-2.0.)
"""Step/condition-triggered hooks for the distributed trainer.

The training loop stays a plain step loop; anything periodic or conditional --
held-out eval, loss-spike detection, grad-norm alarms, future checkpointing --
is a :class:`Hook` dispatched by a :class:`HookRunner` at the end of each step.
This module is plain Python (no torch import at module level), so hook behavior
is independently unit-testable. Distributed synchronization is injected by the
trainer through a boolean reduction callback.

DDP correctness
---------------
Step-based triggers (``every_steps`` / ``at_steps``) are deterministic and
identical on every rank, so a hook doing collective work (e.g. eval over
``DistributedSampler`` shards) fires in lockstep. Value-based conditions (loss,
grad norm) read *rank-local* values -- gradients are all-reduced before the
optimizer step, losses are not -- so ranks may disagree about firing. A hook
whose ``fire`` performs collective work must set ``sync = True``: the runner
then OR-reduces the decision across ranks so either all ranks fire or none do.
Logging-only hooks keep ``sync = False`` and fire rank-locally.

YAML
----
Hooks are declared as a list of ``{type: ..., **kwargs}`` entries::

    hooks:
      - type: eval          # periodic held-out eval (needs a val dataloader)
        every_steps: 200
        eval_steps: 20
      - type: loss_spike    # sample loss vs robust preceding-window mean
        factor: 2.0
        window_batches: 5
        trim_top: 5
      - type: grad_norm     # warn when the pre-clip grad norm exceeds threshold
        threshold: 10.0

:func:`build_hooks` maps ``type`` to a registered hook class and passes the
remaining keys to its constructor. Register out-of-tree hooks with
:func:`register_hook`.
"""

from __future__ import annotations

import json
import logging
import math
import os
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class HookContext:
    """Everything a hook may inspect at one dispatch point.

    Built fresh by the loop each step (cheap: references, no copies). Fields
    that make no sense at a given point (e.g. ``loss`` in ``train_start``) are
    ``None`` / empty.
    """

    step: int
    max_steps: int = 0
    epoch: int = 0
    batch_index: int = 0
    modality: str | None = None
    loss: float | None = None
    loss_name: str | None = None
    metrics: dict = field(default_factory=dict)
    stats: dict = field(default_factory=dict)  # LossMonitor output for this modality
    grad_norm: float | None = None  # pre-clip total grad norm
    model: Any = None  # the unwrapped model
    optimizer: Any = None
    eval_fn: Callable[[int], dict] | None = None  # eval_steps -> run_eval() result
    main_process: bool = True
    batch: Any = None  # prepared batch, retained for diagnostic hooks

    @property
    def is_main(self) -> bool:
        """True on rank 0 or when running single-process."""
        return self.main_process


@dataclass
class Trigger:
    """When a hook fires. The default fires on every step.

    ``hit(step)`` is True when ``step >= after_step`` and either ``step`` is in
    ``at_steps`` or ``every_steps > 0`` and ``step % every_steps == 0``. With
    ``every_steps=0`` only ``at_steps`` fire (never, if it is empty too).
    """

    every_steps: int = 1
    at_steps: tuple[int, ...] = ()
    after_step: int = 0

    def __post_init__(self):
        if self.every_steps < 0:
            raise ValueError(f"every_steps must be >= 0, got {self.every_steps}")
        self.at_steps = tuple(int(s) for s in self.at_steps)

    def hit(self, step: int) -> bool:
        if step < self.after_step:
            return False
        if step in self.at_steps:
            return True
        return self.every_steps > 0 and step % self.every_steps == 0


def _make_trigger(
    every_steps: int | None,
    at_steps: Sequence[int],
    after_step: int,
    *,
    default_every: int,
) -> Trigger:
    """Resolve the (every_steps, at_steps) pair a hook constructor received.

    ``every_steps=None`` means "not specified": it becomes ``default_every``,
    except that explicitly listing ``at_steps`` switches the default to 0 so
    "fire at exactly these steps" means what it says.
    """
    at = tuple(at_steps or ())
    if every_steps is None:
        every_steps = 0 if at else default_every
    return Trigger(
        every_steps=int(every_steps), at_steps=at, after_step=int(after_step)
    )


class Hook:
    """One periodic/conditional behavior attached to the training loop.

    Subclasses override :meth:`fire` (the action) and optionally
    :meth:`should_fire` (the condition; the base class just consults the
    trigger). ``sync=True`` marks hooks whose ``fire`` is collective under DDP
    (see the module docstring).
    """

    name: str = "hook"
    sync: bool = False

    def __init__(self, trigger: Trigger | None = None):
        self.trigger = trigger or Trigger()

    def should_fire(self, ctx: HookContext) -> bool:
        return self.trigger.hit(ctx.step)

    def fire(self, ctx: HookContext) -> None:
        raise NotImplementedError

    def on_train_start(self, ctx: HookContext) -> None:
        pass

    def on_train_end(self, ctx: HookContext) -> None:
        pass


class HookRunner:
    """Dispatch a list of hooks, in order, at each loop point.

    Under DDP, a hook with ``sync=True`` gets its fire decision OR-reduced
    across ranks before anyone acts, so collective work stays in lockstep even
    when the condition was computed from rank-local values.
    """

    def __init__(
        self,
        hooks: Sequence[Hook] | None = None,
        sync_decision: Callable[[bool], bool] | None = None,
    ):
        self.hooks = list(hooks or [])
        self.sync_decision = sync_decision

    def train_start(self, ctx: HookContext) -> None:
        for hook in self.hooks:
            hook.on_train_start(ctx)

    def step_end(self, ctx: HookContext) -> None:
        for hook in self.hooks:
            fired = hook.should_fire(ctx)
            if hook.sync:
                fired = self._any_rank(fired)
            if fired:
                hook.fire(ctx)

    def train_end(self, ctx: HookContext) -> None:
        for hook in self.hooks:
            hook.on_train_end(ctx)

    def _any_rank(self, fired: bool) -> bool:
        return self.sync_decision(fired) if self.sync_decision is not None else fired


# ── built-in hooks ───────────────────────────────────────────────────────────


class EvalHook(Hook):
    """Periodic evaluation-loader pass through ``ctx.eval_fn`` (rank 0 logs it).

    The trainer provides ``eval_fn`` only when it has a validation dataloader;
    without one this hook warns once and skips.
    The step trigger is deterministic across ranks, so every rank runs its eval
    shard together -- no ``sync`` needed.
    """

    name = "eval"

    def __init__(
        self,
        every_steps: int | None = None,
        at_steps: Sequence[int] = (),
        after_step: int = 0,
        eval_steps: int = 20,
    ):
        trigger = _make_trigger(every_steps, at_steps, after_step, default_every=0)
        if trigger.every_steps == 0 and not trigger.at_steps:
            raise ValueError("EvalHook needs every_steps > 0 or a non-empty at_steps")
        if eval_steps <= 0:
            raise ValueError("EvalHook eval_steps must be positive")
        super().__init__(trigger)
        self.eval_steps = int(eval_steps)
        self.last_result: dict | None = None
        self._warned_no_eval = False

    def fire(self, ctx: HookContext) -> None:
        if ctx.eval_fn is None:
            if not self._warned_no_eval:
                logging.warning(
                    "eval hook: no validation dataloader is available; "
                    "skipping all eval firings"
                )
                self._warned_no_eval = True
            return
        self.last_result = ctx.eval_fn(self.eval_steps)
        if ctx.is_main:
            logging.info(
                "step %d | eval_loss=%.4f over %d batches",
                ctx.step,
                self.last_result.get("val_loss", self.last_result.get("loss", 0.0)),
                self.last_result.get("val_batches", self.eval_steps),
            )


class LossSpikeHook(Hook):
    """Record samples whose loss is anomalous relative to a short robust window.

    For each modality, the hook pools sample losses from the preceding
    ``window_batches`` batches, removes the largest ``trim_top`` values, and
    uses the remaining mean as the baseline. A current sample is a spike when
    its loss is greater than ``factor * baseline``. The current batch is added
    only *after* comparison, so an anomalous batch cannot inflate its own
    threshold.
    """

    name = "loss_spike"

    def __init__(
        self,
        factor: float = 2.0,
        window_batches: int = 5,
        trim_top: int = 5,
        save_dir: str | os.PathLike[str] | None = None,
        max_saved: int | None = None,
    ):
        super().__init__(Trigger(every_steps=1))
        if factor <= 1.0:
            raise ValueError(f"factor must be > 1, got {factor}")
        if window_batches <= 0:
            raise ValueError("window_batches must be positive")
        if trim_top < 0:
            raise ValueError("trim_top must be non-negative")
        self.factor = float(factor)
        self.window_batches = int(window_batches)
        self.trim_top = int(trim_top)
        self._history: dict[str, deque[list[float]]] = {}
        self._pending_event: dict | None = None
        self._warned_missing_sample_loss: set[str] = set()
        self.events: list[dict] = []
        self.save_dir = Path(save_dir) if save_dir is not None else None
        self.max_saved = None if max_saved is None else int(max_saved)
        if self.max_saved is not None and self.max_saved <= 0:
            raise ValueError("max_saved must be positive when set")
        self._saved = 0

    def should_fire(self, ctx: HookContext) -> bool:
        self._pending_event = None
        if not self.trigger.hit(ctx.step) or ctx.loss is None or ctx.modality is None:
            return False
        raw_losses = ctx.metrics.get("per_sample_loss")
        if raw_losses is None:
            if ctx.modality not in self._warned_missing_sample_loss:
                logging.warning(
                    "loss spike hook: modality=%s does not report per_sample_loss; skipping",
                    ctx.modality,
                )
                self._warned_missing_sample_loss.add(ctx.modality)
            return False
        sample_losses = [float(value) for value in _json_value(raw_losses)]
        if not sample_losses or not all(
            math.isfinite(value) for value in sample_losses
        ):
            logging.warning(
                "loss spike hook: step %d has empty or non-finite per_sample_loss; skipping",
                ctx.step,
            )
            return False

        history = self._history.setdefault(
            ctx.modality, deque(maxlen=self.window_batches)
        )
        baseline = None
        trimmed_count = 0
        spike_indices: list[int] = []
        if len(history) == self.window_batches:
            pooled = sorted(value for batch_losses in history for value in batch_losses)
            # A tiny batch/window may contain <= trim_top samples. Keep at least
            # one value so batch-size-1 debug runs still have a defined baseline.
            trimmed_count = min(self.trim_top, max(len(pooled) - 1, 0))
            kept = pooled[:-trimmed_count] if trimmed_count else pooled
            if kept:
                baseline = sum(kept) / len(kept)
                threshold = self.factor * baseline
                spike_indices = [
                    index
                    for index, value in enumerate(sample_losses)
                    if value > threshold
                ]
        history.append(sample_losses)
        if baseline is None or not spike_indices:
            return False

        self._pending_event = _build_spike_event(
            ctx,
            sample_losses=sample_losses,
            spike_indices=spike_indices,
            baseline=baseline,
            factor=self.factor,
            window_batches=self.window_batches,
            trim_top=self.trim_top,
            trimmed_count=trimmed_count,
        )
        return True

    def fire(self, ctx: HookContext) -> None:
        if self._pending_event is None:
            return
        event = self._pending_event
        logging.warning(
            "sample loss spike: step %d | modality=%s | samples=%s | "
            "baseline=%.6g threshold=%.6g",
            ctx.step,
            ctx.modality,
            event["spike_sample_indices"],
            event["baseline_sample_loss"],
            event["threshold"],
        )
        if self.max_saved is not None and self._saved >= self.max_saved:
            return
        self.events.append(event)
        self._saved += 1
        if self.save_dir is not None:
            try:
                path = _append_spike_event(self.save_dir, event)
                event["saved_to"] = str(path)
                logging.warning("loss spike references appended to %s", path)
            except Exception:  # diagnostics must never terminate training
                logging.exception(
                    "failed to save loss spike sample at step %d", ctx.step
                )


def _json_value(value: Any) -> Any:
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu()
        if getattr(value, "ndim", 1) == 0:
            return value.item()
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _build_spike_event(
    ctx: HookContext,
    *,
    sample_losses: list[float],
    spike_indices: list[int],
    baseline: float,
    factor: float,
    window_batches: int,
    trim_top: int,
    trimmed_count: int,
) -> dict:
    rank = os.environ.get("RANK", "0")
    samples = [
        _extract_sample_record(
            ctx.batch, index, len(sample_losses), sample_losses[index]
        )
        for index in spike_indices
    ]
    return {
        "global_step": ctx.step,
        "rank": int(rank) if rank.isdigit() else rank,
        "epoch": ctx.epoch,
        "batch_index": ctx.batch_index,
        "modality": ctx.modality,
        "batch_loss": ctx.loss,
        "loss_name": ctx.loss_name,
        "baseline_sample_loss": baseline,
        "factor": factor,
        "threshold": factor * baseline,
        "window_batches": window_batches,
        "trim_top": trim_top,
        "trimmed_count": trimmed_count,
        "batch_sample_losses": sample_losses,
        "spike_sample_indices": spike_indices,
        "samples": samples,
    }


def _extract_sample_record(
    batch: Any, index: int, batch_size: int, sample_loss: float
) -> dict:
    metadata = {}
    if isinstance(batch, dict):
        metadata = _sample_at(batch.get("__sample_metadata__", {}), index, batch_size)
    record = dict(metadata) if isinstance(metadata, dict) else {"metadata": metadata}
    record.update(
        {
            "batch_sample_index": index,
            "per_sample_loss": sample_loss,
            "state": _sample_at(
                _find_batch_field(batch, ("state", "states")), index, batch_size
            ),
            "action_chunk": _sample_at(
                _find_batch_field(batch, ("actions", "action", "real_action")),
                index,
                batch_size,
            ),
            "action_is_pad": _sample_at(
                _find_batch_field(batch, ("action_is_pad", "real_action_is_pad")),
                index,
                batch_size,
            ),
            "state_dim_mask": _sample_at(
                _find_batch_field(batch, ("state_dim_mask",)), index, batch_size
            ),
            "action_dim_mask": _sample_at(
                _find_batch_field(batch, ("action_dim_mask",)), index, batch_size
            ),
        }
    )
    return _json_value(record)


def _sample_at(value: Any, index: int, batch_size: int) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _sample_at(item, index, batch_size) for key, item in value.items()
        }
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        cpu = value.detach().cpu()
        if getattr(cpu, "ndim", 0) > 0 and int(cpu.shape[0]) == batch_size:
            cpu = cpu[index]
        return _json_value(cpu)
    if isinstance(value, (list, tuple)) and len(value) == batch_size:
        return _json_value(value[index])
    return _json_value(value)


def _append_spike_event(root: Path, event: dict) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    rank = event["rank"]
    path = root / f"events_rank_{rank}.jsonl"
    line = (json.dumps(event, ensure_ascii=True, separators=(",", ":")) + "\n").encode()
    descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        os.write(descriptor, line)
    finally:
        os.close(descriptor)
    return path


def _find_batch_field(batch: Any, suffixes: tuple[str, ...]) -> Any:
    """Find action/state tensors while tolerating model-specific field names."""
    if not isinstance(batch, dict):
        return None
    for key, value in batch.items():
        normalized = str(key).lower()
        if any(
            normalized == suffix or normalized.endswith(f".{suffix}")
            for suffix in suffixes
        ):
            return value
    return None


class GradNormHook(Hook):
    """Warn when the pre-clip gradient norm exceeds ``threshold``.

    The loop passes the total norm returned by ``clip_grad_norm_`` -- i.e. the
    norm *before* clipping -- so this fires even when clipping then rescues the
    step. Rank-local (post-all-reduce gradients are identical across ranks, so
    every rank fires together anyway).
    """

    name = "grad_norm"

    def __init__(
        self,
        threshold: float,
        every_steps: int | None = None,
        at_steps: Sequence[int] = (),
        after_step: int = 0,
    ):
        super().__init__(
            _make_trigger(every_steps, at_steps, after_step, default_every=1)
        )
        self.threshold = float(threshold)
        self.events: list[dict] = []

    def should_fire(self, ctx: HookContext) -> bool:
        return (
            self.trigger.hit(ctx.step)
            and ctx.grad_norm is not None
            and ctx.grad_norm > self.threshold
        )

    def fire(self, ctx: HookContext) -> None:
        self.events.append(
            {"step": ctx.step, "modality": ctx.modality, "grad_norm": ctx.grad_norm}
        )
        logging.warning(
            "grad norm alarm: step %d | modality=%s | grad_norm=%.4f > threshold=%.4f (pre-clip)",
            ctx.step,
            ctx.modality,
            ctx.grad_norm,
            self.threshold,
        )


class LambdaHook(Hook):
    """Programmatic hook: run ``fn(ctx)`` on the trigger (not YAML-buildable)."""

    def __init__(
        self,
        fn: Callable[[HookContext], None],
        every_steps: int | None = None,
        at_steps: Sequence[int] = (),
        after_step: int = 0,
        name: str = "lambda",
        sync: bool = False,
    ):
        super().__init__(
            _make_trigger(every_steps, at_steps, after_step, default_every=1)
        )
        self._fn = fn
        self.name = name
        self.sync = sync

    def fire(self, ctx: HookContext) -> None:
        self._fn(ctx)


# ── registry / YAML builder ─────────────────────────────────────────────────

_HOOK_TYPES: dict[str, type[Hook]] = {
    "eval": EvalHook,
    "loss_spike": LossSpikeHook,
    "grad_norm": GradNormHook,
}


def register_hook(name: str, hook_cls: type[Hook], *, override: bool = False) -> None:
    """Register a hook class under a YAML ``type`` name."""
    if name in _HOOK_TYPES and not override:
        raise ValueError(
            f"hook type {name!r} is already registered; pass override=True to replace it"
        )
    _HOOK_TYPES[name] = hook_cls


def hook_types() -> list[str]:
    """Registered hook type names, sorted."""
    return sorted(_HOOK_TYPES)


def build_hooks(specs: Sequence[dict] | None) -> list[Hook]:
    """Instantiate hooks from a YAML ``hooks:`` list of ``{type, **kwargs}``."""
    hooks: list[Hook] = []
    for index, spec in enumerate(specs or []):
        if not isinstance(spec, dict):
            raise TypeError(
                f"hooks[{index}] must be a mapping with a 'type' key, got {spec!r}"
            )
        spec = dict(spec)
        name = spec.pop("type", None)
        if name not in _HOOK_TYPES:
            raise ValueError(
                f"hooks[{index}]: unknown hook type {name!r}. Registered: {hook_types()}"
            )
        try:
            hooks.append(_HOOK_TYPES[name](**spec))
        except TypeError as exc:
            raise TypeError(f"hooks[{index}] ({name!r}): {exc}") from exc
    return hooks


__all__ = [
    "EvalHook",
    "GradNormHook",
    "Hook",
    "HookContext",
    "HookRunner",
    "LambdaHook",
    "LossSpikeHook",
    "Trigger",
    "build_hooks",
    "hook_types",
    "register_hook",
]
