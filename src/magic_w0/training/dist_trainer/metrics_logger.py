"""Small metrics logger shared by trainer entrypoints."""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from .distribution_strategy import is_main_process


_DROPPED_TRAIN_METRIC_KEYS = {
    "valid_action_steps",
    "valid_2d_views",
    "valid_2d_tokens",
    "valid_3d_tokens",
}


_DROPPED_W0_TRAIN_METRIC_KEYS = {
    "valid_action_elements",
    "valid_dino_tokens",
    "valid_geometry_tokens",
    "valid_motion_tokens",
    "mem_reserved_gb",
    "dino_invalid_tokens",
    "dino_raw_nonfinite_elements",
    "geometry_recovered_rows",
    "geometry_dropped_rows",
    "geometry_raw_nonfinite_elements",
    "motion_recovered_rows",
    "motion_dropped_rows",
    "motion_raw_nonfinite_elements",
}


_DROPPED_WANDB_TRAIN_METRIC_KEYS = {
    "epoch",
    "batch",
    "total_loss",
    "action_loss_weight",
    "step_time_seconds",
    "track4world_teacher_seconds",
    # High-frequency diagnostics retained in stdout/metrics.jsonl but omitted
    # from W&B at the user's request.
    "n_fast_tokens",
    "mem_allocated_gb",
    "mem_allocated_min_gb",
    "mem_peak_allocated_gb",
    "mem_peak_allocated_min_gb",
    "mem_reserved_current_gb",
    "mem_reserved_current_min_gb",
    "mem_used_gb",
    "mem_used_min_gb",
    "supervised_tokens",
    "vlm_padding_tokens",
    "vlm_valid_input_tokens",
    "vlm_sequence_tokens",
    "vlm_batch_samples",
    "t_window_s_per_step",
    "t_iter_max",
    "t_iter",
    "t_data",
    "t_data_window_rank_max",
    "t_fwd",
    "t_bwd",
    "t_prep",
    "t_prep_window_rank_max",
    "t_opt",
    # Data/teacher health remains available in stdout and metrics.jsonl.  Keep
    # W&B focused on optimization and model-quality curves.
    "data_retry_attempts_window",
    "data_replaced_samples_window",
    "data_retry_attempts_run_total",
    "data_replaced_samples_run_total",
    "track4world_motion_retry_rows",
    "track4world_motion_recovered_rows",
    "track4world_motion_dropped_rows",
    # Run/model structure. These are useful once at startup, but charting one
    # identical point per train step only bloats the W&B history.
    "max_steps",
    "mem_total_gb",
    "t_window_steps",
    "vlm_loss_weight",
    "dino_teacher_loaded",
    "track4world_teacher_loaded",
    "world_2d_params_with_grad",
    "world_2d_trainable_params",
    "world_2d_optimizer_params",
    "world_3d_params_with_grad",
    "world_3d_trainable_params",
    "world_3d_optimizer_params",
    # Opt-in CUDA diagnostics stay in JSONL/stdout rather than W&B charts.
    "vlm_diag_hidden_ms",
    "vlm_diag_lm_head_ms",
    "vlm_diag_ce_ms",
    "vlm_diag_logits_gel",
    "vlm_diag_logits_gb",
    "vlm_diag_pixel_elements",
    "vlm_diag_image_grid_rows",
    "vlm_diag_fwd_rank_min_ms",
    "vlm_diag_fwd_rank_max_ms",
    "vlm_diag_bwd_rank_min_ms",
    "vlm_diag_bwd_rank_max_ms",
    "vlm_diag_clip_rank_min_ms",
    "vlm_diag_clip_rank_max_ms",
    "vlm_diag_opt_rank_min_ms",
    "vlm_diag_opt_rank_max_ms",
    "vlm_diag_hidden_rank_min_ms",
    "vlm_diag_hidden_rank_max_ms",
    "vlm_diag_lm_head_rank_min_ms",
    "vlm_diag_lm_head_rank_max_ms",
    "vlm_diag_ce_rank_min_ms",
    "vlm_diag_ce_rank_max_ms",
    "vlm_diag_batch_samples_rank_min",
    "vlm_diag_batch_samples_rank_max",
    "vlm_diag_sequence_tokens_rank_min",
    "vlm_diag_sequence_tokens_rank_max",
    "vlm_diag_valid_input_tokens_rank_min",
    "vlm_diag_valid_input_tokens_rank_max",
    "vlm_diag_image_grid_rows_rank_min",
    "vlm_diag_image_grid_rows_rank_max",
}


_DROPPED_WANDB_W0_TRAIN_METRIC_KEYS = {
    "teacher_seconds",
    "teacher_retry_rows",
    "mem_reserved_gb",
    "valid_action_elements",
    "valid_dino_tokens",
    "valid_geometry_tokens",
    "valid_motion_tokens",
    "dino_invalid_tokens",
    "dino_raw_nonfinite_elements",
    "geometry_recovered_rows",
    "geometry_dropped_rows",
    "geometry_raw_nonfinite_elements",
    "motion_recovered_rows",
    "motion_dropped_rows",
    "motion_raw_nonfinite_elements",
    "geometry_params_with_grad",
    "geometry_trainable_params",
    "geometry_optimizer_params",
    "motion_params_with_grad",
    "motion_trainable_params",
    "motion_optimizer_params",
    "dino_params_with_grad",
    "dino_trainable_params",
    "dino_optimizer_params",
    "action_params_with_grad",
    "action_trainable_params",
    "action_optimizer_params",
}


# Not every static field exists on the first physical batch: VLM-only fields
# arrive on the first VLM batch, while teacher/branch fields arrive on the first
# robotics batch after backward. Print each field immediately before its first
# corresponding train line and never again.
_CONSOLE_ONCE_TRAIN_METRIC_KEYS = (
    "max_steps",
    "mem_total_gb",
    "vlm_loss_weight",
    "dino_teacher_loaded",
    "track4world_teacher_loaded",
    "world_2d_params_with_grad",
    "world_2d_trainable_params",
    "world_2d_optimizer_params",
    "world_3d_params_with_grad",
    "world_3d_trainable_params",
    "world_3d_optimizer_params",
)


_CONSOLE_ONCE_W0_TRAIN_METRIC_KEYS = (
    "valid_action_elements",
    "valid_dino_tokens",
    "valid_geometry_tokens",
    "valid_motion_tokens",
    "geometry_params_with_grad",
    "geometry_trainable_params",
    "geometry_optimizer_params",
    "motion_params_with_grad",
    "motion_trainable_params",
    "motion_optimizer_params",
    "dino_params_with_grad",
    "dino_trainable_params",
    "dino_optimizer_params",
    "action_params_with_grad",
    "action_trainable_params",
    "action_optimizer_params",
)


@dataclass(frozen=True, slots=True)
class WandbConfig:
    """Optional Weights & Biases run configuration."""

    enabled: bool = False
    project: str = "magic_w0"
    entity: str | None = None
    name: str | None = None
    group: str | None = None
    job_type: str | None = "train"
    tags: tuple[str, ...] = ()
    notes: str | None = None
    mode: str = "online"
    run_id: str | None = None
    resume: str | None = None
    save_code: bool = False

    def __post_init__(self) -> None:
        if self.mode not in {"online", "offline", "disabled"}:
            raise ValueError("training.wandb.mode must be online, offline, or disabled")
        if self.resume not in {None, "allow", "never", "must", "auto"}:
            raise ValueError(
                "training.wandb.resume must be allow, never, must, auto, or null"
            )
        if self.enabled and not self.project:
            raise ValueError("training.wandb.project must be set when W&B is enabled")

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "WandbConfig":
        raw = dict(payload or {})
        known = {field.name for field in fields(cls)}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(
                f"training.wandb has unknown keys {sorted(unknown)}; allowed {sorted(known)}"
            )
        if isinstance(raw.get("tags"), list):
            raw["tags"] = tuple(str(tag) for tag in raw["tags"])
        return cls(**raw)


class MetricsLogger:
    """Emit concise stdout metrics plus full JSONL and optional W&B metrics."""

    def __init__(
        self,
        *,
        output_dir: str | Path | None = None,
        wandb_config: WandbConfig | None = None,
        run_config: dict[str, Any] | None = None,
    ) -> None:
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self.metrics_path = (
            self.output_dir / "metrics.jsonl" if self.output_dir is not None else None
        )
        self.data_quality_path = (
            self.output_dir / "data_quality.json"
            if self.output_dir is not None
            else None
        )
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        self.wandb_config = wandb_config or WandbConfig()
        self.wandb_run = self._init_wandb(run_config or {})
        # A log record is normally emitted every ten optimizer steps.  The trainer
        # supplies an interval average, which we expand into this small fixed-size
        # window so ``avg50`` always means approximately the last 50 *steps*, not
        # the last 50 log records (which would otherwise be 500 steps).
        self._step_time_window: deque[float] = deque(maxlen=50)
        self._objective_snapshots: dict[str, tuple[float, int]] = {}
        self._printed_train_static_keys: set[str] = set()
        self._data_quality_totals: dict[str, int] = {}
        self._data_quality_nonzero_rank_steps: dict[str, int] = {}
        self._load_data_quality_totals()

    def log(self, payload: dict[str, Any]) -> None:
        """Log one metrics payload from the main process."""
        if not is_main_process():
            return
        # Static fields need the original model payload, but are intentionally
        # removed from the per-step JSONL/W&B payload below.
        static_line = self._console_once_train_static_line(payload)
        record_payload = self._record_payload(payload)
        line = json.dumps(record_payload)
        if static_line is not None:
            print(static_line, flush=True)
        print(self._console_line(record_payload), flush=True)
        profile_line = self._console_vlm_profile_line(record_payload)
        if profile_line is not None:
            print(profile_line, flush=True)
        distributed_profile_line = self._console_distributed_profile_line(
            record_payload
        )
        if distributed_profile_line is not None:
            print(distributed_profile_line, flush=True)
        if self.metrics_path is not None:
            # A metrics-file write must never kill a multi-day run. If the path is
            # not writable (e.g. output_dir owned by another uid), warn once and keep
            # training -- stdout above still carries every metric.
            try:
                with self.metrics_path.open("a", encoding="utf-8") as handle:
                    handle.write(line + "\n")
            except OSError as exc:
                if not getattr(self, "_metrics_write_warned", False):
                    print(
                        f"WARNING: cannot write {self.metrics_path} ({exc}); "
                        "metrics go to stdout only.",
                        flush=True,
                    )
                    self._metrics_write_warned = True
        if self.wandb_run is not None:
            self.wandb_run.log(self._wandb_payload(record_payload))

    def update_data_quality(
        self,
        counts: dict[str, int],
        nonzero_rank_steps: dict[str, int],
        *,
        global_step: int,
    ) -> None:
        """Persist cumulative teacher/data-quality counters outside time series."""
        if not is_main_process() or self.data_quality_path is None:
            return
        for key, value in counts.items():
            self._data_quality_totals[key] = self._data_quality_totals.get(
                key, 0
            ) + int(value)
        for key, value in nonzero_rank_steps.items():
            self._data_quality_nonzero_rank_steps[key] = (
                self._data_quality_nonzero_rank_steps.get(key, 0) + int(value)
            )
        payload = {
            "version": 1,
            "last_global_step": int(global_step),
            "totals": dict(sorted(self._data_quality_totals.items())),
            "nonzero_rank_steps": dict(
                sorted(self._data_quality_nonzero_rank_steps.items())
            ),
        }
        temporary = self.data_quality_path.with_suffix(".json.tmp")
        try:
            temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            temporary.replace(self.data_quality_path)
        except OSError as exc:
            if not getattr(self, "_data_quality_write_warned", False):
                print(
                    f"WARNING: cannot write {self.data_quality_path} ({exc}); "
                    "data-quality counters will remain in memory only.",
                    flush=True,
                )
                self._data_quality_write_warned = True

    def _load_data_quality_totals(self) -> None:
        """Continue cumulative counters when an output directory is resumed."""
        if self.data_quality_path is None or not self.data_quality_path.is_file():
            return
        try:
            payload = json.loads(self.data_quality_path.read_text(encoding="utf-8"))
            if int(payload.get("version", -1)) != 1:
                raise ValueError("unsupported data-quality version")
            self._data_quality_totals = {
                str(key): int(value)
                for key, value in dict(payload.get("totals", {})).items()
            }
            self._data_quality_nonzero_rank_steps = {
                str(key): int(value)
                for key, value in dict(payload.get("nonzero_rank_steps", {})).items()
            }
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            print(
                f"WARNING: cannot resume {self.data_quality_path} ({exc}); "
                "starting data-quality counters from zero.",
                flush=True,
            )

    def _console_line(self, payload: dict[str, Any]) -> str:
        """Render train steps for humans while retaining the original JSONL payload."""
        if payload.get("event") != "train_step":
            return json.dumps(payload)

        step = int(payload.get("global_step", 0))
        self._update_objective_snapshots(payload, step)
        self._update_step_time_window(payload)

        max_steps = int(payload.get("max_steps", 0) or 0)
        if max_steps > 0:
            progress = f"{step}/{max_steps} ({100.0 * step / max_steps:.2f}%)"
        else:
            progress = str(step)

        objective_names = (
            (
                "flow_loss",
                "fast_loss",
                "vlm_ce",
                "dino_flow_loss",
                "motion_flow_loss",
                "geometry_loss",
            )
            if self._is_w0_train_payload(payload)
            else (
                "flow_loss",
                "fast_loss",
                "vlm_ce",
                "gen_2d_loss",
                "geometry_3d_loss",
                "motion_3d_loss",
                "world_3d_loss",
            )
        )
        losses = " ".join(
            self._format_objective(name, step) for name in objective_names
        )
        last = self._as_float(payload.get("t_iter_max", payload.get("t_iter")))
        timing = (
            f"last {self._format_seconds(last)}/step" if last is not None else "last -"
        )
        if self._step_time_window:
            average = sum(self._step_time_window) / len(self._step_time_window)
            timing += f" | avg{len(self._step_time_window)} {self._format_seconds(average)}/step"
            if max_steps > step:
                timing += (
                    f" | ETA {self._format_duration(average * (max_steps - step))}"
                )

        mem_allocated = self._as_float(payload.get("mem_allocated_gb"))
        mem_allocated_min = self._as_float(payload.get("mem_allocated_min_gb"))
        mem_peak_allocated = self._as_float(payload.get("mem_peak_allocated_gb"))
        mem_peak_allocated_min = self._as_float(
            payload.get("mem_peak_allocated_min_gb")
        )
        mem_reserved = self._as_float(payload.get("mem_reserved_current_gb"))
        mem_reserved_min = self._as_float(payload.get("mem_reserved_current_min_gb"))
        mem_used = self._as_float(payload.get("mem_used_gb"))
        mem_total = self._as_float(payload.get("mem_total_gb"))
        mem_min = self._as_float(payload.get("mem_used_min_gb"))
        memory = "mem -"
        if mem_used is not None and mem_total is not None and mem_total > 0:
            memory_parts = []
            if mem_allocated is not None:
                spread = (
                    f"/{mem_allocated_min:.1f}G"
                    if mem_allocated_min is not None
                    else "G"
                )
                memory_parts.append(f"alloc {mem_allocated:.1f}{spread}")
            if mem_peak_allocated is not None:
                spread = (
                    f"/{mem_peak_allocated_min:.1f}G"
                    if mem_peak_allocated_min is not None
                    else "G"
                )
                memory_parts.append(f"peak {mem_peak_allocated:.1f}{spread}")
            if mem_reserved is not None:
                spread = (
                    f"/{mem_reserved_min:.1f}G" if mem_reserved_min is not None else "G"
                )
                memory_parts.append(f"reserve {mem_reserved:.1f}{spread}")
            memory_parts.append(f"device {mem_used:.1f}G")
            memory = " ".join(memory_parts)
            if mem_min is not None:
                memory += f" Δ{mem_used - mem_min:.1f}G"

        lr_values = payload.get("learning_rates", ())
        lr = (
            next(iter(lr_values), None)
            if isinstance(lr_values, (list, tuple))
            else None
        )
        phases = " ".join(
            f"{label}{self._format_seconds(self._as_float(payload.get(key)))}"
            for label, key in (
                ("d", "t_data"),
                ("p", "t_prep"),
                ("f", "t_fwd"),
                ("b", "t_bwd"),
                ("o", "t_opt"),
            )
            if self._as_float(payload.get(key)) is not None
        )
        extras = []
        if phases:
            extras.append(f"phase {phases}")
        profile_labels = (
            ("F", "profile_forward_s"),
            ("B", "profile_backward_s"),
            ("O", "profile_optimizer_s"),
            ("f-vis", "profile_fwd_robot_embed_vision_s"),
            ("f-exp", "profile_fwd_robot_expert_layers_s"),
            ("f-vlm", "profile_fwd_robot_vlm_layers_s"),
            ("o-muon", "profile_opt_optimizer_muon_s"),
            ("o-adamw", "profile_opt_optimizer_adamw_s"),
            ("m-prep", "profile_opt_muon_prepare_s"),
            ("m-ns", "profile_opt_muon_ns_s"),
            ("m-apply", "profile_opt_muon_apply_s"),
            ("C", "profile_clip_grad_s"),
            ("o-update", "profile_opt_optimizer_update_s"),
            ("o-sched", "profile_opt_optimizer_scheduler_s"),
        )
        profile = " ".join(
            f"{label}{self._format_seconds(self._as_float(payload.get(key)))}"
            for label, key in profile_labels
            if self._as_float(payload.get(key)) is not None
        )
        if profile:
            extras.append(f"gpu {profile}")
        if lr is not None:
            extras.append(f"lr {float(lr):.2e}")
        grad_norm = self._as_float(payload.get("grad_norm"))
        if grad_norm is not None:
            extras.append(f"gn {grad_norm:.3g}")
        branches = (
            (
                ("geometry", "geo"),
                ("motion", "motion"),
                ("dino", "dino"),
                ("action", "action"),
            )
            if self._is_w0_train_payload(payload)
            else (("world_2d", "2d"), ("world_3d", "3d"))
        )
        for branch, label in branches:
            branch_grad = self._as_float(payload.get(f"{branch}_grad_norm"))
            if branch_grad is not None:
                extras.append(f"{label}_grad {branch_grad:.3g}")

        loss = self._as_float(payload.get("loss"))
        ema = self._as_float(payload.get("loss_ema"))
        loss_summary = f"loss {self._format_loss(loss)}"
        if ema is not None:
            loss_summary += f" ema {self._format_loss(ema)}"
        modality = str(payload.get("modality", "unknown"))
        suffix = " | ".join(extras)
        return (
            f"[train] {progress} | {modality} | {loss_summary} | {losses} | "
            f"{timing} | {memory}" + (f" | {suffix}" if suffix else "")
        )

    @staticmethod
    def _console_vlm_profile_line(payload: dict[str, Any]) -> str | None:
        """Print opt-in VLM timings separately from the compact train line."""
        if "vlm_diag_bwd_rank_max_ms" not in payload:
            return None

        def seconds(key: str) -> str:
            value = payload.get(key)
            return "-" if value is None else f"{float(value) / 1000.0:.3f}s"

        batch = int(payload.get("vlm_batch_samples", 0) or 0)
        sequence = int(payload.get("vlm_sequence_tokens", 0) or 0)
        valid = int(payload.get("vlm_valid_input_tokens", 0) or 0)
        images = int(payload.get("vlm_diag_image_grid_rows", 0) or 0)
        logits_gb = float(payload.get("vlm_diag_logits_gb", 0.0) or 0.0)
        sequence_min = int(payload.get("vlm_diag_sequence_tokens_rank_min", sequence))
        sequence_max = int(payload.get("vlm_diag_sequence_tokens_rank_max", sequence))
        return (
            f"[vlm-profile] step={int(payload.get('global_step', 0))} "
            f"shape={batch}x{sequence} valid={valid} images={images} "
            f"rank-seq={sequence_min}..{sequence_max} logits={logits_gb:.2f}GB"
            " | model-rank-range "
            f"hidden={seconds('vlm_diag_hidden_rank_min_ms')}.."
            f"{seconds('vlm_diag_hidden_rank_max_ms')} "
            f"head={seconds('vlm_diag_lm_head_rank_min_ms')}.."
            f"{seconds('vlm_diag_lm_head_rank_max_ms')} "
            f"ce={seconds('vlm_diag_ce_rank_min_ms')}.."
            f"{seconds('vlm_diag_ce_rank_max_ms')}"
            " | rank-range "
            f"fwd={seconds('vlm_diag_fwd_rank_min_ms')}.."
            f"{seconds('vlm_diag_fwd_rank_max_ms')} "
            f"bwd={seconds('vlm_diag_bwd_rank_min_ms')}.."
            f"{seconds('vlm_diag_bwd_rank_max_ms')} "
            f"clip={seconds('vlm_diag_clip_rank_min_ms')}.."
            f"{seconds('vlm_diag_clip_rank_max_ms')} "
            f"opt={seconds('vlm_diag_opt_rank_min_ms')}.."
            f"{seconds('vlm_diag_opt_rank_max_ms')}"
        )

    @staticmethod
    def _console_distributed_profile_line(payload: dict[str, Any]) -> str | None:
        """Render sampled CUDA phase times and end-of-step rank skew."""
        if "dist_profile_bwd_rank_max_ms" not in payload:
            return None

        def rank_range(name: str) -> str:
            minimum = float(payload[f"dist_profile_{name}_rank_min_ms"])
            maximum = float(payload[f"dist_profile_{name}_rank_max_ms"])
            return f"{minimum:.1f}..{maximum:.1f}ms"

        return (
            f"[dist-profile] step={int(payload.get('global_step', 0))} "
            f"world={int(payload.get('dist_profile_world_size', 1))} "
            f"rank-range fwd={rank_range('fwd')} "
            f"bwd+grad-reduce={rank_range('bwd')} clip={rank_range('clip')} "
            f"opt={rank_range('opt')} rank-wait={rank_range('rank_wait')}"
        )

    def _console_once_train_static_line(self, payload: dict[str, Any]) -> str | None:
        """Render newly observed non-timeseries fields once, before the step line."""
        if payload.get("event") != "train_step":
            return None
        items = []
        keys = _CONSOLE_ONCE_TRAIN_METRIC_KEYS
        if MetricsLogger._is_w0_train_payload(payload):
            keys += _CONSOLE_ONCE_W0_TRAIN_METRIC_KEYS
        for key in keys:
            if key in self._printed_train_static_keys or key not in payload:
                continue
            value = payload[key]
            if value is None:
                continue
            self._printed_train_static_keys.add(key)
            items.append(f"{key}={value}")
        if not items:
            return None
        return "[train-static] " + " ".join(items)

    def _update_objective_snapshots(self, payload: dict[str, Any], step: int) -> None:
        for key in (
            "flow_loss",
            "fast_loss",
            "vlm_ce",
            "gen_2d_loss",
            "geometry_3d_loss",
            "motion_3d_loss",
            "world_3d_loss",
            "dino_flow_loss",
            "motion_flow_loss",
            "geometry_loss",
        ):
            value = self._as_float(payload.get(key))
            if value is not None:
                self._objective_snapshots[key] = (value, step)

    def _update_step_time_window(self, payload: dict[str, Any]) -> None:
        average = self._as_float(payload.get("t_window_s_per_step"))
        steps = int(payload.get("t_window_steps", 0) or 0)
        if average is None or steps <= 0:
            # Backward-compatible fallback for hand-written test payloads and older
            # trainer records. A single record represents one step in that case.
            average = self._as_float(payload.get("t_iter_max", payload.get("t_iter")))
            steps = 1
        if average is None:
            return
        self._step_time_window.extend(
            [average] * min(steps, self._step_time_window.maxlen)
        )

    def _format_objective(self, key: str, current_step: int) -> str:
        label = {
            "flow_loss": "fm_loss",
            "fast_loss": "fast_loss",
            "vlm_ce": "vlm_loss",
            "gen_2d_loss": "2d_loss",
            "geometry_3d_loss": "3d_geo",
            "motion_3d_loss": "3d_motion",
            "world_3d_loss": "3d_loss",
            "dino_flow_loss": "dino_flow_loss",
            "motion_flow_loss": "motion_flow_loss",
            "geometry_loss": "geometry_loss",
        }[key]
        snapshot = self._objective_snapshots.get(key)
        if snapshot is None:
            return f"{label}=-"
        value, recorded_step = snapshot
        age = current_step - recorded_step
        stale = "" if age <= 0 else f"@-{age}"
        return f"{label}={self._format_loss(value)}{stale}"

    @staticmethod
    def _as_float(value: Any) -> float | None:
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            return float(value)
        return None

    @staticmethod
    def _format_loss(value: float | None) -> str:
        return "-" if value is None else f"{value:.5g}"

    @staticmethod
    def _format_seconds(value: float | None) -> str:
        return "-" if value is None else f"{value:.2f}s"

    @staticmethod
    def _format_duration(seconds: float) -> str:
        seconds = max(0, int(round(seconds)))
        days, seconds = divmod(seconds, 24 * 60 * 60)
        hours, seconds = divmod(seconds, 60 * 60)
        minutes, seconds = divmod(seconds, 60)
        if days:
            return f"{days}d{hours:02d}h"
        if hours:
            return f"{hours}h{minutes:02d}m"
        if minutes:
            return f"{minutes}m{seconds:02d}s"
        return f"{seconds}s"

    def close(self, *, exit_code: int = 0) -> None:
        """Flush and finish the active W&B run."""
        if self.wandb_run is not None:
            self.wandb_run.finish(exit_code=exit_code)
            self.wandb_run = None

    def _init_wandb(self, run_config: dict[str, Any]):
        if not self.wandb_config.enabled or not is_main_process():
            return None
        try:
            import wandb
        except ImportError as exc:
            raise ImportError(
                "W&B logging is enabled but wandb is not installed. "
                "Install requirements.txt and retry."
            ) from exc

        init_kwargs = {
            "project": self.wandb_config.project,
            "entity": self.wandb_config.entity,
            "name": self.wandb_config.name,
            "group": self.wandb_config.group,
            "job_type": self.wandb_config.job_type,
            "tags": self.wandb_config.tags or None,
            "notes": self.wandb_config.notes,
            "mode": self.wandb_config.mode,
            "id": self.wandb_config.run_id,
            "resume": self.wandb_config.resume,
            "save_code": self.wandb_config.save_code,
            "config": run_config,
        }
        if self.output_dir is not None:
            init_kwargs["dir"] = str(self.output_dir)
        run = self._init_run_or_fall_offline(wandb, init_kwargs)
        run.define_metric("global_step")
        run.define_metric("train/*", step_metric="global_step")
        run.define_metric("validation/*", step_metric="global_step")
        return run

    @staticmethod
    def _init_run_or_fall_offline(wandb, init_kwargs: dict[str, Any]):
        """Start the run, degrading ``online`` to ``offline`` rather than dying.

        An unusable API key makes ``wandb.init`` raise, and it raises before the
        first training step -- so the old behaviour was to lose a multi-hour run
        to a missing credential (EIP task 7324 went exactly that way, which is
        why several configs in this repo hard-code ``mode: disabled``).

        Offline keeps every metric on disk and prints the one command that
        uploads it afterwards, so the cost of a bad key is a delay in seeing the
        charts rather than the run.
        """
        try:
            return wandb.init(**init_kwargs)
        except Exception as error:
            if init_kwargs.get("mode") != "online":
                raise
            print(
                f"WARNING: W&B online init failed ({error}); continuing in offline "
                "mode. Metrics are recorded locally -- upload them later with the "
                "`wandb sync` command printed at the end of the run.",
                flush=True,
            )
        init_kwargs["mode"] = "offline"
        run = wandb.init(**init_kwargs)
        print(f"W&B offline run directory: {run.dir}", flush=True)
        return run

    @staticmethod
    def _wandb_payload(payload: dict[str, Any]) -> dict[str, Any]:
        event = str(payload.get("event", "metrics"))
        step = int(payload.get("global_step", 0))
        result: dict[str, Any] = {"global_step": step, "event": event}
        if event == "train_step":
            fixed_keys = ("loss", "loss_ema", "loss_mean", "grad_norm")
            for key in fixed_keys:
                if key in payload:
                    result[f"train/{key}"] = payload[key]
            result["train/modality"] = payload.get("modality", "unknown")
            result["train/loss_name"] = payload.get("loss_name", "unknown")
            learning_rates = payload.get("learning_rates")
            if isinstance(learning_rates, (list, tuple)) and learning_rates:
                result["train/learning_rate"] = learning_rates[0]
            # Forward the model's component metrics (flow_loss / fast_loss / ...) so
            # each loss term charts in W&B alongside the total.
            dropped = _DROPPED_WANDB_TRAIN_METRIC_KEYS
            if MetricsLogger._is_w0_train_payload(payload):
                dropped = dropped | _DROPPED_WANDB_W0_TRAIN_METRIC_KEYS
            skip = {
                "event",
                "global_step",
                "learning_rates",
                "modality",
                "loss_name",
                *fixed_keys,
                *dropped,
            }
            for key, value in payload.items():
                if (
                    key not in skip
                    and isinstance(value, (int, float))
                    and not isinstance(value, bool)
                ):
                    result[f"train/{key}"] = value
        elif event == "validation":
            if "epoch" in payload:
                result["validation/epoch"] = payload["epoch"]
            # Every ``val_*`` scalar charts under validation/, so a new
            # per-modality or per-component key needs no change here. The
            # per-horizon vector is a list and is plotted separately.
            for key, value in payload.items():
                if not key.startswith("val_"):
                    continue
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    result[f"validation/{key[len('val_') :]}"] = value
            has_legacy_modality_loss = any(
                isinstance(key, str) and key.endswith("/loss") for key in payload
            )
            if has_legacy_modality_loss:
                result.pop("validation/loss", None)
                result.pop("validation/batches", None)
                for key, value in payload.items():
                    if (
                        not key.startswith("val_")
                        and key not in {"event", "global_step", "epoch"}
                        and isinstance(value, (int, float))
                        and not isinstance(value, bool)
                    ):
                        result[f"validation/{key}"] = value
            curve = payload.get("val_from_noise_per_horizon")
            if isinstance(curve, list) and curve:
                plot = _per_horizon_plot(curve, step)
                if plot is not None:
                    result["validation/from_noise/per_horizon_curve"] = plot
        else:
            result.update(payload)
        return result

    @staticmethod
    def _record_payload(payload: dict[str, Any]) -> dict[str, Any]:
        if payload.get("event") != "train_step":
            return dict(payload)
        dropped = _DROPPED_TRAIN_METRIC_KEYS
        if MetricsLogger._is_w0_train_payload(payload):
            dropped = dropped | _DROPPED_W0_TRAIN_METRIC_KEYS
        return {key: value for key, value in payload.items() if key not in dropped}

    @staticmethod
    def _is_w0_train_payload(payload: dict[str, Any]) -> bool:
        return str(payload.get("loss_name", "")).startswith("magic_w0_")

    def log_image(self, key: str, path, *, global_step: int) -> None:
        """Send one PNG to W&B under ``key`` (a no-op without a run).

        The file is written regardless -- W&B is where it is convenient to look,
        not where it lives.
        """
        if not is_main_process() or self.wandb_run is None:
            return
        try:
            import wandb

            self.wandb_run.log(
                {key: wandb.Image(str(path)), "global_step": int(global_step)}
            )
        except Exception as error:  # pragma: no cover - depends on the environment
            print(f"WARNING: could not log {key} to W&B ({error})", flush=True)


def _per_horizon_plot(curve: list[float], step: int):
    """A W&B line chart of error against horizon position, or ``None``.

    Native W&B rather than only the PNG so the curve stays zoomable and its
    values readable on hover; the PNG carries the run's whole history instead.
    """
    try:
        import wandb

        table = wandb.Table(
            data=[[index, float(value)] for index, value in enumerate(curve)],
            columns=["horizon", "mse"],
        )
        return wandb.plot.line(
            table, "horizon", "mse", title=f"from-noise MSE by horizon (step {step})"
        )
    except Exception:  # pragma: no cover - depends on the environment
        return None


__all__ = ["MetricsLogger", "WandbConfig"]
