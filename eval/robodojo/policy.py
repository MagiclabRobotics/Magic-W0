"""RoboDojo inference adapter components."""

from __future__ import annotations
import os
from datetime import datetime, timezone
from typing import Any
import numpy as np
from .layout import (
    _CAMERA_ALIASES,
    _CAMERA_EEF_LAYOUTS,
    camera_eef_layout_spec,
    detect_camera_eef_layout,
    detect_checkpoint_action_layout,
    normalize_camera_eef_layout,
    validate_camera_eef_normalization_layout,
)
from .observation import encode_observation
from .action import (
    limit_camera_eef_target_steps,
    limit_joint_target_steps,
    stable_noise_seed,
    unpack_camera_eef_actions,
    unpack_joint_actions,
)
from .normalization import DeltaNormalizationStats


from XPolicyLab.model_template import ModelTemplate
from .loading import CheckpointLoaderMixin
from .diagnostics import DiagnosticsMixin
from .paths import (
    REPO_ROOT,
    _resolve_path,
    resolve_checkpoint,
    resolve_robodojo_root,
    load_robot_action_dim_info,
)
from .history import HistoryBuffer


class Model(CheckpointLoaderMixin, DiagnosticsMixin, ModelTemplate):
    """Serve the Magic_W0 history checkpoint through XPolicyLab."""

    def __init__(self, model_cfg: dict[str, Any]):
        self.magic_w0_root = REPO_ROOT
        self.action_type = str(model_cfg.get("action_type") or "joint")
        if self.action_type not in ("joint", "ee", "camera_ee"):
            raise ValueError("Magic_W0 action_type must be joint, ee, or camera_ee")
        self.env_cfg_type = str(model_cfg.get("env_cfg_type") or "arx_x5")
        self.robodojo_root = resolve_robodojo_root(model_cfg)
        self.robot_action_dim_info = load_robot_action_dim_info(
            self.env_cfg_type, self.robodojo_root
        )
        self.robot_action_dim = (
            sum(self.robot_action_dim_info["arm_dim"])
            + sum(self.robot_action_dim_info["ee_dim"])
            if self.action_type == "joint"
            else 20
        )
        self.camera_eef_layout = "compact20"
        self.checkpoint_action_layout = "joint"
        self.model_input_layout = "action_type"
        self.state_active_robot_indices = tuple(range(self.robot_action_dim))
        self.action_active_robot_indices = tuple(range(self.robot_action_dim))
        self.execute_steps = int(model_cfg.get("execute_steps", 4))
        self.num_inference_steps = int(model_cfg.get("num_inference_steps", 10))
        self.seed = int(model_cfg.get("seed") or 0)
        self.default_task = str(model_cfg.get("task_name") or "").replace("_", " ")
        self.embodiment = str(model_cfg.get("embodiment") or "arx_x5_sim")
        self.clip_normalized_actions = bool(
            model_cfg.get("clip_normalized_actions", True)
        )
        self.reuse_initial_noise = bool(model_cfg.get("reuse_initial_noise", False))
        self.max_joint_step = self._optional_positive_float(
            model_cfg.get("max_joint_step")
        )
        self.max_gripper_step = self._optional_positive_float(
            model_cfg.get("max_gripper_step")
        )
        self.max_eef_translation_step = self._optional_positive_float(
            model_cfg.get("max_eef_translation_step", model_cfg.get("max_joint_step"))
        )
        self.action_diagnostics = bool(
            model_cfg.get("action_diagnostics", False)
        ) or os.environ.get("MAGIC_W0_ACTION_DIAGNOSTICS", "").lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        self.action_diagnostics_interval = int(
            model_cfg.get("action_diagnostics_interval", 10)
        )
        self.save_model_inputs = bool(model_cfg.get("save_model_inputs", False))
        self.save_model_input_sequence = bool(
            model_cfg.get("save_model_input_sequence", False)
        )
        if self.save_model_input_sequence and not self.save_model_inputs:
            raise ValueError(
                "save_model_input_sequence=True requires save_model_inputs=True"
            )
        dump_root = _resolve_path(
            model_cfg.get(
                "model_input_dump_root",
                REPO_ROOT / "runs/robodojo/diagnostics",
            ),
            base=REPO_ROOT,
        )
        self.model_input_dump_dir = dump_root / datetime.now(timezone.utc).strftime(
            "%Y-%m-%d_%H-%M-%S"
        )
        self._trajectory_files_initialized: set[tuple[str, int]] = set()
        self._trajectory_sample_counts: dict[tuple[str, int], int] = {}
        if self.execute_steps <= 0:
            raise ValueError("execute_steps must be positive")
        if self.num_inference_steps <= 0:
            raise ValueError("num_inference_steps must be positive")
        if self.action_diagnostics_interval <= 0:
            raise ValueError("action_diagnostics_interval must be positive")
        self._latest_observations: dict[int, dict[str, Any]] = {}
        self._latest_env_idx_list = [0]
        self._replan_steps: dict[int, int] = {}
        self.checkpoint = resolve_checkpoint(model_cfg, self.magic_w0_root)
        self.policy = self._load_policy(model_cfg)
        self.model = self.policy
        self.action_dim = int(self.policy.config.action_dim)
        self.state_dim = int(self.policy.config.state_dim)
        self.chunk_size = int(self.policy.config.chunk_size)
        if self.execute_steps > self.chunk_size:
            raise ValueError(
                f"execute_steps={self.execute_steps} exceeds chunk_size={self.chunk_size}"
            )
        self._initialize_history()
        self.stats = self._load_normalization_stats(model_cfg)
        self.checkpoint_action_layout = detect_checkpoint_action_layout(self.stats)
        checkpoint_is_joint_eef = self.checkpoint_action_layout == "joint_eef_union32"
        checkpoint_is_eef = self.checkpoint_action_layout in _CAMERA_EEF_LAYOUTS
        requested_is_eef = self.action_type in ("ee", "camera_ee")
        if not checkpoint_is_joint_eef and checkpoint_is_eef != requested_is_eef:
            expected = "ee" if checkpoint_is_eef else "joint"
            raise ValueError(
                f"checkpoint action representation requires action_type={expected!r}, "
                f"got {self.action_type!r}"
            )
        if checkpoint_is_joint_eef:
            requested_layout = normalize_camera_eef_layout(
                str(model_cfg.get("camera_eef_layout") or "auto")
            )
            if requested_layout not in ("auto", "union32"):
                raise ValueError(
                    "joint+EEF checkpoint uses camera_eef_layout='union32', "
                    f"but deployment requested {requested_layout!r}"
                )
            self.camera_eef_layout = "union32"
            self.model_input_layout = "joint_eef_union32"
            self.robot_action_dim = 32
            self.state_active_robot_indices = tuple(range(32))
            self.action_active_robot_indices = tuple(range(32))
        elif requested_is_eef:
            detected_layout = detect_camera_eef_layout(self.stats)
            requested_layout = normalize_camera_eef_layout(
                str(model_cfg.get("camera_eef_layout") or "auto")
            )
            if requested_layout not in ("auto", detected_layout):
                raise ValueError(
                    f"checkpoint uses camera_eef_layout={detected_layout!r}, "
                    f"but deployment requested {requested_layout!r}"
                )
            self.camera_eef_layout = detected_layout
            layout_spec = camera_eef_layout_spec(detected_layout)
            self.robot_action_dim = layout_spec["dim"]
            self.state_active_robot_indices = layout_spec["active_indices"]
            self.action_active_robot_indices = layout_spec["active_indices"]
            validate_camera_eef_normalization_layout(self.stats, detected_layout)
        self.state_active_robot_indices = self._resolve_active_robot_indices(
            model_cfg.get("state_active_robot_indices"),
            default=self.state_active_robot_indices,
            upper_bound=min(self.robot_action_dim, self.state_dim),
            name="state_active_robot_indices",
        )
        self.action_active_robot_indices = self._resolve_active_robot_indices(
            model_cfg.get("action_active_robot_indices"),
            default=self.action_active_robot_indices,
            upper_bound=min(self.robot_action_dim, self.action_dim),
            name="action_active_robot_indices",
        )
        self._validate_normalization_stats()
        print(
            "[Magic_W0] action representation=chunk_delta "
            f"delta_indices={list(self.stats.action_delta_indices)} "
            f"checkpoint_layout={self.checkpoint_action_layout} "
            f"model_input_layout={self.model_input_layout} "
            f"output_action_type={self.action_type} "
            f"camera_eef_layout={self.camera_eef_layout if (requested_is_eef or checkpoint_is_joint_eef) else 'n/a'} "
            f"state_active_indices={list(self.state_active_robot_indices)} "
            f"action_active_indices={list(self.action_active_robot_indices)}",
            flush=True,
        )
        del self._checkpoint_payload
        if self.save_model_inputs:
            self.model_input_dump_dir.mkdir(parents=True, exist_ok=True)
            print(
                f"[Magic_W0] model inputs will be saved to {self.model_input_dump_dir}",
                flush=True,
            )

    def _load_normalization_stats(
        self, model_cfg: dict[str, Any]
    ) -> DeltaNormalizationStats:
        """Load the legacy per-dimension action normalization contract."""
        return DeltaNormalizationStats.from_checkpoint(
            self._checkpoint_payload,
            source_hint=str(model_cfg.get("normalization_source") or "RoboDojo"),
        )

    @staticmethod
    def _resolve_active_robot_indices(
        configured: Any,
        *,
        default: tuple[int, ...],
        upper_bound: int,
        name: str,
    ) -> tuple[int, ...]:
        if configured is None:
            return tuple(default)
        if isinstance(configured, (str, bytes)) or not isinstance(
            configured, (list, tuple)
        ):
            raise TypeError(f"{name} must be a list of integer indices")
        if any(
            isinstance(index, bool) or not isinstance(index, int)
            for index in configured
        ):
            raise TypeError(f"{name} must contain only integer indices")
        indices = tuple(configured)
        if not indices:
            raise ValueError(f"{name} must not be empty")
        if len(set(indices)) != len(indices):
            raise ValueError(f"{name} must not contain duplicate indices")
        invalid = [index for index in indices if index < 0 or index >= upper_bound]
        if invalid:
            raise ValueError(
                f"{name} contains indices outside [0, {upper_bound}): {invalid}"
            )
        return indices

    def _validate_normalization_stats(self) -> None:
        """Check the legacy server's flat state/action statistics."""
        if self.stats.state_low.shape != (self.state_dim,):
            raise ValueError(
                f"normalization state shape={self.stats.state_low.shape}, "
                f"expected ({self.state_dim},)"
            )
        if self.stats.action_low.shape != (self.action_dim,):
            raise ValueError(
                f"normalization action shape={self.stats.action_low.shape}, "
                f"expected ({self.action_dim},)"
            )

    @staticmethod
    def _optional_positive_float(value: Any) -> float | None:
        if value is None:
            return None
        parsed = float(value)
        if parsed <= 0.0:
            raise ValueError("action step limits must be positive or null")
        return parsed

    def _initialize_history(self) -> None:
        """Use the resolved checkpoint configuration for both model and sampler."""
        config = self.policy.config
        self.history_camera = str(config.history_camera_key).split(".")[-1]
        if self.history_camera not in _CAMERA_ALIASES:
            raise ValueError(
                f"unsupported history camera {config.history_camera_key!r}"
            )
        self.history = HistoryBuffer(
            config.history_frames, config.history_interval, self.execute_steps
        )

    def update_obs(self, obs):
        self.update_obs_batch([obs])

    def update_obs_batch(self, obs_list):
        env_indices = [
            int(obs.get("env_idx", index)) for index, obs in enumerate(obs_list)
        ]
        encoded = [
            encode_observation(
                obs,
                action_type=self.action_type,
                robot_action_dim_info=self.robot_action_dim_info,
                fallback_instruction=self.default_task,
                camera_eef_layout=self.camera_eef_layout,
                model_input_layout=self.model_input_layout,
            )
            for obs in obs_list
        ]
        if self.history is not None:
            for env_idx, observation in zip(env_indices, encoded, strict=True):
                self._attach_history(env_idx, observation)
        if self.save_model_inputs:
            self._append_trajectory_rows("measured_state", env_indices, encoded)
        self._latest_env_idx_list = env_indices
        self._latest_observations = dict(zip(env_indices, encoded, strict=True))

    def _attach_history(self, env_idx: int, observation: dict[str, Any]) -> None:
        """Attach the fixed-size strictly-past clip to one encoded observation."""
        if self.history is None:
            return
        step = self.history.step_of(self._replan_steps.get(int(env_idx), 0))
        current = observation["images"][self.history_camera]
        clip, is_pad = self.history.clip(env_idx, step, current)
        observation["images"][f"{self.history_camera}_history"] = clip
        observation["history_is_pad"] = is_pad
        self.history.record(env_idx, step, current)

    def _prepare_model_batch(
        self, env_indices: list[int], observations: list[dict[str, Any]]
    ):
        import torch

        from magic_w0 import (
            ACTION_DIM_MASK,
            EMBODIMENT,
            STATE,
            STATE_DIM_MASK,
            TASK,
            _prepare_image,
        )

        raw_states = np.stack([obs["state"] for obs in observations])
        if raw_states.shape[-1] != self.robot_action_dim:
            raise ValueError(
                f"observation state dim={raw_states.shape[-1]}, "
                f"expected robot dim={self.robot_action_dim}"
            )
        padded_states = np.zeros((len(observations), self.state_dim), dtype=np.float32)
        padded_states[:, : self.robot_action_dim] = raw_states
        state_dim_mask = np.zeros_like(padded_states, dtype=bool)
        state_dim_mask[:, list(self.state_active_robot_indices)] = True
        normalized_states = self.stats.normalize_state(padded_states)
        # Match training exactly even if a future normalization artifact maps a
        # raw padding zero to a non-zero normalized value.
        normalized_states[~state_dim_mask] = 0.0
        action_dim_mask = np.zeros(
            (
                len(observations),
                self.chunk_size,
                self.action_dim,
            ),
            dtype=bool,
        )
        action_dim_mask[:, :, list(self.action_active_robot_indices)] = True
        batch: dict[str, Any] = {
            STATE: torch.from_numpy(normalized_states),
            STATE_DIM_MASK: torch.from_numpy(state_dim_mask),
            ACTION_DIM_MASK: torch.from_numpy(action_dim_mask),
            TASK: [obs["task"] for obs in observations],
            EMBODIMENT: [self.embodiment for _ in observations],
        }
        for key in self.policy.config.camera_keys:
            short_name = key.removeprefix("observation.images.")
            if short_name not in _CAMERA_ALIASES:
                raise ValueError(f"unsupported model camera key {key!r}")
            # Use the exact resize path used to train the checkpoint. In
            # particular, a 480x640 frame becomes 192x256 and is
            # center-padded to 256x256 instead of being stretched or left for
            # the generic Qwen image processor to resize dynamically.
            batch[key] = _prepare_image(
                torch.from_numpy(
                    np.stack([obs["images"][short_name] for obs in observations])
                ),
                self.policy.config.image_size,
            )
        if self.history is not None:
            batch.update(
                self._history_batch(observations, self.policy.config.image_size)
            )
        prepared = self.policy.prepare_inference_batch(batch, self.device)
        prepared["magic_w0.action_dim_mask"] = batch["magic_w0.action_dim_mask"].to(
            device=self.device
        )
        if self.save_model_inputs:
            self._dump_model_inputs(
                env_indices,
                observations,
                padded_states,
                normalized_states,
                state_dim_mask,
                batch,
                prepared,
            )
        return prepared

    def _history_batch(self, observations: list[dict[str, Any]], image_size: int):
        """Prepare temporal images for the model layer's history input contract."""
        import torch
        from magic_w0 import HISTORY_FRAMES, HISTORY_LEN, _prepare_image

        clips = np.stack(
            [obs["images"][f"{self.history_camera}_history"] for obs in observations]
        )
        is_pad = np.stack([obs["history_is_pad"] for obs in observations])
        if is_pad.dtype != np.bool_ or is_pad.ndim != 2:
            raise ValueError("history_is_pad must be a boolean [B,N] array")
        if clips.ndim != 5 or clips.shape[:2] != is_pad.shape:
            raise ValueError("history clip must be [B,N,C,H,W] matching history_is_pad")
        lengths = (~is_pad).sum(axis=1).astype(np.int64)
        capacity = is_pad.shape[1]
        expected = np.arange(capacity)[None] < (capacity - lengths)[:, None]
        if not np.array_equal(is_pad, expected):
            raise ValueError("history_is_pad must be left-padded")
        frames = _prepare_image(
            torch.from_numpy(clips.reshape(-1, *clips.shape[2:])), image_size
        )
        frames = frames.reshape(len(observations), capacity, *frames.shape[1:])
        return {
            HISTORY_FRAMES: frames,
            HISTORY_LEN: torch.from_numpy(lengths),
        }

    def _make_initial_noise(self, env_indices, observations):
        import torch

        chunks = []
        for env_idx, observation in zip(env_indices, observations, strict=True):
            generator = torch.Generator(device=self.device)
            generator.manual_seed(
                stable_noise_seed(
                    self.seed,
                    observation["layout_id"],
                    0
                    if self.reuse_initial_noise
                    else self._replan_steps.get(env_idx, 0),
                )
            )
            chunks.append(
                torch.randn(
                    self.chunk_size,
                    self.action_dim,
                    generator=generator,
                    device=self.device,
                )
            )
        return torch.stack(chunks)

    def get_action(self):
        return self.get_action_batch([self._latest_env_idx_list[0]])[0]

    def get_action_batch(self, env_idx_list=None):
        if not self._latest_observations:
            raise AssertionError(
                self._error_msg("call update_obs/update_obs_batch first")
            )
        env_indices = [
            int(value) for value in (env_idx_list or self._latest_env_idx_list)
        ]
        missing = [
            index for index in env_indices if index not in self._latest_observations
        ]
        if missing:
            raise KeyError(f"missing observations for env_idx {missing}")
        observations = [self._latest_observations[index] for index in env_indices]

        import torch

        prepared = self._prepare_model_batch(env_indices, observations)
        noise = self._make_initial_noise(env_indices, observations)
        action_dim_mask = prepared.get("magic_w0.action_dim_mask")
        with torch.inference_mode():
            if self.device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    normalized = self.policy.generate_actions(
                        prepared,
                        num_steps=self.num_inference_steps,
                        initial_noise=noise,
                        action_dim_mask=action_dim_mask,
                    )
            else:
                normalized = self.policy.generate_actions(
                    prepared,
                    num_steps=self.num_inference_steps,
                    initial_noise=noise,
                    action_dim_mask=action_dim_mask,
                )
        normalized_np = normalized.float().cpu().numpy()
        padded_actions = self.stats.denormalize_action(
            normalized_np, clip=self.clip_normalized_actions
        )[:, : self.execute_steps]
        reconstruct = (
            self.stats.reconstruct_camera_eef_targets
            if self.checkpoint_action_layout == "joint_eef_union32"
            or self.action_type in ("ee", "camera_ee")
            else self.stats.reconstruct_joint_targets
        )
        reconstructed_targets = np.stack(
            [
                reconstruct(chunk, observation["state"])
                for chunk, observation in zip(padded_actions, observations, strict=True)
            ]
        )
        if (
            self.checkpoint_action_layout == "joint_eef_union32"
            and self.action_type == "joint"
        ):
            raw_actions = reconstructed_targets[..., :14]
            execution_states = [
                observation["state"][:14] for observation in observations
            ]
        else:
            raw_actions = reconstructed_targets[..., : self.robot_action_dim]
            execution_states = [observation["state"] for observation in observations]
        if self.save_model_inputs:
            for batch_index, env_idx in enumerate(env_indices):
                self._append_trajectory_rows(
                    "raw_action",
                    [env_idx] * len(raw_actions[batch_index]),
                    list(raw_actions[batch_index]),
                )
        if self.action_type == "joint":
            actions = np.stack(
                [
                    limit_joint_target_steps(
                        chunk,
                        state,
                        self.robot_action_dim_info,
                        max_joint_step=self.max_joint_step,
                        max_gripper_step=self.max_gripper_step,
                    )
                    for chunk, state in zip(raw_actions, execution_states, strict=True)
                ]
            )
        else:
            actions = np.stack(
                [
                    limit_camera_eef_target_steps(
                        chunk,
                        state,
                        max_translation_step=self.max_eef_translation_step,
                        max_gripper_step=self.max_gripper_step,
                        layout=self.camera_eef_layout,
                    )
                    for chunk, state in zip(raw_actions, execution_states, strict=True)
                ]
            )
        if self.save_model_inputs:
            for batch_index, env_idx in enumerate(env_indices):
                self._append_trajectory_rows(
                    "commanded_action",
                    [env_idx] * len(actions[batch_index]),
                    list(actions[batch_index]),
                )
        if self.action_diagnostics:
            for batch_index, env_idx in enumerate(env_indices):
                replan_step = self._replan_steps.get(env_idx, 0)
                if replan_step % self.action_diagnostics_interval == 0:
                    raw_delta = np.diff(
                        np.concatenate(
                            [
                                execution_states[batch_index][None],
                                raw_actions[batch_index],
                            ],
                            axis=0,
                        ),
                        axis=0,
                    )
                    filtered_delta = np.diff(
                        np.concatenate(
                            [
                                execution_states[batch_index][None],
                                actions[batch_index],
                            ],
                            axis=0,
                        ),
                        axis=0,
                    )
                    eef_diag = ""
                    if (
                        self.action_type != "joint"
                        or self.checkpoint_action_layout == "joint_eef_union32"
                    ):
                        spec = camera_eef_layout_spec(self.camera_eef_layout)
                        raw_gripper_delta = raw_delta[:, list(spec["gripper_indices"])]
                        filtered_gripper_delta = filtered_delta[
                            :, list(spec["gripper_indices"])
                        ]
                        raw_eef_translation_delta = np.concatenate(
                            [
                                raw_delta[:, start : start + 3]
                                for start in spec["pose_starts"]
                            ],
                            axis=0,
                        )
                        filtered_eef_translation_delta = np.concatenate(
                            [
                                filtered_delta[:, start : start + 3]
                                for start in spec["pose_starts"]
                            ],
                            axis=0,
                        )
                        eef_diag = (
                            f" raw_gripper_abs_max={np.max(np.abs(raw_gripper_delta)):.4f}"
                            f" filtered_gripper_abs_max={np.max(np.abs(filtered_gripper_delta)):.4f}"
                            f" raw_eef_translation_norm_max={np.max(np.linalg.norm(raw_eef_translation_delta, axis=1)):.4f}"
                            f" filtered_eef_translation_norm_max={np.max(np.linalg.norm(filtered_eef_translation_delta, axis=1)):.4f}"
                        )
                    print(
                        "[Magic_W0][actions] "
                        f"env={env_idx} replan={replan_step} "
                        f"normalized_abs_max={np.max(np.abs(normalized_np[batch_index])):.3f} "
                        f"normalized_oob_frac={np.mean(np.abs(normalized_np[batch_index]) > 1.0):.4f} "
                        f"raw_step_abs_max={np.max(np.abs(raw_delta)):.4f} "
                        f"filtered_step_abs_max={np.max(np.abs(filtered_delta)):.4f}"
                        f"{eef_diag}",
                        flush=True,
                    )
        for env_idx in env_indices:
            self._replan_steps[env_idx] = self._replan_steps.get(env_idx, 0) + 1
        if self.action_type == "joint":
            return [
                unpack_joint_actions(chunk, self.robot_action_dim_info)
                for chunk in actions
            ]
        return [
            unpack_camera_eef_actions(
                chunk, layout=self.camera_eef_layout, action_type=self.action_type
            )
            for chunk in actions
        ]

    def reset(self):
        self._latest_observations = {}
        self._latest_env_idx_list = [0]
        self._replan_steps = {}
        if self.history is not None:
            self.history.reset()
