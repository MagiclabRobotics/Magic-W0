"""RoboDojo inference adapter components."""

from __future__ import annotations
from dataclasses import asdict, fields
from typing import Any
from .layout import (
    detect_camera_eef_layout,
    detect_checkpoint_action_layout,
    normalize_camera_eef_layout,
    validate_camera_eef_normalization_layout,
)
from .normalization import DeltaNormalizationStats


from .paths import REPO_ROOT, _resolve_path
from magic_w0.checkpoint import split_policy_checkpoint_state
from .history import validate_history_schedule

_POLICY_DIR = REPO_ROOT


class CheckpointLoaderMixin:
    def _load_policy(self, model_cfg: dict[str, Any]):
        import torch

        checkpoint = self.checkpoint
        print(f"[Magic_W0] loading {checkpoint}", flush=True)
        payload = torch.load(
            checkpoint, map_location="cpu", weights_only=True, mmap=True
        )
        self._checkpoint_payload = payload
        payload_config = dict(payload.get("config") or {})
        if any(key.startswith("history_") for key in model_cfg):
            raise ValueError(
                "history configuration belongs in checkpoint.config or model_params"
            )
        history_overrides = self._model_config_overrides(model_cfg)
        for key, value in history_overrides.items():
            if (
                key.startswith("history_")
                and key in payload_config
                and value != payload_config[key]
            ):
                raise ValueError(f"history override conflicts with checkpoint: {key}")
        from magic_w0 import MagicW0, MagicW0Config, validate_checkpoint_modes

        validate_checkpoint_modes({**payload_config, **history_overrides})
        known_config_keys = {field.name for field in fields(MagicW0Config)}
        if model_cfg.get("attention_implementation", "sdpa") != "sdpa":
            raise ValueError("Magic_W0 inference supports only SDPA")
        if self.action_type in ("ee", "camera_ee"):
            requested_layout = normalize_camera_eef_layout(
                str(model_cfg.get("camera_eef_layout") or "auto")
            )
            preview_stats = DeltaNormalizationStats.from_checkpoint(
                payload,
                source_hint=str(model_cfg.get("normalization_source") or "RoboDojo"),
            )
            preview_action_layout = detect_checkpoint_action_layout(preview_stats)
            detected_layout = detect_camera_eef_layout(preview_stats)
            if requested_layout != "auto" and requested_layout != detected_layout:
                raise ValueError(
                    f"checkpoint uses camera_eef_layout={detected_layout!r}, "
                    f"but deployment requested {requested_layout!r}"
                )
            if preview_action_layout != "joint_eef_union32":
                validate_camera_eef_normalization_layout(preview_stats, detected_layout)
        state = payload.get("model", payload)
        config_values = asdict(MagicW0Config())
        # Read inference settings only; unrelated checkpoint metadata is ignored.
        config_values.update(
            {
                key: value
                for key, value in payload_config.items()
                if key in known_config_keys
            }
        )
        # Mode flags are validated above; only actual inference dimensions are configurable.
        from magic_w0 import SUPPORTED_MODES

        unknown_overrides = (
            set(history_overrides) - known_config_keys - set(SUPPORTED_MODES)
        )
        if unknown_overrides:
            raise ValueError(
                f"unsupported inference overrides: {sorted(unknown_overrides)}"
            )
        config_values.update(
            {
                key: value
                for key, value in history_overrides.items()
                if key in known_config_keys
            }
        )
        config_values["vlm_checkpoint"] = str(
            _resolve_path(
                model_cfg.get(
                    "vlm_checkpoint", REPO_ROOT / "checkpoints/qwen3.5-2b-assets"
                ),
                base=self.magic_w0_root,
            )
        )
        config = MagicW0Config.from_dict(config_values)
        validate_history_schedule(config.history_interval, self.execute_steps)
        if config.action_dim < self.robot_action_dim:
            raise ValueError(
                f"checkpoint action_dim={config.action_dim} is smaller than "
                f"environment action dim={self.robot_action_dim}"
            )
        if config.state_dim < self.robot_action_dim:
            raise ValueError(
                f"checkpoint state_dim={config.state_dim} is smaller than "
                f"environment state dim={self.robot_action_dim}"
            )
        vlm_state, policy_state = split_policy_checkpoint_state(state)
        policy = MagicW0(config, vlm_state_dict=vlm_state)
        incompatible = policy.load_state_dict(policy_state, strict=False)
        illegal_missing = [
            key
            for key in incompatible.missing_keys
            if not key.startswith(("vlm.", "_lm_head."))
        ]
        illegal_unexpected = incompatible.unexpected_keys
        if illegal_missing or illegal_unexpected:
            raise RuntimeError(
                "checkpoint mismatch outside the VLM loaded by Transformers: "
                f"missing={illegal_missing}, unexpected={illegal_unexpected}"
            )
        requested_device = str(model_cfg.get("device") or "cuda")
        if requested_device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for Magic_W0 evaluation")
        self.device = torch.device(requested_device)
        policy.to(self.device).eval()
        torch.manual_seed(self.seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(self.seed)
        print(
            f"[Magic_W0] ready device={self.device} "
            f"flow_steps={self.num_inference_steps} execute_steps={self.execute_steps}",
            flush=True,
        )
        return policy

    @staticmethod
    def _model_config_overrides(model_cfg: dict[str, Any]) -> dict[str, Any]:
        """Return architecture values required when a model-only file has no config."""
        values = model_cfg.get("model_params") or {}
        if not isinstance(values, dict):
            raise TypeError("model_params must be a mapping")
        return dict(values)
