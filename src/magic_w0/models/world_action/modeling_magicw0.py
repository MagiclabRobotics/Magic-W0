"""New MagicW0 Base with synchronous 2D/3D world experts."""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from magic_w0.models.base import ModelForwardOutput
from magic_w0.models.action_backbone.modeling_magicw0_base import (
    MagicW0BasePolicy,
    QwenHybridExpertLayer,
    QwenRMSNorm,
    _maybe_to,
    _repeat_kv,
    _run_expert_deltanet,
)
from magic_w0.models.action_backbone.processor_magicw0_base import (
    ACTION_DIM_MASK,
    ACTION_IS_PAD,
    ACTIONS,
    CAMERA_VALID,
    EMBODIMENT,
    STATE,
    STATE_DIM_MASK,
    TASK,
)

from .configuration_magicw0 import MagicW0TrainingConfig
from .history_frames import (
    HISTORY_FRAMES,
    HISTORY_IMAGES,
    HISTORY_LEN,
    HISTORY_PLACEHOLDER_TOKEN,
    history_block_text,
    image_prompt_label,
    pool_frame_features,
)
from .processor_magicw0 import (
    FUTURE_2D_IMAGES,
    FUTURE_2D_PIXEL_VALUES,
    FUTURE_3D_IMAGES,
    GEOMETRY_3D_TARGET,
    MOTION_3D_TARGET,
    VIEW_VALID_MASK,
    WORLD_3D_CURRENT_IMAGES,
    WORLD_3D_VALID_MASK,
)
from .world_teachers import (
    TRACK4WORLD_MOTION_DROPPED_ROWS,
    TRACK4WORLD_MOTION_RECOVERED_ROWS,
    TRACK4WORLD_MOTION_RETRY_ROWS,
    TRACK4WORLD_TENSORRT_RELEASED_GIB,
    TRACK4WORLD_TEACHER_SECONDS,
    WorldTeacherMixin,
)

logger = logging.getLogger(__name__)

_ACTION_WORLD_KEEP_BOTH = 0
_ACTION_WORLD_MASK_2D = 1
_ACTION_WORLD_MASK_3D = 2
_ACTION_WORLD_MASK_MODE_NAMES = {
    _ACTION_WORLD_KEEP_BOTH: "KEEP_BOTH",
    _ACTION_WORLD_MASK_2D: "MASK_2D",
    _ACTION_WORLD_MASK_3D: "MASK_3D",
}
_ACTION_WORLD_STREAM_MASK_LOG_RECORD = "_action_world_stream_mask"


def _build_query_stream_key_mask(
    prefix_mask: torch.Tensor,
    stream_masks: list[torch.Tensor],
    stream_names: tuple[str, ...],
    visible_sources: tuple[str, ...],
    *,
    masked_sources: tuple[str, ...] = (),
) -> torch.Tensor:
    """Build one expert query stream's K/V mask from its visibility row."""
    if len(stream_masks) != len(stream_names):
        raise ValueError("stream masks/names must have the same length")
    if len(set(stream_names)) != len(stream_names):
        raise ValueError(f"expert stream names must be unique: {stream_names}")
    present_sources = {"vlm", *stream_names}
    unknown_visible = set(visible_sources) - present_sources
    if unknown_visible:
        raise ValueError(
            "visible K/V sources are absent from this model: "
            f"{sorted(unknown_visible)}; present={sorted(present_sources)}"
        )
    unknown_masked = set(masked_sources) - present_sources
    if unknown_masked:
        raise ValueError(
            "masked K/V sources are absent from this model: "
            f"{sorted(unknown_masked)}; present={sorted(present_sources)}"
        )
    effective_sources = set(visible_sources) - set(masked_sources)
    if not effective_sources:
        raise ValueError("an expert query must retain at least one visible K/V source")

    pieces = [
        prefix_mask if "vlm" in effective_sources else torch.zeros_like(prefix_mask)
    ]
    for name, mask in zip(stream_names, stream_masks, strict=True):
        pieces.append(mask if name in effective_sources else torch.zeros_like(mask))
    return torch.cat(pieces, dim=1)


def _build_action_query_key_mask(
    prefix_mask: torch.Tensor,
    stream_masks: list[torch.Tensor],
    stream_names: tuple[str, ...],
    mode: int,
) -> torch.Tensor:
    """Build the Action-only KV mask without changing any World query mask."""
    if mode not in _ACTION_WORLD_MASK_MODE_NAMES:
        raise ValueError(f"unknown action/world stream mask mode {mode}")
    masked_stream = {
        _ACTION_WORLD_KEEP_BOTH: None,
        _ACTION_WORLD_MASK_2D: "world_2d",
        _ACTION_WORLD_MASK_3D: "world_3d",
    }[mode]
    if masked_stream is not None and masked_stream not in stream_names:
        raise ValueError(
            f"cannot mask absent {masked_stream!r} stream; present={stream_names}"
        )
    return _build_query_stream_key_mask(
        prefix_mask,
        stream_masks,
        stream_names,
        ("vlm", *stream_names),
        masked_sources=() if masked_stream is None else (masked_stream,),
    )


@dataclass(frozen=True, slots=True)
class _MagicW0LossWeights:
    flow: float
    fast_ce: float
    world_2d: float
    world_3d: float
    geometry_effective: float
    motion_effective: float

    def metrics(self) -> dict[str, float]:
        """Names are part of the metrics.jsonl/W&B monitoring contract."""
        return {
            "flow_loss_weight": self.flow,
            "fast_ce_loss_weight": self.fast_ce,
            "gen_2d_loss_weight": self.world_2d,
            "geometry_effective_weight": self.geometry_effective,
            "motion_effective_weight": self.motion_effective,
            "world_3d_loss_weight": self.world_3d,
        }


def _scheduled_auxiliary_weight(
    maximum: float,
    minimum: float,
    *,
    global_step: int,
    max_steps: int,
) -> float:
    """20% max hold, 60% cosine decay, 20% min hold."""
    if max_steps <= 0:
        raise ValueError("loss weight scheduling requires training.max_steps > 0")
    progress = min(max(float(global_step) / float(max_steps), 0.0), 1.0)
    if progress <= 0.2:
        return float(maximum)
    if progress >= 0.8:
        return float(minimum)
    decay_progress = (progress - 0.2) / 0.6
    cosine = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
    return float(minimum + (maximum - minimum) * cosine)


def _strip_dinov3_teacher_from_state_dict(
    module: nn.Module,
    state_dict: dict[str, torch.Tensor],
    prefix: str,
    local_metadata: dict[str, Any],
) -> None:
    """Keep the frozen target network out of training checkpoints.

    DINO remains a registered module so ordinary ``model.to(device)`` and
    ``model.eval()`` semantics stay intact. It is reconstructed from
    ``dinov3_checkpoint`` and is not part of the learned MagicW0 state.
    """
    del module, local_metadata
    teacher_prefix = f"{prefix}dinov3_teacher."
    for key in tuple(state_dict):
        if key.startswith(teacher_prefix):
            del state_dict[key]


def _ignore_dinov3_teacher_checkpoint_keys(
    module: nn.Module, incompatible_keys: Any
) -> None:
    """Make strict loading accept old-with-DINO and new-without-DINO files."""
    del module
    for keys in (incompatible_keys.missing_keys, incompatible_keys.unexpected_keys):
        keys[:] = [key for key in keys if not key.startswith("dinov3_teacher.")]


def _selected_sdpa_backend(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor,
    dropout_p: float,
    scale: float,
) -> str:
    """Best-effort report of the kernel PyTorch will select for this SDPA call."""
    choice_fn = getattr(torch, "_fused_sdp_choice", None)
    if choice_fn is None:
        return "unknown(torch._fused_sdp_choice unavailable)"
    try:
        try:
            choice = choice_fn(
                query,
                key,
                value,
                attention_mask,
                dropout_p,
                False,
                scale=scale,
            )
        except TypeError:
            choice = choice_fn(
                query,
                key,
                value,
                attention_mask,
                dropout_p,
                False,
            )
        # torch.nn.attention.SDPBackend uses these stable numeric values.
        return {
            -1: "error",
            0: "math",
            1: "flash_attention",
            2: "efficient_attention",
            3: "cudnn_attention",
            4: "overrideable",
        }.get(int(choice), f"unknown(choice={choice})")
    except Exception as exc:  # noqa: BLE001 - diagnostics must never break training
        return f"unknown({type(exc).__name__})"


def _qualified_name(value: Any) -> str:
    if value is None:
        return "unavailable"
    module = getattr(value, "__module__", None)
    name = getattr(value, "__qualname__", None) or getattr(value, "__name__", None)
    if module and name:
        return f"{module}.{name}"
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _module_name(module: nn.Module | None) -> str:
    if module is None:
        return "unavailable"
    module_type = type(module)
    return f"{module_type.__module__}.{module_type.__qualname__}"


def _describe_deltanet(mixer: nn.Module | None) -> str:
    if mixer is None:
        return "module=unavailable rule=unavailable causal_conv=unavailable"
    rule = getattr(mixer, "chunk_gated_delta_rule", None)
    if rule is None:
        rule = getattr(mixer, "_delta_rule", None)
    causal_conv = getattr(mixer, "causal_conv1d_fn", None)
    if causal_conv is None:
        causal_conv = getattr(mixer, "conv1d", None)
    return (
        f"module={_module_name(mixer)} "
        f"rule={_qualified_name(rule)} "
        f"causal_conv={_qualified_name(causal_conv)}"
    )


def _resolved_deltanet_backend(mixer: nn.Module | None) -> str:
    if mixer is None:
        return "unavailable"
    rule = getattr(mixer, "chunk_gated_delta_rule", None)
    rule_module = str(getattr(rule, "__module__", ""))
    if getattr(mixer, "_magicvla_transformers_fla", False) or rule_module.startswith(
        "fla."
    ):
        return "transformers_fla"
    if type(mixer).__name__ == "QwenGatedDeltaNet":
        return "torch_parallel"
    return "transformers_native_or_fallback"


def _resolved_vlm_attention_backend(vlm: nn.Module, requested: str) -> str:
    configs = (getattr(vlm.config, "text_config", None), vlm.config)
    for candidate in configs:
        if candidate is None:
            continue
        for attribute in ("_attn_implementation", "attn_implementation"):
            value = getattr(candidate, attribute, None)
            if value:
                return str(value)
    return requested or "unspecified"


class QwenHybrid2DWorldExpert(nn.Module):
    """Hybrid DeltaNet/full-attention queries supervised by frozen DINOv3."""

    def __init__(self, config: MagicW0TrainingConfig, dino_hidden_size: int) -> None:
        super().__init__()
        self.config = config
        hidden = config.action_expert_hidden_size
        grid_h, grid_w = config.gen_2d_grid_size
        self.query_tokens = nn.Parameter(
            torch.empty(1, config.gen_2d_num_query_tokens_per_view, hidden)
        )
        self.row_embedding = nn.Parameter(torch.empty(grid_h, hidden))
        self.col_embedding = nn.Parameter(torch.empty(grid_w, hidden))
        self.camera_embedding = (
            nn.Parameter(torch.empty(config.gen_2d_max_views, hidden))
            if config.gen_2d_camera_embedding
            else None
        )
        self.blocks = nn.ModuleList(
            QwenHybridExpertLayer(
                config,
                use_full_attention=((index + 1) % config.full_attention_interval == 0),
                layer_idx=index,
            )
            for index in range(config.action_expert_num_layers)
        )
        self.output_norm = QwenRMSNorm(hidden, config.rms_norm_eps)
        self.projector = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, dino_hidden_size),
        )
        self.residual_dropout = nn.Dropout(config.gen_2d_dropout)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.query_tokens, std=0.02)
        nn.init.normal_(self.row_embedding, std=0.02)
        nn.init.normal_(self.col_embedding, std=0.02)
        if self.camera_embedding is not None:
            nn.init.normal_(self.camera_embedding, std=0.02)
        for block in self.blocks:
            block.reset_parameters()
        nn.init.zeros_(self.output_norm.weight)
        nn.init.ones_(self.projector[0].weight)
        nn.init.zeros_(self.projector[0].bias)
        nn.init.normal_(self.projector[1].weight, std=0.02)
        nn.init.zeros_(self.projector[1].bias)

    def embed_queries(self, batch_size: int, device: torch.device) -> torch.Tensor:
        grid_h, grid_w = self.config.gen_2d_grid_size
        spatial = (self.row_embedding[:, None] + self.col_embedding[None]).reshape(
            grid_h * grid_w, -1
        )
        per_view = self.query_tokens + spatial.unsqueeze(0)
        per_view = per_view.expand(self.config.gen_2d_max_views, -1, -1)
        if self.camera_embedding is not None:
            per_view = per_view + self.camera_embedding[:, None]
        tokens = per_view.reshape(-1, per_view.shape[-1]).to(device)
        return tokens.unsqueeze(0).expand(batch_size, -1, -1)

    def project(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = self.output_norm(hidden)
        return self.projector(hidden.to(self.projector[1].weight.dtype))


class QwenHybrid3DWorldExpert(nn.Module):
    """Hybrid world queries supervised by Track4World geometry and motion latents."""

    def __init__(self, config: MagicW0TrainingConfig) -> None:
        super().__init__()
        self.config = config
        hidden = config.action_expert_hidden_size
        grid_h, grid_w = config.gen_3d_grid_size
        self.query_tokens = nn.Parameter(
            torch.empty(1, config.gen_3d_num_query_tokens, hidden)
        )
        self.row_embedding = nn.Parameter(torch.empty(grid_h, hidden))
        self.col_embedding = nn.Parameter(torch.empty(grid_w, hidden))
        self.blocks = nn.ModuleList(
            QwenHybridExpertLayer(
                config,
                use_full_attention=((index + 1) % config.full_attention_interval == 0),
                layer_idx=index,
            )
            for index in range(config.action_expert_num_layers)
        )
        self.output_norm = QwenRMSNorm(hidden, config.rms_norm_eps)
        self.geometry_projector = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1024),
        )
        self.motion_projector = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1024),
        )
        self.residual_dropout = nn.Dropout(config.gen_3d_dropout)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.query_tokens, std=0.02)
        nn.init.normal_(self.row_embedding, std=0.02)
        nn.init.normal_(self.col_embedding, std=0.02)
        for block in self.blocks:
            block.reset_parameters()
        nn.init.zeros_(self.output_norm.weight)
        for projector in (self.geometry_projector, self.motion_projector):
            nn.init.ones_(projector[0].weight)
            nn.init.zeros_(projector[0].bias)
            for module in projector:
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, std=0.02)
                    nn.init.zeros_(module.bias)

    def embed_queries(self, batch_size: int, device: torch.device) -> torch.Tensor:
        grid_h, grid_w = self.config.gen_3d_grid_size
        spatial = (self.row_embedding[:, None] + self.col_embedding[None]).reshape(
            grid_h * grid_w, -1
        )
        tokens = (self.query_tokens + spatial.unsqueeze(0)).to(device)
        return tokens.expand(batch_size, -1, -1)

    def project_geometry(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = self.output_norm(hidden)
        return self.geometry_projector(
            hidden.to(self.geometry_projector[1].weight.dtype)
        )

    def project_motion(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = self.output_norm(hidden)
        return self.motion_projector(hidden.to(self.motion_projector[1].weight.dtype))


class MagicW0TrainingPolicy(WorldTeacherMixin, MagicW0BasePolicy):
    """New Base action/VQA policy augmented with 2D and 3D world streams."""

    config_class = MagicW0TrainingConfig
    name = "magicw0"
    loss_name = "magic_w0_robotics"
    supports_cpu_prepare = True

    def __init__(
        self,
        config: MagicW0TrainingConfig,
        *,
        vlm_state_dict: dict[str, torch.Tensor] | None = None,
    ) -> None:
        super().__init__(config, vlm_state_dict=vlm_state_dict)
        # Set by the trainer before every optimizer-step forward. These are
        # ordinary Python values rather than persistent buffers, so enabling the
        # schedule does not change checkpoint state_dict compatibility.
        self._training_global_step = 0
        self._training_max_steps: int | None = None
        # DINO is a frozen, reproducible target generator rather than learned
        # policy state. Saving it duplicated the local HF checkpoint in every
        # trainer checkpoint even though the optimizer already excluded it.
        self.register_state_dict_post_hook(_strip_dinov3_teacher_from_state_dict)
        self.register_load_state_dict_post_hook(_ignore_dinov3_teacher_checkpoint_keys)
        dtype = self._torch_dtype(config.dtype)
        self.action_residual_dropout = nn.Dropout(config.action_expert_dropout)
        self.dinov3_teacher: nn.Module | None = None
        self.world_2d_expert: QwenHybrid2DWorldExpert | None = None
        self.world_3d_expert: QwenHybrid3DWorldExpert | None = None

        if config.enable_2d_world_expert:
            dino_hidden = int(config.dinov3_target_hidden_size)
            if config.enable_world_teacher_supervision:
                from transformers import AutoModel

                self.dinov3_teacher = AutoModel.from_pretrained(
                    config.dinov3_checkpoint,
                    local_files_only=config.dinov3_local_files_only,
                    dtype=dtype,
                )
                self.dinov3_teacher.requires_grad_(False).eval()
                loaded_hidden = int(self.dinov3_teacher.config.hidden_size)
                if loaded_hidden != dino_hidden:
                    raise ValueError(
                        "DINOv3 hidden size does not match "
                        "dinov3_target_hidden_size: "
                        f"{loaded_hidden} != {dino_hidden}"
                    )
            self.world_2d_expert = QwenHybrid2DWorldExpert(config, dino_hidden).to(
                dtype=dtype
            )
            if config.freeze_2d_world_expert:
                self.world_2d_expert.requires_grad_(False)

        if config.enable_3d_world_expert:
            self.world_3d_expert = QwenHybrid3DWorldExpert(config).to(dtype=dtype)
            if config.freeze_3d_world_expert:
                self.world_3d_expert.requires_grad_(False)

        if os.environ.get("RANK", "0") == "0":
            self._log_kernel_backends()
            self._log_joint_attention_visibility_config()
            self._log_action_world_stream_mask_config()

    def set_training_progress(
        self,
        *,
        global_step: int,
        max_steps: int | None,
    ) -> None:
        """Supply optimizer-step progress for the stateless loss schedule."""
        step = int(global_step)
        if step < 0:
            raise ValueError(f"global_step must be non-negative, got {step}")
        if self.config.loss_weight_schedule_enabled:
            if max_steps is None or int(max_steps) <= 0:
                raise ValueError(
                    "loss_weight_schedule_enabled=True requires training.max_steps"
                )
            self._training_max_steps = int(max_steps)
        else:
            self._training_max_steps = None if max_steps is None else int(max_steps)
        self._training_global_step = step

    def _active_loss_weights(self) -> _MagicW0LossWeights:
        config = self.config
        fast_ce = float(config.fast_ce_loss_weight)
        world_2d = float(config.gen_2d_loss_weight)
        world_3d = float(config.world_3d_loss_weight)
        if config.loss_weight_schedule_enabled:
            max_steps = getattr(self, "_training_max_steps", None)
            if max_steps is None:
                raise RuntimeError(
                    "loss weight schedule has no training progress; the trainer must "
                    "call set_training_progress() before a robotics forward"
                )
            step = int(getattr(self, "_training_global_step", 0))
            fast_ce = _scheduled_auxiliary_weight(
                fast_ce,
                config.fast_ce_loss_weight_min,
                global_step=step,
                max_steps=max_steps,
            )
            world_2d = _scheduled_auxiliary_weight(
                world_2d,
                config.gen_2d_loss_weight_min,
                global_step=step,
                max_steps=max_steps,
            )
            world_3d = _scheduled_auxiliary_weight(
                world_3d,
                config.world_3d_loss_weight_min,
                global_step=step,
                max_steps=max_steps,
            )
        return _MagicW0LossWeights(
            flow=float(config.flow_matching_loss_weight),
            fast_ce=fast_ce,
            world_2d=world_2d,
            world_3d=world_3d,
            geometry_effective=world_3d * float(config.geometry_latent_weight),
            motion_effective=world_3d * float(config.motion_latent_weight),
        )

    def _log_kernel_backends(self) -> None:
        """Report configured and actually bound attention kernels before training."""
        language_model = self.vlm.model.language_model
        full_layer_indices = tuple(
            index + 1
            for index, layer_type in enumerate(self.vlm_layer_types)
            if layer_type == "full_attention"
        )
        linear_layer_indices = tuple(
            index + 1
            for index, layer_type in enumerate(self.vlm_layer_types)
            if layer_type == "linear_attention"
        )
        vlm_full_mixer = language_model.layers[full_layer_indices[0] - 1].self_attn
        vlm_linear_mixer = language_model.layers[
            linear_layer_indices[0] - 1
        ].linear_attn
        resolved_vlm_backend = _resolved_vlm_attention_backend(
            self.vlm, self.config.attention_implementation
        )
        logger.warning(
            "[kernel-backends] component=Qwen3.5-VLM "
            "full_attention_requested=%s full_attention_resolved=%s "
            "external_flash_attn2=%s full_attention_module=%s "
            "full_attention_layers=%s; DeltaNet_resolved=%s %s "
            "linear_attention_layers=%s",
            self.config.attention_implementation,
            resolved_vlm_backend,
            "yes" if resolved_vlm_backend == "flash_attention_2" else "no",
            _module_name(vlm_full_mixer),
            full_layer_indices,
            _resolved_deltanet_backend(vlm_linear_mixer),
            _describe_deltanet(vlm_linear_mixer),
            linear_layer_indices,
        )

        expert_layers = tuple(
            index + 1
            for index in range(self.config.action_expert_num_layers)
            if (index + 1) % self.config.full_attention_interval == 0
        )
        linear_expert_layers = tuple(
            index + 1
            for index in range(self.config.action_expert_num_layers)
            if (index + 1) % self.config.full_attention_interval != 0
        )
        experts = (
            ("Action", self.action_expert),
            ("2D", self.world_2d_expert),
            ("3D", self.world_3d_expert),
        )
        for name, expert in experts:
            if expert is None:
                logger.warning("[kernel-backends] component=%s-Expert disabled", name)
                continue
            full_mixer = next(
                block.mixer for block in expert.blocks if block.use_full_attention
            )
            linear_mixer = next(
                block.mixer for block in expert.blocks if not block.use_full_attention
            )
            full_backend = str(full_mixer.full_attention_backend)
            if full_backend == "flash_attention_2":
                full_detail = (
                    "external_flash_attn2=yes function="
                    f"{_qualified_name(full_mixer._flash_attn_varlen_func)} "
                    f"version={full_mixer.flash_attn_version}"
                )
            else:
                full_detail = (
                    "external_flash_attn2=no "
                    "function=torch.nn.functional.scaled_dot_product_attention"
                )
            logger.warning(
                "[kernel-backends] component=%s-Expert "
                "full_attention_backend=%s %s full_attention_layers=%s; "
                "DeltaNet_requested=%s DeltaNet_resolved=%s %s "
                "linear_attention_layers=%s",
                name,
                full_backend,
                full_detail,
                expert_layers,
                self.config.expert_deltanet_backend,
                _resolved_deltanet_backend(linear_mixer),
                _describe_deltanet(linear_mixer),
                linear_expert_layers,
            )

    def _log_action_world_stream_mask_config(self) -> None:
        config = self.config.action_world_stream_mask
        both_world_streams = (
            self.config.enable_2d_world_expert and self.config.enable_3d_world_expert
        )
        effective = config.enabled and both_world_streams
        reason = (
            "active_during_training"
            if effective
            else (
                "disabled_by_config"
                if not config.enabled
                else "requires_2d_and_3d_world_experts"
            )
        )
        logger.warning(
            "[action-world-stream-mask] configured_enabled=%s effective=%s "
            "training_only=yes local_batch_shared_mode=yes ddp_broadcast=no "
            "keep_both_prob=%.6f mask_2d_prob=%.6f mask_3d_prob=%.6f reason=%s",
            config.enabled,
            effective,
            config.keep_both_prob,
            config.mask_2d_prob,
            config.mask_3d_prob,
            reason,
        )

    def _log_joint_attention_visibility_config(self) -> None:
        visibility = self.config.joint_attention_visibility
        logger.warning(
            "[joint-attention-visibility] training_and_inference=yes "
            "vlm_query_reads_experts=no world_2d=%s world_3d=%s action=%s",
            ",".join(visibility.world_2d),
            ",".join(visibility.world_3d),
            ",".join(visibility.action),
        )

    def _joint_attention_visible_sources(
        self,
        query_stream: str,
        stream_names: tuple[str, ...],
        *,
        action_world_stream_mask_mode: int = _ACTION_WORLD_KEEP_BOTH,
    ) -> tuple[str, ...]:
        """Resolve configured visibility against streams present in this run."""
        present_sources = ("vlm", *stream_names)
        if query_stream in {"world_2d", "world_3d", "action"}:
            configured = set(
                self.config.joint_attention_visibility.sources_for(query_stream)
            )
        else:
            # Keep direct low-level calls without semantic stream names backward
            # compatible. Production paths always pass the three canonical names.
            configured = set(present_sources)

        if query_stream == "action":
            if action_world_stream_mask_mode not in _ACTION_WORLD_MASK_MODE_NAMES:
                raise ValueError(
                    "unknown action/world stream mask mode "
                    f"{action_world_stream_mask_mode}"
                )
            masked_source = {
                _ACTION_WORLD_KEEP_BOTH: None,
                _ACTION_WORLD_MASK_2D: "world_2d",
                _ACTION_WORLD_MASK_3D: "world_3d",
            }[action_world_stream_mask_mode]
            if masked_source is not None:
                configured.discard(masked_source)

        visible = tuple(source for source in present_sources if source in configured)
        if not visible:
            raise RuntimeError(
                "joint attention leaves query stream "
                f"{query_stream!r} with no K/V source; present={present_sources}"
            )
        return visible

    def _sample_action_world_stream_mask_mode(self) -> int:
        config = self.config.action_world_stream_mask
        if (
            not self.training
            or not config.enabled
            or not self.config.enable_2d_world_expert
            or not self.config.enable_3d_world_expert
        ):
            return _ACTION_WORLD_KEEP_BOTH

        # One CPU scalar per robotics forward: the whole local batch shares it.
        # Trainer seeding uses seed + rank, and there is deliberately no DDP
        # collective here, so ranks draw independently without a CUDA sync.
        draw = float(torch.rand(()).item())
        if draw < config.keep_both_prob:
            return _ACTION_WORLD_KEEP_BOTH
        if draw < config.keep_both_prob + config.mask_2d_prob:
            return _ACTION_WORLD_MASK_2D
        return _ACTION_WORLD_MASK_3D

    def _action_world_stream_mask_log_record(
        self,
        mode: int,
        valid_elements: torch.Tensor,
    ) -> dict[str, str | int] | None:
        """Return trainer-only metadata for windowed mask-mode statistics."""
        config = self.config.action_world_stream_mask
        if (
            not self.training
            or not config.enabled
            or not self.config.enable_2d_world_expert
            or not self.config.enable_3d_world_expert
        ):
            return None
        if mode not in _ACTION_WORLD_MASK_MODE_NAMES:
            raise ValueError(f"unknown action/world stream mask mode {mode}")
        # A sample is effective when at least one action element participates in
        # flow loss. Empty/padded rows must not affect either the mode ratio or
        # the sample-weighted mode loss.
        sample_count = int(
            valid_elements.flatten(start_dim=1).any(dim=1).sum().detach().cpu()
        )
        return {
            "mode": _ACTION_WORLD_MASK_MODE_NAMES[mode].lower(),
            "sample_count": sample_count,
        }

    def train(self, mode: bool = True):
        super().train(mode)
        if self.dinov3_teacher is not None:
            self.dinov3_teacher.eval()
        if self.config.freeze_2d_world_expert and self.world_2d_expert is not None:
            self.world_2d_expert.eval()
        if self.config.freeze_3d_world_expert and self.world_3d_expert is not None:
            self.world_3d_expert.eval()
        teacher = getattr(self, "_track4world_teacher", None)
        if teacher is not None:
            teacher.eval()
        return self

    def _prepare_robotics_prefix(
        self,
        batch,
        states,
        tasks,
        camera_valid,
        embodiments=None,
        state_dim_masks=None,
        subtasks=None,
        action_dim_masks=None,
    ):
        """MagicW0 layout when history is on; the Base prompt otherwise.

        ``task line + History images: <past frames>`` ->
        ``Head image: <img> Left wrist image: <img> ...`` -> state + closing.
        """
        if not self.config.history_enabled:
            return super()._prepare_robotics_prefix(
                batch,
                states,
                tasks,
                camera_valid,
                embodiments=embodiments,
                state_dim_masks=state_dim_masks,
                subtasks=subtasks,
                action_dim_masks=action_dim_masks,
            )
        if HISTORY_FRAMES not in batch or HISTORY_LEN not in batch:
            raise KeyError(
                "history_enabled=True but the batch carries no "
                f"{HISTORY_FRAMES!r}/{HISTORY_LEN!r}; build them with "
                "history_frames.past_frame_indices()"
            )
        messages = []
        past_frames: list[torch.Tensor] = []
        for sample_index in range(int(states.shape[0])):
            length = int(batch[HISTORY_LEN][sample_index])
            if length:
                past_frames.extend(
                    batch[HISTORY_FRAMES][sample_index][-length:].detach().cpu()
                )
            task_text, tail_text = self._prefix_text_parts(
                sample_index,
                states,
                tasks,
                embodiments,
                state_dim_masks,
                action_dim_masks,
            )
            content: list[dict[str, Any]] = [
                {
                    "type": "text",
                    "text": task_text
                    + history_block_text(length, self.config.history_tokens_per_frame),
                }
            ]
            for camera_index, camera_key in enumerate(self.config.camera_keys):
                if not bool(camera_valid[sample_index, camera_index]):
                    continue
                image = batch[camera_key][sample_index]
                if image.ndim == 4:
                    image = image[-1]
                content.append(
                    {
                        "type": "text",
                        "text": f"{image_prompt_label(camera_key)} image: ",
                    }
                )
                content.append({"type": "image", "image": image.detach().cpu()})
            content.append({"type": "text", "text": tail_text})
            messages.append([{"role": "user", "content": content}])
        vlm_inputs, prefix_len, subtask_ids, subtask_valid = (
            self._finalize_robotics_prefix(messages, subtasks)
        )
        if past_frames:
            # Same image processor as the current views, so past frames get the
            # same normalisation and grid.  A nested dict on purpose: every
            # helper that grows or trims the prefix touches only 2-D tensors.
            encoded = self.processor.image_processor(
                images=past_frames, return_tensors="pt"
            )
            pixel_values = encoded["pixel_values"]
            if self.config.dtype == "bfloat16":
                # The vision tower casts its input to its own dtype first, so a
                # bf16 copy is numerically the same input at half the host
                # memory: 25 frames x 48 samples is ~1.9 GB per batch in fp32,
                # held by every dataloader worker's prefetched batches.
                pixel_values = pixel_values.to(torch.bfloat16)
            vlm_inputs = dict(vlm_inputs)
            vlm_inputs[HISTORY_IMAGES] = {
                "pixel_values": pixel_values,
                "image_grid_thw": encoded["image_grid_thw"],
            }
        return vlm_inputs, prefix_len, subtask_ids, subtask_valid

    def _history_placeholder_id(self) -> int:
        cached = getattr(self, "_history_placeholder_cache", None)
        if cached is None:
            cached = int(
                self.processor.tokenizer.convert_tokens_to_ids(
                    HISTORY_PLACEHOLDER_TOKEN
                )
            )
            self._history_placeholder_cache = cached
        return cached

    def _embed_vlm_inputs(self, vlm_inputs, states=None, state_dim_mask=None):
        """Base embedding, then pooled past frames over their placeholders.

        Placeholders are ordinary text tokens to the position builder, so each
        pooled token takes the next 1-D text position, as in MagicW0.
        """
        inputs_embeds, attention_mask, position_ids = super()._embed_vlm_inputs(
            vlm_inputs, states, state_dim_mask
        )
        if not self.config.history_enabled:
            return inputs_embeds, attention_mask, position_ids
        placeholder = vlm_inputs["input_ids"] == self._history_placeholder_id()
        expected = int(placeholder.sum())
        frames = vlm_inputs.get(HISTORY_IMAGES)
        if frames is None:
            if expected:
                raise ValueError(
                    f"{expected} history placeholders but no history frames"
                )
            return inputs_embeds, attention_mask, position_ids
        device = inputs_embeds.device
        grid_thw = frames["image_grid_thw"].to(device)
        outputs = self.vlm.model.get_image_features(
            frames["pixel_values"].to(device), grid_thw, return_dict=True
        )
        pooled = pool_frame_features(
            list(outputs.pooler_output),
            grid_thw,
            merge=int(self.vlm.config.vision_config.spatial_merge_size),
            pool_size=int(self.config.history_pool_size),
        )
        if int(pooled.shape[0]) != expected:
            raise ValueError(
                f"{expected} history placeholders but {int(pooled.shape[0])} pooled tokens"
            )
        inputs_embeds = inputs_embeds.masked_scatter(
            placeholder.to(device).unsqueeze(-1), pooled.to(device, inputs_embeds.dtype)
        )
        return inputs_embeds, attention_mask, position_ids

    def prepare_batch(
        self, batch: dict[str, Any], device: torch.device | None
    ) -> dict[str, Any]:
        prepared = super().prepare_batch(batch, device)
        modality = str(
            prepared.get(
                self.config.cotrain_modality_key, self.config.cotrain_native_modality
            )
        )
        if modality in self.config.vlm_modalities:
            return prepared
        if self.config.enable_2d_world_expert:
            if FUTURE_2D_IMAGES not in batch or VIEW_VALID_MASK not in batch:
                raise ValueError("MagicW0 2D robotics batch is missing future images/mask")
            pixels = self._prepare_dino_pixel_values(batch[FUTURE_2D_IMAGES])
            prepared[FUTURE_2D_PIXEL_VALUES] = _maybe_to(pixels, device)
            prepared[VIEW_VALID_MASK] = _maybe_to(batch[VIEW_VALID_MASK], device)
        if self.config.enable_3d_world_expert:
            required = (
                WORLD_3D_CURRENT_IMAGES,
                FUTURE_3D_IMAGES,
                WORLD_3D_VALID_MASK,
            )
            missing = [key for key in required if key not in batch]
            if missing:
                raise ValueError(f"MagicW0 3D robotics batch is missing {missing}")
            for key in required:
                prepared[key] = _maybe_to(batch[key], device)
        return prepared

    def forward(self, batch: dict[str, Any]) -> ModelForwardOutput:
        if "vlm_inputs" not in batch:
            raise ValueError("magicw0 received an unprepared batch")
        modality = str(
            batch.get(
                self.config.cotrain_modality_key, self.config.cotrain_native_modality
            )
        )
        if modality in self.config.vlm_modalities:
            loss, metrics = self._vlm_loss(batch)
            return ModelForwardOutput(
                loss=loss,
                metrics=metrics,
                loss_name="magic_w0_vlm_cross_entropy",
            )
        if not self.config.enable_world_teacher_supervision and (
            self.config.enable_2d_world_expert or self.config.enable_3d_world_expert
        ):
            raise RuntimeError(
                "robotics training forward requires world teacher supervision; "
                "enable_world_teacher_supervision=False is inference-only"
            )
        loss_weights = self._active_loss_weights()

        actions = batch[ACTIONS]
        states = batch[STATE]
        valid_steps = ~batch[ACTION_IS_PAD].bool()
        valid_elements = valid_steps.unsqueeze(-1) & batch[ACTION_DIM_MASK].bool()
        timesteps = self._sample_timesteps(actions.shape[0], actions.device)
        noise = torch.randn_like(actions).masked_fill(~valid_elements, 0.0)
        time_view = timesteps[:, None, None].to(actions.dtype)
        noisy_actions = (1.0 - time_view) * actions + time_view * noise
        target_velocity = noise - actions

        targets_3d: dict[str, Any] = {}
        if self.config.enable_3d_world_expert:
            targets_3d = self._track4world_teacher_targets(
                batch[WORLD_3D_CURRENT_IMAGES],
                batch[FUTURE_3D_IMAGES],
                batch[WORLD_3D_VALID_MASK],
                actions.device,
            )

        prefix_len = int(
            batch.get("prefix_len", batch["vlm_inputs"]["input_ids"].shape[1])
        )
        action_world_stream_mask_mode = self._sample_action_world_stream_mask_mode()
        predicted_velocity, vlm_hidden, world_2d_hidden, world_3d_hidden = (
            self._run_joint_trunks(
                batch["vlm_inputs"],
                noisy_actions,
                timesteps,
                states,
                valid_steps,
                prefix_len,
                batch.get(STATE_DIM_MASK),
                batch.get(VIEW_VALID_MASK),
                batch.get(WORLD_3D_VALID_MASK),
                action_world_stream_mask_mode=action_world_stream_mask_mode,
            )
        )

        mask = valid_elements.expand_as(predicted_velocity)
        flow_loss = (
            (predicted_velocity.float() - target_velocity.float()).square() * mask
        ).sum() / mask.sum().clamp_min(1)
        loss = loss_weights.flow * flow_loss
        metrics: dict[str, Any] = {
            "modality": modality,
            "flow_loss": float(flow_loss.detach().cpu()),
            **loss_weights.metrics(),
            "valid_action_steps": int(valid_steps.sum().detach().cpu()),
            "valid_action_elements": int(valid_elements.sum().detach().cpu()),
            "dino_teacher_loaded": int(self.dinov3_teacher is not None),
            "track4world_teacher_loaded": int(
                getattr(self, "_track4world_teacher", None) is not None
            ),
        }
        mask_log_record = self._action_world_stream_mask_log_record(
            action_world_stream_mask_mode,
            valid_elements,
        )
        if mask_log_record is not None:
            # Structured/internal metadata is consumed by DistributedTrainer and
            # deliberately ignored by the scalar JSONL/W&B pass-through.
            metrics[_ACTION_WORLD_STREAM_MASK_LOG_RECORD] = mask_log_record
        if TRACK4WORLD_TEACHER_SECONDS in targets_3d:
            metrics[TRACK4WORLD_TEACHER_SECONDS] = float(
                targets_3d[TRACK4WORLD_TEACHER_SECONDS]
            )
        for key in (
            TRACK4WORLD_MOTION_RETRY_ROWS,
            TRACK4WORLD_MOTION_RECOVERED_ROWS,
            TRACK4WORLD_MOTION_DROPPED_ROWS,
        ):
            if key in targets_3d:
                metrics[key] = int(targets_3d[key])
        if TRACK4WORLD_TENSORRT_RELEASED_GIB in targets_3d:
            metrics[TRACK4WORLD_TENSORRT_RELEASED_GIB] = float(
                targets_3d[TRACK4WORLD_TENSORRT_RELEASED_GIB]
            )
        # DISABLED: CUDA Events for memory leak investigation
        # CPU time is now always used in world_teachers.py
        # elif _TRACK4WORLD_TEACHER_EVENTS in targets_3d:
        #     # Creating flow_loss above materializes a CUDA scalar on CPU, so all
        #     # preceding default-stream work (including the teacher stop event) has
        #     # completed. Reading elapsed_time here adds no new synchronization.
        #     start_event, stop_event = targets_3d[_TRACK4WORLD_TEACHER_EVENTS]
        #     metrics[TRACK4WORLD_TEACHER_SECONDS] = (
        #         float(start_event.elapsed_time(stop_event)) / 1000.0
        #     )

        if self.config.use_fast_action_tokens:
            fast_loss, fast_tokens = self._fast_ce_loss(
                vlm_hidden,
                prefix_len,
                batch["fast_ids"],
                batch["fast_valid"],
            )
            loss = loss + loss_weights.fast_ce * fast_loss
            metrics["fast_loss"] = float(fast_loss.detach().cpu())
            metrics["n_fast_tokens"] = fast_tokens

        if self.config.enable_2d_world_expert:
            if self.world_2d_expert is None or world_2d_hidden is None:
                raise RuntimeError("enabled MagicW0 2D stream produced no hidden states")
            predicted_dino = self.world_2d_expert.project(world_2d_hidden)
            target_dino = (
                self._dino_patch_targets(batch[FUTURE_2D_PIXEL_VALUES])
                .reshape_as(predicted_dino)
                .to(predicted_dino.dtype)
            )
            world_2d_mask = self._world_2d_mask(batch[VIEW_VALID_MASK])
            token_cosine = F.cosine_similarity(predicted_dino, target_dino, dim=-1)
            denominator = world_2d_mask.sum().clamp_min(1)
            loss_2d = ((1.0 - token_cosine) * world_2d_mask).sum() / denominator
            loss = loss + loss_weights.world_2d * loss_2d
            metrics.update(
                {
                    "gen_2d_loss": float(loss_2d.detach().cpu()),
                    "gen_2d_cosine_similarity": float(
                        ((token_cosine * world_2d_mask).sum() / denominator)
                        .detach()
                        .cpu()
                    ),
                    "valid_2d_tokens": int(world_2d_mask.sum().detach().cpu()),
                }
            )

        if self.config.enable_3d_world_expert:
            if self.world_3d_expert is None or world_3d_hidden is None:
                raise RuntimeError("enabled MagicW0 3D stream produced no hidden states")
            predicted_geometry = self.world_3d_expert.project_geometry(world_3d_hidden)
            predicted_motion = self.world_3d_expert.project_motion(world_3d_hidden)
            target_geometry = targets_3d[GEOMETRY_3D_TARGET].to(
                predicted_geometry.dtype
            )
            target_motion = targets_3d[MOTION_3D_TARGET].to(predicted_motion.dtype)
            if predicted_geometry.shape != target_geometry.shape:
                raise RuntimeError(
                    "3D geometry prediction/target mismatch: "
                    f"{tuple(predicted_geometry.shape)} != {tuple(target_geometry.shape)}"
                )
            if predicted_motion.shape != target_motion.shape:
                raise RuntimeError(
                    "3D motion prediction/target mismatch: "
                    f"{tuple(predicted_motion.shape)} != {tuple(target_motion.shape)}"
                )
            world_3d_mask = self._world_3d_mask(targets_3d[WORLD_3D_VALID_MASK])
            geometry_loss = _masked_layernorm_mse(
                predicted_geometry, target_geometry, world_3d_mask
            )
            motion_loss = _masked_layernorm_mse(
                predicted_motion, target_motion, world_3d_mask
            )
            loss_3d = (
                self.config.geometry_latent_weight * geometry_loss
                + self.config.motion_latent_weight * motion_loss
            )
            loss = loss + loss_weights.world_3d * loss_3d
            metrics.update(
                {
                    "geometry_3d_loss": float(geometry_loss.detach().cpu()),
                    "motion_3d_loss": float(motion_loss.detach().cpu()),
                    "world_3d_loss": float(loss_3d.detach().cpu()),
                    "valid_3d_tokens": int(world_3d_mask.sum().detach().cpu()),
                }
            )

        metrics["loss"] = float(loss.detach().cpu())
        return ModelForwardOutput(loss=loss, metrics=metrics, loss_name=self.loss_name)

    def prepare_inference_batch(
        self, batch: dict[str, Any], device: torch.device | None
    ) -> dict[str, Any]:
        """Build a robotics prefix without actions or future teacher targets.

        Images follow ``config.camera_keys`` and may be uint8 CHW or a temporal
        stack. State normalization remains the caller's responsibility, matching
        the training processor contract.
        """
        if STATE not in batch or TASK not in batch:
            raise ValueError(f"inference batch requires {STATE!r} and {TASK!r}")
        states = batch[STATE]
        if not torch.is_tensor(states) or states.ndim != 2:
            raise ValueError("inference state must be a [B,state_dim] tensor")
        if states.shape[1] != self.config.state_dim:
            raise ValueError(
                f"inference state width {states.shape[1]} != {self.config.state_dim}"
            )
        batch_size = int(states.shape[0])
        camera_valid = batch.get(CAMERA_VALID)
        if camera_valid is None:
            camera_valid = torch.ones(
                (batch_size, len(self.config.camera_keys)), dtype=torch.bool
            )
        if tuple(camera_valid.shape) != (batch_size, len(self.config.camera_keys)):
            raise ValueError(
                "camera_valid must be [B,num_configured_cameras], got "
                f"{tuple(camera_valid.shape)}"
            )
        missing_images = [
            key
            for index, key in enumerate(self.config.camera_keys)
            if bool(camera_valid[:, index].any()) and key not in batch
        ]
        if missing_images:
            raise ValueError(
                f"inference batch is missing camera images {missing_images}"
            )

        state_dim_mask = batch.get(STATE_DIM_MASK)
        if state_dim_mask is not None and tuple(state_dim_mask.shape) != tuple(
            states.shape
        ):
            raise ValueError("state_dim_mask must have the same shape as state")
        (
            vlm_inputs,
            prefix_len,
            _subtask_ids,
            _subtask_valid,
        ) = self._prepare_robotics_prefix(
            batch,
            states,
            batch[TASK],
            camera_valid,
            batch.get(EMBODIMENT),
            state_dim_mask,
            action_dim_masks=batch.get(ACTION_DIM_MASK),
        )
        prepared: dict[str, Any] = {
            self.config.cotrain_modality_key: self.config.cotrain_native_modality,
            "prefix_len": prefix_len,
            "vlm_inputs": {
                key: _maybe_to(value, device) if torch.is_tensor(value) else value
                for key, value in vlm_inputs.items()
            },
            STATE: _maybe_to(states, device),
        }
        if state_dim_mask is not None:
            prepared[STATE_DIM_MASK] = _maybe_to(state_dim_mask, device)

        camera_index = {key: index for index, key in enumerate(self.config.camera_keys)}
        if self.world_2d_expert is not None:
            view_mask = torch.zeros(
                (batch_size, self.config.gen_2d_max_views), dtype=torch.bool
            )
            for view_index, key in enumerate(self.config.gen_2d_camera_keys):
                if key not in camera_index:
                    raise ValueError(f"2D camera {key!r} is absent from camera_keys")
                view_mask[:, view_index] = camera_valid[:, camera_index[key]].bool()
            prepared[VIEW_VALID_MASK] = _maybe_to(view_mask, device)
        if self.world_3d_expert is not None:
            key = self.config.gen_3d_camera_key
            if key not in camera_index:
                raise ValueError(f"3D camera {key!r} is absent from camera_keys")
            prepared[WORLD_3D_VALID_MASK] = _maybe_to(
                camera_valid[:, camera_index[key]].bool(), device
            )
        return prepared

    @torch.no_grad()
    def generate_actions(
        self,
        batch: dict[str, Any],
        num_steps: int | None = None,
        initial_noise: torch.Tensor | None = None,
        action_dim_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Integrate the MagicW0 flow field from noise (t=1) to actions (t=0)."""
        if "vlm_inputs" not in batch or STATE not in batch:
            raise ValueError("generate_actions expects prepare_inference_batch output")
        n_steps = int(
            self.config.num_inference_steps if num_steps is None else num_steps
        )
        if n_steps <= 0:
            raise ValueError("num_steps must be positive")
        states = batch[STATE]
        expected = (states.shape[0], self.config.chunk_size, self.config.action_dim)
        if initial_noise is None:
            actions = torch.randn(expected, device=states.device, dtype=states.dtype)
        else:
            if tuple(initial_noise.shape) != expected:
                raise ValueError(
                    f"initial_noise shape {tuple(initial_noise.shape)} != {expected}"
                )
            actions = initial_noise.to(device=states.device, dtype=states.dtype).clone()
        valid_dimensions = None
        if action_dim_mask is None:
            action_attention_mask = torch.ones(
                expected[:2], dtype=torch.bool, device=states.device
            )
        else:
            raw_mask = torch.as_tensor(
                action_dim_mask, dtype=torch.bool, device=states.device
            )
            if raw_mask.ndim == 1 and tuple(raw_mask.shape) == (expected[2],):
                valid_dimensions = raw_mask[None, None, :].expand(expected)
            elif raw_mask.ndim == 2 and tuple(raw_mask.shape) == (
                expected[0],
                expected[2],
            ):
                valid_dimensions = raw_mask[:, None, :].expand(expected)
            elif raw_mask.ndim == 3 and tuple(raw_mask.shape) == expected:
                valid_dimensions = raw_mask
            else:
                raise ValueError(
                    "action_dim_mask must be [D], [B,D], or [B,T,D], got "
                    f"{tuple(raw_mask.shape)} for actions {expected}"
                )
            action_attention_mask = valid_dimensions.any(dim=-1)
            actions = actions.masked_fill(~valid_dimensions, 0.0)
        dt = 1.0 / n_steps
        was_training = self.training
        self.eval()
        try:
            flow_cache = self._prepare_flow_inference_cache(batch)
            for step in range(n_steps):
                timestep = torch.full(
                    (expected[0],),
                    1.0 - step * dt,
                    device=states.device,
                    dtype=torch.float32,
                )
                velocity = self._run_cached_joint_expert_trunks(
                    actions,
                    timestep,
                    states,
                    action_attention_mask,
                    flow_cache,
                )
                actions = actions - dt * velocity.to(actions.dtype)
                if valid_dimensions is not None:
                    actions = actions.masked_fill(~valid_dimensions, 0.0)
        finally:
            self.train(was_training)
        return actions

    def _active_joint_expert_components(
        self,
    ) -> list[tuple[str, nn.ModuleList, nn.Dropout]]:
        """Return enabled expert stacks in their Joint Attention stream order."""
        components: list[tuple[str, nn.ModuleList, nn.Dropout]] = []
        if self.world_2d_expert is not None:
            components.append(
                (
                    "world_2d",
                    self.world_2d_expert.blocks,
                    self.world_2d_expert.residual_dropout,
                )
            )
        if self.world_3d_expert is not None:
            components.append(
                (
                    "world_3d",
                    self.world_3d_expert.blocks,
                    self.world_3d_expert.residual_dropout,
                )
            )
        components.append(
            ("action", self.action_expert.blocks, self.action_residual_dropout)
        )
        return components

    def _prepare_flow_inference_cache(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Run the action-independent VLM branch once for one flow trajectory.

        Joint Attention is asymmetric: every expert reads the VLM prefix, while
        the VLM never reads world/action tokens. Consequently the VLM hidden
        states and its projected per-layer K/V are identical for all reverse-flow
        steps. World experts are intentionally not cached because their full
        attention reads the changing noisy Action stream.
        """
        states = batch[STATE]
        vlm_hidden, vlm_padding_mask, vlm_position_ids = self._embed_vlm_inputs(
            batch["vlm_inputs"], states, batch.get(STATE_DIM_MASK)
        )
        language_model = self.vlm.model.language_model
        vlm_position_embeddings = language_model.rotary_emb(
            vlm_hidden, vlm_position_ids
        )
        from transformers.masking_utils import create_causal_mask

        vlm_causal_mask = create_causal_mask(
            config=language_model.config,
            inputs_embeds=vlm_hidden,
            attention_mask=vlm_padding_mask,
            past_key_values=None,
            position_ids=vlm_position_ids[0],
        )
        prefix_len = int(batch["prefix_len"])
        prefix_mask = vlm_padding_mask[:, :prefix_len]
        prefix_position_ids = vlm_position_ids[:, :, :prefix_len]
        prefix_rope = tuple(value[:, :prefix_len] for value in vlm_position_embeddings)
        valid_prefix_positions = prefix_position_ids.masked_fill(
            ~prefix_mask.unsqueeze(0), 0
        )
        world_start = valid_prefix_positions.amax(dim=(0, 2)).add(1)

        world_hidden_states: list[torch.Tensor] = []
        world_masks: list[torch.Tensor] = []
        world_position_embeddings: list[tuple[torch.Tensor, torch.Tensor]] = []
        max_world_len = 0

        if self.world_2d_expert is not None:
            view_valid_mask = batch.get(VIEW_VALID_MASK)
            if view_valid_mask is None:
                raise ValueError("2D stream requires view_valid_mask")
            world_2d = self.world_2d_expert.embed_queries(
                states.shape[0], states.device
            )
            positions_2d = world_start[:, None] + torch.arange(
                world_2d.shape[1], device=states.device
            )
            world_hidden_states.append(world_2d)
            world_masks.append(self._world_2d_mask(view_valid_mask))
            world_position_embeddings.append(
                language_model.rotary_emb(
                    world_2d, positions_2d.unsqueeze(0).expand(3, -1, -1)
                )
            )
            max_world_len = max(max_world_len, world_2d.shape[1])

        if self.world_3d_expert is not None:
            world_3d_valid_mask = batch.get(WORLD_3D_VALID_MASK)
            if world_3d_valid_mask is None:
                raise ValueError("3D stream requires world_3d_valid_mask")
            world_3d = self.world_3d_expert.embed_queries(
                states.shape[0], states.device
            )
            positions_3d = world_start[:, None] + torch.arange(
                world_3d.shape[1], device=states.device
            )
            world_hidden_states.append(world_3d)
            world_masks.append(self._world_3d_mask(world_3d_valid_mask))
            world_position_embeddings.append(
                language_model.rotary_emb(
                    world_3d, positions_3d.unsqueeze(0).expand(3, -1, -1)
                )
            )
            max_world_len = max(max_world_len, world_3d.shape[1])

        action_start = world_start + max_world_len
        action_positions = action_start[:, None] + torch.arange(
            self.config.chunk_size, device=states.device
        )
        action_rope_reference = vlm_hidden.new_empty(
            (
                states.shape[0],
                self.config.chunk_size,
                self.config.action_expert_hidden_size,
            )
        )
        action_position_embeddings = language_model.rotary_emb(
            action_rope_reference,
            action_positions.unsqueeze(0).expand(3, -1, -1),
        )

        components = self._active_joint_expert_components()
        if any(
            len(blocks) != len(language_model.layers) for _, blocks, _ in components
        ):
            raise RuntimeError("VLM and MagicW0 expert layer counts differ")
        vlm_key_values: list[tuple[torch.Tensor, torch.Tensor] | None] = []
        for layer_index, vlm_layer in enumerate(language_model.layers):
            blocks = [component[1][layer_index] for component in components]
            is_full = blocks[0].use_full_attention
            if any(block.use_full_attention != is_full for block in blocks):
                raise RuntimeError("MagicW0 expert full-attention schedules diverged")
            if is_full:
                vlm_context = vlm_layer.input_layernorm(vlm_hidden)[:, :prefix_len]
                if self.config.knowledge_insulation:
                    vlm_context = vlm_context.detach()
                vlm_key_values.append(
                    blocks[0].mixer._vlm_kv(
                        vlm_context,
                        vlm_layer.self_attn,
                        prefix_rope,
                        detach_parameters=self.config.knowledge_insulation,
                    )
                )
                attention_mask = vlm_causal_mask
            else:
                vlm_key_values.append(None)
                attention_mask = vlm_padding_mask
            vlm_hidden = vlm_layer(
                vlm_hidden,
                position_embeddings=vlm_position_embeddings,
                attention_mask=attention_mask,
                position_ids=vlm_position_ids[0],
                past_key_values=None,
                use_cache=False,
            )

        return {
            "stream_names": tuple(name for name, _, _ in components),
            "world_hidden_states": tuple(world_hidden_states),
            "world_masks": tuple(world_masks),
            "world_position_embeddings": tuple(world_position_embeddings),
            "action_position_embeddings": action_position_embeddings,
            "prefix_mask": prefix_mask,
            "vlm_key_values": tuple(vlm_key_values),
        }

    def _run_cached_joint_expert_trunks(
        self,
        noisy_actions: torch.Tensor,
        timesteps: torch.Tensor,
        states: torch.Tensor,
        action_attention_mask: torch.Tensor,
        cache: dict[str, Any],
        diagnostics: Any | None = None,
    ) -> torch.Tensor:
        """Run world/action experts using cached per-layer VLM prefix K/V."""
        if diagnostics is not None and self.training:
            raise RuntimeError("joint-attention diagnostics require model.eval()")
        components = self._active_joint_expert_components()
        stream_names = tuple(name for name, _, _ in components)
        if stream_names != cache["stream_names"]:
            raise RuntimeError("enabled MagicW0 streams changed after cache construction")

        hidden_states = list(cache["world_hidden_states"])
        hidden_states.append(
            self.action_expert.embed_inputs(noisy_actions, timesteps, states)
        )
        stream_masks = [*cache["world_masks"], action_attention_mask.bool()]
        stream_rope = [
            *cache["world_position_embeddings"],
            cache["action_position_embeddings"],
        ]
        dropouts = [dropout for _, _, dropout in components]

        for layer_index, cached_vlm_key_value in enumerate(cache["vlm_key_values"]):
            blocks = [component[1][layer_index] for component in components]
            is_full = blocks[0].use_full_attention
            if any(block.use_full_attention != is_full for block in blocks):
                raise RuntimeError("MagicW0 expert full-attention schedules diverged")
            if not is_full:
                updated: list[torch.Tensor] = []
                for hidden, mask, block, dropout in zip(
                    hidden_states, stream_masks, blocks, dropouts, strict=True
                ):
                    hidden = hidden + dropout(
                        _run_expert_deltanet(
                            block.mixer,
                            block.input_layernorm(hidden),
                            mask,
                        )
                    )
                    hidden = hidden + dropout(
                        block.mlp(block.post_attention_layernorm(hidden))
                    )
                    updated.append(hidden)
                hidden_states = updated
                continue

            if cached_vlm_key_value is None:
                raise RuntimeError("missing cached VLM K/V for a full-attention layer")
            normalized = [
                block.input_layernorm(value)
                for block, value in zip(blocks, hidden_states, strict=True)
            ]
            if diagnostics is not None:
                normalized = [
                    diagnostics.transform_hidden(name, value, layer_index)
                    for name, value in zip(stream_names, normalized, strict=True)
                ]
            qkv = [
                block.mixer._expert_qkv(value, rope)
                for block, value, rope in zip(
                    blocks, normalized, stream_rope, strict=True
                )
            ]
            vlm_key, vlm_value = cached_vlm_key_value
            key = torch.cat([vlm_key, *(entry[1] for entry in qkv)], dim=2)
            value = torch.cat([vlm_value, *(entry[2] for entry in qkv)], dim=2)
            full_key_mask = torch.cat((cache["prefix_mask"], *stream_masks), dim=1)
            first_mixer = blocks[0].mixer
            prepared_by_visibility: dict[
                tuple[str, ...], tuple[torch.Tensor, tuple[Any, ...]]
            ] = {}

            def prepared_for_query(
                query_stream: str,
            ) -> tuple[torch.Tensor, tuple[Any, ...]]:
                visible_sources = self._joint_attention_visible_sources(
                    query_stream, stream_names
                )
                cached_prepared = prepared_by_visibility.get(visible_sources)
                if cached_prepared is not None:
                    return cached_prepared
                query_mask = _build_query_stream_key_mask(
                    cache["prefix_mask"],
                    stream_masks,
                    stream_names,
                    visible_sources,
                )
                prepared = first_mixer.prepare_attention_kv(key, value, query_mask)
                result = (query_mask, prepared)
                prepared_by_visibility[visible_sources] = result
                return result

            updated = []
            for stream_name, hidden, block, dropout, (
                query,
                _key,
                _value,
                gate,
            ) in zip(stream_names, hidden_states, blocks, dropouts, qkv, strict=True):
                query_key_mask, query_prepared_kv = prepared_for_query(stream_name)
                if diagnostics is not None:
                    intervention_mask = diagnostics.key_mask(
                        stream_name,
                        full_key_mask,
                        cache["prefix_mask"],
                        stream_names,
                        stream_masks,
                        layer_index,
                    )
                    combined_mask = query_key_mask & intervention_mask
                    if not torch.equal(combined_mask, query_key_mask):
                        query_key_mask = combined_mask
                        query_prepared_kv = first_mixer.prepare_attention_kv(
                            key, value, query_key_mask
                        )
                    diagnostics.observe(
                        stream_name,
                        layer_index,
                        query,
                        key,
                        query_key_mask,
                        stream_masks[stream_names.index(stream_name)],
                    )
                dropout_p = (
                    self.config.joint_attention_dropout if self.training else 0.0
                )
                self._log_expert_full_attention_backend_once(
                    stream_name,
                    layer_index,
                    query,
                    key,
                    value,
                    query_key_mask,
                    dropout_p,
                    first_mixer,
                )
                attended = first_mixer.attend_prepared(
                    query,
                    query_prepared_kv,
                    dropout_p=dropout_p,
                )
                attended = attended.transpose(1, 2).reshape(
                    hidden.shape[0], hidden.shape[1], -1
                )
                mixed = block.mixer.o_proj(attended * torch.sigmoid(gate))
                hidden = hidden + dropout(mixed)
                hidden = hidden + dropout(
                    block.mlp(block.post_attention_layernorm(hidden))
                )
                updated.append(hidden)
            hidden_states = updated

        output_by_name = dict(zip(stream_names, hidden_states, strict=True))
        return self.action_expert.project_output(output_by_name["action"])

    def _run_joint_trunks(
        self,
        vlm_inputs: dict[str, Any],
        noisy_actions: torch.Tensor,
        timesteps: torch.Tensor,
        states: torch.Tensor,
        action_attention_mask: torch.Tensor,
        prefix_len: int,
        state_dim_mask: torch.Tensor | None = None,
        view_valid_mask: torch.Tensor | None = None,
        world_3d_valid_mask: torch.Tensor | None = None,
        action_world_stream_mask_mode: int = _ACTION_WORLD_KEEP_BOTH,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        vlm_hidden, vlm_padding_mask, vlm_position_ids = self._embed_vlm_inputs(
            vlm_inputs, states, state_dim_mask
        )
        language_model = self.vlm.model.language_model
        vlm_position_embeddings = language_model.rotary_emb(
            vlm_hidden, vlm_position_ids
        )
        from transformers.masking_utils import create_causal_mask

        vlm_causal_mask = create_causal_mask(
            config=language_model.config,
            inputs_embeds=vlm_hidden,
            attention_mask=vlm_padding_mask,
            past_key_values=None,
            position_ids=vlm_position_ids[0],
        )
        prefix_mask = vlm_padding_mask[:, :prefix_len]
        prefix_position_ids = vlm_position_ids[:, :, :prefix_len]
        prefix_rope = tuple(value[:, :prefix_len] for value in vlm_position_embeddings)
        valid_prefix_positions = prefix_position_ids.masked_fill(
            ~prefix_mask.unsqueeze(0), 0
        )
        world_start = valid_prefix_positions.amax(dim=(0, 2)).add(1)

        names: list[str] = []
        hidden_states: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        block_stacks: list[nn.ModuleList] = []
        dropouts: list[nn.Dropout] = []
        position_embeddings: list[tuple[torch.Tensor, torch.Tensor]] = []

        max_world_len = 0
        if self.world_2d_expert is not None:
            if view_valid_mask is None:
                raise ValueError("2D stream requires view_valid_mask")
            world_2d = self.world_2d_expert.embed_queries(
                states.shape[0], states.device
            )
            mask_2d = self._world_2d_mask(view_valid_mask)
            positions_2d = world_start[:, None] + torch.arange(
                world_2d.shape[1], device=states.device
            )
            names.append("world_2d")
            hidden_states.append(world_2d)
            masks.append(mask_2d)
            block_stacks.append(self.world_2d_expert.blocks)
            dropouts.append(self.world_2d_expert.residual_dropout)
            position_embeddings.append(
                language_model.rotary_emb(
                    world_2d, positions_2d.unsqueeze(0).expand(3, -1, -1)
                )
            )
            max_world_len = max(max_world_len, world_2d.shape[1])

        if self.world_3d_expert is not None:
            if world_3d_valid_mask is None:
                raise ValueError("3D stream requires world_3d_valid_mask")
            world_3d = self.world_3d_expert.embed_queries(
                states.shape[0], states.device
            )
            mask_3d = self._world_3d_mask(world_3d_valid_mask)
            positions_3d = world_start[:, None] + torch.arange(
                world_3d.shape[1], device=states.device
            )
            names.append("world_3d")
            hidden_states.append(world_3d)
            masks.append(mask_3d)
            block_stacks.append(self.world_3d_expert.blocks)
            dropouts.append(self.world_3d_expert.residual_dropout)
            position_embeddings.append(
                language_model.rotary_emb(
                    world_3d, positions_3d.unsqueeze(0).expand(3, -1, -1)
                )
            )
            max_world_len = max(max_world_len, world_3d.shape[1])

        action_hidden = self.action_expert.embed_inputs(
            noisy_actions, timesteps, states
        )
        action_start = world_start + max_world_len
        action_positions = action_start[:, None] + torch.arange(
            action_hidden.shape[1], device=states.device
        )
        names.append("action")
        hidden_states.append(action_hidden)
        masks.append(action_attention_mask.bool())
        block_stacks.append(self.action_expert.blocks)
        dropouts.append(self.action_residual_dropout)
        position_embeddings.append(
            language_model.rotary_emb(
                action_hidden, action_positions.unsqueeze(0).expand(3, -1, -1)
            )
        )

        use_checkpoint = self.config.gradient_checkpointing and self.training
        preserve_rng = any(
            value > 0
            for value in (
                self.config.joint_attention_dropout,
                self.config.action_expert_dropout,
                self.config.gen_2d_dropout,
                self.config.gen_3d_dropout,
            )
        )
        if use_checkpoint:
            from torch.utils.checkpoint import checkpoint

        for layer_index, vlm_layer in enumerate(language_model.layers):
            current_blocks = [stack[layer_index] for stack in block_stacks]

            def run_layer(
                vlm_value,
                *stream_values,
                _current_blocks=current_blocks,
                _vlm_layer=vlm_layer,
                _stream_names=tuple(names),
                _layer_index=layer_index,
                _action_world_stream_mask_mode=action_world_stream_mask_mode,
            ):
                return self._forward_joint_layer(
                    vlm_value,
                    list(stream_values),
                    masks,
                    position_embeddings,
                    _current_blocks,
                    dropouts,
                    _vlm_layer,
                    vlm_position_embeddings,
                    vlm_padding_mask,
                    vlm_causal_mask,
                    vlm_position_ids,
                    prefix_len,
                    prefix_mask,
                    prefix_rope,
                    stream_names=_stream_names,
                    layer_index=_layer_index,
                    action_world_stream_mask_mode=(_action_world_stream_mask_mode),
                )

            if use_checkpoint:
                outputs = checkpoint(
                    run_layer,
                    vlm_hidden,
                    *hidden_states,
                    use_reentrant=False,
                    preserve_rng_state=preserve_rng,
                )
            else:
                outputs = run_layer(vlm_hidden, *hidden_states)
            vlm_hidden, *hidden_states = outputs

        vlm_hidden = language_model.norm(vlm_hidden)
        output_by_name = dict(zip(names, hidden_states, strict=True))
        predicted_velocity = self.action_expert.project_output(output_by_name["action"])
        return (
            predicted_velocity,
            vlm_hidden,
            output_by_name.get("world_2d"),
            output_by_name.get("world_3d"),
        )

    def _forward_joint_layer(
        self,
        vlm_hidden: torch.Tensor,
        stream_hidden: list[torch.Tensor],
        stream_masks: list[torch.Tensor],
        stream_rope: list[tuple[torch.Tensor, torch.Tensor]],
        blocks: list[QwenHybridExpertLayer],
        dropouts: list[nn.Dropout],
        vlm_layer: nn.Module,
        vlm_rope: tuple[torch.Tensor, torch.Tensor],
        vlm_padding_mask: torch.Tensor,
        vlm_causal_mask: torch.Tensor,
        vlm_position_ids: torch.Tensor,
        prefix_len: int,
        prefix_mask: torch.Tensor,
        prefix_rope: tuple[torch.Tensor, torch.Tensor],
        stream_names: tuple[str, ...] | None = None,
        layer_index: int = -1,
        action_world_stream_mask_mode: int = _ACTION_WORLD_KEEP_BOTH,
    ) -> tuple[torch.Tensor, ...]:
        is_full = blocks[0].use_full_attention
        if any(block.use_full_attention != is_full for block in blocks):
            raise RuntimeError("MagicW0 expert full-attention schedules diverged")
        if is_full:
            normalized = [
                block.input_layernorm(value)
                for block, value in zip(blocks, stream_hidden)
            ]
            qkv = [
                block.mixer._expert_qkv(value, rope)
                for block, value, rope in zip(blocks, normalized, stream_rope)
            ]
            vlm_context = vlm_layer.input_layernorm(vlm_hidden)[:, :prefix_len]
            if self.config.knowledge_insulation:
                vlm_context = vlm_context.detach()
            first_mixer = blocks[0].mixer
            vlm_key, vlm_value = first_mixer._vlm_kv(
                vlm_context,
                vlm_layer.self_attn,
                prefix_rope,
                detach_parameters=self.config.knowledge_insulation,
            )
            keys = [vlm_key, *(entry[1] for entry in qkv)]
            values = [vlm_value, *(entry[2] for entry in qkv)]
            key = torch.cat(keys, dim=2)
            value = torch.cat(values, dim=2)
            if stream_names is None:
                stream_names = tuple(
                    f"stream_{index}" for index in range(len(stream_hidden))
                )
            if len(stream_names) != len(stream_hidden):
                raise ValueError("stream names/hidden states must have the same length")

            # K/V tensors are shared, but every query stream owns a visibility
            # row. Reuse the expensive backend preparation whenever rows resolve
            # to the same set of present sources (the default graph prepares once).
            prepared_by_visibility: dict[
                tuple[str, ...], tuple[torch.Tensor, tuple[Any, ...]]
            ] = {}

            def prepared_for_query(
                query_stream: str,
            ) -> tuple[torch.Tensor, tuple[Any, ...]]:
                visible_sources = self._joint_attention_visible_sources(
                    query_stream,
                    stream_names,
                    action_world_stream_mask_mode=(action_world_stream_mask_mode),
                )
                cached = prepared_by_visibility.get(visible_sources)
                if cached is not None:
                    return cached
                query_key_mask = _build_query_stream_key_mask(
                    prefix_mask,
                    stream_masks,
                    stream_names,
                    visible_sources,
                )
                prepared = first_mixer.prepare_attention_kv(key, value, query_key_mask)
                result = (query_key_mask, prepared)
                prepared_by_visibility[visible_sources] = result
                return result

            updated: list[torch.Tensor] = []
            for stream_name, hidden, block, dropout, (
                query,
                _key,
                _value,
                gate,
            ) in zip(stream_names, stream_hidden, blocks, dropouts, qkv, strict=True):
                dropout_p = (
                    self.config.joint_attention_dropout if self.training else 0.0
                )
                query_key_mask, query_prepared_kv = prepared_for_query(stream_name)
                self._log_expert_full_attention_backend_once(
                    stream_name,
                    layer_index,
                    query,
                    key,
                    value,
                    query_key_mask,
                    dropout_p,
                    first_mixer,
                )
                attended = first_mixer.attend_prepared(
                    query,
                    query_prepared_kv,
                    dropout_p=dropout_p,
                )
                attended = attended.transpose(1, 2).reshape(
                    hidden.shape[0], hidden.shape[1], -1
                )
                mixed = block.mixer.o_proj(attended * torch.sigmoid(gate))
                hidden = hidden + dropout(mixed)
                hidden = hidden + dropout(
                    block.mlp(block.post_attention_layernorm(hidden))
                )
                updated.append(hidden)
            vlm_hidden = vlm_layer(
                vlm_hidden,
                position_embeddings=vlm_rope,
                attention_mask=vlm_causal_mask,
                position_ids=vlm_position_ids[0],
                past_key_values=None,
                use_cache=False,
            )
            return (vlm_hidden, *updated)

        updated = []
        for hidden, mask, block, dropout in zip(
            stream_hidden, stream_masks, blocks, dropouts, strict=True
        ):
            hidden = hidden + dropout(
                _run_expert_deltanet(
                    block.mixer,
                    block.input_layernorm(hidden),
                    mask,
                )
            )
            hidden = hidden + dropout(block.mlp(block.post_attention_layernorm(hidden)))
            updated.append(hidden)
        vlm_hidden = vlm_layer(
            vlm_hidden,
            position_embeddings=vlm_rope,
            attention_mask=vlm_padding_mask,
            position_ids=vlm_position_ids[0],
            past_key_values=None,
            use_cache=False,
        )
        return (vlm_hidden, *updated)

    def _log_expert_full_attention_backend_once(
        self,
        expert_name: str,
        layer_index: int,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_mask: torch.Tensor,
        dropout_p: float,
        mixer: nn.Module,
    ) -> None:
        if layer_index < 0:
            return
        key_id = (expert_name, layer_index)
        already_logged = getattr(self, "_logged_expert_full_attention_layers", None)
        if already_logged is None:
            already_logged = set()
            self._logged_expert_full_attention_layers = already_logged
        if key_id in already_logged:
            return
        already_logged.add(key_id)
        if os.environ.get("RANK", "0") != "0":
            return
        if mixer.full_attention_backend == "flash_attention_2":
            logger.warning(
                "[expert-full-attention] expert=%s layer=%d/%d "
                "code_path=external_flash_attn_varlen "
                "selected_backend=flash_attention_2 external_flash_attn2=yes "
                "function=%s version=%s dtype=%s q_len=%d kv_len=%d "
                "dropout_p=%.4f mask_shape=%s",
                expert_name,
                layer_index + 1,
                self.config.action_expert_num_layers,
                _qualified_name(mixer._flash_attn_varlen_func),
                mixer.flash_attn_version,
                query.dtype,
                query.shape[-2],
                key.shape[-2],
                dropout_p,
                tuple(key_mask.shape),
            )
            return
        repeated_key = _repeat_kv(key, mixer.kv_repeats)
        repeated_value = _repeat_kv(value, mixer.kv_repeats)
        attention_mask = key_mask[:, None, None, :]
        selected = _selected_sdpa_backend(
            query,
            repeated_key,
            repeated_value,
            attention_mask,
            dropout_p,
            mixer.head_dim**-0.5,
        )
        logger.warning(
            "[expert-full-attention] expert=%s layer=%d/%d "
            "code_path=torch_sdpa selected_backend=%s external_flash_attn2=no "
            "dtype=%s q_len=%d kv_len=%d dropout_p=%.4f mask_shape=%s. "
            "selected_backend=flash_attention means PyTorch fused Flash SDPA.",
            expert_name,
            layer_index + 1,
            self.config.action_expert_num_layers,
            selected,
            query.dtype,
            query.shape[-2],
            repeated_key.shape[-2],
            dropout_p,
            tuple(attention_mask.shape),
        )

    def _world_2d_mask(self, view_mask: torch.Tensor) -> torch.Tensor:
        return view_mask.bool().repeat_interleave(
            self.config.gen_2d_num_query_tokens_per_view, dim=1
        )

    def _world_3d_mask(self, sample_mask: torch.Tensor) -> torch.Tensor:
        return (
            sample_mask.bool()
            .unsqueeze(1)
            .expand(-1, self.config.gen_3d_num_query_tokens)
        )


MagicW0Training = MagicW0TrainingPolicy


def _masked_layernorm_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    prediction = F.layer_norm(prediction.float(), (prediction.shape[-1],))
    target = F.layer_norm(target.detach().float(), (target.shape[-1],))
    token_loss = (prediction - target).square().mean(dim=-1)
    denominator = mask.sum().clamp_min(1)
    return (token_loss * mask.bool()).sum() / denominator
