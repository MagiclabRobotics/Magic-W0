"""Configuration for the shared-backbone MagicW0 training model."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields
from typing import Any

from magic_w0.models.action_backbone.configuration_magicw0_base import (
    DEFAULT_ROBODOJO_CAMERAS,
    MagicW0BaseConfig,
)

DEFAULT_DINOV3_PATH = "checkpoints/third_party_models/dinov3-vitl16-pretrain-lvd1689m"
DEFAULT_TRACK4WORLD_ROOT = "third_party/Track4World"
DEFAULT_TRACK4WORLD_DA3_BACKBONE = (
    "checkpoints/third_party_models/DA3NESTED-GIANT-LARGE-1.1"
)
DEFAULT_TRACK4WORLD_CONVNEXT_TINY = (
    "checkpoints/third_party_models/convnext_tiny-983f1562.pth"
)

JOINT_ATTENTION_QUERY_STREAMS = ("world_2d", "world_3d", "action")
JOINT_ATTENTION_KEY_STREAMS = ("vlm", *JOINT_ATTENTION_QUERY_STREAMS)


@dataclass(frozen=True, slots=True)
class JointAttentionVisibilityConfig:
    """Allowed K/V sources for each expert query at full-attention layers."""

    world_2d: tuple[str, ...] = JOINT_ATTENTION_KEY_STREAMS
    world_3d: tuple[str, ...] = JOINT_ATTENTION_KEY_STREAMS
    action: tuple[str, ...] = JOINT_ATTENTION_KEY_STREAMS

    def __post_init__(self) -> None:
        for query_stream in JOINT_ATTENTION_QUERY_STREAMS:
            raw_sources = getattr(self, query_stream)
            if isinstance(raw_sources, list):
                sources = tuple(raw_sources)
                object.__setattr__(self, query_stream, sources)
            elif isinstance(raw_sources, tuple):
                sources = raw_sources
            else:
                raise TypeError(
                    f"joint_attention_visibility.{query_stream} must be a list or tuple"
                )
            if not sources:
                raise ValueError(
                    f"joint_attention_visibility.{query_stream} must not be empty"
                )
            if any(not isinstance(source, str) for source in sources):
                raise TypeError(
                    f"joint_attention_visibility.{query_stream} entries must be strings"
                )
            unknown = set(sources) - set(JOINT_ATTENTION_KEY_STREAMS)
            if unknown:
                raise ValueError(
                    "joint_attention_visibility."
                    f"{query_stream} has unknown K/V sources {sorted(unknown)}; "
                    f"allowed {list(JOINT_ATTENTION_KEY_STREAMS)}"
                )
            if len(set(sources)) != len(sources):
                raise ValueError(
                    "joint_attention_visibility."
                    f"{query_stream} contains duplicate K/V sources"
                )

    def sources_for(self, query_stream: str) -> tuple[str, ...]:
        if query_stream not in JOINT_ATTENTION_QUERY_STREAMS:
            raise ValueError(f"unknown expert query stream {query_stream!r}")
        return getattr(self, query_stream)

    @classmethod
    def from_dict(
        cls, payload: dict[str, Any] | None
    ) -> "JointAttentionVisibilityConfig":
        raw = dict(payload or {})
        unknown = set(raw) - set(JOINT_ATTENTION_QUERY_STREAMS)
        if unknown:
            raise ValueError(
                "JointAttentionVisibilityConfig: unknown query streams "
                f"{sorted(unknown)}; allowed {list(JOINT_ATTENTION_QUERY_STREAMS)}"
            )
        return cls(**raw)


@dataclass(frozen=True, slots=True)
class ActionWorldStreamMaskConfig:
    """Training-only visibility dropout from Action queries to World K/V."""

    enabled: bool = False
    keep_both_prob: float = 0.8
    mask_2d_prob: float = 0.1
    mask_3d_prob: float = 0.1

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("action_world_stream_mask.enabled must be a boolean")
        probabilities = {
            "keep_both_prob": self.keep_both_prob,
            "mask_2d_prob": self.mask_2d_prob,
            "mask_3d_prob": self.mask_3d_prob,
        }
        invalid = [
            name
            for name, value in probabilities.items()
            if not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or not 0.0 <= float(value) <= 1.0
        ]
        if invalid:
            raise ValueError(
                "action_world_stream_mask probabilities must be finite and lie "
                f"in [0, 1]: {invalid}"
            )
        total = sum(float(value) for value in probabilities.values())
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(
                f"action_world_stream_mask probabilities must sum to 1.0; got {total}"
            )

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "ActionWorldStreamMaskConfig":
        raw = dict(payload or {})
        known = {item.name for item in fields(cls)}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(
                "ActionWorldStreamMaskConfig: unknown keys "
                f"{sorted(unknown)}; allowed {sorted(known)}"
            )
        return cls(**raw)


@dataclass
class MagicW0TrainingConfig(MagicW0BaseConfig):
    """New MagicW0 Base plus synchronous 2D and 3D world streams.

    The VLM follows the shared backbone co-training contract.  Robotics batches run
    Action/2D/3D experts; VLM batches run assistant-only language modelling and
    never invoke either frozen world teacher.  Knowledge insulation is enabled
    by default so action/world losses cannot update the VLM.
    """

    policy_type = "magicw0"

    # Production defaults match the new 8-source Base pre-training setup.
    action_dim: int = 32
    state_dim: int = 32
    chunk_size: int = 50
    camera_keys: tuple[str, ...] = DEFAULT_ROBODOJO_CAMERAS
    train_action_expert_only: bool = False
    knowledge_insulation: bool = True
    freeze_vision_encoder: bool = False
    cotrain_vlm_modalities: tuple[str, ...] = ("vlm_sft",)
    vlm_loss_weight: float = 0.1
    use_fast_action_tokens: bool = True
    fast_tokenizer_path: str | None = "checkpoints/fast_tokenizer"
    fast_ce_loss_weight: float = 0.05

    # Optional action-first auxiliary-loss schedule. The existing loss weights
    # above/below are the maxima. When enabled, FAST/2D/3D outer weights stay at
    # their maxima for the first 20% of training, cosine-decay over the next
    # 60%, and stay at these minima for the final 20%. The trainer supplies its
    # effective max_steps; no second, potentially divergent step budget lives in
    # the model config.
    loss_weight_schedule_enabled: bool = False
    fast_ce_loss_weight_min: float = 0.005
    gen_2d_loss_weight_min: float = 0.05
    world_3d_loss_weight_min: float = 0.05

    # ``dropout`` is inherited only because Base blocks consume it.  MagicW0 YAMLs
    # must use the explicit names below; from_dict rejects the ambiguous alias.
    joint_attention_dropout: float = 0.0
    action_expert_dropout: float = 0.0
    gen_2d_dropout: float = 0.0
    gen_3d_dropout: float = 0.0

    # Deterministic expert-query -> K/V-source graph used at every Joint Full
    # Attention layer in both training and inference. ``vlm`` means the VLM
    # prefix; VLM queries still use their own causal self-attention and never
    # read expert K/V. The legacy training-only Action/World dropout below is
    # composed on top of the Action row.
    joint_attention_visibility: JointAttentionVisibilityConfig = field(
        default_factory=JointAttentionVisibilityConfig
    )

    # One process-local draw is made for each robotics training step. The
    # selected mode applies to the whole local batch and changes only what the
    # Action query can read at Joint Full Attention layers. It owns no model
    # state, so adding it is strictly compatible with existing checkpoints.
    action_world_stream_mask: ActionWorldStreamMaskConfig = field(
        default_factory=ActionWorldStreamMaskConfig
    )

    enable_2d_world_expert: bool = True
    freeze_2d_world_expert: bool = False
    # Teachers are training-only supervision. Deployment keeps the learned
    # world streams active while disabling all DINO/Track4World construction.
    enable_world_teacher_supervision: bool = True
    dinov3_target_hidden_size: int = 1024
    gen_2d_grid_size: tuple[int, int] = (16, 16)
    gen_2d_num_query_tokens_per_view: int = 256
    gen_2d_max_views: int = 1
    gen_2d_camera_keys: tuple[str, ...] = ("observation.images.cam_high",)
    gen_2d_camera_embedding: bool = True
    gen_2d_loss_weight: float = 0.2
    dinov3_checkpoint: str = DEFAULT_DINOV3_PATH
    dinov3_local_files_only: bool = True

    enable_3d_world_expert: bool = True
    freeze_3d_world_expert: bool = False
    gen_3d_grid_size: tuple[int, int] = (16, 16)
    gen_3d_num_query_tokens: int = 256
    gen_3d_camera_key: str = "observation.images.cam_high"
    geometry_latent_weight: float = 0.5
    motion_latent_weight: float = 1.0
    world_3d_loss_weight: float = 0.3

    # ``None`` means the end of the source action chunk.  This is source-aware:
    # a 50-token chunk at action_upsample_rate=2 predicts t+25 source frames,
    # while rate=1 predicts t+50.  Set an integer only for a fixed-frame ablation.
    gen_2d_future_frame_offset: int | None = None
    gen_3d_future_frame_offset: int | None = None
    gen_world_data_format: str = "auto"

    track4world_inference_backend: str = "pytorch"
    track4world_tensorrt_engine_dir: str = "checkpoints/teacher_engines"
    track4world_tensorrt_keep_execution_memory: bool = False
    track4world_tensorrt_expected_engine_batch_sizes: dict[str, int] | None = None
    track4world_project_root: str = DEFAULT_TRACK4WORLD_ROOT
    track4world_ckpt: str = "checkpoints/third_party_models/track4world_da3.pth"
    track4world_config_path: str = "track4world/config/eval/v1.json"
    track4world_da3_backbone_path: str = DEFAULT_TRACK4WORLD_DA3_BACKBONE
    track4world_local_files_only: bool = True
    track4world_convnext_tiny_weights_path: str = DEFAULT_TRACK4WORLD_CONVNEXT_TINY
    track4world_image_size: int = 256
    track4world_multiple: int = 64
    track4world_preprocess_mode: str = "already_letterboxed_256"
    track4world_inference_iters: int = 4
    # Maximum number of frame pairs refined together by Track4World's
    # correlation/update blocks. Larger values reduce micro-batch loop overhead
    # but increase peak VRAM; this is independent of the action chunk_size.
    track4world_motion_chunk_size: int = 12
    track4world_autocast_dtype: str = "float16"

    #: MagicW0 visual history (see ``history_frames.py``).  Off by default so
    #: every existing checkpoint and recipe is unchanged; it adds no parameter,
    #: so a history run loads a history-free checkpoint strictly.
    history_enabled: bool = False
    #: Strictly past frames in the window.
    history_frames: int = 25
    #: Spacing in ACTION steps; divided by the source's action_upsample_rate
    #: to get video frames.  25 x 20 at RoboDojo's 25 fps = 0.8 s apart, 20 s.
    history_interval: int = 20
    #: Side of the pooled grid per frame (8x8 merged grid on the 256 canvas).
    history_pool_size: int = 4
    history_camera_key: str = "observation.images.cam_high"
    #: Training-sample history dropout, drawn per sample in the dataset:
    #: keep the whole window, keep only a random most-recent part (1..n-1
    #: frames), or drop it all.  Both dropped forms are exactly what an episode
    #: start looks like, so no new input form is introduced.  Must sum to 1.
    history_keep_prob: float = 1.0
    history_partial_drop_prob: float = 0.0
    history_full_drop_prob: float = 0.0
    #: Sources whose root contains any of these substrings always keep their
    #: whole window (e.g. ``memory__``: the tasks the history exists for).
    history_dropout_exempt_sources: tuple[str, ...] = ()

    @property
    def history_tokens_per_frame(self) -> int:
        return int(self.history_pool_size) ** 2

    def __post_init__(self) -> None:
        if isinstance(self.joint_attention_visibility, dict):
            self.joint_attention_visibility = JointAttentionVisibilityConfig.from_dict(
                self.joint_attention_visibility
            )
        elif not isinstance(
            self.joint_attention_visibility, JointAttentionVisibilityConfig
        ):
            raise TypeError(
                "joint_attention_visibility must be a mapping or "
                "JointAttentionVisibilityConfig"
            )
        if isinstance(self.action_world_stream_mask, dict):
            self.action_world_stream_mask = ActionWorldStreamMaskConfig.from_dict(
                self.action_world_stream_mask
            )
        elif not isinstance(self.action_world_stream_mask, ActionWorldStreamMaskConfig):
            raise TypeError(
                "action_world_stream_mask must be a mapping or "
                "ActionWorldStreamMaskConfig"
            )
        # Base's full-attention module reads ``dropout``.  Keep one uniform
        # attention-weight dropout across OBS/2D/3D/ACTION.
        self.dropout = self.joint_attention_dropout
        super().__post_init__()

        probabilities = {
            "joint_attention_dropout": self.joint_attention_dropout,
            "action_expert_dropout": self.action_expert_dropout,
            "gen_2d_dropout": self.gen_2d_dropout,
            "gen_3d_dropout": self.gen_3d_dropout,
        }
        invalid = [
            name for name, value in probabilities.items() if not 0.0 <= value < 1.0
        ]
        if invalid:
            raise ValueError(
                f"dropout probabilities must satisfy 0 <= p < 1: {invalid}"
            )
        if self.image_size != self.track4world_image_size:
            raise ValueError(
                "image_size and track4world_image_size must match for the shared "
                "letterboxed robotics images"
            )
        if self.gen_2d_grid_size[0] * self.gen_2d_grid_size[1] != (
            self.gen_2d_num_query_tokens_per_view
        ):
            raise ValueError("2D query count must equal gen_2d_grid_size product")
        if (
            self.gen_3d_grid_size[0] * self.gen_3d_grid_size[1]
            != self.gen_3d_num_query_tokens
        ):
            raise ValueError("3D query count must equal gen_3d_grid_size product")
        if self.enable_2d_world_expert and not self.gen_2d_camera_keys:
            raise ValueError("gen_2d_camera_keys must be non-empty when 2D is enabled")
        active_key_streams = {"vlm", "action"}
        if self.enable_2d_world_expert:
            active_key_streams.add("world_2d")
        if self.enable_3d_world_expert:
            active_key_streams.add("world_3d")
        active_query_streams = active_key_streams - {"vlm"}
        for query_stream in active_query_streams:
            configured = set(self.joint_attention_visibility.sources_for(query_stream))
            active_sources = configured & active_key_streams
            if not active_sources:
                raise ValueError(
                    "joint_attention_visibility."
                    f"{query_stream} has no enabled K/V source; configured "
                    f"{sorted(configured)}, enabled {sorted(active_key_streams)}"
                )
            if (
                query_stream == "action"
                and self.action_world_stream_mask.enabled
                and self.enable_2d_world_expert
                and self.enable_3d_world_expert
            ):
                for masked_source, probability in (
                    ("world_2d", self.action_world_stream_mask.mask_2d_prob),
                    ("world_3d", self.action_world_stream_mask.mask_3d_prob),
                ):
                    if probability > 0.0 and not (active_sources - {masked_source}):
                        raise ValueError(
                            "action_world_stream_mask can remove the last enabled "
                            "Action K/V source: "
                            f"joint_attention_visibility.action={sorted(configured)}, "
                            f"masked_source={masked_source!r}"
                        )
        if self.dinov3_target_hidden_size <= 0:
            raise ValueError("dinov3_target_hidden_size must be positive")
        if len(self.gen_2d_camera_keys) > self.gen_2d_max_views:
            raise ValueError("gen_2d_camera_keys cannot exceed gen_2d_max_views")
        for name, value in (
            ("gen_2d_future_frame_offset", self.gen_2d_future_frame_offset),
            ("gen_3d_future_frame_offset", self.gen_3d_future_frame_offset),
        ):
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive or None")
        if self.gen_world_data_format not in {
            "auto",
            "lerobot_v21",
            "lerobot_v30",
            "global_index",
        }:
            raise ValueError("invalid gen_world_data_format")
        if self.gen_2d_loss_weight < 0:
            raise ValueError("gen_2d_loss_weight must be non-negative")
        if (
            min(
                self.geometry_latent_weight,
                self.motion_latent_weight,
                self.world_3d_loss_weight,
            )
            < 0
        ):
            raise ValueError("3D loss weights must be non-negative")
        scheduled_ranges = (
            (
                "fast_ce_loss_weight",
                self.fast_ce_loss_weight,
                self.fast_ce_loss_weight_min,
            ),
            (
                "gen_2d_loss_weight",
                self.gen_2d_loss_weight,
                self.gen_2d_loss_weight_min,
            ),
            (
                "world_3d_loss_weight",
                self.world_3d_loss_weight,
                self.world_3d_loss_weight_min,
            ),
        )
        for name, maximum, minimum in scheduled_ranges:
            if minimum < 0:
                raise ValueError(f"{name}_min must be non-negative")
            if self.loss_weight_schedule_enabled and minimum > maximum:
                raise ValueError(
                    f"{name}_min cannot exceed {name} when loss scheduling is "
                    f"enabled: {minimum} > {maximum}"
                )
        if self.track4world_preprocess_mode not in {
            "already_letterboxed_256",
            "pad_to_multiple",
            "demo_resize",
        }:
            raise ValueError("invalid track4world_preprocess_mode")
        if self.track4world_inference_backend not in {"pytorch", "tensorrt"}:
            raise ValueError(
                "track4world_inference_backend must be 'pytorch' or 'tensorrt'"
            )
        expected_engine_names = {
            "da3_anyview",
            "da3_metric",
            "flow_feature",
            "motion_iteration",
        }
        expected_batches = self.track4world_tensorrt_expected_engine_batch_sizes
        if expected_batches is not None and (
            set(expected_batches) != expected_engine_names
            or any(batch_size <= 0 for batch_size in expected_batches.values())
        ):
            raise ValueError(
                "track4world_tensorrt_expected_engine_batch_sizes must contain "
                "positive batches for all four TensorRT engines"
            )
        if self.track4world_autocast_dtype not in {"float16", "bfloat16", "float32"}:
            raise ValueError("invalid track4world_autocast_dtype")
        if (
            self.track4world_inference_iters <= 0
            or self.track4world_motion_chunk_size <= 0
            or self.track4world_multiple <= 0
        ):
            raise ValueError(
                "Track4World iteration/motion-chunk/multiple values must be positive"
            )
        if not isinstance(self.history_enabled, bool):
            raise TypeError("history_enabled must be a boolean")
        if self.history_enabled:
            if self.history_frames < 1 or self.history_interval < 1:
                raise ValueError("history_frames and history_interval must be >= 1")
            if not 1 <= self.history_pool_size <= 8:
                raise ValueError(
                    "history_pool_size must lie in [1, 8], the side of the merged "
                    f"vision grid of a 256x256 frame, got {self.history_pool_size}"
                )
            if self.image_size != 256:
                raise ValueError("MagicW0 history pooling assumes the 256x256 canvas")
            if self.history_camera_key not in self.camera_keys:
                raise ValueError(
                    f"history_camera_key {self.history_camera_key!r} is not one of "
                    f"camera_keys {list(self.camera_keys)}"
                )
            if self.gen_world_data_format == "global_index":
                raise ValueError(
                    "history needs a LeRobot reader with get_episode_frames; "
                    "gen_world_data_format='global_index' has none"
                )
            probabilities = (
                self.history_keep_prob,
                self.history_partial_drop_prob,
                self.history_full_drop_prob,
            )
            if any(
                not math.isfinite(float(p)) or not 0.0 <= float(p) <= 1.0
                for p in probabilities
            ) or not math.isclose(sum(probabilities), 1.0, abs_tol=1e-9):
                raise ValueError(
                    "history keep/partial/full drop probabilities must lie in [0,1] "
                    f"and sum to 1, got {probabilities}"
                )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> MagicW0TrainingConfig:
        raw = dict(payload or {})
        if "dropout" in raw:
            raise ValueError(
                "MagicW0TrainingConfig does not accept ambiguous 'dropout'; use "
                "joint_attention_dropout/action_expert_dropout/gen_2d_dropout/"
                "gen_3d_dropout"
            )
        known = {field.name for field in fields(cls)}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(
                f"MagicW0TrainingConfig: unknown keys {sorted(unknown)}; allowed {sorted(known)}"
            )
        for key in (
            "camera_keys",
            "cotrain_vlm_modalities",
            "gen_2d_camera_keys",
            "gen_2d_grid_size",
            "gen_3d_grid_size",
            "history_dropout_exempt_sources",
        ):
            if isinstance(raw.get(key), list):
                raw[key] = tuple(raw[key])
        if "gen_2d_camera_keys" in raw and "gen_2d_max_views" not in raw:
            raw["gen_2d_max_views"] = len(raw["gen_2d_camera_keys"])
        if isinstance(raw.get("action_world_stream_mask"), dict):
            raw["action_world_stream_mask"] = ActionWorldStreamMaskConfig.from_dict(
                raw["action_world_stream_mask"]
            )
        if isinstance(raw.get("joint_attention_visibility"), dict):
            raw["joint_attention_visibility"] = (
                JointAttentionVisibilityConfig.from_dict(
                    raw["joint_attention_visibility"]
                )
            )
        return cls(**raw)
