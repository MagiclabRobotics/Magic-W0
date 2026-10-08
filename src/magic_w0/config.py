"""Magic-W0 inference components; tensor names and computations are preserved."""

from __future__ import annotations
from dataclasses import dataclass, fields
from pathlib import Path


EMBODIMENT_DESCRIPTIONS = {
    "arx_x5_sim": "simulated ARX X5 dual-arm robot with parallel grippers",
}


def describe(name: str) -> str | None:
    return EMBODIMENT_DESCRIPTIONS.get(name)


STATE = "magic_w0.state"


ACTION_DIM_MASK = "magic_w0.action_dim_mask"


CAMERA_VALID = "magic_w0.camera_valid"


TASK = "magic_w0.task"


EMBODIMENT = "magic_w0.embodiment"


STATE_DIM_MASK = "magic_w0.state_dim_mask"


VIEW_VALID_MASK = "magic_w0.view_valid_mask"


WORLD_3D_VALID_MASK = "magic_w0.world_3d_valid_mask"


HISTORY_FRAMES = "magic_w0.history_frames"


HISTORY_LEN = "magic_w0.history_len"


HISTORY_IMAGES = "history_images"


HISTORY_PLACEHOLDER_TOKEN = "<|vision_pad|>"


_IMAGE_LABELS = {
    "cam_high": "Head",
    "cam_left_wrist": "Left wrist",
    "cam_right_wrist": "Right wrist",
}


SUPPORTED_MODES = {
    "state_prompt_mode": "embedding",
    "state_prompt_position": "append",
    "history_enabled": True,
    "subtask_generation": False,
    "enable_2d_world_expert": True,
    "enable_3d_world_expert": True,
    "expert_full_attention_backend": "sdpa",
    "expert_deltanet_backend": "torch_parallel",
    "gen_2d_camera_embedding": True,
}


def validate_checkpoint_modes(config):
    for name, expected in SUPPORTED_MODES.items():
        actual = config.get(name, expected)
        if actual != expected:
            raise ValueError(
                f"unsupported {name}={actual!r}; this runtime requires {expected!r}"
            )
    if config.get("compile_model", False):
        raise ValueError("compiled execution is not supported")
    graph = config.get("joint_attention_visibility")
    streams = ("world_2d", "world_3d", "action")
    if graph is not None and (
        set(graph) != set(streams)
        or any(set(graph[s]) != {"vlm", *streams} for s in streams)
    ):
        raise ValueError("this checkpoint must enable full joint expert visibility")


@dataclass
class MagicW0Config:
    """Dimensions and runtime inputs for the supported inference graph."""

    action_dim: int = 34
    action_expert_hidden_size: int = 1024
    action_expert_intermediate_size: int = 3072
    action_expert_num_layers: int = 24
    camera_keys: tuple[str, ...] = (
        "observation.images.cam_high",
        "observation.images.cam_left_wrist",
        "observation.images.cam_right_wrist",
    )
    chunk_size: int = 50
    deltanet_conv_kernel: int = 4
    deltanet_key_head_dim: int = 128
    deltanet_num_key_heads: int = 16
    deltanet_num_value_heads: int = 16
    deltanet_value_head_dim: int = 128
    dinov3_target_hidden_size: int = 1024
    dtype: str = "bfloat16"
    full_attention_interval: int = 4
    full_attn_head_dim: int = 256
    full_attn_num_heads: int = 8
    full_attn_num_kv_heads: int = 2
    gen_2d_camera_keys: tuple[str, ...] = ("observation.images.cam_high",)
    gen_2d_grid_size: tuple[int, int] = (16, 16)
    gen_2d_max_views: int = 1
    gen_2d_num_query_tokens_per_view: int = 256
    gen_3d_camera_key: str = "observation.images.cam_high"
    gen_3d_grid_size: tuple[int, int] = (16, 16)
    gen_3d_num_query_tokens: int = 256
    history_camera_key: str = "observation.images.cam_high"
    history_frames: int = 25
    history_interval: int = 20
    history_pool_size: int = 4
    image_size: int = 256
    num_inference_steps: int = 10
    rms_norm_eps: float = 1e-06
    robot_prompt_metadata: str = "Robot action space: {action_space}.{joint_dimensions}"
    state_dim: int = 34
    state_prompt_tokens: int = 1
    vlm_checkpoint: str = "checkpoints/qwen3.5-2b-assets"
    vlm_model_type: str = "qwen3_5"

    def __post_init__(self):
        if self.dtype not in {"bfloat16", "float32"}:
            raise ValueError("dtype must be bfloat16 or float32")
        for name in (
            "history_frames",
            "history_interval",
            "history_pool_size",
            "state_prompt_tokens",
            "action_dim",
            "state_dim",
            "chunk_size",
            "action_expert_num_layers",
            "full_attention_interval",
            "num_inference_steps",
            "full_attn_num_heads",
            "full_attn_num_kv_heads",
            "full_attn_head_dim",
            "deltanet_num_key_heads",
            "deltanet_num_value_heads",
            "deltanet_key_head_dim",
            "deltanet_value_head_dim",
            "deltanet_conv_kernel",
            "action_expert_hidden_size",
            "action_expert_intermediate_size",
            "gen_2d_max_views",
            "dinov3_target_hidden_size",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.image_size != 256 or self.history_pool_size > 8:
            raise ValueError(
                "history requires image_size=256 and history_pool_size <= 8"
            )
        if self.history_camera_key not in self.camera_keys:
            raise ValueError("history_camera_key must appear in camera_keys")
        if self.action_expert_num_layers % self.full_attention_interval:
            raise ValueError(
                "expert depth must be divisible by full_attention_interval"
            )
        if (
            self.gen_2d_grid_size[0] * self.gen_2d_grid_size[1]
            != self.gen_2d_num_query_tokens_per_view
        ):
            raise ValueError("2D query count must equal grid size product")
        if (
            self.gen_3d_grid_size[0] * self.gen_3d_grid_size[1]
            != self.gen_3d_num_query_tokens
        ):
            raise ValueError("3D query count must equal grid size product")
        if (
            not self.gen_2d_camera_keys
            or len(self.gen_2d_camera_keys) > self.gen_2d_max_views
        ):
            raise ValueError("invalid 2D camera count")
        if any(
            k not in self.camera_keys
            for k in (*self.gen_2d_camera_keys, self.gen_3d_camera_key)
        ):
            raise ValueError("world expert cameras must appear in camera_keys")
        if self.full_attn_num_heads % self.full_attn_num_kv_heads:
            raise ValueError("attention head counts must be divisible")
        if self.deltanet_num_value_heads % self.deltanet_num_key_heads:
            raise ValueError("DeltaNet head counts must be divisible")

    @property
    def history_tokens_per_frame(self):
        return self.history_pool_size**2

    @classmethod
    def from_dict(cls, payload):
        raw = dict(payload)
        known = {f.name for f in fields(cls)}
        if set(raw) - known:
            raise ValueError(f"unknown inference settings: {sorted(set(raw) - known)}")
        for name in (
            "camera_keys",
            "gen_2d_camera_keys",
            "gen_2d_grid_size",
            "gen_3d_grid_size",
        ):
            if name in raw:
                raw[name] = tuple(raw[name])
        return cls(**raw)

    def validate_checkpoint_layout(self):
        path = Path(self.vlm_checkpoint)
        if not (path / "config.json").is_file():
            raise FileNotFoundError(
                f"missing Qwen config/tokenizer/processor directory: {path}"
            )
        return path
