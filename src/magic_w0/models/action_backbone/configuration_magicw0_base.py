"""Configuration for the Qwen3.5 based MagicW0 policy."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from magic_w0.models.action_loss_weighting import validate_action_loss_weighting


DEFAULT_QWEN35_PATH = "checkpoints/Qwen3.5-2B"
DEFAULT_ROBODOJO_CAMERAS = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)


@dataclass
class MagicW0BaseConfig:
    """Training configuration for ``magicw0_base``.

    The action expert mirrors the Qwen3.5 text-layer schedule at a smaller
    width.  Qwen3.5-2B has six repeats of three Gated DeltaNet layers followed
    by one full-attention layer.  Only the full-attention layers mix VLM and
    action tokens; the linear-attention layers remain modality-specific.
    """

    policy_type = "magicw0_base"

    vlm_checkpoint: str = DEFAULT_QWEN35_PATH
    vlm_model_type: str = "qwen3_5"
    local_files_only: bool = True
    trust_remote_code: bool = False
    dtype: str = "bfloat16"
    attention_implementation: str = "sdpa"
    # Execution-only backend for the asymmetric Action/World full-attention
    # layers.  The VLM's own full attention is controlled independently by
    # ``attention_implementation`` above.
    expert_full_attention_backend: str = "sdpa"
    gradient_checkpointing: bool = True
    # torch.compile over the joint trunk loop only -- not the whole forward, which
    # would drag the processor and loss bookkeeping into the graph and break it.
    # Off by default: at batch 64 the step is already 97.6% forward+backward with
    # a 93% busy GPU, so this is a fusion play against kernel-launch gaps, and it
    # has to earn its place against two things that make compiling awkward here --
    # co-training alternates two sub-graphs (robotics vs vlm_sft) and the padded
    # prefix length changes batch to batch.  compile_dynamic exists for the second.
    compile_model: bool = False
    compile_mode: str = "default"  # "default" | "reduce-overhead" | "max-autotune"
    compile_dynamic: bool = (
        True  # mark shapes dynamic so varying prefix_len does not recompile
    )

    action_expert_hidden_size: int = 1024
    action_expert_intermediate_size: int = 3072
    action_expert_num_layers: int = 24
    full_attention_interval: int = 4
    full_attn_num_heads: int = 8
    full_attn_num_kv_heads: int = 2
    full_attn_head_dim: int = 256
    # cuDNN SDPA builds a new execution plan for every VLM prefix length (~1 s
    # forward + ~1 s backward on H20). The efficient backend supports our
    # non-null padding mask and avoids that shape-keyed latency.
    action_expert_efficient_sdpa: bool = False
    deltanet_num_key_heads: int = 16
    deltanet_num_value_heads: int = 16
    deltanet_key_head_dim: int = 128
    deltanet_value_head_dim: int = 128
    deltanet_conv_kernel: int = 4
    # Execution-only choice. It is intentionally not part of model state_dicts,
    # so one checkpoint can resume with either numerically compatible backend.
    expert_deltanet_backend: str = "torch_parallel"
    rms_norm_eps: float = 1e-6
    action_dim: int = 14
    chunk_size: int = 16
    num_inference_steps: int = 10
    dropout: float = 0.0
    time_min: float = 0.001
    time_max: float = 0.999
    time_sampling_beta_alpha: float = 1.5
    time_sampling_beta_beta: float = 1.0

    state_dim: int = 14
    state_key: str = "observation.state"
    action_key: str = "action"
    camera_keys: tuple[str, ...] = DEFAULT_ROBODOJO_CAMERAS
    image_size: int = 256
    #: How the proprioceptive state reaches the VLM.  ``"embedding"`` projects it
    #: through an MLP into the prefix as continuous token embeddings, the way
    #: images already enter.  ``"text"`` renders the numbers as digits, which is
    #: what this model did originally and costs 294 tokens per sample -- 88% of
    #: the non-image prefix -- to express 32 numbers the MLP carries in one.
    state_prompt_mode: str = "embedding"
    #: Number of prefix positions the state embedding occupies. One is enough for
    #: 32 numbers; more only helps if the trunk needs somewhere to put per-group
    #: structure.
    state_prompt_tokens: int = 1
    #: Where the state embedding sits when ``state_prompt_mode == "embedding"``.
    #: ``"append"`` puts it after the closed chat turn, which is what every
    #: existing checkpoint learned -- the prefix is causal, so nothing in the
    #: user turn can attend to it there, including the sentence that names it.
    #: ``"inline"`` splices it into the turn ahead of the closing sentence, which
    #: is the only arrangement where the prompt text can actually condition on
    #: the state, and makes the embedding mode structurally parallel to the text
    #: one.  It changes the layout every checkpoint was trained with, so it is
    #: off by default and wants its own single-variable ablation.
    state_prompt_position: str = "append"
    #: Decimal places when ``state_prompt_mode == "text"``.
    state_prompt_decimals: int = 5
    #: Optional one-line text metadata inserted before the task line.  The two
    #: supported placeholders are rendered from the effective 34-D action/state
    #: mask: ``{action_space}`` is ``joint+EEF`` or ``EEF`` and
    #: ``{joint_dimensions}`` is either a 6/7-DoF description or empty.
    robot_prompt_metadata: str = ""
    #: Probability of hiding the entire assistant subtask turn from the action
    #: expert during training. Applies to conditioning and generation alike;
    #: generation CE remains supervised. Evaluation never applies this dropout.
    subtask_dropout: float = 0.0
    #: Add subtask text CE and decode a subtask at inference when none is supplied.
    #: Both modes place supplied text in the same causal assistant turn after
    #: the observation/state prefix. False consumes that text without text CE
    #: or automatic decoding; True also learns to generate it.
    subtask_generation: bool = False
    #: Decode budget for the subtask turn.  RoboDojo's longest annotation is 27
    #: characters against a median of 13; 48 leaves room for the terminator and
    #: for sources with longer descriptions.
    subtask_max_new_tokens: int = 48
    #: Weight on the subtask text CE.  pi_0.5's Eq. (1) puts alpha=10 on the flow
    #: term against a coefficient of 1 on the cross-entropy;
    #: ``flow_matching_loss_weight`` is 1.0 here, so 0.1 reproduces that 1:10
    #: ratio.  It matters more here than it does there: this recipe is flow-only
    #: (no FAST CE, no VLM co-train branch), so the subtask CE is the only
    #: language-shaped signal the VLM gets.
    subtask_ce_loss_weight: float = 0.1
    #: How many replans reuse one decoded subtask.  pi_0.5 runs its high level
    #: "at a lower frequency than low-level action inference"; with
    #: ``execute_steps: 16`` at 25 fps a replan lands every 0.64 s, which is 5.4
    #: replans inside the median 87-frame segment.  1 regenerates every replan.
    subtask_replan_interval: int = 1
    normalization: str = "quantile"
    train_action_expert_only: bool = True
    # Knowledge insulation (KI): when enabled, the flow expert reads detached
    # VLM states while the optional FAST CE remains the VLM training signal.
    knowledge_insulation: bool = False
    freeze_vision_encoder: bool = True
    flow_matching_loss_weight: float = 1.0
    # Optional temporal weighting inside an action chunk. ``None``/uniform keeps
    # the historical objective; piecewise is useful when the controller replans
    # after a shorter execution horizon than the predicted chunk.
    action_loss_weighting: dict[str, Any] | None = None
    fast_ce_loss_weight: float = 1.0
    # Batch modalities routed to the shared VLM language-model loss.  The base
    # policy remains robotics-only by default for backwards compatibility.
    cotrain_vlm_modalities: tuple[str, ...] = ()
    cotrain_native_modality: str = "robotics"
    cotrain_modality_key: str = "modality"
    cotrain_ignore_index: int = -100
    vlm_loss_weight: float = 1.0
    # FAST+ universal tokenizer (a local directory or a Hugging Face repo id).
    use_fast_action_tokens: bool = False
    fast_tokenizer_path: str | None = None
    fast_max_action_tokens: int = 96
    fast_skip_tokens: int = 128
    device: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.robot_prompt_metadata, str):
            raise TypeError("robot_prompt_metadata must be a string")
        if self.dtype not in {"bfloat16", "float32"}:
            raise ValueError("dtype must be 'bfloat16' or 'float32'")
        if (
            self.action_expert_hidden_size <= 0
            or self.action_expert_intermediate_size <= 0
        ):
            raise ValueError("action expert hidden/intermediate sizes must be positive")
        if self.action_expert_num_layers <= 0 or self.full_attention_interval <= 0:
            raise ValueError(
                "action_expert_num_layers/full_attention_interval must be positive"
            )
        if self.action_expert_num_layers % self.full_attention_interval:
            raise ValueError(
                "action_expert_num_layers must be divisible by full_attention_interval"
            )
        if self.full_attn_num_heads <= 0 or self.full_attn_num_kv_heads <= 0:
            raise ValueError("full-attention head counts must be positive")
        if self.full_attn_num_heads % self.full_attn_num_kv_heads:
            raise ValueError(
                "full_attn_num_heads must be divisible by full_attn_num_kv_heads"
            )
        if self.full_attn_head_dim <= 0:
            raise ValueError("full_attn_head_dim must be positive")
        if self.expert_full_attention_backend not in {
            "sdpa",
            "flash_attention_2",
        }:
            raise ValueError(
                "expert_full_attention_backend must be 'sdpa' or 'flash_attention_2'"
            )
        if self.deltanet_num_key_heads <= 0 or self.deltanet_num_value_heads <= 0:
            raise ValueError("DeltaNet head counts must be positive")
        if self.deltanet_num_value_heads % self.deltanet_num_key_heads:
            raise ValueError(
                "deltanet_num_value_heads must be divisible by deltanet_num_key_heads"
            )
        if self.deltanet_key_head_dim <= 0 or self.deltanet_value_head_dim <= 0:
            raise ValueError("DeltaNet head dimensions must be positive")
        if self.deltanet_conv_kernel <= 0:
            raise ValueError("deltanet_conv_kernel must be positive")
        if self.expert_deltanet_backend not in {
            "torch_parallel",
            "transformers_fla",
        }:
            raise ValueError(
                "expert_deltanet_backend must be 'torch_parallel' or 'transformers_fla'"
            )
        if self.rms_norm_eps <= 0:
            raise ValueError("rms_norm_eps must be positive")
        if self.action_dim <= 0 or self.state_dim <= 0 or self.chunk_size <= 0:
            raise ValueError("action_dim, state_dim, and chunk_size must be positive")
        if self.normalization not in {"quantile", "mean_std", "none"}:
            raise ValueError("normalization must be 'quantile', 'mean_std', or 'none'")
        if self.flow_matching_loss_weight < 0 or self.fast_ce_loss_weight < 0:
            raise ValueError("loss weights must be non-negative")
        validate_action_loss_weighting(
            self.action_loss_weighting, chunk_size=self.chunk_size
        )
        if self.vlm_loss_weight < 0:
            raise ValueError("vlm_loss_weight must be non-negative")
        if self.cotrain_vlm_modalities and self.train_action_expert_only:
            raise ValueError(
                "VLM co-training requires train_action_expert_only=False so the shared "
                "VLM can receive the language-model loss"
            )
        if self.use_fast_action_tokens and not self.fast_tokenizer_path:
            raise ValueError(
                "use_fast_action_tokens=True requires fast_tokenizer_path (a local FAST "
                "directory or a Hugging Face repo id such as physical-intelligence/fast)"
            )
        if self.use_fast_action_tokens and self.train_action_expert_only:
            raise ValueError(
                "FAST CE trains the VLM, so use_fast_action_tokens=True requires "
                "train_action_expert_only=False"
            )
        if (
            self.knowledge_insulation
            and not self.train_action_expert_only
            and not self.use_fast_action_tokens
            and not self.cotrain_vlm_modalities
        ):
            raise ValueError(
                "trainable-VLM knowledge insulation requires FAST action tokens as the "
                "VLM training signal"
            )
        if self.fast_max_action_tokens <= 1:
            raise ValueError("fast_max_action_tokens must be > 1 (room for action EOS)")
        if self.fast_skip_tokens < 0:
            raise ValueError("fast_skip_tokens must be non-negative")
        if not 0.0 <= self.time_min < self.time_max <= 1.0:
            raise ValueError("time_min/time_max must satisfy 0 <= min < max <= 1")
        if self.time_sampling_beta_alpha <= 0 or self.time_sampling_beta_beta <= 0:
            raise ValueError("time-sampling Beta parameters must be positive")
        if not self.camera_keys:
            raise ValueError("camera_keys must contain at least one camera")
        if self.image_size <= 0:
            raise ValueError("image_size must be positive")
        if not 0.0 <= self.subtask_dropout <= 1.0:
            raise ValueError("subtask_dropout must lie in [0, 1]")
        if self.state_prompt_position not in {"append", "inline"}:
            raise ValueError(
                "state_prompt_position must be 'append' or 'inline', got "
                f"{self.state_prompt_position!r}"
            )
        if self.subtask_max_new_tokens <= 0:
            raise ValueError("subtask_max_new_tokens must be positive")
        if self.subtask_ce_loss_weight < 0:
            raise ValueError("subtask_ce_loss_weight must be non-negative")
        if self.subtask_replan_interval < 1:
            raise ValueError("subtask_replan_interval must be at least 1")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MagicW0BaseConfig":
        raw = dict(payload or {})
        known = {field.name for field in fields(cls)}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(
                f"MagicW0BaseConfig: unknown keys {sorted(unknown)}; allowed {sorted(known)}"
            )
        if isinstance(raw.get("camera_keys"), list):
            raw["camera_keys"] = tuple(raw["camera_keys"])
        if isinstance(raw.get("cotrain_vlm_modalities"), list):
            raw["cotrain_vlm_modalities"] = tuple(raw["cotrain_vlm_modalities"])
        return cls(**raw)

    @property
    def vlm_modalities(self) -> frozenset[str]:
        return frozenset(self.cotrain_vlm_modalities)

    def validate_checkpoint_layout(self) -> Path:
        path = Path(self.vlm_checkpoint)
        config_path = path / "config.json"
        if self.local_files_only and not config_path.is_file():
            raise FileNotFoundError(
                f"Qwen3.5 checkpoint is incomplete or missing: {config_path}. "
                "Prepare the Qwen assets/weights following docs/training.md."
            )
        return path


def subtask_replan_due(replan_step: int, interval: int, has_cached: bool) -> bool:
    """Whether stage one should decode a fresh subtask for this env now.

    Lives here, next to ``subtask_replan_interval``, rather than inside the
    inference server: the server imports the simulator at module scope, so a
    rule buried in it cannot be tested without one.
    """
    if not has_cached:
        return True
    return replan_step % interval == 0
