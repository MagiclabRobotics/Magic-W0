"""Reject silent training-semantic changes when resuming an MagicW0 run."""

from dataclasses import asdict

from magic_w0.models.world_action.configuration_magicw0 import MagicW0TrainingConfig


# Relocation and execution choices may change without changing the training recipe.
_EXECUTION_FIELDS = frozenset(
    {
        "vlm_checkpoint",
        "local_files_only",
        "trust_remote_code",
        "dtype",
        "device",
        "attention_implementation",
        "expert_full_attention_backend",
        "expert_deltanet_backend",
        "gradient_checkpointing",
        "compile_model",
        "compile_mode",
        "compile_dynamic",
        "action_expert_efficient_sdpa",
        "fast_tokenizer_path",
        "dinov3_checkpoint",
        "dinov3_local_files_only",
        "track4world_project_root",
        "track4world_ckpt",
        "track4world_config_path",
        "track4world_da3_backbone_path",
        "track4world_local_files_only",
        "track4world_convnext_tiny_weights_path",
        "track4world_motion_chunk_size",
        "track4world_inference_backend",
        "track4world_tensorrt_engine_dir",
        "track4world_tensorrt_keep_execution_memory",
        "track4world_tensorrt_expected_engine_batch_sizes",
        "track4world_autocast_dtype",
    }
)


def validate_resume_model_config(
    saved: dict | None, current: MagicW0TrainingConfig
) -> None:
    """Compare canonical configuration, including defaults and nested settings.

    File relocation is allowed; callers must still supply the same assets.
    Changing the learning task requires initialization, rather than resume.
    """
    if not isinstance(saved, dict):
        raise ValueError("resume checkpoint has no model config; use --init-checkpoint")
    raw = dict(saved)
    # Serialized dataclasses contain Base's unused zero-dropout field; MagicW0 YAML
    # deliberately rejects it in favor of four explicit dropout settings.
    if raw.get("dropout") == 0:
        raw.pop("dropout")
    previous = asdict(MagicW0TrainingConfig.from_dict(raw))
    requested = asdict(current)
    changed = sorted(
        key
        for key in requested
        if key not in _EXECUTION_FIELDS and previous[key] != requested[key]
    )
    if changed:
        raise ValueError(
            "resume requires unchanged model training settings; changed: "
            + ", ".join(changed)
            + ". Use --init-checkpoint for a new training stage."
        )


def check_resume_checkpoint(path, params: dict) -> None:
    """Check metadata before allocating CUDA or constructing a production model."""
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if (
        not isinstance(payload.get("step"), int)
        or isinstance(payload.get("step"), bool)
        or payload["step"] < 0
        or not (
            isinstance(payload.get("optimizer"), dict)
            or isinstance(payload.get("optimizer_file"), str)
            and bool(payload["optimizer_file"])
        )
    ):
        raise ValueError(
            "resume requires a training checkpoint with step and optimizer state; "
            "use --init-checkpoint for inference weights"
        )
    validate_resume_model_config(
        payload.get("config"), MagicW0TrainingConfig.from_dict(params)
    )
