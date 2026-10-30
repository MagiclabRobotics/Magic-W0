"""Strict decomposition of policy checkpoint tensors."""

from __future__ import annotations
from typing import Any


def inference_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Export an MagicW0 training payload for the existing inference loader.

    Validate the public graph before dropping training-only configuration.
    This prevents accidental export of the incompatible four-stream model.
    """
    from dataclasses import fields
    from .config import MagicW0Config, SUPPORTED_MODES, validate_checkpoint_modes

    raw_config = payload.get("config")
    if not isinstance(raw_config, dict):
        raise ValueError("Checkpoint must contain a model config mapping")
    config = dict(raw_config)
    if config.get("normalization", "quantile") != "quantile":
        raise ValueError("The public runtime requires quantile normalization")
    # Kernel selection changes execution, not tensor layout. The public runtime
    # executes SDPA / torch DeltaNet even when training used accelerated kernels.
    config.update(
        expert_full_attention_backend="sdpa",
        expert_deltanet_backend="torch_parallel",
        compile_model=False,
    )
    validate_checkpoint_modes(config)
    state = payload.get("model")
    if not isinstance(state, dict):
        raise ValueError("Checkpoint must contain a model state dictionary")
    state = {key.removeprefix("module."): value for key, value in state.items()}
    allowed = (
        "vlm.",
        "action_expert.",
        "world_2d_expert.",
        "world_3d_expert.",
        "state_prompt_projection.",
    )
    training_only = (
        "_lm_head.",
        "dinov3_teacher.",
        "track4world_teacher.",
        "_track4world_teacher.",
    )
    unknown = [key for key in state if not key.startswith((*allowed, *training_only))]
    if unknown:
        raise ValueError(f"Unsupported model tensors (MagicW0 required): {unknown[:10]}")
    state = {key: value for key, value in state.items() if key.startswith(allowed)}
    split_policy_checkpoint_state(state)
    for prefix in allowed[1:]:
        if not any(key.startswith(prefix) for key in state):
            raise ValueError(f"Missing required inference branch: {prefix}")
    normalization = payload.get("normalization")
    if not isinstance(normalization, dict) or not normalization:
        raise ValueError("Inference export requires checkpoint normalization metadata")
    if "normalization" in raw_config:
        # Training clamps quantile scales to 1e-6, including constant observed
        # dimensions. Record that exact transform, without changing the bounds.
        normalization = {
            name: {**entry, "quantile_scale_epsilon": 1e-6}
            for name, entry in normalization.items()
        }
    keys = (
        {field.name for field in fields(MagicW0Config)}
        | set(SUPPORTED_MODES)
        | {"joint_attention_visibility"}
    )
    config = {key: value for key, value in config.items() if key in keys}
    runtime_config = MagicW0Config.from_dict(
        {
            key: value
            for key, value in config.items()
            if key in {f.name for f in fields(MagicW0Config)}
        }
    )
    validate_policy_shapes(state, runtime_config)
    result = {"model": state, "config": config, "normalization": normalization}
    if "step" in payload:
        result["step"] = payload["step"]
    return result


def validate_policy_shapes(state: dict[str, Any], config) -> None:
    """Check every non-VLM tensor against the public runtime without allocating weights."""
    import torch
    from torch import nn
    from .modules import HybridActionExpert, Hybrid2DWorldExpert, Hybrid3DWorldExpert

    embeddings = [
        value
        for key, value in state.items()
        if key.startswith("vlm.") and key.endswith("embed_tokens.weight")
    ]
    if len(embeddings) != 1 or embeddings[0].ndim != 2:
        raise ValueError("Expected one VLM token embedding matrix")
    hidden = embeddings[0].shape[1]
    with torch.device("meta"):
        modules = {
            "action_expert": HybridActionExpert(config),
            "world_2d_expert": Hybrid2DWorldExpert(
                config, config.dinov3_target_hidden_size
            ),
            "world_3d_expert": Hybrid3DWorldExpert(config),
            "state_prompt_projection": nn.Sequential(
                nn.Linear(config.state_dim * 2, hidden),
                nn.SiLU(),
                nn.Linear(hidden, hidden * config.state_prompt_tokens),
            ),
        }
    expected = {
        f"{prefix}.{key}": tuple(value.shape)
        for prefix, module in modules.items()
        for key, value in module.state_dict().items()
    }
    actual = {
        key: tuple(value.shape)
        for key, value in state.items()
        if not key.startswith("vlm.")
    }
    mismatches = [
        key for key in expected.keys() & actual.keys() if expected[key] != actual[key]
    ]
    if expected.keys() != actual.keys() or mismatches:
        raise ValueError(
            f"Inference tensor layout mismatch: missing={sorted(expected.keys() - actual.keys())[:8]}, unexpected={sorted(actual.keys() - expected.keys())[:8]}, shape={mismatches[:8]}"
        )


def split_policy_checkpoint_state(
    state: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split a full policy checkpoint into HF VLM and policy-owned tensors."""
    normalized = {key.removeprefix("module."): value for key, value in state.items()}
    vlm_state = {
        key.removeprefix("vlm."): value
        for key, value in normalized.items()
        if key.startswith("vlm.")
    }
    if not vlm_state:
        raise ValueError("checkpoint does not contain the complete vlm.* state")
    policy_state = {
        key: value
        for key, value in normalized.items()
        if not key.startswith(("vlm.", "_lm_head."))
    }
    return vlm_state, policy_state
