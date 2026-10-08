"""Strict decomposition of policy checkpoint tensors."""

from __future__ import annotations
from typing import Any


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
