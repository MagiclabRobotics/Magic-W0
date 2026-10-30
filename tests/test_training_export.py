"""Training exports must satisfy the unchanged public inference graph."""

from dataclasses import asdict

import pytest
import torch
from torch import nn

from magic_w0.checkpoint import inference_payload
from magic_w0.config import MagicW0Config
from magic_w0.modules import (
    HybridActionExpert,
    Hybrid2DWorldExpert,
    Hybrid3DWorldExpert,
)


@pytest.fixture
def payload():
    config = MagicW0Config(
        action_expert_hidden_size=32,
        action_expert_intermediate_size=64,
        action_expert_num_layers=4,
        full_attn_num_heads=2,
        full_attn_num_kv_heads=1,
        full_attn_head_dim=16,
        deltanet_num_key_heads=2,
        deltanet_num_value_heads=2,
        deltanet_key_head_dim=16,
        deltanet_value_head_dim=16,
        dtype="float32",
    )
    modules = {
        "action_expert": HybridActionExpert(config),
        "world_2d_expert": Hybrid2DWorldExpert(
            config, config.dinov3_target_hidden_size
        ),
        "world_3d_expert": Hybrid3DWorldExpert(config),
        "state_prompt_projection": nn.Sequential(
            nn.Linear(68, 32), nn.SiLU(), nn.Linear(32, 32)
        ),
    }
    state = {
        f"{prefix}.{key}": value
        for prefix, module in modules.items()
        for key, value in module.state_dict().items()
    }
    state["vlm.model.language_model.embed_tokens.weight"] = torch.ones(32, 32)
    return {
        "model": state,
        "config": asdict(config),
        "normalization": {"robot": {}},
        "optimizer": {"unused": True},
        "step": 2,
    }


def test_export_retains_weights_and_normalization_and_omits_training_state(payload):
    payload["config"].update(
        expert_deltanet_backend="transformers_fla",
        expert_full_attention_backend="flash_attention_2",
    )
    payload["model"]["dinov3_teacher.weight"] = torch.ones(1)
    result = inference_payload(payload)
    assert "optimizer" not in result
    assert result["normalization"] == payload["normalization"]
    assert result["config"]["expert_deltanet_backend"] == "torch_parallel"
    assert "dinov3_teacher.weight" not in result["model"]
    assert all(value is payload["model"][key] for key, value in result["model"].items())


@pytest.mark.parametrize(
    "mode",
    ["missing_norm", "four_stream", "wrong_shape", "missing_tensor", "history_off"],
)
def test_export_rejects_incompatible_or_incomplete_checkpoint(payload, mode):
    if mode == "missing_norm":
        payload.pop("normalization")
    if mode == "four_stream":
        payload["model"]["geometry_expert.weight"] = torch.ones(1)
    if mode == "wrong_shape":
        payload["model"]["state_prompt_projection.0.weight"] = torch.ones(32, 67)
    if mode == "missing_tensor":
        payload["model"].pop("world_2d_expert.query_tokens")
    if mode == "history_off":
        payload["config"]["history_enabled"] = False
    with pytest.raises(ValueError):
        inference_payload(payload)
