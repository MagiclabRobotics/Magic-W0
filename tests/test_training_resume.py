"""Same-shaped policies can have incompatible history and training semantics."""

from dataclasses import asdict, replace

import pytest
import torch

from magic_w0.models.world_action.configuration_magicw0 import MagicW0TrainingConfig
from magic_w0.training.resume import (
    check_resume_checkpoint,
    validate_resume_model_config,
)


@pytest.mark.parametrize(
    "changes",
    [
        {"history_interval": 10},
        {"flow_matching_loss_weight": 0.7},
        {"knowledge_insulation": False},
        {
            "history_keep_prob": 0.5,
            "history_partial_drop_prob": 0.3,
            "history_full_drop_prob": 0.2,
        },
    ],
)
def test_resume_rejects_same_shape_semantic_changes(changes):
    config = MagicW0TrainingConfig()
    with pytest.raises(ValueError, match="unchanged model training settings"):
        validate_resume_model_config(asdict(config), replace(config, **changes))


@pytest.mark.parametrize("device", ["cpu", "cuda:1"])
def test_resume_allows_relocation_and_execution_settings(tmp_path, device):
    config = MagicW0TrainingConfig()
    saved = asdict(config)
    saved["device"] = "cuda:0"
    saved.pop("expert_deltanet_backend")
    saved.pop("expert_full_attention_backend")
    checkpoint = tmp_path / "checkpoint-step-2.pt"
    torch.save(
        {
            "config": saved,
            "model": {"weight": torch.ones(1)},
            "step": 2,
            "optimizer": {},
        },
        checkpoint,
    )
    requested = replace(
        config,
        vlm_checkpoint="/relocated/qwen",
        track4world_ckpt="/relocated/teacher.pth",
        gradient_checkpointing=False,
        device=device,
    )
    params = asdict(requested)
    params.pop("dropout")
    check_resume_checkpoint(checkpoint, params)


def test_resume_requires_configuration():
    with pytest.raises(ValueError, match="no model config"):
        validate_resume_model_config(None, MagicW0TrainingConfig())


def test_resume_rejects_inference_only_payload(tmp_path):
    checkpoint = tmp_path / "inference.pt"
    torch.save({"config": asdict(MagicW0TrainingConfig()), "model": {}}, checkpoint)
    with pytest.raises(ValueError, match="step and optimizer state"):
        check_resume_checkpoint(checkpoint, {})
