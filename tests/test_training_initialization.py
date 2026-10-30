"""A public inference checkpoint must initialize a fresh trainable policy."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from magic_w0.models.action_backbone.modeling_magicw0_base import MagicW0BasePolicy


class TinyPolicy(nn.Module):
    def __init__(self, config, *, vlm_state_dict):
        super().__init__()
        self.vlm = nn.Linear(2, 2, bias=False)
        self.vlm.load_state_dict(vlm_state_dict, strict=True)
        self._lm_head = self.vlm
        self.action_expert = nn.Linear(2, 2, bias=False)


def test_inference_initialization_restores_only_known_head_alias(tmp_path):
    path = tmp_path / "policy.pt"
    state = {
        "vlm.weight": torch.ones(2, 2),
        "action_expert.weight": torch.full((2, 2), 2.0),
    }
    torch.save({"model": state}, path)
    load = MagicW0BasePolicy.from_pretrained.__func__
    model = load(TinyPolicy, path, config=SimpleNamespace())
    assert torch.equal(model.action_expert.weight, state["action_expert.weight"])
    assert model._lm_head.weight is model.vlm.weight
    state["_lm_head.weight"] = torch.zeros(2, 2)
    torch.save({"model": state}, path)
    with pytest.raises(ValueError, match="alias conflicts"):
        load(TinyPolicy, path, config=SimpleNamespace())


def test_initialization_still_rejects_missing_policy_weights(tmp_path):
    path = tmp_path / "broken.pt"
    torch.save({"model": {"vlm.weight": torch.ones(2, 2)}}, path)
    with pytest.raises(RuntimeError, match="Missing key"):
        MagicW0BasePolicy.from_pretrained.__func__(
            TinyPolicy, path, config=SimpleNamespace()
        )


def test_magicw0_initialization_forwards_checkpoint_vlm_to_base(tmp_path, monkeypatch):
    from magic_w0.models.world_action import MagicW0TrainingConfig, MagicW0TrainingPolicy

    def tiny_base_init(self, config, *, vlm_state_dict=None):
        nn.Module.__init__(self)
        self.config = config
        self.vlm = nn.Linear(2, 2, bias=False)
        self.vlm.load_state_dict(vlm_state_dict, strict=True)
        self._lm_head = self.vlm
        self.action_expert = nn.Linear(2, 2, bias=False)

    # Keep the real MagicW0 constructor and inherited checkpoint-loading path.
    # Only replace the expensive VLM allocation with a small trainable module.
    monkeypatch.setattr(MagicW0BasePolicy, "__init__", tiny_base_init)
    monkeypatch.setenv("RANK", "1")
    state = {
        "vlm.weight": torch.ones(2, 2),
        "action_expert.weight": torch.full((2, 2), 2.0),
    }
    path = tmp_path / "pretrain.pt"
    torch.save({"model": state}, path)
    cfg = MagicW0TrainingConfig(
        enable_2d_world_expert=False, enable_3d_world_expert=False,
        enable_world_teacher_supervision=False, use_fast_action_tokens=False,
    )
    model = MagicW0TrainingPolicy.from_pretrained(path, config=cfg)
    assert isinstance(model, MagicW0TrainingPolicy)
    assert model._training_global_step == 0
    for key, expected in state.items():
        torch.testing.assert_close(model.state_dict()[key], expected)
    assert all(parameter.requires_grad for parameter in model.parameters())
