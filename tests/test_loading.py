"""Checkpoint metadata must not change inference configuration or bypass checks."""

from types import ModuleType, SimpleNamespace
import sys
import pytest
import magic_w0
from eval.robodojo.loading import CheckpointLoaderMixin


@pytest.fixture
def loader(monkeypatch, tmp_path):
    payload = {
        "config": {
            "history_enabled": True,
            "history_frames": 25,
            "history_interval": 20,
            "history_pool_size": 4,
        },
        "model": {"vlm.stub": object(), "action.stub": object()},
    }
    torch = ModuleType("torch")
    torch.load = lambda *args, **kwargs: payload
    torch.device = lambda value: SimpleNamespace(type=value)
    torch.manual_seed = lambda value: None
    monkeypatch.setitem(sys.modules, "torch", torch)

    class Policy:
        def __init__(self, config, *, vlm_state_dict):
            self.config = config
            assert list(vlm_state_dict) == ["stub"]

        def load_state_dict(self, state, strict=False):
            assert list(state) == ["action.stub"]
            return SimpleNamespace(missing_keys=[], unexpected_keys=[])

        def to(self, device):
            assert device.type == "cpu"
            return self

        def eval(self):
            return self

    monkeypatch.setitem(magic_w0.__dict__, "MagicW0", Policy)
    adapter = CheckpointLoaderMixin()
    adapter.checkpoint = tmp_path / "policy.pt"
    adapter.magic_w0_root = tmp_path
    adapter.action_type = "joint"
    adapter.execute_steps = 20
    adapter.num_inference_steps = 10
    adapter.robot_action_dim = 34
    adapter.seed = 0
    return adapter, payload, {"device": "cpu"}


def test_auxiliary_checkpoint_metadata_does_not_change_inference(loader):
    adapter, payload, cfg = loader
    baseline = adapter._load_policy(cfg).config
    payload["config"].update(
        auxiliary_metadata={"note": "not a runtime setting"},
        history_export_annotation=["ignored metadata"],
    )
    loaded = adapter._load_policy(cfg).config
    assert loaded == baseline
    assert loaded.history_frames == 25
    assert loaded.history_interval == 20


def test_unknown_runtime_override_still_fails(loader):
    adapter, _, cfg = loader
    cfg["model_params"] = {"history_export_annotation": True}
    with pytest.raises(ValueError, match="unsupported inference overrides"):
        adapter._load_policy(cfg)


def test_checkpoint_history_override_still_must_match(loader):
    adapter, _, cfg = loader
    cfg["model_params"] = {"history_frames": 2}
    with pytest.raises(ValueError, match="history override conflicts"):
        adapter._load_policy(cfg)


def test_checkpoint_modes_still_validated(loader):
    adapter, payload, cfg = loader
    payload["config"]["history_enabled"] = False
    with pytest.raises(ValueError, match="unsupported history_enabled"):
        adapter._load_policy(cfg)


def test_checkpoint_history_schedule_still_validated(loader):
    adapter, payload, cfg = loader
    payload["config"]["history_interval"] = 30
    with pytest.raises(ValueError, match="divisible"):
        adapter._load_policy(cfg)
