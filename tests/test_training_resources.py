"""Missing user-provided resources must fail before a model is allocated."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from magic_w0.training.resources import check_resources


def recipe(tmp_path):
    qwen = tmp_path / "qwen"
    qwen.mkdir()
    (qwen / "config.json").write_text("{}")
    init = tmp_path / "policy.pt"
    init.write_bytes(b"placeholder for path-only validation")
    return {
        "model": {
            "name": "magic_w0",
            "checkpoint": str(init),
            "params": {
                "vlm_checkpoint": str(qwen),
                "use_fast_action_tokens": False,
                "enable_world_teacher_supervision": False,
            },
        }
    }


def test_init_requires_policy_file_and_pretraining_requires_base_weights(tmp_path):
    payload = recipe(tmp_path)
    check_resources(payload, verbose=False)
    Path(payload["model"]["checkpoint"]).unlink()
    with pytest.raises(FileNotFoundError, match="initial policy checkpoint"):
        check_resources(payload, verbose=False)
    payload["model"].pop("checkpoint")
    with pytest.raises(FileNotFoundError, match="base weights"):
        check_resources(payload, verbose=False)


def test_fast_directory_alone_is_not_a_complete_resource(tmp_path):
    payload = recipe(tmp_path)
    fast = tmp_path / "fast"
    fast.mkdir()
    payload["model"]["params"].update(
        use_fast_action_tokens=True, fast_tokenizer_path=str(fast)
    )
    with pytest.raises(FileNotFoundError, match="FAST processor config"):
        check_resources(payload, verbose=False)


def test_teacher_path_check_is_explicit_and_does_not_download(tmp_path, monkeypatch):
    payload = recipe(tmp_path)
    teacher = tmp_path / "teacher"
    teacher.mkdir()
    params = payload["model"]["params"]
    params.update(
        enable_world_teacher_supervision=True,
        track4world_project_root=str(teacher),
        dinov3_checkpoint=str(tmp_path / "dino"),
        track4world_ckpt=str(tmp_path / "teacher.pt"),
        track4world_da3_backbone_path=str(tmp_path / "da3"),
        track4world_convnext_tiny_weights_path=str(tmp_path / "convnext.pt"),
    )
    with pytest.raises(FileNotFoundError, match="DINOv3 directory"):
        check_resources(payload, verbose=False)
    for key in ["dinov3_checkpoint", "track4world_da3_backbone_path"]:
        Path(params[key]).mkdir()
    for key in ["track4world_ckpt", "track4world_convnext_tiny_weights_path"]:
        Path(params[key]).write_bytes(b"path-only placeholder")
    config = teacher / "track4world/config/eval/v1.json"
    config.parent.mkdir(parents=True)
    config.write_text("{}")

    class Unpatched:
        def infer_pair(self, images):
            pass

    import importlib

    original = importlib.import_module
    monkeypatch.setattr(
        "magic_w0.training.resources.importlib.import_module",
        lambda name: (
            SimpleNamespace(Track4World=Unpatched)
            if name == "track4world.nets.model"
            else original(name)
        ),
    )
    with pytest.raises(ValueError, match="feature patch"):
        check_resources(payload, check_imports=True, verbose=False)
