"""Portable recipes must preserve model contracts and strict resume semantics."""

from argparse import Namespace

import pytest

from magic_w0.training.config import load_recipe
from magic_w0.training.cli import _apply_train_cli_overrides


def test_include_override_environment_and_no_source_mutation(tmp_path, monkeypatch):
    base = tmp_path / "base.yaml"
    base.write_text(
        'model: {name: magic_w0, params: {action_dim: 34, chunk_size: 50}}\ndata: {root: "${TEST_DATA_ROOT:-data}"}\n'
    )
    child = tmp_path / "child.yaml"
    child.write_text("include: base.yaml\nmodel: {params: {chunk_size: 25}}\n")
    monkeypatch.setenv("TEST_DATA_ROOT", "/external/data")
    recipe = load_recipe(child)
    assert recipe["model"]["params"] == {"action_dim": 34, "chunk_size": 25}
    assert recipe["data"]["root"] == "/external/data"
    assert load_recipe(base)["model"]["params"]["chunk_size"] == 50


def test_include_cycle_and_missing_environment_are_errors(tmp_path, monkeypatch):
    a = tmp_path / "a.yaml"
    b = tmp_path / "b.yaml"
    a.write_text("include: b.yaml\n")
    b.write_text("include: a.yaml\n")
    with pytest.raises(ValueError, match="Cyclic"):
        load_recipe(a)
    monkeypatch.delenv("ABSENT_TRAINING_TEST_ROOT", raising=False)
    a.write_text('path: "${ABSENT_TRAINING_TEST_ROOT}"\n')
    with pytest.raises(ValueError, match="unset"):
        load_recipe(a)


def test_resume_never_silently_starts_fresh(tmp_path):
    args = Namespace(
        output_dir=None, resume=tmp_path, max_steps=None, skip_checkpoints=False
    )
    with pytest.raises(FileNotFoundError, match="no checkpoint"):
        _apply_train_cli_overrides({}, args)
    (tmp_path / "checkpoint-step-9.pt").touch()
    (tmp_path / "checkpoint-step-100.pt").touch()
    config = {}
    _apply_train_cli_overrides(config, args)
    assert config["resume_from"].endswith("checkpoint-step-100.pt")
