"""Explicit repository, benchmark and checkpoint paths."""

from __future__ import annotations
import os
import re
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]


def _resolve_path(value: str | Path, *, base: Path = REPO_ROOT) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def resolve_robodojo_root(config: dict[str, Any]) -> Path:
    raw = config.get("robodojo_root") or os.environ.get("ROBODOJO_ROOT")
    if not raw:
        raise ValueError(
            "Set --robodojo-root or ROBODOJO_ROOT to your RoboDojo checkout"
        )
    return _resolve_path(raw)


def resolve_checkpoint(
    config: dict[str, Any], magic_w0_root: Path | None = None
) -> Path:
    base = magic_w0_root or REPO_ROOT
    raw = config.get("checkpoint_path") or config.get("ckpt_name")
    if not raw:
        raw = "checkpoints/magic_w0_robodojo.pt"
    path = _resolve_path(raw, base=base)
    if path.is_file():
        return path
    if path.is_dir():

        def step(file: Path):
            match = re.search(r"step-?(\d+)", file.name)
            return (int(match[1]) if match else -1, file.name)

        files = sorted(
            {
                *path.glob("magic_w0_robodojo*.pt"),
                *path.glob("checkpoint-epoch-*-step-*.pt"),
            },
            key=step,
            reverse=True,
        )
        inference = [file for file in files if "inference" in file.stem]
        if inference or files:
            return (inference or files)[0]
    raise FileNotFoundError(
        f"Checkpoint not found: {path}; set --checkpoint to a .pt file"
    )


def load_robot_action_dim_info(
    env_cfg_type: str, robodojo_root: Path
) -> dict[str, Any]:
    from XPolicyLab.utils import process_data

    actual_root = Path(process_data.__file__).resolve().parents[2]
    if actual_root != robodojo_root.resolve():
        raise RuntimeError(
            f"XPolicyLab imported from {actual_root}, expected {robodojo_root}"
        )
    return dict(process_data.get_robot_action_dim_info(env_cfg_type))
