"""Persist the effective recipe and software versions independently of W&B."""

from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess

import yaml


def write_run_metadata(config: dict, *, rank: int) -> None:
    output = config.get("training", {}).get("output_dir")
    if rank != 0 or not output:
        return
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    suffix = f"-{stamp}" if (output / "resolved_config.yaml").exists() else ""
    (output / f"resolved_config{suffix}.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False)
    )
    versions = {}
    for name in ("torch", "transformers", "numpy", "magic-w0"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        revision = None
        dirty = None
    (output / f"run_info{suffix}.json").write_text(
        json.dumps(
            {
                "created_utc": stamp,
                "python": platform.python_version(),
                "versions": versions,
                "git_revision": revision,
                "git_dirty": dirty,
            },
            indent=2,
        )
        + "\n"
    )
