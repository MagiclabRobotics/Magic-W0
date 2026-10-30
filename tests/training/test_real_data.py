"""Optional real-data check; no dataset is bundled or downloaded by the test."""

import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from magic_w0.training.config import load_recipe


@pytest.mark.skipif(
    not os.environ.get("MAGIC_W0_REAL_DATA"),
    reason="Set MAGIC_W0_REAL_DATA to a RoboDojo 32D union LeRobot v2.1 export",
)
def test_real_robodojo_normalization_and_history(tmp_path):
    root = Path(os.environ["MAGIC_W0_REAL_DATA"]).resolve()
    project = Path(__file__).resolve().parents[2]
    config = load_recipe(project / "configs/train/robodojo_sft.yaml")
    source = copy.deepcopy(config["data"]["datasets"][0])
    source["root"] = str(root)
    with (root / "meta/episodes.jsonl").open() as stream:
        episode = json.loads(next(stream))["episode_index"]
    source["episodes"] = [episode]
    norm = tmp_path / "norm.json"
    source["params"].update(norm_stats_path=str(norm), action_stats_samples=64)
    config["data"]["datasets"] = [source]
    recipe = tmp_path / "real_data.yaml"
    recipe.write_text(yaml.safe_dump(config))
    for args in [
        [
            "scripts/compute_norm_stats.py",
            "--config",
            str(recipe),
            "--dataset-index",
            "0",
            "--output",
            str(norm),
        ],
        ["scripts/check_data.py", "--config", str(recipe), "--samples", "2"],
    ]:
        result = subprocess.run(
            [sys.executable, *args], cwd=project, text=True, capture_output=True
        )
        assert result.returncode == 0, result.stdout + result.stderr
