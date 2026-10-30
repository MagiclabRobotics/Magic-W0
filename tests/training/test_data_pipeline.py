"""Exercise the public commands on actual LeRobot parquet/video files."""

import json
import os
from pathlib import Path
import subprocess
import sys

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

ROOT = Path(__file__).resolve().parents[2]


def create_dataset(root):
    frames = 64
    cameras = [
        "observation.images.cam_high",
        "observation.images.cam_left_wrist",
        "observation.images.cam_right_wrist",
    ]
    values = np.zeros((frames, 32), dtype=np.float32)
    values[:, 0] = np.linspace(0, 1, frames)
    for start in (17, 26):
        values[:, start : start + 6] = [1, 0, 0, 0, 1, 0]
    meta = root / "meta"
    meta.mkdir(parents=True)
    features = {
        "observation.state": {"dtype": "float32", "shape": [32]},
        "action": {"dtype": "float32", "shape": [32]},
    }
    features.update(
        {key: {"dtype": "bool", "shape": [32]} for key in ("state_mask", "action_mask")}
    )
    features.update({key: {"dtype": "video", "shape": [64, 64, 3]} for key in cameras})
    info = {
        "codebase_version": "v2.1",
        "fps": 25,
        "chunks_size": 1000,
        "total_episodes": 1,
        "total_frames": frames,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
    }
    (meta / "info.json").write_text(json.dumps(info))
    (meta / "episodes.jsonl").write_text(
        json.dumps({"episode_index": 0, "length": frames, "tasks": ["move the block"]})
        + "\n"
    )
    (meta / "tasks.jsonl").write_text(
        json.dumps({"task_index": 0, "task": "move the block"}) + "\n"
    )
    stats = {
        "mean": values.mean(0).tolist(),
        "std": values.std(0).tolist(),
        "min": values.min(0).tolist(),
        "max": values.max(0).tolist(),
        "count": [frames],
    }
    (meta / "episodes_stats.jsonl").write_text(
        json.dumps(
            {"episode_index": 0, "stats": {"observation.state": stats, "action": stats}}
        )
        + "\n"
    )
    columns = {
        "observation.state": values.tolist(),
        "action": values.tolist(),
        "state_mask": np.ones_like(values, dtype=bool).tolist(),
        "action_mask": np.ones_like(values, dtype=bool).tolist(),
        "episode_index": [0] * frames,
        "task_index": [0] * frames,
        "frame_index": list(range(frames)),
        "index": list(range(frames)),
        "timestamp": (np.arange(frames) / 25).tolist(),
    }
    data = root / "data/chunk-000"
    data.mkdir(parents=True)
    pq.write_table(pa.table(columns), data / "episode_000000.parquet")
    for camera in cameras:
        path = root / "videos/chunk-000" / camera / "episode_000000.mp4"
        path.parent.mkdir(parents=True)
        with av.open(str(path), "w") as container:
            stream = container.add_stream("mpeg4", rate=25)
            stream.width = stream.height = 64
            stream.pix_fmt = "yuv420p"
            for index in range(frames):
                frame = av.VideoFrame.from_ndarray(
                    np.full((64, 64, 3), index * 3, np.uint8), format="rgb24"
                )
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)


def test_normalization_and_history_data_commands(tmp_path, monkeypatch):
    root = tmp_path / "data/robodojo"
    create_dataset(root)
    norm = tmp_path / "cache/norm/robodojo.json"
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text(
        yaml.safe_dump({"include": str(ROOT / "configs/train/robodojo_sft.yaml")})
    )
    env = {
        **os.environ,
        "MAGIC_W0_DATA_ROOT": str(tmp_path / "data"),
        "MAGIC_W0_CACHE_ROOT": str(tmp_path / "cache"),
        "PYTHONPATH": str(ROOT / "src"),
    }

    def run(*args):
        result = subprocess.run(
            [sys.executable, *args], cwd=ROOT, env=env, text=True, capture_output=True
        )
        assert result.returncode == 0, result.stdout + result.stderr
        return result.stdout

    run(
        "scripts/compute_norm_stats.py",
        "--config",
        str(recipe),
        "--dataset-index",
        "0",
        "--output",
        str(norm),
    )
    assert norm.is_file()
    output = run("scripts/check_data.py", "--config", str(recipe), "--samples", "2")
    assert '"sample": 1' in output
    assert "history" in output
    # Verify the actual generated statistics also work at the deployment
    # boundary, including constant/padded rotation and joint dimensions.
    from types import SimpleNamespace
    from magic_w0.training.config import load_recipe
    from magic_w0.data.config import source_config_from_dict
    from magic_w0.data.backends import get_data_backend
    from magic_w0.data.norm_export import collect_norm_metadata
    from magic_w0.models._normalization import normalize_tensor
    from eval.robodojo.normalization import DeltaNormalizationStats

    for name in ("MAGIC_W0_DATA_ROOT", "MAGIC_W0_CACHE_ROOT"):
        monkeypatch.setenv(name, env[name])
    config = load_recipe(recipe)
    source = source_config_from_dict(config["data"]["datasets"][0])
    source.lerobot_version = "v2.1"
    dataset = get_data_backend("lerobot").make_dataset(source)
    entry = next(iter(collect_norm_metadata(SimpleNamespace(dataset=dataset)).values()))
    runtime = DeltaNormalizationStats.from_entry(
        {**entry, "quantile_scale_epsilon": 1e-6}
    )
    state = dataset[0]["observation.state"]
    np.testing.assert_allclose(
        runtime.normalize_state(state.numpy()),
        normalize_tensor(
            state, dataset.meta_stats["observation.state"], "quantile"
        ).numpy(),
    )
