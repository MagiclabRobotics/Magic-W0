"""Run one real checkpoint inference with a synthetic observation on CUDA."""

import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
import yaml
from .bootstrap import bootstrap
from .paths import REPO_ROOT


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--robodojo-root", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--vlm-assets", type=Path, required=True)
    p.add_argument("--output", type=Path, default=REPO_ROOT / "runs/gpu-smoke.json")
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    bootstrap(args.robodojo_root.resolve())
    from .policy import Model

    config = yaml.safe_load((REPO_ROOT / "configs/robodojo.yaml").read_text())
    config.update(
        robodojo_root=str(args.robodojo_root.resolve()),
        checkpoint_path=str(args.checkpoint.resolve()),
        vlm_checkpoint=str(args.vlm_assets.resolve()),
        task_name="stack_bowls",
    )
    start = time.monotonic()
    model = Model(config)
    torch.cuda.synchronize()
    load_seconds = time.monotonic() - start
    robot = model.robot_action_dim_info
    observation = {
        "env_idx": 0,
        "instruction": "Stack the bowls.",
        "images": {
            key: np.full((256, 256, 3), 128, dtype=np.uint8)
            for key in ("cam_high", "cam_left_wrist", "cam_right_wrist")
        },
        "state": {},
    }
    for index, side in enumerate(("left", "right")):
        observation["state"][f"{side}_arm_joint_state"] = np.zeros(
            robot["arm_dim"][index], dtype=np.float32
        )
        observation["state"][f"{side}_ee_joint_state"] = np.full(
            robot["ee_dim"][index], 0.5, dtype=np.float32
        )
        observation["state"][f"{side}_camera_ee_pose"] = np.array(
            [0.1, 0.2, 0.3, 1, 0, 0, 0], dtype=np.float32
        )
    model.update_obs(observation)
    torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    actions = model.get_action()
    torch.cuda.synchronize()
    inference_seconds = time.monotonic() - start
    if len(actions) != config["execute_steps"] or not actions:
        raise RuntimeError(
            f"Expected {config['execute_steps']} actions, got {len(actions)}"
        )
    for action in actions:
        for key, value in action.items():
            if not np.isfinite(value).all():
                raise RuntimeError(f"Nonfinite predicted action: {key}")
    result = {
        "status": "passed",
        "observation": "synthetic; actual checkpoint and CUDA model",
        "checkpoint": str(args.checkpoint.resolve()),
        "gpu": torch.cuda.get_device_name(0),
        "load_seconds": load_seconds,
        "inference_seconds": inference_seconds,
        "action_count": len(actions),
        "action_shapes": {
            key: list(np.asarray(value).shape) for key, value in actions[0].items()
        },
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
