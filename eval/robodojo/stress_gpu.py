"""Stress real checkpoint inference across seeds, histories and active batches."""

import argparse
import itertools
import json
from pathlib import Path
import time
import numpy as np
import torch
import yaml
from .bootstrap import bootstrap
from .paths import REPO_ROOT

CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")


def observation(env, step, robot):
    state = {}
    for index, side in enumerate(("left", "right")):
        state[f"{side}_arm_joint_state"] = np.zeros(robot["arm_dim"][index], np.float32)
        state[f"{side}_ee_joint_state"] = np.full(
            robot["ee_dim"][index], 0.5, np.float32
        )
        state[f"{side}_camera_ee_pose"] = np.array(
            [0.1, 0.2, 0.3, 1, 0, 0, 0], np.float32
        )
    return {
        "env_idx": env,
        "layout_id": env,
        "instruction": "Stack the bowls.",
        "state": state,
        "images": {
            name: np.full((480, 640, 3), 30 + index * 50 + (step + env) % 20, np.uint8)
            for index, name in enumerate(CAMERAS)
        },
    }


def flatten(actions):
    return np.concatenate(
        [
            np.asarray(value).reshape(-1)
            for environment in actions
            for action in environment
            for value in action.values()
        ]
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--robodojo-root", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--vlm-assets", type=Path, required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--replans", type=int, default=28)
    p.add_argument("--output", type=Path, default=REPO_ROOT / "runs/gpu-stress.json")
    args = p.parse_args()
    bootstrap(args.robodojo_root.resolve())
    from .policy import Model
    from magic_w0 import _prepare_image

    config = yaml.safe_load((REPO_ROOT / "configs/robodojo.yaml").read_text())
    config.update(
        robodojo_root=str(args.robodojo_root.resolve()),
        checkpoint_path=str(args.checkpoint.resolve()),
        vlm_checkpoint=str(args.vlm_assets.resolve()),
    )
    model = Model(config)
    if args.replans < model.policy.config.history_frames + 2:
        p.error("replans must extend beyond full history capacity")
    original = model.policy.prepare_inference_batch
    checks = 0

    def checked(batch, device):
        nonlocal checks
        for key in model.policy.config.camera_keys:
            name = key.removeprefix("observation.images.")
            expected = torch.from_numpy(
                np.stack(
                    [
                        model._latest_observations[env]["images"][name]
                        for env in model._latest_env_idx_list
                    ]
                )
            )
            expected = _prepare_image(expected, model.policy.config.image_size)
            if not torch.equal(batch[key], expected):
                raise AssertionError(f"Camera identity mismatch: {key}")
            checks += 1
        return original(batch, device)

    model.policy.prepare_inference_batch = checked
    permutations = list(itertools.permutations(CAMERAS))
    cases = []
    torch.cuda.reset_peak_memory_stats()
    for seed in args.seeds:
        model.seed = seed
        model.reset()
        started = time.monotonic()
        baseline = None
        full_history = False
        for step in range(args.replans):
            model.policy.config.camera_keys = tuple(
                "observation.images." + name
                for name in permutations[step % len(permutations)]
            )
            indices = [0, 7] if step < 16 else [7]
            model.update_obs_batch(
                [observation(env, step, model.robot_action_dim_info) for env in indices]
            )
            if step >= model.policy.config.history_frames:
                full_history = all(
                    not value["history_is_pad"].any()
                    for value in model._latest_observations.values()
                )
                if not full_history:
                    raise AssertionError("History did not fill at expected replan")
            actions = model.get_action_batch(indices)
            values = flatten(actions)
            if (
                len(actions) != len(indices)
                or any(len(chunk) != model.execute_steps for chunk in actions)
                or not np.isfinite(values).all()
            ):
                raise AssertionError("Invalid batched action chunk")
            if step == 0:
                baseline = values.copy()
            print(
                f"seed={seed} replan={step + 1}/{args.replans} active={indices} full_history={full_history}",
                flush=True,
            )
        model.reset()
        model.policy.config.camera_keys = tuple(
            "observation.images." + name for name in CAMERAS
        )
        model.update_obs_batch(
            [observation(env, 0, model.robot_action_dim_info) for env in (0, 7)]
        )
        replay = flatten(model.get_action_batch([0, 7]))
        if not np.allclose(replay, baseline, rtol=1e-3, atol=1e-3):
            raise AssertionError("Reset/seed replay differs from initial action chunk")
        torch.cuda.synchronize()
        cases.append(
            {
                "seed": seed,
                "replans": args.replans,
                "full_history": full_history,
                "reset_max_abs_error": float(np.max(np.abs(replay - baseline))),
                "seconds": time.monotonic() - started,
            }
        )
    result = {
        "status": "passed",
        "observation": "synthetic; real checkpoint and CUDA",
        "checkpoint": str(args.checkpoint),
        "cases": cases,
        "camera_tensor_checks": checks,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "gpu": torch.cuda.get_device_name(0),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
