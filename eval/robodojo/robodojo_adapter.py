"""Translate Magic_W0 camera EEF data at the official RoboDojo deployment boundary."""

from __future__ import annotations

import numpy as np

from XPolicyLab.utils.camera_extrinsics import _as_numpy, camera_to_world_matrix

from XPolicyLab.utils.pose_transform import (
    camera_pose_to_world_pose,
    world_pose_to_camera_pose,
)


def head_camera_to_env(task_env, env_idx):
    """Read the actual ROS optical pose, expressed relative to this environment."""
    manager = task_env.camera_manager
    names = manager.camera_names[env_idx]
    aliases = ("cam_head", "cam_high", "head_camera", "top_camera")
    camera_id = next((names.index(name) for name in aliases if name in names), None)
    if camera_id is None:
        raise ValueError(
            f"Magic_W0 camera EEF requires a head camera; available={names}"
        )
    transform = camera_to_world_matrix(manager.cameras[env_idx][camera_id])
    transform[:3, 3] -= _as_numpy(task_env.env_origins[env_idx])
    return transform.astype(np.float32)


def prepare_observation(task_env, observation, *, camera_to_env=None):
    """Add camera EEF state to a copy, leaving official observations untouched.

    Simulation observations contain environment-local EEF poses. Other clients
    may already supply camera EEF state and do not expose a simulation camera.
    """
    if not hasattr(task_env, "camera_manager"):
        return observation
    state = dict(observation["state"])
    pose_keys = [
        key for key in ("ee_pose", "left_ee_pose", "right_ee_pose") if key in state
    ]
    if pose_keys:
        transform = camera_to_env
        if transform is None:
            transform = head_camera_to_env(task_env, observation.get("env_idx", 0))
        for key in pose_keys:
            camera_key = key.removesuffix("ee_pose") + "camera_ee_pose"
            state[camera_key] = world_pose_to_camera_pose(state[key], transform)
    return {**observation, "state": state}


def prepare_action(task_env, action, env_idx=0, *, camera_to_env=None):
    """Convert targets using the same frozen extrinsic as the planning observation."""
    pose_keys = [
        key
        for key in ("camera_ee_pose", "left_camera_ee_pose", "right_camera_ee_pose")
        if key in action
    ]
    if not pose_keys:
        return action
    if camera_to_env is None:
        raise ValueError(
            "Camera EEF actions require the planning camera_to_env transform"
        )
    transform = camera_to_env
    converted = dict(action)
    for key in pose_keys:
        target_key = key.replace("camera_ee_pose", "ee_pose")
        if target_key in converted:
            raise ValueError(
                f"Conflicting camera and environment EEF targets: {key}, {target_key}"
            )
        converted[target_key] = camera_pose_to_world_pose(converted.pop(key), transform)
    return converted
