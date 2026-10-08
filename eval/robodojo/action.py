"""RoboDojo inference adapter components."""

from __future__ import annotations
import hashlib
from typing import Any
import numpy as np
from .rotation6d import rot6d_to_matrix
from .layout import camera_eef_layout_spec
from .geometry import _matrix_to_quat_wxyz
from .observation import _state_keys


def unpack_camera_eef_actions(
    actions: np.ndarray, *, layout: str = "compact20", action_type: str = "camera_ee"
) -> list[dict[str, np.ndarray]]:
    """Convert absolute EEF targets to the selected execution action schema.

    The checkpoint layout is independent from the execution schema: both
    ``ee`` and ``camera_ee`` can use the same reconstructed EEF chunk, while
    the XPolicyLab deployment adapter converts camera targets for RoboDojo.
    """
    if action_type not in ("ee", "camera_ee"):
        raise ValueError(
            f"EEF action unpacking requires action_type='ee' or 'camera_ee', got {action_type!r}"
        )
    values = np.asarray(actions, dtype=np.float32)
    spec = camera_eef_layout_spec(layout)
    if values.ndim != 2 or values.shape[-1] != spec["dim"]:
        raise ValueError(
            f"expected camera EEF action chunk [T,{spec['dim']}], got {values.shape}"
        )
    result = []
    for step in values:
        item = {
            "left_ee_joint_state": step[
                spec["gripper_indices"][0] : spec["gripper_indices"][0] + 1
            ].copy(),
            "right_ee_joint_state": step[
                spec["gripper_indices"][1] : spec["gripper_indices"][1] + 1
            ].copy(),
        }
        pose_key = "camera_ee_pose" if action_type == "camera_ee" else "ee_pose"
        for prefix, start in zip(("left", "right"), spec["pose_starts"], strict=True):
            rotation = rot6d_to_matrix(step[start + 3 : start + 9])
            item[f"{prefix}_{pose_key}"] = np.concatenate(
                (step[start : start + 3], _matrix_to_quat_wxyz(rotation))
            ).astype(np.float32)
        result.append(item)
    return result


def unpack_joint_actions(
    actions: np.ndarray, robot_info: dict[str, Any]
) -> list[dict[str, np.ndarray]]:
    arm_dims = list(robot_info["arm_dim"])
    ee_dims = list(robot_info["ee_dim"])
    arm_keys, ee_keys = _state_keys(len(arm_dims))
    values = np.asarray(actions, dtype=np.float32)
    expected = sum(arm_dims) + sum(ee_dims)
    if values.ndim != 2 or values.shape[-1] != expected:
        raise ValueError(f"expected action chunk [T,{expected}], got {values.shape}")
    result: list[dict[str, np.ndarray]] = []
    for step in values:
        item: dict[str, np.ndarray] = {}
        offset = 0
        for arm_key, ee_key, arm_dim, ee_dim in zip(
            arm_keys, ee_keys, arm_dims, ee_dims, strict=True
        ):
            item[arm_key] = step[offset : offset + arm_dim].copy()
            offset += arm_dim
            item[ee_key] = step[offset : offset + ee_dim].copy()
            offset += ee_dim
        result.append(item)
    return result


def stable_noise_seed(base_seed: int, layout_id: int, replan_step: int) -> int:
    payload = f"{int(base_seed)}:{int(layout_id)}:{int(replan_step)}".encode()
    digest = hashlib.blake2b(payload, digest_size=8, person=b"magic" + b"vla").digest()
    return int.from_bytes(digest, "little") & ((1 << 63) - 1)


def limit_joint_target_steps(
    targets: np.ndarray,
    current_state: np.ndarray,
    robot_info: dict[str, Any],
    *,
    max_joint_step: float | None,
    max_gripper_step: float | None,
) -> np.ndarray:
    """Apply only per-frame safety limits to reconstructed joint targets."""
    values = np.asarray(targets, dtype=np.float32)
    state = np.asarray(current_state, dtype=np.float32).reshape(-1)
    expected = sum(robot_info["arm_dim"]) + sum(robot_info["ee_dim"])
    if values.ndim != 2 or values.shape[-1] != expected:
        raise ValueError(f"expected action chunk [T,{expected}], got {values.shape}")
    if state.shape[0] != expected:
        raise ValueError(f"expected current state [{expected}], got {state.shape}")
    if not np.isfinite(values).all() or not np.isfinite(state).all():
        raise ValueError("actions and current state must be finite")
    for name, limit in (
        ("max_joint_step", max_joint_step),
        ("max_gripper_step", max_gripper_step),
    ):
        if limit is not None and limit <= 0.0:
            raise ValueError(f"{name} must be positive or null")

    joint_mask = np.zeros(expected, dtype=bool)
    gripper_mask = np.zeros(expected, dtype=bool)
    offset = 0
    for arm_dim, ee_dim in zip(
        robot_info["arm_dim"], robot_info["ee_dim"], strict=True
    ):
        joint_mask[offset : offset + arm_dim] = True
        offset += arm_dim
        gripper_mask[offset : offset + ee_dim] = True
        offset += ee_dim

    limited = np.empty_like(values)
    previous = state.copy()
    for index, target in enumerate(values):
        delta = target - previous
        if max_joint_step is not None:
            delta[joint_mask] = np.clip(
                delta[joint_mask], -max_joint_step, max_joint_step
            )
        if max_gripper_step is not None:
            delta[gripper_mask] = np.clip(
                delta[gripper_mask], -max_gripper_step, max_gripper_step
            )
        previous = previous + delta
        limited[index] = previous
    return limited


def limit_camera_eef_target_steps(
    targets: np.ndarray,
    current_state: np.ndarray,
    *,
    max_translation_step: float | None,
    max_gripper_step: float | None,
    layout: str = "compact20",
) -> np.ndarray:
    """Limit gripper and camera-frame EEF translation changes per command."""
    values = np.asarray(targets, dtype=np.float32).copy()
    state = np.asarray(current_state, dtype=np.float32).reshape(-1)
    spec = camera_eef_layout_spec(layout)
    expected = spec["dim"]
    if values.ndim != 2 or values.shape[-1] != expected or state.shape != (expected,):
        raise ValueError(
            f"expected targets [T,{expected}] and state [{expected}], "
            f"got {values.shape}/{state.shape}"
        )
    previous = state.copy()
    for index, target in enumerate(values):
        if max_gripper_step is not None:
            grippers = list(spec["gripper_indices"])
            target[grippers] = previous[grippers] + np.clip(
                target[grippers] - previous[grippers],
                -max_gripper_step,
                max_gripper_step,
            )
        if max_translation_step is not None:
            for start in spec["pose_starts"]:
                delta = target[start : start + 3] - previous[start : start + 3]
                norm = float(np.linalg.norm(delta))
                if norm > max_translation_step:
                    target[start : start + 3] = (
                        previous[start : start + 3]
                        + delta * max_translation_step / norm
                    )
        previous = target.copy()
        values[index] = target
    return values
