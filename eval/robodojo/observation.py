"""RoboDojo inference adapter components."""

from __future__ import annotations
from typing import Any
import numpy as np
from .rotation6d import matrix_to_rot6d
from .layout import _CAMERA_ALIASES, camera_eef_layout_spec
from .geometry import _quat_wxyz_to_matrix


def _extract_image(observation: dict[str, Any], aliases: tuple[str, ...]) -> Any:
    images = observation.get("images")
    if isinstance(images, dict):
        for name in aliases:
            if name in images:
                return images[name]
    vision = observation.get("vision", {})
    for name in aliases:
        if name not in vision:
            continue
        image = vision[name]
        if isinstance(image, dict):
            for field in ("color", "rgb"):
                if field in image:
                    return image[field]
        else:
            return image
    raise KeyError(f"no camera found for aliases {aliases}")


def ensure_chw_uint8(image: Any) -> np.ndarray:
    """Convert an already-decoded HWC or CHW observation to RGB uint8 CHW."""
    if isinstance(image, (bytes, bytearray, memoryview)):
        raise TypeError("model observations must contain decoded RGB image arrays")
    array = np.asarray(image)
    if array.ndim == 1 and array.dtype == np.uint8:
        raise TypeError("model observations must contain decoded RGB image arrays")
    if array.ndim != 3:
        raise ValueError(f"expected a 3-D image, got shape {array.shape}")
    if np.issubdtype(array.dtype, np.floating):
        if array.size and float(np.nanmax(array)) <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0.0, 255.0).astype(np.uint8)
    elif array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    if array.shape[-1] in (1, 3):
        array = np.transpose(array, (2, 0, 1))
    elif array.shape[0] not in (1, 3):
        raise ValueError(f"cannot infer channels for image shape {array.shape}")
    if array.shape[0] == 1:
        array = np.repeat(array, 3, axis=0)
    return np.ascontiguousarray(array, dtype=np.uint8)


def _state_keys(num_arms: int) -> tuple[list[str], list[str]]:
    if num_arms == 1:
        return ["arm_joint_state"], ["ee_joint_state"]
    if num_arms == 2:
        return (
            ["left_arm_joint_state", "right_arm_joint_state"],
            ["left_ee_joint_state", "right_ee_joint_state"],
        )
    raise ValueError(f"unsupported arm count: {num_arms}")


def pack_joint_state(state: dict[str, Any], robot_info: dict[str, Any]) -> np.ndarray:
    arm_dims = list(robot_info["arm_dim"])
    ee_dims = list(robot_info["ee_dim"])
    if len(arm_dims) != len(ee_dims):
        raise ValueError("arm_dim and ee_dim lengths differ")
    arm_keys, ee_keys = _state_keys(len(arm_dims))
    parts: list[np.ndarray] = []
    for arm_key, ee_key, arm_dim, ee_dim in zip(
        arm_keys, ee_keys, arm_dims, ee_dims, strict=True
    ):
        arm = np.asarray(state[arm_key], dtype=np.float32).reshape(-1)
        ee = np.asarray(state[ee_key], dtype=np.float32).reshape(-1)
        if arm.shape[0] != arm_dim or ee.shape[0] != ee_dim:
            raise ValueError(
                f"state dimensions for {arm_key}/{ee_key} are "
                f"{arm.shape[0]}/{ee.shape[0]}, expected {arm_dim}/{ee_dim}"
            )
        parts.extend((arm, ee))
    return np.concatenate(parts)


def _pack_eef_state(
    state: dict[str, Any],
    robot_info: dict[str, Any],
    *,
    layout: str,
    pose_suffix: str,
) -> np.ndarray:
    """Pack dual-arm EEF state from one explicit pose frame."""
    arm_dims = list(robot_info["arm_dim"])
    ee_dims = list(robot_info["ee_dim"])
    if len(arm_dims) != 2 or len(ee_dims) != 2 or ee_dims != [1, 1]:
        raise ValueError(
            "EEF inference currently requires dual arms with scalar grippers"
        )
    grippers = [
        float(np.asarray(state[key], dtype=np.float32).reshape(-1)[0])
        for key in ("left_ee_joint_state", "right_ee_joint_state")
    ]
    poses = []
    for key in (f"left_{pose_suffix}", f"right_{pose_suffix}"):
        pose = np.asarray(state[key], dtype=np.float32).reshape(-1)
        if pose.shape != (7,) or not np.isfinite(pose).all():
            raise ValueError(f"{key} must be a finite 7-D xyz+wxyz pose")
        rot6d = matrix_to_rot6d(_quat_wxyz_to_matrix(pose[3:])).astype(np.float32)
        poses.append(np.concatenate((pose[:3], rot6d)).astype(np.float32))
    spec = camera_eef_layout_spec(layout)
    packed = np.zeros(spec["dim"], dtype=np.float32)
    packed[list(spec["gripper_indices"])] = grippers
    for start, pose in zip(spec["pose_starts"], poses, strict=True):
        packed[start : start + 9] = pose
    return packed


def pack_camera_eef_state(
    state: dict[str, Any], robot_info: dict[str, Any], *, layout: str = "compact20"
) -> np.ndarray:
    """Pack dual-arm camera-frame EEF state."""
    return _pack_eef_state(
        state, robot_info, layout=layout, pose_suffix="camera_ee_pose"
    )


def pack_ee_state(
    state: dict[str, Any], robot_info: dict[str, Any], *, layout: str = "compact20"
) -> np.ndarray:
    """Pack dual-arm world/base-frame EEF state."""
    return _pack_eef_state(state, robot_info, layout=layout, pose_suffix="ee_pose")


def pack_joint_eef_state(
    state: dict[str, Any], robot_info: dict[str, Any]
) -> np.ndarray:
    """Pack all valid slots from the training-time joint+EEF union32 schema."""
    joint = pack_joint_state(state, robot_info)
    if joint.shape != (14,):
        raise ValueError(
            f"joint+EEF union32 currently requires a 14-D dual-arm joint state, got {joint.shape}"
        )
    camera_eef = pack_camera_eef_state(state, robot_info, layout="union32")
    packed = camera_eef.copy()
    packed[:14] = joint
    if not np.allclose(packed[[6, 13]], camera_eef[[6, 13]]):
        raise ValueError("joint and camera EEF packers disagree on gripper slots 6/13")
    return packed


def encode_observation(
    observation: dict[str, Any],
    *,
    action_type: str,
    robot_action_dim_info: dict[str, Any],
    fallback_instruction: str,
    camera_eef_layout: str = "compact20",
    model_input_layout: str = "action_type",
) -> dict[str, Any]:
    if action_type not in ("joint", "ee", "camera_ee"):
        raise ValueError(
            f"unsupported Magic_W0 action_type={action_type!r}; "
            "expected 'joint', 'ee', or 'camera_ee'"
        )
    images = {
        name: ensure_chw_uint8(_extract_image(observation, aliases))
        for name, aliases in _CAMERA_ALIASES.items()
    }
    raw_state = observation.get("state")
    if isinstance(raw_state, dict):
        if model_input_layout == "joint_eef_union32":
            state = pack_joint_eef_state(raw_state, robot_action_dim_info)
        elif model_input_layout == "action_type":
            state = (
                pack_joint_state(raw_state, robot_action_dim_info)
                if action_type == "joint"
                else (pack_ee_state if action_type == "ee" else pack_camera_eef_state)(
                    raw_state, robot_action_dim_info, layout=camera_eef_layout
                )
            )
        else:
            raise ValueError(f"unsupported model_input_layout={model_input_layout!r}")
    else:
        state = np.asarray(raw_state, dtype=np.float32).reshape(-1)
    instruction = (
        observation.get("instruction")
        or observation.get("instructions")
        or observation.get("prompt")
        or fallback_instruction
    )
    if not str(instruction).strip():
        raise ValueError("observation has no instruction and task_name is empty")
    layout_id = observation.get("layout_id", observation.get("env_seed"))
    if layout_id is None:
        layout_id = observation.get("env_idx", 0)
    return {
        "images": images,
        "state": np.asarray(state, dtype=np.float32).reshape(-1),
        "task": str(instruction).strip(),
        "layout_id": int(layout_id),
    }
