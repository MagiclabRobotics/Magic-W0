"""RoboDojo inference adapter components."""

from __future__ import annotations
from typing import Any
import numpy as np
from .normalization import DeltaNormalizationStats


_CAMERA_ALIASES = {
    "cam_high": ("cam_high", "cam_head", "head_camera", "top_camera"),
    "cam_left_wrist": ("cam_left_wrist", "left_camera", "left_wrist", "wrist_left"),
    "cam_right_wrist": (
        "cam_right_wrist",
        "right_camera",
        "right_wrist",
        "wrist_right",
    ),
}


_CAMERA_EEF_LAYOUTS = {
    "compact20": {
        "dim": 20,
        "active_indices": tuple(range(20)),
        "gripper_indices": (0, 1),
        "pose_starts": (2, 11),
        "delta_indices": tuple(range(2, 20)),
        "rotation_groups": (tuple(range(5, 11)), tuple(range(14, 20))),
    },
    "union32": {
        "dim": 32,
        "active_indices": (6, 13, *range(14, 32)),
        "gripper_indices": (6, 13),
        "pose_starts": (14, 23),
        "delta_indices": tuple(range(14, 32)),
        "rotation_groups": (tuple(range(17, 23)), tuple(range(26, 32))),
    },
    "eef18gripper2in34": {
        "dim": 34,
        "active_indices": (7, 15, *range(16, 34)),
        "gripper_indices": (7, 15),
        "pose_starts": (16, 25),
        "delta_indices": tuple(range(16, 34)),
        "rotation_groups": (tuple(range(19, 25)), tuple(range(28, 34))),
    },
}


_CAMERA_EEF_LAYOUT_ALIASES = {
    "union34": "eef18gripper2in34",
    "eef18_gripper2_in34": "eef18gripper2in34",
}


def normalize_camera_eef_layout(layout: str) -> str:
    return _CAMERA_EEF_LAYOUT_ALIASES.get(str(layout), str(layout))


_JOINT_UNION32_INDICES = tuple(range(14))


_JOINT_UNION32_DELTA_INDICES = (*range(0, 6), *range(7, 13))


_JOINT_EEF_UNION32_DELTA_INDICES = (
    *_JOINT_UNION32_DELTA_INDICES,
    *range(14, 32),
)


_JOINT_EEF_UNION32_ROTATION_GROUPS = (
    tuple(range(17, 23)),
    tuple(range(26, 32)),
)


def detect_checkpoint_action_layout(stats: "DeltaNormalizationStats") -> str:
    """Identify pure joint, pure EEF, or the fully active joint+EEF schema."""
    delta_indices = tuple(stats.action_delta_indices)
    rotation_groups = tuple(stats.action_rotation_groups)
    if not rotation_groups:
        return "joint"
    if (
        delta_indices == _JOINT_EEF_UNION32_DELTA_INDICES
        and rotation_groups == _JOINT_EEF_UNION32_ROTATION_GROUPS
    ):
        return "joint_eef_union32"
    for name, spec in _CAMERA_EEF_LAYOUTS.items():
        if (
            delta_indices == spec["delta_indices"]
            and rotation_groups == spec["rotation_groups"]
        ):
            return name
    raise ValueError(
        "cannot infer action slot layout from checkpoint metadata: "
        f"delta_indices={delta_indices}, rotation_groups={rotation_groups}"
    )


def camera_eef_layout_spec(layout: str) -> dict[str, Any]:
    layout = normalize_camera_eef_layout(layout)
    try:
        return _CAMERA_EEF_LAYOUTS[str(layout)]
    except KeyError as exc:
        raise ValueError(
            f"unknown camera EEF layout {layout!r}; expected "
            f"{', '.join(_CAMERA_EEF_LAYOUTS)}"
        ) from exc


def camera_eef_labels(layout: str) -> list[str]:
    layout = normalize_camera_eef_layout(layout)
    if layout == "compact20":
        return [
            "left_gripper",
            "right_gripper",
            "left_x",
            "left_y",
            "left_z",
            *[f"left_rot6d_{i}" for i in range(6)],
            "right_x",
            "right_y",
            "right_z",
            *[f"right_rot6d_{i}" for i in range(6)],
        ]
    if layout == "union32":
        return [
            *[f"left_joint_masked_{i}" for i in range(6)],
            "left_gripper",
            *[f"right_joint_masked_{i}" for i in range(6)],
            "right_gripper",
            "left_x",
            "left_y",
            "left_z",
            *[f"left_rot6d_{i}" for i in range(6)],
            "right_x",
            "right_y",
            "right_z",
            *[f"right_rot6d_{i}" for i in range(6)],
        ]
    if layout == "eef18gripper2in34":
        return [
            *[f"left_joint_masked_{i}" for i in range(7)],
            "left_gripper",
            *[f"right_joint_masked_{i}" for i in range(7)],
            "right_gripper",
            "left_x",
            "left_y",
            "left_z",
            *[f"left_rot6d_{i}" for i in range(6)],
            "right_x",
            "right_y",
            "right_z",
            *[f"right_rot6d_{i}" for i in range(6)],
        ]
    camera_eef_layout_spec(layout)
    raise AssertionError(f"unhandled camera EEF layout {layout!r}")


def joint_eef_union32_labels() -> list[str]:
    return [
        "left_j0",
        "left_j1",
        "left_j2",
        "left_j3",
        "left_j4",
        "left_j5",
        "left_gripper",
        "right_j0",
        "right_j1",
        "right_j2",
        "right_j3",
        "right_j4",
        "right_j5",
        "right_gripper",
        "left_x",
        "left_y",
        "left_z",
        *[f"left_rot6d_{i}" for i in range(6)],
        "right_x",
        "right_y",
        "right_z",
        *[f"right_rot6d_{i}" for i in range(6)],
    ]


def detect_camera_eef_layout(stats: DeltaNormalizationStats) -> str:
    layout = detect_checkpoint_action_layout(stats)
    if layout == "joint_eef_union32":
        return "union32"
    if layout not in _CAMERA_EEF_LAYOUTS:
        raise ValueError(
            "cannot infer camera EEF slot layout from checkpoint metadata: "
            f"delta_indices={stats.action_delta_indices}, "
            f"rotation_groups={stats.action_rotation_groups}"
        )
    return layout


def validate_camera_eef_normalization_layout(
    stats: DeltaNormalizationStats, layout: str
) -> None:
    """Reject checkpoints whose padded/selected slots disagree with the layout."""
    spec = camera_eef_layout_spec(layout)
    q01 = np.asarray(stats.action_low, dtype=np.float32)
    q99 = np.asarray(stats.action_high, dtype=np.float32)
    if layout in ("union32", "eef18gripper2in34"):
        grippers = list(spec["gripper_indices"])
        masked = [
            index for index in range(spec["dim"]) if index not in spec["active_indices"]
        ]
        gripper_ok = all(
            q01[index] >= -0.05 and q99[index] <= 1.05 for index in grippers
        )
        padding_ok = all(q01[index] <= -0.90 and q99[index] >= 0.90 for index in masked)
        if not (gripper_ok and padding_ok):
            raise ValueError(
                f"checkpoint normalization is not a true {layout} layout: "
                f"gripper_bounds={[(float(q01[i]), float(q99[i])) for i in grippers]}, "
                f"masked_bounds={[(float(q01[i]), float(q99[i])) for i in masked]}. "
                "The training backend likely compacted selected slots because "
                "preserve_selected_slots was not forwarded; retrain this checkpoint "
                "after updating the LeRobot backend."
            )
