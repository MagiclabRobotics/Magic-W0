"""RoboDojo inference adapter components."""

from __future__ import annotations
from typing import Any
import numpy as np


def _quat_wxyz_to_matrix(quaternion: Any) -> np.ndarray:
    quat = np.asarray(quaternion, dtype=np.float64).reshape(-1)
    if quat.shape != (4,) or not np.isfinite(quat).all():
        raise ValueError(f"expected a finite wxyz quaternion, got {quat}")
    norm = np.linalg.norm(quat)
    if norm < 1e-12:
        raise ValueError("quaternion norm is zero")
    w, x, y, z = quat / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _matrix_to_quat_wxyz(matrix: Any) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError(f"expected a finite 3x3 rotation matrix, got {matrix.shape}")
    # Stable branch-based conversion, returned scalar-first for RoboDojo.
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        quat = np.array(
            [
                0.25 * scale,
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
            ]
        )
    else:
        axis = int(np.argmax(np.diag(matrix)))
        if axis == 0:
            scale = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            quat = np.array(
                [
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                    0.25 * scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                ]
            )
        elif axis == 1:
            scale = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            quat = np.array(
                [
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    0.25 * scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                ]
            )
        else:
            scale = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            quat = np.array(
                [
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    0.25 * scale,
                ]
            )
    norm = np.linalg.norm(quat)
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError("rotation matrix produced an invalid quaternion")
    return (quat / norm).astype(np.float32)


def _as_numpy(value: Any) -> np.ndarray:
    """Convert NumPy or Torch values without importing Torch in the simulator."""
    for method in ("detach", "cpu", "numpy"):
        if hasattr(value, method):
            value = getattr(value, method)()
    return np.asarray(value, dtype=np.float64)


def camera_to_world_matrix(camera: Any) -> np.ndarray:
    """Read Isaac's ROS optical camera frame; USD axes differ from training."""
    position, quaternion = camera.get_world_pose(camera_axes="ros")
    position = _as_numpy(position).reshape(-1)
    if position.shape != (3,) or not np.isfinite(position).all():
        raise ValueError("camera position must be a finite 3D vector")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = _quat_wxyz_to_matrix(_as_numpy(quaternion))
    transform[:3, 3] = position
    return transform


def _transform_pose(pose: Any, transform: Any, *, inverse: bool) -> np.ndarray:
    pose = _as_numpy(pose).reshape(-1)
    transform = _as_numpy(transform)
    if pose.shape != (7,) or transform.shape != (4, 4):
        raise ValueError("expected pose [x,y,z,qw,qx,qy,qz] and a 4x4 transform")
    if not np.isfinite(pose).all() or not np.isfinite(transform).all():
        raise ValueError("pose and camera transform must be finite")
    rotation, translation = transform[:3, :3], transform[:3, 3]
    orientation = _quat_wxyz_to_matrix(pose[3:])
    if inverse:
        position = rotation.T @ (pose[:3] - translation)
        orientation = rotation.T @ orientation
    else:
        position = rotation @ pose[:3] + translation
        orientation = rotation @ orientation
    quaternion = _matrix_to_quat_wxyz(orientation)
    if quaternion[0] < 0:
        quaternion = -quaternion
    return np.concatenate([position, quaternion]).astype(np.float32)


def camera_pose_to_world_pose(pose: Any, camera_to_world: Any) -> np.ndarray:
    """Convert a scalar-first quaternion pose from camera to world coordinates."""
    return _transform_pose(pose, camera_to_world, inverse=False)


def world_pose_to_camera_pose(pose: Any, camera_to_world: Any) -> np.ndarray:
    """Convert a world pose into the ROS optical camera frame."""
    return _transform_pose(pose, camera_to_world, inverse=True)
