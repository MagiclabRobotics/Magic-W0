"""Community deployment must not depend on private XPolicyLab pose helpers."""

from types import SimpleNamespace
import numpy as np
import pytest

from eval.robodojo.geometry import (
    _quat_wxyz_to_matrix,
    camera_to_world_matrix,
    camera_pose_to_world_pose,
    world_pose_to_camera_pose,
)


def test_camera_requests_ros_axes_and_normalizes_quaternion():
    calls = []

    def pose(*, camera_axes):
        calls.append(camera_axes)
        return np.array([1, 2, 3]), np.array([2, 0, 0, 2])

    matrix = camera_to_world_matrix(SimpleNamespace(get_world_pose=pose))
    assert calls == ["ros"]
    np.testing.assert_allclose(matrix[:3, 3], [1, 2, 3])
    np.testing.assert_allclose(matrix[:3, :3], [[0, -1, 0], [1, 0, 0], [0, 0, 1]], atol=1e-7)


def test_pose_roundtrip_with_nontrivial_camera_and_eef_rotations():
    rng = np.random.default_rng(2026)
    for _ in range(100):
        transform = np.eye(4)
        transform[:3, :3] = _quat_wxyz_to_matrix(rng.normal(size=4))
        transform[:3, 3] = rng.normal(size=3)
        pose = rng.normal(size=7)
        pose[3:] /= np.linalg.norm(pose[3:])
        converted = world_pose_to_camera_pose(pose, transform)
        restored = camera_pose_to_world_pose(converted, transform)
        np.testing.assert_allclose(restored[:3], pose[:3], atol=1e-6)
        np.testing.assert_allclose(_quat_wxyz_to_matrix(restored[3:]), _quat_wxyz_to_matrix(pose[3:]), atol=1e-6)


@pytest.mark.parametrize("pose", [[0] * 7, [0, 0, np.nan, 1, 0, 0, 0]])
def test_invalid_poses_are_rejected(pose):
    with pytest.raises(ValueError):
        camera_pose_to_world_pose(pose, np.eye(4))
