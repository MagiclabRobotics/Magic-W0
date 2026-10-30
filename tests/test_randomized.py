"""Randomized physical invariants across layouts and temporal schedules."""

import numpy as np
import pytest
from eval.robodojo.geometry import _quat_wxyz_to_matrix
from eval.robodojo.history import HistoryBuffer
from eval.robodojo.layout import camera_eef_layout_spec
from eval.robodojo.normalization import DeltaNormalizationStats
from eval.robodojo.rotation6d import matrix_to_rot6d, rot6d_to_matrix


@pytest.mark.parametrize("seed", range(128))
@pytest.mark.parametrize("layout", ["compact20", "union32", "eef18gripper2in34"])
def test_random_rotation_delta_composition(seed, layout):
    rng = np.random.default_rng(seed)
    spec = camera_eef_layout_spec(layout)
    entry = {
        "action_representation": "chunk_delta",
        "action_delta_indices": spec["delta_indices"],
        "action_rotation_groups": spec["rotation_groups"],
        "stats": {
            key: {"min": [-1] * spec["dim"], "max": [1] * spec["dim"]}
            for key in ("observation.state", "action")
        },
    }
    stats = DeltaNormalizationStats.from_entry(entry)
    state = rng.normal(size=spec["dim"]).astype(np.float32)
    delta = rng.normal(size=(8, spec["dim"])).astype(np.float32)
    references = []
    relative = []
    for group in spec["rotation_groups"]:
        quat = rng.normal(size=4)
        quat /= np.linalg.norm(quat)
        reference = _quat_wxyz_to_matrix(quat)
        rotations = []
        for _ in range(8):
            quat = rng.normal(size=4)
            quat /= np.linalg.norm(quat)
            rotations.append(_quat_wxyz_to_matrix(quat))
        rotations = np.stack(rotations)
        state[list(group)] = matrix_to_rot6d(reference)
        delta[:, list(group)] = matrix_to_rot6d(rotations)
        references.append(reference)
        relative.append(rotations)
    untouched = delta.copy()
    output = stats.reconstruct_camera_eef_targets(delta, state)
    for group, reference, rotations in zip(
        spec["rotation_groups"], references, relative, strict=True
    ):
        actual = rot6d_to_matrix(output[:, list(group)])
        np.testing.assert_allclose(actual, rotations @ reference, atol=2e-6)
        np.testing.assert_allclose(np.linalg.det(actual), 1, atol=2e-6)
    np.testing.assert_array_equal(delta, untouched)
    np.testing.assert_array_equal(
        output[:, list(spec["gripper_indices"])],
        delta[:, list(spec["gripper_indices"])],
    )


@pytest.mark.parametrize("interval", [20, 40, 60, 100])
@pytest.mark.parametrize("execute", [1, 4, 5, 10, 20])
@pytest.mark.parametrize("capacity", [1, 2, 25])
def test_long_history_retains_only_past_frames_per_environment(
    interval, execute, capacity
):
    buffer = HistoryBuffer(capacity, interval, execute)
    for replan in range(160):
        step = replan * execute
        for env in (2, 9):
            current = np.full((1, 2, 2), step + env, dtype=np.int32)
            clip, pad = buffer.clip(env, step, current)
            expected = [
                t for t in range(step - capacity * interval, step, interval) if t >= 0
            ]
            assert (~pad).sum() == len(expected)
            assert not clip[pad].any()
            if expected:
                np.testing.assert_array_equal(
                    clip[~pad, 0, 0, 0], np.array(expected) + env
                )
            buffer.record(env, step, current)
            current.fill(-1)
    buffer.reset()
    assert not buffer._frames
