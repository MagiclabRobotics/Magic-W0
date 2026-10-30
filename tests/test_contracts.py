import numpy as np
import pytest

from magic_w0 import MagicW0Config, validate_checkpoint_modes
from magic_w0.checkpoint import split_policy_checkpoint_state
from eval.robodojo.action import unpack_camera_eef_actions, limit_joint_target_steps
from eval.robodojo.history import HistoryBuffer, validate_history_schedule
from eval.robodojo.layout import camera_eef_layout_spec, detect_checkpoint_action_layout
from eval.robodojo.normalization import DeltaNormalizationStats
from eval.robodojo.observation import pack_camera_eef_state, pack_joint_state
from eval.robodojo.rotation6d import matrix_to_rot6d, rot6d_to_matrix

ROBOT = {"arm_dim": [6, 6], "ee_dim": [1, 1]}


def state():
    return {
        "left_arm_joint_state": np.arange(6),
        "right_arm_joint_state": np.arange(6) + 10,
        "left_ee_joint_state": np.array([0.2]),
        "right_ee_joint_state": np.array([0.8]),
        "left_camera_ee_pose": np.array([1, 2, 3, 1, 0, 0, 0]),
        "right_camera_ee_pose": np.array([4, 5, 6, 1, 0, 0, 0]),
    }


def stats_for(layout):
    spec = camera_eef_layout_spec(layout)
    return DeltaNormalizationStats.from_entry(
        {
            "action_representation": "chunk_delta",
            "action_delta_indices": spec["delta_indices"],
            "action_rotation_groups": spec["rotation_groups"],
            "stats": {
                key: {"min": [-1] * spec["dim"], "max": [1] * spec["dim"]}
                for key in ["observation.state", "action"]
            },
        }
    )


@pytest.mark.parametrize("layout", ["compact20", "union32", "eef18gripper2in34"])
def test_camera_eef_roundtrip_and_padding(layout):
    original = state()
    packed = pack_camera_eef_state(original, ROBOT, layout=layout)
    spec = camera_eef_layout_spec(layout)
    inactive = sorted(set(range(spec["dim"])) - set(spec["active_indices"]))
    np.testing.assert_array_equal(packed[inactive], 0)
    recovered = unpack_camera_eef_actions(packed[None], layout=layout)[0]
    for key in recovered:
        np.testing.assert_allclose(recovered[key], original[key], atol=1e-6)
    assert detect_checkpoint_action_layout(stats_for(layout)) == layout


@pytest.mark.parametrize("layout", ["compact20", "union32", "eef18gripper2in34"])
def test_rotation_deltas_compose_on_so3_without_changing_grippers(layout):
    packed = pack_camera_eef_state(state(), ROBOT, layout=layout)
    spec = camera_eef_layout_spec(layout)
    # A nonidentity reference distinguishes rotation composition from addition.
    reference = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)
    relative = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float32)
    delta = np.zeros((2, spec["dim"]), dtype=np.float32)
    delta[:, list(spec["gripper_indices"])] = [0.3, 0.7]
    for start, group in zip(spec["pose_starts"], spec["rotation_groups"], strict=True):
        packed[list(group)] = matrix_to_rot6d(reference)
        delta[:, list(group)] = matrix_to_rot6d(relative)
        delta[:, start : start + 3] = [0.1, 0.2, 0.3]
    result = stats_for(layout).reconstruct_camera_eef_targets(delta, packed)
    for start, group in zip(spec["pose_starts"], spec["rotation_groups"], strict=True):
        np.testing.assert_allclose(
            rot6d_to_matrix(result[0, list(group)]), relative @ reference, atol=1e-6
        )
        np.testing.assert_allclose(
            result[0, start : start + 3],
            packed[start : start + 3] + [0.1, 0.2, 0.3],
            atol=1e-6,
        )
    np.testing.assert_allclose(
        result[:, list(spec["gripper_indices"])], [[0.3, 0.7]] * 2
    )
    np.testing.assert_array_equal(
        delta[:, list(spec["gripper_indices"])],
        [[np.float32(0.3), np.float32(0.7)]] * 2,
    )


def test_joint_pack_unpack_and_per_frame_limits():
    packed = pack_joint_state(state(), ROBOT)
    assert packed[6] == pytest.approx(0.2)
    assert packed[13] == pytest.approx(0.8)
    limited = limit_joint_target_steps(
        np.ones((3, 14)), np.zeros(14), ROBOT, max_joint_step=0.1, max_gripper_step=0.2
    )
    np.testing.assert_allclose(limited[:, 0], [0.1, 0.2, 0.3], atol=1e-6)
    np.testing.assert_allclose(limited[:, 6], [0.2, 0.4, 0.6], atol=1e-6)
    with pytest.raises(ValueError, match="finite"):
        limit_joint_target_steps(
            np.full((1, 14), np.nan),
            packed,
            ROBOT,
            max_joint_step=None,
            max_gripper_step=None,
        )


def test_history_padding_isolation_schedule_and_reset():
    buffer = HistoryBuffer(2, 40, 20)
    frame = np.ones((3, 2, 2), dtype=np.uint8)
    clip, pad = buffer.clip(0, 0, frame)
    assert pad.tolist() == [True, True]
    assert not clip.any()
    buffer.record(0, 0, frame)
    frame[:] = 9  # The stored snapshot must not alias the caller.
    clip, pad = buffer.clip(0, 40, frame)
    assert pad.tolist() == [True, False]
    np.testing.assert_array_equal(clip[-1], 1)
    with pytest.raises(RuntimeError, match="missing history"):
        buffer.clip(1, 40, frame)
    buffer.reset()
    with pytest.raises(RuntimeError):
        buffer.clip(0, 40, frame)
    with pytest.raises(ValueError, match="divisible"):
        validate_history_schedule(30, 20)


def test_normalization_rejects_missing_or_incompatible_contract():
    with pytest.raises(KeyError, match="normalization"):
        DeltaNormalizationStats.from_checkpoint({})
    with pytest.raises(ValueError, match="chunk_delta"):
        DeltaNormalizationStats.from_entry({"action_representation": "absolute"})
    stats = stats_for("compact20")
    np.testing.assert_allclose(stats.denormalize_action(np.zeros((1, 20))), 0)
    with pytest.raises(ValueError, match="dim mismatch"):
        stats.normalize_state(np.zeros(19))


def test_checkpoint_split_preserves_vlm_and_rejects_partial_policy():
    vlm, policy = split_policy_checkpoint_state(
        {"module.vlm.a": 1, "module.action.a": 2, "module._lm_head.a": 3}
    )
    assert vlm == {"a": 1}
    assert policy == {"action.a": 2}
    with pytest.raises(ValueError, match="complete vlm"):
        split_policy_checkpoint_state({"action.a": 1})


def test_config_refuses_training_modes_and_unknown_dimensions():
    assert MagicW0Config.from_dict(
        {"camera_keys": ["observation.images.cam_high"]}
    ).camera_keys == ("observation.images.cam_high",)
    with pytest.raises(ValueError, match="unknown"):
        MagicW0Config.from_dict({"typo": 1})
    with pytest.raises(ValueError, match="positive integer"):
        MagicW0Config(chunk_size=0)
    with pytest.raises(ValueError, match="unsupported"):
        validate_checkpoint_modes({"state_prompt_mode": "text"})
