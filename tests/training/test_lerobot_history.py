"""Temporal-observation tests for standalone LeRobot readers and pi0.5."""

from __future__ import annotations

import sys
import unittest
import importlib.util
import inspect
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from magic_w0.data.lerobot_history import (  # noqa: E402
    episode_observation_indices,
    validate_observation_sampling,
)

HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_NUMPY = importlib.util.find_spec("numpy") is not None
HAS_READER_DEPS = HAS_TORCH and HAS_NUMPY
if HAS_READER_DEPS:
    import numpy as np
    import torch

    from magic_w0.data.backends.lerobot_v21 import (  # noqa: E402
        LeRobotV21Dataset,
        _adapt_last_dim,
        _mask_inactive_dimensions,
        aggregate_v21_stats,
        _parse_action_representation,
        _parse_camera_rotations,
        _quality_mask,
    )
    from magic_w0.data.backends.lerobot_v30 import LeRobotV30Dataset  # noqa: E402
    from magic_w0.data.norm_stats import save_norm_stats_artifact  # noqa: E402
elif HAS_TORCH:
    import torch


class TemporalSamplingTest(unittest.TestCase):
    @unittest.skipUnless(HAS_READER_DEPS, "torch and numpy are required")
    def test_episode_end_padding_supervision_is_disabled_by_default(self) -> None:
        for dataset_type in (LeRobotV21Dataset, LeRobotV30Dataset):
            parameter = inspect.signature(dataset_type).parameters[
                "supervise_episode_end_padding"
            ]
            self.assertIs(parameter.default, False)

    @unittest.skipUnless(HAS_READER_DEPS, "torch and numpy are required")
    def test_first_14_dimensions_pad_to_32_with_false_mask(self) -> None:
        values = np.arange(64, dtype=np.float32).reshape(2, 32)
        raw_mask = np.ones((2, 32), dtype=np.bool_)
        indices = tuple(range(14))

        adapted = _adapt_last_dim(values, indices=indices, target_dim=32, pad_value=0.0)
        adapted_mask = _adapt_last_dim(
            raw_mask, indices=indices, target_dim=32, pad_value=False
        )

        self.assertEqual(adapted.shape, (2, 32))
        self.assertEqual(adapted_mask.shape, (2, 32))
        np.testing.assert_array_equal(adapted[:, :14], values[:, :14])
        np.testing.assert_array_equal(adapted[:, 14:], 0.0)
        self.assertTrue(adapted_mask[:, :14].all())
        self.assertFalse(adapted_mask[:, 14:].any())

    @unittest.skipUnless(HAS_READER_DEPS, "torch and numpy are required")
    def test_active_indices_preserve_eef_and_gripper_model_slots(self) -> None:
        values = np.arange(64, dtype=np.float32).reshape(2, 32)
        original = values.copy()
        dim_mask = np.ones_like(values, dtype=np.bool_)
        active = (6, 13, *range(14, 32))

        _mask_inactive_dimensions(
            values,
            dim_mask,
            active_indices=active,
        )

        inactive = (0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12)
        np.testing.assert_array_equal(values[:, inactive], 0.0)
        self.assertFalse(dim_mask[:, inactive].any())
        np.testing.assert_array_equal(values[:, active], original[:, active])
        self.assertTrue(dim_mask[:, active].all())

    @unittest.skipUnless(HAS_READER_DEPS, "torch and numpy are required")
    def test_v21_stats_respect_episode_selection(self) -> None:
        import json
        import tempfile
        from pathlib import Path

        def row(episode_index, mean, count):
            return {
                "episode_index": episode_index,
                "stats": {
                    "action": {
                        "mean": [mean],
                        "std": [0.0],
                        "min": [mean],
                        "max": [mean],
                        "q01": [mean],
                        "q99": [mean],
                        "count": [count],
                    }
                },
            }

        with tempfile.TemporaryDirectory() as directory:
            meta = Path(directory) / "meta"
            meta.mkdir()
            stats_path = meta / "episodes_stats.jsonl"
            stats_path.write_text(
                "\n".join(
                    json.dumps(value) for value in (row(0, 1.0, 2), row(1, 9.0, 6))
                )
            )

            stats = aggregate_v21_stats(directory, episodes=[0])

        self.assertEqual(stats["action"]["mean"].tolist(), [1.0])
        self.assertEqual(stats["action"]["count"].tolist(), [2])

    @unittest.skipUnless(HAS_READER_DEPS, "torch and numpy are required")
    def test_v21_reader_loads_only_matching_external_norm_artifact(self) -> None:
        import json
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            meta = root / "meta"
            meta.mkdir()
            info = {
                "codebase_version": "v2.1",
                "fps": 30,
                "chunks_size": 1000,
                "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
                "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
                "features": {
                    "observation.state": {"dtype": "float32", "shape": [1]},
                    "action": {"dtype": "float32", "shape": [1]},
                },
            }
            (meta / "info.json").write_text(json.dumps(info), encoding="utf-8")
            (meta / "episodes.jsonl").write_text(
                json.dumps({"episode_index": 0, "length": 2, "tasks": ["move"]}) + "\n",
                encoding="utf-8",
            )
            raw_feature = {
                "count": [2],
                "mean": [1.0],
                "std": [0.5],
                "min": [0.0],
                "max": [2.0],
            }
            (meta / "episodes_stats.jsonl").write_text(
                json.dumps(
                    {
                        "episode_index": 0,
                        "stats": {
                            "observation.state": raw_feature,
                            "action": raw_feature,
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            base = LeRobotV21Dataset(root, chunk_size=1, estimate_action_stats=False)
            legacy_signature = base.norm_stats_signature()
            self.assertNotIn("active_indices", legacy_signature["state"])
            self.assertNotIn("active_indices", legacy_signature["action"])
            active = LeRobotV21Dataset(
                root,
                chunk_size=1,
                state_active_indices=[0],
                action_active_indices=[0],
                estimate_action_stats=False,
            )
            active_signature = active.norm_stats_signature()
            self.assertEqual(active_signature["state"]["active_indices"], (0,))
            self.assertEqual(active_signature["action"]["active_indices"], (0,))
            norm_feature = {
                "count": np.asarray([2]),
                "mean": np.asarray([10.0]),
                "std": np.asarray([2.0]),
                "min": np.asarray([8.0]),
                "max": np.asarray([12.0]),
                "q01": np.asarray([8.1]),
                "q99": np.asarray([11.9]),
            }
            artifact = root / "norm.json"
            save_norm_stats_artifact(
                artifact,
                signature=base.norm_stats_signature(),
                stats={"observation.state": norm_feature, "action": norm_feature},
                sampling={"moment_anchors": 2},
            )

            loaded = LeRobotV21Dataset(root, chunk_size=1, norm_stats_path=artifact)
            self.assertAlmostEqual(
                float(loaded.meta_stats["observation.state"]["q01"][0]), 8.1, places=6
            )
            with self.assertRaisesRegex(ValueError, "signature mismatch"):
                LeRobotV21Dataset(root, chunk_size=2, norm_stats_path=artifact)

    @unittest.skipUnless(HAS_READER_DEPS, "torch and numpy are required")
    def test_current_frame_and_action_chunk_quality_rules_are_independent(self) -> None:
        table = {
            "image": np.array([True, False, True, False]),
            "head_tail": np.array([False, False, True, True]),
            "eef_visual": np.array([True, True, True, False]),
        }
        current_frame_rule = {
            "require_true": ("image", "eef_visual"),
            "require_false": ("head_tail",),
        }
        action_chunk_rule = {
            "require_true": ("eef_visual",),
            "require_false": (),
        }

        self.assertEqual(
            _quality_mask(table, current_frame_rule, 4).tolist(),
            [True, False, False, False],
        )
        self.assertEqual(
            _quality_mask(table, action_chunk_rule, 4).tolist(),
            [True, True, True, False],
        )

    def test_seconds_are_quantized_with_dataset_fps(self) -> None:
        stride, offsets = validate_observation_sampling(
            obs_steps=3, obs_interval_seconds=0.1, fps=20
        )
        self.assertEqual(stride, 2)
        self.assertEqual(offsets, (-4, -2, 0))
        self.assertEqual(episode_observation_indices(3, offsets), (0, 1, 3))

    def test_invalid_history_settings_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "obs_steps"):
            validate_observation_sampling(obs_steps=0, obs_interval_seconds=0.1, fps=30)
        with self.assertRaisesRegex(ValueError, "obs_interval_seconds"):
            validate_observation_sampling(obs_steps=2, obs_interval_seconds=0.0, fps=30)

    @unittest.skipUnless(HAS_READER_DEPS, "torch and numpy are required")
    def test_action_representation_names_and_aliases(self) -> None:
        self.assertEqual(_parse_action_representation("abs-action"), "abs")
        self.assertEqual(_parse_action_representation("chunk-delta"), "chunk_delta")
        self.assertEqual(_parse_action_representation("frame_delta"), "frame_delta")
        with self.assertRaisesRegex(ValueError, "action_representation"):
            _parse_action_representation("velocity")

    @unittest.skipUnless(HAS_READER_DEPS, "torch and numpy are required")
    def test_v21_camera_rotations_are_canonical_and_quarter_turns(self) -> None:
        parsed = _parse_camera_rotations(
            {"left": 90, "right": -90, "head": 360},
            camera_keys=("head", "left", "right"),
        )
        self.assertEqual(parsed, {"left": 90, "right": 270})

        with self.assertRaisesRegex(ValueError, "unselected canonical camera"):
            _parse_camera_rotations({"source.left": 90}, camera_keys=("left",))
        with self.assertRaisesRegex(ValueError, "multiple of 90"):
            _parse_camera_rotations({"left": 45}, camera_keys=("left",))

    @unittest.skipUnless(HAS_READER_DEPS, "torch and numpy are required")
    def test_v21_camera_rotations_cover_decoded_stacks_and_future_frames(self) -> None:
        ds = LeRobotV21Dataset.__new__(LeRobotV21Dataset)
        ds.camera_keys = ["left", "right"]
        ds._source_camera_keys = ["raw_left", "raw_right"]
        ds.camera_rotations = {"left": 90, "right": 270}
        ds._ep_indices = np.array([4], dtype=np.int64)
        ds._ep_lengths = np.array([3], dtype=np.int64)
        ds.fps = 30.0
        decoded = torch.tensor([[[[1, 2, 3], [4, 5, 6]]]], dtype=torch.uint8)
        ds._decode_frames = lambda *_args: decoded.clone()

        clockwise = ds._rotate_camera_frames(decoded, "left")
        self.assertEqual(clockwise.tolist(), [[[[4, 1], [5, 2], [6, 3]]]])

        future = ds.get_episode_frame_images(4, 1, ["right"])["right"]
        self.assertEqual(future.tolist(), [[[3, 6], [2, 5], [1, 4]]])

    @unittest.skipUnless(HAS_READER_DEPS, "torch and numpy are required")
    def test_v21_reader_stacks_history_and_pads_at_episode_start(self) -> None:
        ds = LeRobotV21Dataset.__new__(LeRobotV21Dataset)
        ds._cum = np.array([0, 6])
        ds._ep_indices = [9]
        ds._ep_lengths = [6]
        ds._ep_tasks = [["pick"]]
        ds._task_override = None
        ds.chunk_size = 2
        ds.obs_steps = 3
        ds.observation_delta_indices = (-4, -2, 0)
        ds.state_key = "observation.state"
        ds.action_key = "action"
        ds.camera_keys = ["observation.images.top"]
        ds.fps = 20.0
        table = {
            ds.state_key: np.arange(12, dtype=np.float32).reshape(6, 2),
            ds.action_key: np.arange(12, dtype=np.float32).reshape(6, 2),
        }
        ds._episode_table = lambda _ep: table
        ds._decode_frames = lambda _cam, _ep, frames: torch.stack(
            [torch.full((3, 2, 2), frame, dtype=torch.uint8) for frame in frames]
        )

        item = ds[3]

        self.assertEqual(tuple(item[ds.state_key].shape), (3, 2))
        self.assertTrue(
            torch.equal(
                item[ds.state_key], torch.from_numpy(table[ds.state_key][[0, 1, 3]])
            )
        )
        self.assertEqual(tuple(item[ds.camera_keys[0]].shape), (3, 3, 2, 2))
        self.assertEqual(item[ds.camera_keys[0]][:, 0, 0, 0].tolist(), [0, 1, 3])

    @unittest.skipUnless(HAS_READER_DEPS, "torch and numpy are required")
    def test_v21_reader_remaps_source_camera_keys_before_emitting_item(self) -> None:
        ds = LeRobotV21Dataset.__new__(LeRobotV21Dataset)
        ds._cum = np.array([0, 2])
        ds._ep_indices = [1]
        ds._ep_lengths = [2]
        ds._ep_tasks = [["task"]]
        ds._task_override = None
        ds.chunk_size = 1
        ds.obs_steps = 1
        ds.observation_delta_indices = (0,)
        ds.state_key = "observation.state"
        ds.action_key = "action"
        ds._source_camera_keys = ["observation.images.head_rgb"]
        ds.camera_keys = ["observation.images.cam_high"]
        ds.fps = 10.0
        table = {
            ds.state_key: np.ones((2, 1), dtype=np.float32),
            f"{ds.state_key}_dim_mask": np.ones((2, 1), dtype=np.bool_),
            f"{ds.action_key}_dim_mask": np.ones((2, 1), dtype=np.bool_),
            "_action_chunk_quality_mask": np.ones(2, dtype=np.bool_),
        }
        ds._episode_table = lambda _ep: table
        ds._build_action_chunk = lambda _table, **_kwargs: (
            np.ones((1, 1), dtype=np.float32),
            np.zeros(1, dtype=np.bool_),
            np.ones((1, 1), dtype=np.bool_),
        )
        decoded = []

        def decode(camera, _ep, _frames):
            decoded.append(camera)
            return torch.ones((1, 3, 2, 2), dtype=torch.uint8)

        ds._decode_frames = decode
        item = ds[0]

        self.assertEqual(decoded, ["observation.images.head_rgb"])
        self.assertIn("observation.images.cam_high", item)
        self.assertNotIn("observation.images.head_rgb", item)

    @unittest.skipUnless(HAS_READER_DEPS, "torch and numpy are required")
    def test_obs_steps_one_preserves_legacy_shapes(self) -> None:
        ds = LeRobotV21Dataset.__new__(LeRobotV21Dataset)
        ds._cum = np.array([0, 2])
        ds._ep_indices = [1]
        ds._ep_lengths = [2]
        ds._ep_tasks = [["task"]]
        ds._task_override = None
        ds.chunk_size = 1
        ds.obs_steps = 1
        ds.observation_delta_indices = (0,)
        ds.state_key = "observation.state"
        ds.action_key = "action"
        ds.camera_keys = ["cam"]
        ds.fps = 10.0
        table = {
            ds.state_key: np.ones((2, 4), dtype=np.float32),
            ds.action_key: np.ones((2, 4), dtype=np.float32),
        }
        ds._episode_table = lambda _ep: table
        ds._decode_frames = lambda _cam, _ep, _frames: torch.zeros(
            (1, 3, 4, 5), dtype=torch.uint8
        )

        item = ds[0]
        self.assertEqual(tuple(item[ds.state_key].shape), (4,))
        self.assertEqual(tuple(item["cam"].shape), (3, 4, 5))

    @unittest.skipUnless(HAS_READER_DEPS, "torch and numpy are required")
    def test_v21_quality_index_and_chunk_dimension_masks(self) -> None:
        ds = LeRobotV21Dataset.__new__(LeRobotV21Dataset)
        ds._cum = np.array([0, 4])
        ds._sample_indices = np.array([1], dtype=np.int64)
        ds._ep_indices = [7]
        ds._ep_lengths = [4]
        ds._ep_tasks = [["pick"]]
        ds._task_override = None
        ds.chunk_size = 3
        ds.obs_steps = 1
        ds.observation_delta_indices = (0,)
        ds.state_key = "observation.state"
        ds.action_key = "action"
        ds.camera_keys = ["cam"]
        ds.target_action_dim = 4
        ds.fps = 20.0
        table = {
            ds.state_key: np.arange(16, dtype=np.float32).reshape(4, 4),
            ds.action_key: np.arange(16, dtype=np.float32).reshape(4, 4),
            f"{ds.state_key}_dim_mask": np.ones((4, 4), dtype=np.bool_),
            f"{ds.action_key}_dim_mask": np.array(
                [[1, 1, 0, 0], [1, 1, 0, 0], [1, 0, 0, 0], [1, 1, 1, 1]],
                dtype=np.bool_,
            ),
            "_action_chunk_quality_mask": np.array([True, True, False, True]),
        }
        ds._episode_table = lambda _ep: table
        ds._decode_frames = lambda _cam, _ep, _frames: torch.zeros(
            (1, 3, 2, 2), dtype=torch.uint8
        )

        item = ds[0]

        self.assertEqual(item["index"].item(), 1)
        self.assertEqual(item["frame_index"].item(), 1)
        self.assertEqual(item["action_is_pad"].tolist(), [False, True, False])
        self.assertEqual(
            item["action_dim_mask"].tolist(),
            [
                [True, True, False, False],
                [False, False, False, False],
                [True, True, True, True],
            ],
        )

    @unittest.skipUnless(HAS_READER_DEPS, "torch and numpy are required")
    def test_v21_repeat_upsamples_actions_and_all_supervision_masks(self) -> None:
        ds = LeRobotV21Dataset.__new__(LeRobotV21Dataset)
        ds._cum = np.array([0, 4])
        ds._sample_indices = None
        ds._ep_indices = [3]
        ds._ep_lengths = [4]
        ds._ep_tasks = [["move"]]
        ds._task_override = None
        ds.chunk_size = 5
        ds.action_upsample_rate = 2
        ds.action_upsample_mode = "repeat"
        ds.obs_steps = 1
        ds.observation_delta_indices = (0,)
        ds.state_key = "observation.state"
        ds.action_key = "action"
        ds.camera_keys = ["cam"]
        ds.target_action_dim = 2
        ds.fps = 30.0
        table = {
            ds.state_key: np.arange(8, dtype=np.float32).reshape(4, 2),
            ds.action_key: np.array(
                [[10, 11], [20, 21], [30, 31], [40, 41]], dtype=np.float32
            ),
            f"{ds.state_key}_dim_mask": np.ones((4, 2), dtype=np.bool_),
            f"{ds.action_key}_dim_mask": np.array(
                [[1, 0], [1, 1], [0, 1], [1, 1]], dtype=np.bool_
            ),
            "_action_chunk_quality_mask": np.array([True, False, True, True]),
        }
        ds._episode_table = lambda _ep: table
        ds._decode_frames = lambda _cam, _ep, _frames: torch.zeros(
            (1, 3, 2, 2), dtype=torch.uint8
        )

        item = ds[0]

        self.assertEqual(
            item["action"].tolist(),
            [[10, 11], [10, 11], [20, 21], [20, 21], [30, 31]],
        )
        self.assertEqual(
            item["action_is_pad"].tolist(), [False, False, True, True, False]
        )
        self.assertEqual(
            item["action_dim_mask"].tolist(),
            [
                [True, False],
                [True, False],
                [False, False],
                [False, False],
                [False, True],
            ],
        )

    @unittest.skipUnless(HAS_READER_DEPS, "torch and numpy are required")
    def test_v21_repeat_upsampling_preserves_episode_end_padding(self) -> None:
        ds = LeRobotV21Dataset.__new__(LeRobotV21Dataset)
        ds._cum = np.array([0, 4])
        ds._sample_indices = np.array([3], dtype=np.int64)
        ds._ep_indices = [3]
        ds._ep_lengths = [4]
        ds._ep_tasks = [["move"]]
        ds._task_override = None
        ds.chunk_size = 5
        ds.action_upsample_rate = 2
        ds.action_upsample_mode = "repeat"
        ds.supervise_episode_end_padding = False
        ds.obs_steps = 1
        ds.observation_delta_indices = (0,)
        ds.state_key = "observation.state"
        ds.action_key = "action"
        ds.camera_keys = ["cam"]
        ds.target_action_dim = 1
        ds.fps = 30.0
        table = {
            ds.state_key: np.arange(4, dtype=np.float32).reshape(4, 1),
            ds.action_key: np.array([[10], [20], [30], [40]], dtype=np.float32),
            f"{ds.state_key}_dim_mask": np.ones((4, 1), dtype=np.bool_),
            f"{ds.action_key}_dim_mask": np.ones((4, 1), dtype=np.bool_),
            "_action_chunk_quality_mask": np.ones(4, dtype=np.bool_),
        }
        ds._episode_table = lambda _ep: table
        ds._decode_frames = lambda _cam, _ep, _frames: torch.zeros(
            (1, 3, 2, 2), dtype=torch.uint8
        )

        item = ds[0]

        self.assertEqual(item["action"].flatten().tolist(), [40, 40, 40, 40, 40])
        self.assertEqual(
            item["action_is_pad"].tolist(), [False, False, True, True, True]
        )
        self.assertEqual(
            item["action_dim_mask"].flatten().tolist(),
            [True, True, False, False, False],
        )

    @unittest.skipUnless(HAS_READER_DEPS, "torch and numpy are required")
    def test_v21_can_supervise_episode_end_padding(self) -> None:
        ds = LeRobotV21Dataset.__new__(LeRobotV21Dataset)
        ds.chunk_size = 4
        ds.action_key = "action"
        ds.state_key = "observation.state"
        ds.supervise_episode_end_padding = True
        table = {
            ds.state_key: np.array([[0], [1]], dtype=np.float32),
            ds.action_key: np.array([[10], [20]], dtype=np.float32),
            f"{ds.action_key}_dim_mask": np.ones((2, 1), dtype=np.bool_),
        }

        action, is_pad, dim_mask = ds._build_action_chunk(table, t=1, length=2)

        self.assertEqual(action.flatten().tolist(), [20, 20, 20, 20])
        self.assertEqual(is_pad.tolist(), [False, False, False, False])
        self.assertEqual(dim_mask.flatten().tolist(), [True, True, True, True])

    @unittest.skipUnless(HAS_READER_DEPS, "torch and numpy are required")
    def test_v21_chunk_delta_is_relative_to_current_state(self) -> None:
        ds = LeRobotV21Dataset.__new__(LeRobotV21Dataset)
        ds.chunk_size = 3
        ds.action_upsample_rate = 1
        ds.action_representation = "chunk_delta"
        ds.action_delta_indices = (0,)
        ds.state_key = "observation.state"
        ds.action_key = "action"
        table = {
            ds.state_key: np.array([[10, 100], [11, 101], [12, 102]], dtype=np.float32),
            ds.action_key: np.array(
                [[12, 0.2], [15, 0.3], [19, 0.4]], dtype=np.float32
            ),
            f"{ds.state_key}_dim_mask": np.ones((3, 2), dtype=np.bool_),
            f"{ds.action_key}_dim_mask": np.ones((3, 2), dtype=np.bool_),
            "_action_chunk_quality_mask": np.array([True, False, True]),
        }

        chunk, is_pad, dim_mask = ds._build_action_chunk(table, t=0, length=3)

        self.assertTrue(np.allclose(chunk, [[2, 0.2], [5, 0.3], [9, 0.4]]))
        self.assertEqual(is_pad.tolist(), [False, True, False])
        self.assertEqual(
            dim_mask.tolist(), [[True, True], [False, False], [True, True]]
        )

    @unittest.skipUnless(HAS_READER_DEPS, "torch and numpy are required")
    def test_v21_delta_keeps_configured_gripper_absolute(self) -> None:
        from magic_w0.data.action_interpolation import ActionInterpolationSpec

        ds = LeRobotV21Dataset.__new__(LeRobotV21Dataset)
        ds.chunk_size = 3
        ds.action_upsample_rate = 1
        ds.action_representation = "chunk_delta"
        # Even an explicit delta list cannot convert a declared gripper channel.
        ds.action_delta_indices = (0, 1)
        ds.action_interpolation_spec = ActionInterpolationSpec(
            joint_indices=(0,),
            gripper_indices=(1,),
            eef_position_indices=(),
            rotation_groups=(),
        )
        ds.state_key = "observation.state"
        ds.action_key = "action"
        table = {
            ds.state_key: np.array([[10, 1], [11, 1], [12, 1]], dtype=np.float32),
            ds.action_key: np.array(
                [[12, 0.2], [15, 0.3], [19, 0.4]], dtype=np.float32
            ),
            f"{ds.state_key}_dim_mask": np.ones((3, 2), dtype=np.bool_),
            f"{ds.action_key}_dim_mask": np.ones((3, 2), dtype=np.bool_),
            "_action_chunk_quality_mask": np.ones(3, dtype=np.bool_),
        }

        chunk, _, _ = ds._build_action_chunk(table, t=0, length=3)

        self.assertTrue(np.allclose(chunk, [[2, 0.2], [5, 0.3], [9, 0.4]]))

    @unittest.skipUnless(HAS_READER_DEPS, "torch and numpy are required")
    def test_v21_frame_delta_runs_after_temporal_upsampling(self) -> None:
        ds = LeRobotV21Dataset.__new__(LeRobotV21Dataset)
        ds.chunk_size = 6
        ds.action_upsample_rate = 2
        ds.action_upsample_mode = "repeat"
        ds.action_representation = "frame_delta"
        ds.action_delta_indices = (0,)
        ds.state_key = "observation.state"
        ds.action_key = "action"
        table = {
            ds.state_key: np.array([[10], [13], [20]], dtype=np.float32),
            ds.action_key: np.array([[12], [17], [25]], dtype=np.float32),
            f"{ds.state_key}_dim_mask": np.ones((3, 1), dtype=np.bool_),
            f"{ds.action_key}_dim_mask": np.ones((3, 1), dtype=np.bool_),
            "_action_chunk_quality_mask": np.array([True, False, True]),
        }

        chunk, is_pad, dim_mask = ds._build_action_chunk(table, t=0, length=3)

        self.assertEqual(chunk.flatten().tolist(), [2, 0, 5, 0, 8, 0])
        self.assertEqual(is_pad.tolist(), [False, False, True, True, False, False])
        # A quality-invalid target does not invalidate the next frame's numeric
        # reference; the quality rule masks exactly the rows where it is false.
        self.assertEqual(
            dim_mask.flatten().tolist(), [True, True, False, False, True, True]
        )

    @unittest.skipUnless(HAS_READER_DEPS, "torch and numpy are required")
    def test_v21_frame_delta_runs_after_semantic_interpolation(self) -> None:
        from magic_w0.data.action_interpolation import ActionInterpolationSpec

        ds = LeRobotV21Dataset.__new__(LeRobotV21Dataset)
        ds.chunk_size = 6
        ds.action_upsample_rate = 2
        ds.action_upsample_mode = "semantic"
        # Dimension 0 is an EEF position, not a gripper.  Both interpolate
        # linearly, so the absolute curve is unchanged -- but grippers are now
        # held absolute under either delta representation, which would empty the
        # delta set and turn this into an interpolation test by accident.  What
        # it is for is the ORDER: frame_delta must run after interpolation, not
        # before.
        ds.action_interpolation_spec = ActionInterpolationSpec(
            joint_indices=(),
            gripper_indices=(),
            eef_position_indices=(0,),
            rotation_groups=(),
        )
        ds.action_representation = "frame_delta"
        ds.action_delta_indices = (0,)
        ds.state_key = "observation.state"
        ds.action_key = "action"
        table = {
            ds.state_key: np.array([[10], [13], [20]], dtype=np.float32),
            ds.action_key: np.array([[12], [17], [25]], dtype=np.float32),
            f"{ds.state_key}_dim_mask": np.ones((3, 1), dtype=np.bool_),
            f"{ds.action_key}_dim_mask": np.ones((3, 1), dtype=np.bool_),
            "_action_chunk_quality_mask": np.ones(3, dtype=np.bool_),
        }

        chunk, is_pad, dim_mask = ds._build_action_chunk(table, t=0, length=3)

        # Absolute semantic interpolation is [12, 14.5, 17, 21, 25, 25].
        self.assertEqual(chunk.flatten().tolist(), [2, 2.5, 2.5, 4, 4, 0])
        self.assertFalse(is_pad.any())
        self.assertTrue(dim_mask.all())

    @unittest.skipUnless(HAS_READER_DEPS, "torch and numpy are required")
    def test_v21_delta_stats_use_transformed_supervised_actions(self) -> None:
        ds = LeRobotV21Dataset.__new__(LeRobotV21Dataset)
        ds._cum = np.array([0, 2])
        ds._sample_indices = None
        ds._ep_indices = [5]
        ds._ep_lengths = [2]
        ds.chunk_size = 1
        ds.action_upsample_rate = 1
        ds.action_representation = "chunk_delta"
        ds.action_delta_indices = (0,)
        ds.action_stats_samples = 2
        ds.target_action_dim = 1
        ds.state_key = "observation.state"
        ds.action_key = "action"
        table = {
            ds.state_key: np.array([[10], [13]], dtype=np.float32),
            ds.action_key: np.array([[12], [15]], dtype=np.float32),
            f"{ds.state_key}_dim_mask": np.ones((2, 1), dtype=np.bool_),
            f"{ds.action_key}_dim_mask": np.ones((2, 1), dtype=np.bool_),
            "_action_chunk_quality_mask": np.ones(2, dtype=np.bool_),
        }
        ds._episode_table = lambda _ep: table

        stats = ds._estimate_action_stats()

        self.assertEqual(stats["mean"].tolist(), [2.0])
        self.assertEqual(stats["min"].tolist(), [2.0])
        self.assertEqual(stats["max"].tolist(), [2.0])
        self.assertEqual(ds.action_stats_summary["sampled_anchors"], 2)

    @unittest.skipUnless(HAS_READER_DEPS, "torch and numpy are required")
    def test_v30_history_rows_remain_inside_current_episode(self) -> None:
        ds = LeRobotV30Dataset.__new__(LeRobotV30Dataset)
        ds._ep_from = np.array([100], dtype=np.int64)
        ds._ep_to = np.array([106], dtype=np.int64)
        ds._ep_index = np.array([4], dtype=np.int64)
        ds._ep_data_file = [(0, 0)]
        ds._file_offset = {(0, 0): 90}
        ds._ep_tasks = [["place"]]
        ds.chunk_size = 2
        ds.obs_steps = 3
        ds.observation_delta_indices = (-4, -2, 0)
        ds.state_key = "observation.state"
        ds.action_key = "action"
        ds.camera_keys = ["cam"]
        ds.fps = 20.0
        table = {
            ds.state_key: np.arange(60, dtype=np.float32).reshape(30, 2),
            ds.action_key: np.arange(60, dtype=np.float32).reshape(30, 2),
        }
        ds._data_table = lambda _key: table
        ds._decode_frames = lambda _cam, _pos, frames: torch.stack(
            [torch.full((3, 2, 2), frame, dtype=torch.uint8) for frame in frames]
        )

        item = ds[103]  # file row 13, episode-local frame 3 -> history frames 0,1,3

        expected_rows = torch.from_numpy(table[ds.state_key][[10, 11, 13]])
        self.assertTrue(torch.equal(item[ds.state_key], expected_rows))
        self.assertEqual(item["cam"][:, 0, 0, 0].tolist(), [0, 1, 3])

    @unittest.skipUnless(HAS_READER_DEPS, "torch and numpy are required")
    def test_v30_can_supervise_episode_end_padding(self) -> None:
        ds = LeRobotV30Dataset.__new__(LeRobotV30Dataset)
        ds._ep_from = np.array([0], dtype=np.int64)
        ds._ep_to = np.array([2], dtype=np.int64)
        ds._ep_index = np.array([0], dtype=np.int64)
        ds._ep_data_file = [(0, 0)]
        ds._file_offset = {(0, 0): 0}
        ds._ep_tasks = [["hold"]]
        ds.chunk_size = 3
        ds.supervise_episode_end_padding = True
        ds.obs_steps = 1
        ds.observation_delta_indices = (0,)
        ds.state_key = "observation.state"
        ds.action_key = "action"
        ds.camera_keys = []
        ds.fps = 20.0
        table = {
            ds.state_key: np.array([[0], [1]], dtype=np.float32),
            ds.action_key: np.array([[10], [20]], dtype=np.float32),
        }
        ds._data_table = lambda _key: table

        item = ds[1]

        self.assertEqual(item[ds.action_key].flatten().tolist(), [20, 20, 20])
        self.assertEqual(item[f"{ds.action_key}_is_pad"].tolist(), [False] * 3)


if __name__ == "__main__":
    unittest.main()
