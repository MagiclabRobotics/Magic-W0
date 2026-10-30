"""Lightweight contracts for the shared-backbone MagicW0 variant."""

from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from typing import ClassVar
from unittest.mock import patch
from unittest import mock

import torch
from torch import nn

SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from magic_w0.models.action_backbone.modeling_magicw0_base import (
    QwenGatedDeltaNet,
    QwenHybridActionExpert,
    QwenHybridExpertLayer,
    QwenRMSNorm,
)
from magic_w0.models.action_backbone.processor_magicw0_base import (
    ACTION_DIM_MASK,
    ACTIONS,
    CAMERA_VALID,
    EMBODIMENT,
    STATE,
    STATE_DIM_MASK,
    TASK,
    TEXT,
)
from magic_w0.models.world_action import (
    MagicW0TrainingConfig,
    MagicW0TrainingDataProcessor,
    MagicW0TrainingPolicy,
    QwenHybrid2DWorldExpert,
    QwenHybrid3DWorldExpert,
)
from magic_w0.models.world_action.modeling_magicw0 import (
    _ACTION_WORLD_KEEP_BOTH,
    _ACTION_WORLD_MASK_2D,
    _ACTION_WORLD_MASK_3D,
    _build_action_query_key_mask,
    _build_query_stream_key_mask,
    _ignore_dinov3_teacher_checkpoint_keys,
    _scheduled_auxiliary_weight,
    _strip_dinov3_teacher_from_state_dict,
)
from magic_w0.models.world_action.processor_magicw0 import (
    FUTURE_2D_IMAGES,
    FUTURE_3D_IMAGES,
    MOTION_3D_TARGET,
    VIEW_VALID_MASK,
    WORLD_3D_VALID_MASK,
    _MagicW0TrainingDataset,
)
from magic_w0.models.world_action.world_teachers import (
    TRACK4WORLD_MOTION_DROPPED_ROWS,
    TRACK4WORLD_MOTION_RECOVERED_ROWS,
    TRACK4WORLD_MOTION_RETRY_ROWS,
    TRACK4WORLD_TENSORRT_RELEASED_GIB,
    WorldTeacherMixin,
    _resolve_magicw0_path,
    _track4world_constructor_kwargs,
)
from magic_w0.models.world_action.track4world_tensorrt import (
    Track4WorldTensorRT,
    _Correlation,
    _TensorRTWorkspace,
    _coords_grid,
)


def _tiny_config(**overrides) -> MagicW0TrainingConfig:
    values = {
        "dtype": "float32",
        "action_expert_hidden_size": 32,
        "action_expert_intermediate_size": 96,
        "action_expert_num_layers": 4,
        "full_attention_interval": 4,
        "full_attn_num_heads": 2,
        "full_attn_num_kv_heads": 1,
        "full_attn_head_dim": 16,
        "deltanet_num_key_heads": 2,
        "deltanet_num_value_heads": 2,
        "deltanet_key_head_dim": 16,
        "deltanet_value_head_dim": 16,
        "deltanet_conv_kernel": 4,
        "action_dim": 4,
        "state_dim": 4,
        "chunk_size": 4,
        "camera_keys": ("cam", "left"),
        "gen_2d_camera_keys": ("cam",),
        "gen_2d_grid_size": (2, 2),
        "gen_2d_num_query_tokens_per_view": 4,
        "gen_3d_grid_size": (2, 2),
        "gen_3d_num_query_tokens": 4,
        "gen_3d_camera_key": "cam",
        "use_fast_action_tokens": False,
        "cotrain_vlm_modalities": ("vlm_sft",),
        "train_action_expert_only": False,
        "knowledge_insulation": True,
    }
    values.update(overrides)
    return MagicW0TrainingConfig(**values)


def _teacher_policy() -> MagicW0TrainingPolicy:
    policy = MagicW0TrainingPolicy.__new__(MagicW0TrainingPolicy)
    nn.Module.__init__(policy)
    policy.config = _tiny_config(
        gen_3d_grid_size=(16, 16),
        gen_3d_num_query_tokens=256,
        track4world_autocast_dtype="float32",
    )
    return policy


class TensorRTCorrelationTests(unittest.TestCase):
    def test_sampling_grid_matches_correlation_dtype(self) -> None:
        first = torch.randn(1, 4, 32, 32, dtype=torch.float64)
        second = torch.randn(1, 4, 32, 32, dtype=torch.float64)
        coords = _coords_grid(1, 32, 32, torch.device("cpu"), torch.float32)

        output = _Correlation(first, second)(coords)

        self.assertEqual(output.dtype, torch.float64)
        self.assertEqual(output.shape, (1, 405, 32, 32))
        self.assertTrue(bool(torch.isfinite(output).all()))


class Track4WorldPathTests(unittest.TestCase):
    def test_relative_path_is_resolved_from_magicw0_repository(self) -> None:
        expected = Path(__file__).resolve().parents[2] / "third_party" / "Track4World"

        resolved = _resolve_magicw0_path("third_party/Track4World")

        self.assertEqual(resolved, expected.resolve())


class DINOInputScalingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.teacher = WorldTeacherMixin()
        self.teacher.config = types.SimpleNamespace(image_size=8)

    def test_uint8_scaling_does_not_depend_on_other_batch_images(self) -> None:
        dark = torch.ones((1, 1, 3, 8, 8), dtype=torch.uint8)
        bright = torch.full((1, 1, 3, 8, 8), 255, dtype=torch.uint8)

        alone = self.teacher._prepare_dino_pixel_values(dark)
        mixed = self.teacher._prepare_dino_pixel_values(
            torch.cat((dark, bright), dim=0)
        )[:1]

        self.assertTrue(torch.equal(alone, mixed))
        expected = (1.0 / 255.0 - 0.485) / 0.229
        self.assertAlmostEqual(float(alone[0, 0, 0, 0]), expected, places=6)

    def test_float_images_are_treated_as_already_scaled(self) -> None:
        images = torch.full((1, 1, 3, 8, 8), 0.5, dtype=torch.float32)

        pixels = self.teacher._prepare_dino_pixel_values(images)

        expected = (0.5 - 0.485) / 0.229
        self.assertAlmostEqual(float(pixels[0, 0, 0, 0]), expected, places=6)

    def test_non_uint8_integer_images_are_rejected(self) -> None:
        images = torch.zeros((1, 1, 3, 8, 8), dtype=torch.int16)

        with self.assertRaisesRegex(TypeError, "uint8"):
            self.teacher._prepare_dino_pixel_values(images)


class Track4WorldMotionFallbackTests(unittest.TestCase):
    def test_nonfinite_motion_row_is_retried_with_batch_size_one(self) -> None:
        class BatchSensitiveTeacher:
            def __init__(self) -> None:
                self.batch_sizes = []
                self.release_calls = 0

            def infer_pair(self, frame_pair, **kwargs):
                del kwargs
                self.batch_sizes.append(int(frame_pair.shape[0]))
                marker = frame_pair[:, 0, 0, 0, 0]
                geometry = marker[:, None, None].expand(-1, 324, 1024).clone()
                motion = marker[:, None, None, None].expand(-1, 256, 32, 32).clone()
                if frame_pair.shape[0] > 1:
                    motion[0].fill_(float("nan"))
                return {
                    "geometry_latent_t": geometry,
                    "motion_latent_t_to_target": motion,
                }

            def release_execution_memory(self) -> int:
                self.release_calls += 1
                return 3 * 2**30

        policy = _teacher_policy()
        teacher = BatchSensitiveTeacher()
        object.__setattr__(policy, "_track4world_teacher", teacher)
        current = torch.stack(
            [torch.full((3, 256, 256), value) for value in (10.0, 20.0)]
        )
        targets = policy._track4world_teacher_targets(
            current,
            current + 1.0,
            torch.ones(2, dtype=torch.bool),
            torch.device("cpu"),
        )

        self.assertEqual(teacher.batch_sizes, [2, 1])
        self.assertEqual(teacher.release_calls, 1)
        self.assertEqual(targets[TRACK4WORLD_TENSORRT_RELEASED_GIB], 3.0)
        self.assertTrue(torch.isfinite(targets[MOTION_3D_TARGET]).all())
        self.assertTrue(torch.all(targets[MOTION_3D_TARGET][0] == 10.0))
        self.assertTrue(torch.all(targets[MOTION_3D_TARGET][1] == 20.0))
        self.assertEqual(targets[TRACK4WORLD_MOTION_RETRY_ROWS], 1)
        self.assertEqual(targets[TRACK4WORLD_MOTION_RECOVERED_ROWS], 1)
        self.assertEqual(targets[TRACK4WORLD_MOTION_DROPPED_ROWS], 0)
        self.assertTrue(targets[WORLD_3D_VALID_MASK].all())

    def test_unrecoverable_motion_row_is_zeroed_and_masked(self) -> None:
        class AlwaysBadForMarkerTeacher:
            def __init__(self) -> None:
                self.batch_sizes = []

            def infer_pair(self, frame_pair, **kwargs):
                del kwargs
                self.batch_sizes.append(int(frame_pair.shape[0]))
                marker = frame_pair[:, 0, 0, 0, 0]
                geometry = marker[:, None, None].expand(-1, 324, 1024).clone()
                motion = marker[:, None, None, None].expand(-1, 256, 32, 32).clone()
                motion[marker == 10.0] = float("nan")
                return {
                    "geometry_latent_t": geometry,
                    "motion_latent_t_to_target": motion,
                }

        policy = _teacher_policy()
        teacher = AlwaysBadForMarkerTeacher()
        object.__setattr__(policy, "_track4world_teacher", teacher)
        current = torch.stack(
            [torch.full((3, 256, 256), value) for value in (10.0, 20.0)]
        )
        targets = policy._track4world_teacher_targets(
            current,
            current + 1.0,
            torch.ones(2, dtype=torch.bool),
            torch.device("cpu"),
        )

        self.assertEqual(teacher.batch_sizes, [2, 1])
        self.assertTrue(torch.isfinite(targets[MOTION_3D_TARGET]).all())
        self.assertEqual(int(torch.count_nonzero(targets[MOTION_3D_TARGET][0])), 0)
        self.assertTrue(torch.all(targets[MOTION_3D_TARGET][1] == 20.0))
        self.assertEqual(targets[TRACK4WORLD_MOTION_RETRY_ROWS], 1)
        self.assertEqual(targets[TRACK4WORLD_MOTION_RECOVERED_ROWS], 0)
        self.assertEqual(targets[TRACK4WORLD_MOTION_DROPPED_ROWS], 1)
        self.assertEqual(targets[WORLD_3D_VALID_MASK].tolist(), [False, True])


class Track4WorldTensorRTLifecycleTests(unittest.TestCase):
    def test_constructor_rejects_unexpected_engine_batches(self) -> None:
        batches = {
            "da3_anyview": 24,
            "da3_metric": 48,
            "flow_feature": 48,
            "motion_iteration": 48,
        }

        class FakeEngine:
            def __init__(self, path: Path) -> None:
                batch_size = batches[path.stem]
                self.inputs = {"input": ((batch_size, 1), torch.float32)}
                self.outputs = {"output": ((batch_size, 1), torch.float32)}
                self._engine = types.SimpleNamespace(device_memory_size_v2=1)

            @property
            def batch_size(self) -> int:
                return next(iter(self.inputs.values()))[0][0]

        expected = dict(batches, da3_metric=24)
        with (
            patch(
                "magic_w0.models.world_action.track4world_tensorrt._TensorRTEngine",
                FakeEngine,
            ),
            self.assertRaisesRegex(ValueError, "actual=.*expected="),
        ):
            Track4WorldTensorRT(
                "/engines",
                "/track4world",
                expected_engine_batch_sizes=expected,
            )

    def test_run_padded_chunks_batch_72_for_fixed_batch_engines(self) -> None:
        class FakeEngine:
            def __init__(self, batch_size: int) -> None:
                self.batch_size = batch_size
                self.inputs = {
                    "samples": ((batch_size, 1), torch.float32),
                    "paired": ((2 * batch_size, 1), torch.float32),
                }
                self.outputs = {"result": ((2 * batch_size, 1), torch.float32)}
                self.calls = 0

            def __call__(self, **inputs: torch.Tensor) -> dict[str, torch.Tensor]:
                self.calls += 1
                self.assert_shapes(inputs)
                return {
                    "result": inputs["paired"]
                    + inputs["samples"].repeat_interleave(2, 0)
                }

            def assert_shapes(self, inputs: dict[str, torch.Tensor]) -> None:
                if inputs["samples"].shape != (self.batch_size, 1):
                    raise AssertionError(inputs["samples"].shape)
                if inputs["paired"].shape != (2 * self.batch_size, 1):
                    raise AssertionError(inputs["paired"].shape)

        teacher = Track4WorldTensorRT.__new__(Track4WorldTensorRT)
        nn.Module.__init__(teacher)
        samples = torch.arange(72, dtype=torch.float32).unsqueeze(1)
        paired = torch.arange(144, dtype=torch.float32).unsqueeze(1)

        for engine_batch, expected_calls in ((24, 3), (48, 2)):
            with self.subTest(engine_batch=engine_batch):
                engine = FakeEngine(engine_batch)
                outputs = teacher._run_padded(engine, samples=samples, paired=paired)

                self.assertEqual(engine.calls, expected_calls)
                self.assertEqual(outputs["result"].shape, (144, 1))
                self.assertTrue(
                    torch.equal(
                        outputs["result"],
                        paired + samples.repeat_interleave(2, 0),
                    )
                )

    def test_release_execution_memory_is_idempotent(self) -> None:
        class FakeEngine:
            def __init__(self) -> None:
                self.release_calls = 0

            def release_context(self) -> None:
                self.release_calls += 1

        teacher = Track4WorldTensorRT.__new__(Track4WorldTensorRT)
        nn.Module.__init__(teacher)
        teacher.keep_execution_memory = False
        teacher.workspace = _TensorRTWorkspace(3 * 2**30)
        teacher.workspace._allocation = torch.empty(1)
        teacher.workspace._device = torch.device("cpu")
        engines = [FakeEngine() for _ in range(4)]
        (
            teacher.da3_anyview,
            teacher.da3_metric,
            teacher.flow_feature,
            teacher.motion_iteration,
        ) = engines
        stream = types.SimpleNamespace(synchronize=lambda: None)

        with patch("torch.cuda.current_stream", return_value=stream) as current_stream:
            self.assertEqual(teacher.release_execution_memory(), 3 * 2**30)
            self.assertEqual(teacher.release_execution_memory(), 0)

        current_stream.assert_called_once_with(torch.device("cpu"))
        self.assertTrue(all(engine.release_calls == 1 for engine in engines))
        self.assertIsNone(teacher.workspace._allocation)
        self.assertIsNone(teacher.workspace._device)

    def test_release_is_skipped_when_execution_memory_is_persistent(self) -> None:
        teacher = Track4WorldTensorRT.__new__(Track4WorldTensorRT)
        nn.Module.__init__(teacher)
        teacher.keep_execution_memory = True
        teacher.workspace = _TensorRTWorkspace(3 * 2**30)
        allocation = torch.empty(1)
        teacher.workspace._allocation = allocation

        self.assertEqual(teacher.release_execution_memory(), 0)
        self.assertIs(teacher.workspace._allocation, allocation)


class MagicW0ConfigTests(unittest.TestCase):
    def test_production_defaults_select_scheme_b_and_new_action_space(self) -> None:
        config = MagicW0TrainingConfig()
        self.assertEqual(
            (config.state_dim, config.action_dim, config.chunk_size), (32, 32, 50)
        )
        self.assertTrue(config.knowledge_insulation)
        self.assertEqual(config.vlm_modalities, frozenset({"vlm_sft"}))
        self.assertTrue(config.use_fast_action_tokens)
        self.assertTrue(config.enable_world_teacher_supervision)
        self.assertEqual(config.dinov3_target_hidden_size, 1024)
        self.assertIsNone(config.gen_2d_future_frame_offset)
        self.assertIsNone(config.gen_3d_future_frame_offset)
        self.assertFalse(config.action_world_stream_mask.enabled)
        all_sources = ("vlm", "world_2d", "world_3d", "action")
        self.assertEqual(config.joint_attention_visibility.world_2d, all_sources)
        self.assertEqual(config.joint_attention_visibility.world_3d, all_sources)
        self.assertEqual(config.joint_attention_visibility.action, all_sources)

    def test_joint_attention_visibility_parses_per_query_rows(self) -> None:
        config = MagicW0TrainingConfig.from_dict(
            {
                "joint_attention_visibility": {
                    "world_2d": ["vlm", "world_2d"],
                    "world_3d": ["vlm", "world_2d", "world_3d"],
                    "action": ["vlm", "world_3d", "action"],
                }
            }
        )

        self.assertEqual(
            config.joint_attention_visibility.world_2d,
            ("vlm", "world_2d"),
        )
        self.assertEqual(
            config.joint_attention_visibility.world_3d,
            ("vlm", "world_2d", "world_3d"),
        )
        self.assertEqual(
            config.joint_attention_visibility.action,
            ("vlm", "world_3d", "action"),
        )

    def test_joint_attention_visibility_rejects_unknown_source(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown K/V sources"):
            MagicW0TrainingConfig.from_dict(
                {
                    "joint_attention_visibility": {
                        "action": ["vlm", "unknown_expert"],
                    }
                }
            )

    def test_joint_attention_visibility_requires_an_enabled_source(self) -> None:
        with self.assertRaisesRegex(ValueError, "has no enabled K/V source"):
            _tiny_config(
                enable_3d_world_expert=False,
                joint_attention_visibility={"action": ["world_3d"]},
            )

    def test_action_dropout_cannot_remove_its_last_visible_source(self) -> None:
        with self.assertRaisesRegex(ValueError, "last enabled Action K/V source"):
            _tiny_config(
                joint_attention_visibility={"action": ["world_2d"]},
                action_world_stream_mask={
                    "enabled": True,
                    "keep_both_prob": 0.8,
                    "mask_2d_prob": 0.1,
                    "mask_3d_prob": 0.1,
                },
            )

    def test_action_world_stream_mask_parses_nested_yaml_mapping(self) -> None:
        config = MagicW0TrainingConfig.from_dict(
            {
                "action_world_stream_mask": {
                    "enabled": True,
                    "keep_both_prob": 0.8,
                    "mask_2d_prob": 0.1,
                    "mask_3d_prob": 0.1,
                }
            }
        )
        self.assertTrue(config.action_world_stream_mask.enabled)
        self.assertEqual(config.action_world_stream_mask.keep_both_prob, 0.8)
        self.assertEqual(config.action_world_stream_mask.mask_2d_prob, 0.1)
        self.assertEqual(config.action_world_stream_mask.mask_3d_prob, 0.1)

    def test_action_world_stream_mask_probabilities_must_sum_to_one(self) -> None:
        with self.assertRaisesRegex(ValueError, "must sum to 1.0"):
            MagicW0TrainingConfig.from_dict(
                {
                    "action_world_stream_mask": {
                        "enabled": True,
                        "keep_both_prob": 0.8,
                        "mask_2d_prob": 0.2,
                        "mask_3d_prob": 0.1,
                    }
                }
            )
        self.assertEqual(_tiny_config().track4world_inference_backend, "pytorch")

    def test_tensorrt_teacher_backend_is_supported(self) -> None:
        config = _tiny_config(
            track4world_inference_backend="tensorrt",
            track4world_tensorrt_engine_dir="/models/track4world-engines",
        )

        self.assertEqual(config.track4world_inference_backend, "tensorrt")
        self.assertEqual(
            config.track4world_tensorrt_engine_dir,
            "/models/track4world-engines",
        )

    def test_invalid_teacher_backend_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "track4world_inference_backend"):
            _tiny_config(track4world_inference_backend="onnxruntime")

    def test_ambiguous_dropout_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "ambiguous 'dropout'"):
            MagicW0TrainingConfig.from_dict({"dropout": 0.1})

    def test_enabled_loss_schedule_rejects_minimum_above_maximum(self) -> None:
        with self.assertRaisesRegex(ValueError, "fast_ce_loss_weight_min"):
            _tiny_config(
                loss_weight_schedule_enabled=True,
                fast_ce_loss_weight=0.01,
                fast_ce_loss_weight_min=0.02,
            )

    def test_loss_schedule_minima_must_be_non_negative(self) -> None:
        with self.assertRaisesRegex(ValueError, "gen_2d_loss_weight_min"):
            _tiny_config(gen_2d_loss_weight_min=-0.01)


class MagicW0LossWeightScheduleTests(unittest.TestCase):
    @staticmethod
    def _policy(**overrides) -> MagicW0TrainingPolicy:
        policy = MagicW0TrainingPolicy.__new__(MagicW0TrainingPolicy)
        nn.Module.__init__(policy)
        policy.config = _tiny_config(**overrides)
        return policy

    def test_schedule_holds_decays_and_holds_at_configured_fractions(self) -> None:
        policy = self._policy(
            loss_weight_schedule_enabled=True,
            flow_matching_loss_weight=1.5,
            fast_ce_loss_weight=0.02,
            fast_ce_loss_weight_min=0.005,
            gen_2d_loss_weight=0.2,
            gen_2d_loss_weight_min=0.05,
            geometry_latent_weight=1.0,
            motion_latent_weight=0.75,
            world_3d_loss_weight=0.2,
            world_3d_loss_weight_min=0.05,
        )

        policy.set_training_progress(global_step=200, max_steps=1000)
        at_twenty_percent = policy._active_loss_weights()
        self.assertEqual(at_twenty_percent.fast_ce, 0.02)
        self.assertEqual(at_twenty_percent.world_2d, 0.2)
        self.assertEqual(at_twenty_percent.world_3d, 0.2)

        # 50% is halfway through the [20%, 80%] decay interval, so cosine
        # interpolation is exactly the arithmetic midpoint.
        policy.set_training_progress(global_step=500, max_steps=1000)
        midpoint = policy._active_loss_weights()
        self.assertAlmostEqual(midpoint.fast_ce, 0.0125)
        self.assertAlmostEqual(midpoint.world_2d, 0.125)
        self.assertAlmostEqual(midpoint.world_3d, 0.125)
        self.assertAlmostEqual(midpoint.geometry_effective, 0.125)
        self.assertAlmostEqual(midpoint.motion_effective, 0.09375)
        self.assertEqual(midpoint.flow, 1.5)
        weight_metrics = midpoint.metrics()
        expected_metrics = {
            "flow_loss_weight": 1.5,
            "fast_ce_loss_weight": 0.0125,
            "gen_2d_loss_weight": 0.125,
            "geometry_effective_weight": 0.125,
            "motion_effective_weight": 0.09375,
            "world_3d_loss_weight": 0.125,
        }
        self.assertEqual(weight_metrics.keys(), expected_metrics.keys())
        for name, expected in expected_metrics.items():
            self.assertAlmostEqual(weight_metrics[name], expected)

        policy.set_training_progress(global_step=800, max_steps=1000)
        at_eighty_percent = policy._active_loss_weights()
        self.assertEqual(at_eighty_percent.fast_ce, 0.005)
        self.assertEqual(at_eighty_percent.world_2d, 0.05)
        self.assertEqual(at_eighty_percent.world_3d, 0.05)

        policy.set_training_progress(global_step=1000, max_steps=1000)
        self.assertEqual(policy._active_loss_weights(), at_eighty_percent)

    def test_disabled_schedule_preserves_static_weights(self) -> None:
        policy = self._policy(
            loss_weight_schedule_enabled=False,
            flow_matching_loss_weight=1.25,
            fast_ce_loss_weight=0.03,
            fast_ce_loss_weight_min=0.001,
            gen_2d_loss_weight=0.4,
            gen_2d_loss_weight_min=0.002,
            geometry_latent_weight=0.5,
            motion_latent_weight=1.0,
            world_3d_loss_weight=0.3,
            world_3d_loss_weight_min=0.003,
        )
        policy.set_training_progress(global_step=999, max_steps=None)
        weights = policy._active_loss_weights()
        self.assertEqual(weights.flow, 1.25)
        self.assertEqual(weights.fast_ce, 0.03)
        self.assertEqual(weights.world_2d, 0.4)
        self.assertEqual(weights.world_3d, 0.3)
        self.assertEqual(weights.geometry_effective, 0.15)
        self.assertEqual(weights.motion_effective, 0.3)

    def test_enabled_schedule_requires_max_steps_and_training_progress(self) -> None:
        policy = self._policy(loss_weight_schedule_enabled=True)
        with self.assertRaisesRegex(ValueError, "requires training.max_steps"):
            policy.set_training_progress(global_step=0, max_steps=None)
        with self.assertRaisesRegex(RuntimeError, "no training progress"):
            policy._active_loss_weights()

    def test_schedule_clamps_steps_outside_training_interval(self) -> None:
        self.assertEqual(
            _scheduled_auxiliary_weight(0.2, 0.05, global_step=-10, max_steps=100),
            0.2,
        )
        self.assertEqual(
            _scheduled_auxiliary_weight(0.2, 0.05, global_step=110, max_steps=100),
            0.05,
        )


class MagicW0RuntimeConfigTests(unittest.TestCase):
    def test_startup_kernel_report_covers_vlm_and_all_experts(self) -> None:
        config = _tiny_config(attention_implementation="flash_attention_2")
        policy = MagicW0TrainingPolicy.__new__(MagicW0TrainingPolicy)
        nn.Module.__init__(policy)
        policy.config = config
        policy.vlm_layer_types = (
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
        )
        layers = [
            types.SimpleNamespace(linear_attn=QwenGatedDeltaNet(config))
            for _ in range(3)
        ]
        layers.append(types.SimpleNamespace(self_attn=nn.Identity()))
        text_config = types.SimpleNamespace(_attn_implementation="flash_attention_2")
        policy.vlm = types.SimpleNamespace(
            config=types.SimpleNamespace(text_config=text_config),
            model=types.SimpleNamespace(
                language_model=types.SimpleNamespace(layers=layers)
            ),
        )
        policy.action_expert = QwenHybridActionExpert(config)
        policy.world_2d_expert = QwenHybrid2DWorldExpert(config, dino_hidden_size=12)
        policy.world_3d_expert = QwenHybrid3DWorldExpert(config)

        with self.assertLogs(
            "magic_w0.models.world_action.modeling_magicw0", level="WARNING"
        ) as captured:
            policy._log_kernel_backends()

        report = "\n".join(captured.output)
        for component in ("Qwen3.5-VLM", "Action-Expert", "2D-Expert", "3D-Expert"):
            self.assertIn(f"component={component}", report)
        self.assertIn("full_attention_resolved=flash_attention_2", report)
        self.assertIn("DeltaNet_resolved=torch_parallel", report)

    def test_track4world_runtime_paths_override_external_json(self) -> None:
        kwargs = _track4world_constructor_kwargs(
            {
                "model": {
                    "encoder": "dinov2_vitl14",
                    "depthanythingv3_model_path": "/old/server/da3",
                    "depthanythingv3_local_files_only": False,
                }
            },
            backbone_path=Path("/new/server/da3"),
            local_files_only=True,
            convnext_path=Path("/new/server/convnext.pth"),
            motion_chunk_size=16,
        )
        self.assertEqual(kwargs["depthanythingv3_model_path"], "/new/server/da3")
        self.assertTrue(kwargs["depthanythingv3_local_files_only"])
        self.assertEqual(
            kwargs["convnext_tiny_weights_path"], "/new/server/convnext.pth"
        )
        self.assertEqual(kwargs["seqlen"], 16)
        self.assertTrue(kwargs["use_3d"])
        self.assertEqual(kwargs["motion_chunk_size"], 16)

    def test_track4world_motion_chunk_size_must_be_positive(self) -> None:
        with self.assertRaisesRegex(ValueError, "motion-chunk"):
            _tiny_config(track4world_motion_chunk_size=0)


class MagicW0ProcessorTests(unittest.TestCase):
    class RawDataset:
        camera_keys: ClassVar[tuple[str, ...]] = ("cam",)
        state_key = "observation.state"
        action_key = "action"
        meta_stats: ClassVar[dict] = {state_key: {}, action_key: {}}
        chunk_size = 4
        action_upsample_rate = 2
        supervise_episode_end_padding = False
        _cum: ClassVar[tuple[int, ...]] = (0, 4)

        def __len__(self):
            return 4

        def __getitem__(self, index):
            return {
                self.state_key: torch.tensor([1.0, 2.0, 3.0, 4.0]),
                f"{self.state_key}_dim_mask": torch.tensor(
                    [1, 0, 1, 0], dtype=torch.bool
                ),
                self.action_key: torch.arange(16, dtype=torch.float32).reshape(4, 4),
                f"{self.action_key}_dim_mask": torch.tensor(
                    [[1, 1, 0, 0]] * 4, dtype=torch.bool
                ),
                f"{self.action_key}_is_pad": torch.zeros(4, dtype=torch.bool),
                "task": "pick",
                "embodiment": "arx_x5",
                "episode_index": torch.tensor(0),
                "frame_index": torch.tensor(index),
                "cam": torch.full((3, 8, 8), index, dtype=torch.uint8),
            }

        def get_episode_frame_images(self, episode, frame, camera_keys):
            return {
                key: torch.full((3, 8, 8), frame, dtype=torch.uint8)
                for key in camera_keys
            }

        def get_episode_length(self, episode):
            return 4

    def test_robotics_wrapper_preserves_base_masks_and_source_horizon(self) -> None:
        config = _tiny_config(normalization="none")
        dataset = _MagicW0TrainingDataset(self.RawDataset(), config)
        # By default ceil(chunk=4 / upsample=2) == 2 future source frames are
        # required, so the final two anchors are filtered out.
        self.assertEqual(len(dataset), 2)
        item = dataset[0]
        self.assertEqual(item[STATE].tolist(), [1.0, 0.0, 3.0, 0.0])
        self.assertEqual(item[ACTION_DIM_MASK].shape, item[ACTIONS].shape)
        self.assertEqual(item[STATE_DIM_MASK].tolist(), [True, False, True, False])
        self.assertEqual(item[CAMERA_VALID].tolist(), [True, False])
        self.assertEqual(item[EMBODIMENT], "arx_x5")
        self.assertEqual(item[VIEW_VALID_MASK].tolist(), [True])
        self.assertTrue(item[WORLD_3D_VALID_MASK])
        self.assertEqual(item[FUTURE_2D_IMAGES].shape, (1, 3, 256, 256))
        self.assertEqual(item[FUTURE_3D_IMAGES].shape, (3, 256, 256))

    def _counting_dataset(self):
        """RawDataset that records every future-frame fetch it is asked for."""
        calls: list[tuple[int, int, tuple[str, ...]]] = []
        outer = self

        class CountingDataset(outer.RawDataset):
            def get_episode_frame_images(self, episode, frame, camera_keys):
                calls.append((int(episode), int(frame), tuple(camera_keys)))
                return super().get_episode_frame_images(episode, frame, camera_keys)

        return CountingDataset(), calls

    def test_shared_future_offset_decodes_the_frame_once(self) -> None:
        """2D and 3D default to the same camera and horizon -- fetch it once.

        Both branches resolve their offset through _resolve_future_offset and both
        point at gen_3d_camera_key by default, so the un-deduped version issued
        two identical get_episode_frame_images calls per sample, each a fresh
        av.open plus a seek, in 256 dataloader workers per node.
        """
        source, calls = self._counting_dataset()
        dataset = _MagicW0TrainingDataset(source, _tiny_config(normalization="none"))
        item = dataset[0]

        self.assertEqual(len(calls), 1, f"expected one future fetch, got {calls}")
        episode, frame, camera_keys = calls[0]
        self.assertEqual((episode, frame), (0, 2))
        self.assertEqual(camera_keys, ("cam",))
        # Both branches still see the future frame, not a zero-filled placeholder.
        self.assertTrue(item[VIEW_VALID_MASK].all())
        self.assertTrue(item[WORLD_3D_VALID_MASK])
        self.assertGreater(float(item[FUTURE_2D_IMAGES].sum()), 0.0)
        self.assertGreater(float(item[FUTURE_3D_IMAGES].sum()), 0.0)

    def test_distinct_future_offsets_are_still_fetched_separately(self) -> None:
        source, calls = self._counting_dataset()
        config = _tiny_config(
            normalization="none",
            gen_2d_future_frame_offset=1,
            gen_3d_future_frame_offset=2,
        )
        _MagicW0TrainingDataset(source, config)[0]

        self.assertEqual(sorted(frame for _, frame, _ in calls), [1, 2])

    def test_episode_tail_anchor_is_filtered_by_default(self) -> None:
        dataset = _MagicW0TrainingDataset(self.RawDataset(), _tiny_config())
        self.assertEqual(dataset.indices.tolist(), [0, 1])
        self.assertEqual(len(dataset), 2)

    def test_episode_tail_anchor_reuses_terminal_world_frame_when_enabled(self) -> None:
        source, calls = self._counting_dataset()
        source.supervise_episode_end_padding = True
        dataset = _MagicW0TrainingDataset(source, _tiny_config(normalization="none"))
        self.assertIsNone(dataset.indices)
        self.assertEqual(len(dataset), 4)

        item = dataset[3]

        self.assertEqual(calls, [(0, 3, ("cam",))])
        self.assertTrue(item[VIEW_VALID_MASK].all())
        self.assertTrue(item[WORLD_3D_VALID_MASK])
        self.assertGreater(float(item[FUTURE_2D_IMAGES].sum()), 0.0)
        self.assertGreater(float(item[FUTURE_3D_IMAGES].sum()), 0.0)

    def test_vlm_source_uses_base_passthrough(self) -> None:
        class VlmDataset:
            def __len__(self):
                return 1

            def __getitem__(self, index):
                return {"text": "Describe the scene."}

        processor = MagicW0TrainingDataProcessor(_tiny_config())
        transformed = processor.transform_dataset(
            VlmDataset(), types.SimpleNamespace(modality="vlm_sft")
        )
        self.assertEqual(transformed[0], {TEXT: "Describe the scene."})


class MagicW0ActionWorldStreamMaskTests(unittest.TestCase):
    @staticmethod
    def _policy(**overrides) -> MagicW0TrainingPolicy:
        policy = MagicW0TrainingPolicy.__new__(MagicW0TrainingPolicy)
        nn.Module.__init__(policy)
        policy.config = _tiny_config(
            action_world_stream_mask={
                "enabled": True,
                "keep_both_prob": 0.8,
                "mask_2d_prob": 0.1,
                "mask_3d_prob": 0.1,
            },
            **overrides,
        )
        policy.dinov3_teacher = None
        return policy

    def test_action_key_mask_removes_only_the_selected_world_block(self) -> None:
        prefix = torch.tensor([[True, True]])
        masks = [
            torch.tensor([[True, False]]),
            torch.tensor([[True, True]]),
            torch.tensor([[True, False]]),
        ]
        names = ("world_2d", "world_3d", "action")

        mask_2d = _build_action_query_key_mask(
            prefix, masks, names, _ACTION_WORLD_MASK_2D
        )
        mask_3d = _build_action_query_key_mask(
            prefix, masks, names, _ACTION_WORLD_MASK_3D
        )

        self.assertEqual(
            mask_2d.tolist(),
            [[True, True, False, False, True, True, True, False]],
        )
        self.assertEqual(
            mask_3d.tolist(),
            [[True, True, True, False, False, False, True, False]],
        )
        # Building an Action-only view must not mutate the World masks used by
        # 2D/3D queries or by their supervision losses.
        self.assertEqual(masks[0].tolist(), [[True, False]])
        self.assertEqual(masks[1].tolist(), [[True, True]])

    def test_query_key_mask_applies_an_arbitrary_visibility_row(self) -> None:
        prefix = torch.tensor([[True, True]])
        masks = [
            torch.tensor([[True, False]]),
            torch.tensor([[True, True]]),
            torch.tensor([[True, False]]),
        ]
        result = _build_query_stream_key_mask(
            prefix,
            masks,
            ("world_2d", "world_3d", "action"),
            ("vlm", "world_3d"),
        )

        self.assertEqual(
            result.tolist(),
            [[True, True, False, False, True, True, False, False]],
        )

    def test_sampling_is_training_only_and_requires_both_world_streams(self) -> None:
        policy = self._policy()
        policy.train()
        with patch(
            "magic_w0.models.world_action.modeling_magicw0.torch.rand",
            return_value=torch.tensor(0.50),
        ):
            self.assertEqual(
                policy._sample_action_world_stream_mask_mode(),
                _ACTION_WORLD_KEEP_BOTH,
            )

        with patch(
            "magic_w0.models.world_action.modeling_magicw0.torch.rand",
            return_value=torch.tensor(0.85),
        ):
            self.assertEqual(
                policy._sample_action_world_stream_mask_mode(),
                _ACTION_WORLD_MASK_2D,
            )

        with patch(
            "magic_w0.models.world_action.modeling_magicw0.torch.rand",
            return_value=torch.tensor(0.95),
        ):
            self.assertEqual(
                policy._sample_action_world_stream_mask_mode(),
                _ACTION_WORLD_MASK_3D,
            )

        policy.eval()
        with patch(
            "magic_w0.models.world_action.modeling_magicw0.torch.rand"
        ) as random_draw:
            self.assertEqual(
                policy._sample_action_world_stream_mask_mode(),
                _ACTION_WORLD_KEEP_BOTH,
            )
            random_draw.assert_not_called()

        for enabled_2d, enabled_3d in ((True, False), (False, True), (False, False)):
            with self.subTest(enabled_2d=enabled_2d, enabled_3d=enabled_3d):
                policy = self._policy(
                    enable_2d_world_expert=enabled_2d,
                    enable_3d_world_expert=enabled_3d,
                )
                policy.train()
                with patch(
                    "magic_w0.models.world_action.modeling_magicw0.torch.rand"
                ) as random_draw:
                    self.assertEqual(
                        policy._sample_action_world_stream_mask_mode(),
                        _ACTION_WORLD_KEEP_BOTH,
                    )
                    random_draw.assert_not_called()

    def test_mask_log_record_counts_only_samples_with_valid_actions(self) -> None:
        policy = self._policy()
        policy.train()
        valid_elements = torch.tensor(
            [
                [[True, False], [False, False]],
                [[False, False], [False, False]],
                [[False, True], [True, False]],
            ]
        )

        record = policy._action_world_stream_mask_log_record(
            _ACTION_WORLD_MASK_2D,
            valid_elements,
        )

        self.assertEqual(record, {"mode": "mask_2d", "sample_count": 2})
        policy.eval()
        self.assertIsNone(
            policy._action_world_stream_mask_log_record(
                _ACTION_WORLD_MASK_2D,
                valid_elements,
            )
        )

    def test_mask_config_adds_no_checkpoint_state(self) -> None:
        class TinyTopology(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.action = QwenHybridActionExpert(config)
                self.world_2d = QwenHybrid2DWorldExpert(config, dino_hidden_size=12)
                self.world_3d = QwenHybrid3DWorldExpert(config)

        old_model = TinyTopology(_tiny_config())
        enabled_model = TinyTopology(
            _tiny_config(
                action_world_stream_mask={
                    "enabled": True,
                    "keep_both_prob": 0.8,
                    "mask_2d_prob": 0.1,
                    "mask_3d_prob": 0.1,
                }
            )
        )
        old_state = old_model.state_dict()
        self.assertEqual(old_state.keys(), enabled_model.state_dict().keys())
        enabled_model.load_state_dict(old_state, strict=True)

    def test_startup_log_reports_probabilities_once_when_called(self) -> None:
        policy = self._policy()
        with self.assertLogs(
            "magic_w0.models.world_action.modeling_magicw0", level="WARNING"
        ) as captured:
            policy._log_action_world_stream_mask_config()
        report = "\n".join(captured.output)
        self.assertIn("configured_enabled=True", report)
        self.assertIn("effective=True", report)
        self.assertIn("keep_both_prob=0.800000", report)
        self.assertIn("mask_2d_prob=0.100000", report)
        self.assertIn("mask_3d_prob=0.100000", report)
        self.assertIn("ddp_broadcast=no", report)

    def test_each_query_uses_its_configured_visibility_row(self) -> None:
        config = _tiny_config(
            joint_attention_visibility={
                "world_2d": ["vlm", "world_2d", "world_3d", "action"],
                "world_3d": ["vlm", "world_3d"],
                "action": ["vlm", "world_2d", "world_3d", "action"],
            }
        )
        policy = MagicW0TrainingPolicy.__new__(MagicW0TrainingPolicy)
        nn.Module.__init__(policy)
        policy.config = config
        policy._log_expert_full_attention_backend_once = types.MethodType(
            lambda _self, *args, **kwargs: None, policy
        )

        stream_names = ("world_2d", "world_3d", "action")
        blocks = [
            QwenHybridExpertLayer(config, use_full_attention=True) for _ in stream_names
        ]
        first_mixer = blocks[0].mixer
        prepared_masks: list[torch.Tensor] = []
        attended_masks: list[torch.Tensor] = []

        def fake_prepare(_self, key, value, key_mask):
            del key, value
            saved = key_mask.clone()
            prepared_masks.append(saved)
            return ("captured", saved)

        def fake_attend(_self, query, prepared_kv, *, dropout_p):
            del dropout_p
            attended_masks.append(prepared_kv[1])
            return torch.zeros_like(query)

        first_mixer.prepare_attention_kv = types.MethodType(fake_prepare, first_mixer)
        first_mixer.attend_prepared = types.MethodType(fake_attend, first_mixer)

        class FakeAttention(nn.Module):
            def __init__(self):
                super().__init__()
                self.head_dim = 16
                self.k_proj = nn.Linear(32, 16, bias=False)
                self.v_proj = nn.Linear(32, 16, bias=False)
                self.k_norm = QwenRMSNorm(16, 1e-6)

        class FakeVlmLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.input_layernorm = QwenRMSNorm(32, 1e-6)
                self.self_attn = FakeAttention()

            def forward(self, hidden, **kwargs):
                return hidden

        prefix_mask = torch.tensor([[True, True]])
        stream_masks = [
            torch.tensor([[True, True]]),
            torch.tensor([[True, True]]),
            torch.tensor([[True, False]]),
        ]
        vlm = torch.randn(1, 2, 32)
        streams = [torch.randn(1, 2, 32) for _ in stream_names]
        stream_rope = [
            (torch.ones(1, 2, 16), torch.zeros(1, 2, 16)) for _ in stream_names
        ]
        vlm_rope = (torch.ones(1, 2, 16), torch.zeros(1, 2, 16))

        policy._forward_joint_layer(
            vlm,
            streams,
            stream_masks,
            stream_rope,
            blocks,
            [nn.Dropout(0.0) for _ in stream_names],
            FakeVlmLayer(),
            vlm_rope,
            prefix_mask,
            None,
            torch.zeros(3, 1, 2, dtype=torch.long),
            2,
            prefix_mask,
            vlm_rope,
            stream_names=stream_names,
            action_world_stream_mask_mode=_ACTION_WORLD_MASK_2D,
        )

        original = torch.tensor([[True, True, True, True, True, True, True, False]])
        action_only = torch.tensor(
            [[True, True, False, False, True, True, True, False]]
        )
        world_3d_only = torch.tensor(
            [[True, True, False, False, True, True, False, False]]
        )
        self.assertEqual(len(prepared_masks), 3)
        self.assertTrue(torch.equal(prepared_masks[0], original))
        self.assertTrue(torch.equal(prepared_masks[1], world_3d_only))
        self.assertTrue(torch.equal(prepared_masks[2], action_only))
        self.assertEqual(len(attended_masks), 3)
        self.assertTrue(torch.equal(attended_masks[0], original))
        self.assertTrue(torch.equal(attended_masks[1], world_3d_only))
        self.assertTrue(torch.equal(attended_masks[2], action_only))


class _FakeVLMFullAttention(nn.Module):
    head_dim = 16

    def __init__(self) -> None:
        super().__init__()
        self.k_proj = nn.Linear(32, 16, bias=False)
        self.v_proj = nn.Linear(32, 16, bias=False)
        self.k_norm = nn.RMSNorm(16, eps=1e-6)


class _FakeVLMLayer(nn.Module):
    def __init__(self, full_attention: bool) -> None:
        super().__init__()
        self.calls = 0
        self.input_layernorm = nn.RMSNorm(32, eps=1e-6)
        if full_attention:
            self.self_attn = _FakeVLMFullAttention()
        self.projection = nn.Linear(32, 32, bias=False)

    def forward(self, hidden_states, **_kwargs):
        self.calls += 1
        return hidden_states + torch.tanh(self.projection(hidden_states))


class _FakeLanguageModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = types.SimpleNamespace()
        self.layers = nn.ModuleList(
            [
                _FakeVLMLayer(False),
                _FakeVLMLayer(False),
                _FakeVLMLayer(False),
                _FakeVLMLayer(True),
            ]
        )
        self.norm = nn.RMSNorm(32, eps=1e-6)

    def rotary_emb(self, hidden_states, position_ids):
        del hidden_states
        positions = position_ids[0] if position_ids.ndim == 3 else position_ids
        angles = positions.float().unsqueeze(-1).expand(-1, -1, 4)
        return angles.cos(), angles.sin()


class _FakeMultimodalModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.language_model = _FakeLanguageModel()


class _FakePolicyVLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _FakeMultimodalModel()


def _bare_cached_magicw0_policy(
    *,
    enable_2d: bool = True,
    enable_3d: bool = True,
    **config_overrides,
) -> MagicW0TrainingPolicy:
    config = _tiny_config(
        enable_2d_world_expert=enable_2d,
        enable_3d_world_expert=enable_3d,
        enable_world_teacher_supervision=False,
        gradient_checkpointing=False,
        expert_deltanet_backend="torch_parallel",
        expert_full_attention_backend="sdpa",
        **config_overrides,
    )
    policy = MagicW0TrainingPolicy.__new__(MagicW0TrainingPolicy)
    nn.Module.__init__(policy)
    policy.config = config
    policy.vlm = _FakePolicyVLM()
    policy.action_expert = QwenHybridActionExpert(config)
    policy.action_residual_dropout = nn.Dropout(config.action_expert_dropout)
    policy.world_2d_expert = (
        QwenHybrid2DWorldExpert(config, dino_hidden_size=12) if enable_2d else None
    )
    policy.world_3d_expert = QwenHybrid3DWorldExpert(config) if enable_3d else None
    policy.dinov3_teacher = None
    nn.init.normal_(policy.action_expert.output_projection.weight, std=0.05)
    return policy


def _install_fake_magicw0_prefix(
    policy: MagicW0TrainingPolicy,
    *,
    batch_size: int = 2,
    prefix_len: int = 5,
) -> dict[str, object]:
    sequence_length = prefix_len + 3
    prefix = torch.randn(batch_size, sequence_length, 32)
    attention_mask = torch.ones(batch_size, sequence_length, dtype=torch.bool)
    position_ids = (
        torch.arange(sequence_length).view(1, 1, -1).expand(3, batch_size, -1)
    )
    policy._embed_vlm_inputs = lambda _inputs, *_state: (
        prefix,
        attention_mask,
        position_ids,
    )
    return {
        "vlm_inputs": {"input_ids": torch.ones(batch_size, sequence_length)},
        "prefix_len": prefix_len,
        STATE: torch.randn(batch_size, policy.config.state_dim),
        VIEW_VALID_MASK: torch.ones(batch_size, 1, dtype=torch.bool),
        WORLD_3D_VALID_MASK: torch.ones(batch_size, dtype=torch.bool),
    }


class MagicW0ExpertTests(unittest.TestCase):
    def test_dino_teacher_is_not_serialized_and_strict_load_stays_compatible(
        self,
    ) -> None:
        class TinyPolicy(nn.Module):
            def __init__(self):
                super().__init__()
                self.student = nn.Linear(2, 2)
                self.dinov3_teacher = nn.Linear(2, 2)
                self.dinov3_teacher.requires_grad_(False)
                self.register_state_dict_post_hook(
                    _strip_dinov3_teacher_from_state_dict
                )
                self.register_load_state_dict_post_hook(
                    _ignore_dinov3_teacher_checkpoint_keys
                )

        original = TinyPolicy()
        state = original.state_dict()
        self.assertTrue(any(key.startswith("student.") for key in state))
        self.assertFalse(any(key.startswith("dinov3_teacher.") for key in state))

        # New checkpoints omit DINO and must still load under strict=True.
        restored = TinyPolicy()
        restored.load_state_dict(state, strict=True)
        self.assertTrue(torch.equal(restored.student.weight, original.student.weight))

        # Old checkpoints carried DINO. They remain loadable during migration.
        legacy_state = dict(state)
        legacy_state.update(
            {
                "dinov3_teacher.weight": original.dinov3_teacher.weight.clone(),
                "dinov3_teacher.bias": original.dinov3_teacher.bias.clone(),
            }
        )
        restored.load_state_dict(legacy_state, strict=True)

    def test_world_experts_follow_hybrid_schedule_and_shapes(self) -> None:
        config = _tiny_config()
        expert_2d = QwenHybrid2DWorldExpert(config, dino_hidden_size=12)
        expert_3d = QwenHybrid3DWorldExpert(config)
        self.assertEqual(
            [block.use_full_attention for block in expert_2d.blocks],
            [False, False, False, True],
        )
        hidden_2d = expert_2d.embed_queries(2, torch.device("cpu"))
        hidden_3d = expert_3d.embed_queries(2, torch.device("cpu"))
        self.assertEqual(expert_2d.project(hidden_2d).shape, (2, 4, 12))
        self.assertEqual(expert_3d.project_geometry(hidden_3d).shape, (2, 4, 1024))
        self.assertEqual(expert_3d.project_motion(hidden_3d).shape, (2, 4, 1024))

    def test_joint_expert_loss_has_no_gradient_to_vlm_context_or_kv(self) -> None:
        config = _tiny_config()
        policy = MagicW0TrainingPolicy.__new__(MagicW0TrainingPolicy)
        nn.Module.__init__(policy)
        policy.config = config
        blocks = [
            QwenHybridExpertLayer(config, use_full_attention=True),
            QwenHybridExpertLayer(config, use_full_attention=True),
        ]
        for block in blocks:
            block.reset_parameters()

        class FakeAttention(nn.Module):
            def __init__(self):
                super().__init__()
                self.head_dim = 16
                self.k_proj = nn.Linear(32, 16, bias=False)
                self.v_proj = nn.Linear(32, 16, bias=False)
                self.k_norm = QwenRMSNorm(16, 1e-6)

        class FakeVlmLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.input_layernorm = QwenRMSNorm(32, 1e-6)
                self.self_attn = FakeAttention()

            def forward(self, hidden, **kwargs):
                return hidden

        vlm_layer = FakeVlmLayer()
        vlm = torch.randn(2, 3, 32, requires_grad=True)
        streams = [torch.randn(2, 4, 32, requires_grad=True) for _ in range(2)]
        masks = [torch.ones(2, 4, dtype=torch.bool) for _ in range(2)]
        rope_stream = [(torch.ones(2, 4, 16), torch.zeros(2, 4, 16)) for _ in range(2)]
        rope_vlm = (torch.ones(2, 3, 16), torch.zeros(2, 3, 16))
        outputs = policy._forward_joint_layer(
            vlm,
            streams,
            masks,
            rope_stream,
            blocks,
            [nn.Dropout(0.0), nn.Dropout(0.0)],
            vlm_layer,
            rope_vlm,
            torch.ones(2, 3, dtype=torch.bool),
            None,
            torch.zeros(3, 2, 3, dtype=torch.long),
            3,
            torch.ones(2, 3, dtype=torch.bool),
            rope_vlm,
        )
        (outputs[1].sum() + outputs[2].sum()).backward()
        self.assertIsNone(vlm.grad)
        self.assertIsNone(vlm_layer.self_attn.k_proj.weight.grad)
        self.assertIsNone(vlm_layer.self_attn.v_proj.weight.grad)
        self.assertTrue(
            any(
                parameter.grad is not None
                for block in blocks
                for parameter in block.parameters()
            )
        )

    def test_each_joint_stream_loss_reaches_vlm_when_insulation_is_disabled(
        self,
    ) -> None:
        class FakeAttention(nn.Module):
            def __init__(self):
                super().__init__()
                self.head_dim = 16
                self.k_proj = nn.Linear(32, 16, bias=False)
                self.v_proj = nn.Linear(32, 16, bias=False)
                self.k_norm = QwenRMSNorm(16, 1e-6)

        class FakeVlmLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.input_layernorm = QwenRMSNorm(32, 1e-6)
                self.self_attn = FakeAttention()

            def forward(self, hidden, **kwargs):
                return hidden

        stream_names = ("world_2d", "world_3d", "action")
        for target_index, target_name in enumerate(stream_names):
            with self.subTest(stream=target_name):
                torch.manual_seed(0)
                config = _tiny_config(knowledge_insulation=False)
                policy = MagicW0TrainingPolicy.__new__(MagicW0TrainingPolicy)
                nn.Module.__init__(policy)
                policy.config = config
                blocks = [
                    QwenHybridExpertLayer(config, use_full_attention=True)
                    for _ in stream_names
                ]
                vlm_layer = FakeVlmLayer()
                vlm = torch.randn(2, 3, 32, requires_grad=True)
                streams = [
                    torch.randn(2, 4, 32, requires_grad=True) for _ in stream_names
                ]
                masks = [torch.ones(2, 4, dtype=torch.bool) for _ in stream_names]
                rope_stream = [
                    (torch.ones(2, 4, 16), torch.zeros(2, 4, 16)) for _ in stream_names
                ]
                rope_vlm = (
                    torch.ones(2, 3, 16),
                    torch.zeros(2, 3, 16),
                )
                outputs = policy._forward_joint_layer(
                    vlm,
                    streams,
                    masks,
                    rope_stream,
                    blocks,
                    [nn.Dropout(0.0) for _ in stream_names],
                    vlm_layer,
                    rope_vlm,
                    torch.ones(2, 3, dtype=torch.bool),
                    None,
                    torch.zeros(3, 2, 3, dtype=torch.long),
                    3,
                    torch.ones(2, 3, dtype=torch.bool),
                    rope_vlm,
                    stream_names=stream_names,
                )
                outputs[target_index + 1].square().mean().backward()

                self.assertIsNotNone(vlm.grad)
                self.assertGreater(float(vlm.grad.abs().sum()), 0.0)
                for parameter in (
                    vlm_layer.input_layernorm.weight,
                    vlm_layer.self_attn.k_proj.weight,
                    vlm_layer.self_attn.v_proj.weight,
                    vlm_layer.self_attn.k_norm.weight,
                ):
                    self.assertIsNotNone(parameter.grad)
                    self.assertGreater(float(parameter.grad.abs().sum()), 0.0)
                self.assertIsNotNone(streams[target_index].grad)
                self.assertGreater(
                    float(streams[target_index].grad.abs().sum()),
                    0.0,
                )

    def test_cached_joint_expert_trunks_match_original_joint_trunk(self) -> None:
        masking_module = types.ModuleType("transformers.masking_utils")
        masking_module.create_causal_mask = lambda **_kwargs: None
        transformers_module = types.ModuleType("transformers")

        for enable_2d, enable_3d in (
            (True, True),
            (True, False),
            (False, True),
            (False, False),
        ):
            with self.subTest(enable_2d=enable_2d, enable_3d=enable_3d):
                torch.manual_seed(7)
                visibility_override = (
                    {
                        "joint_attention_visibility": {
                            "world_2d": ["vlm", "world_2d"],
                            "world_3d": ["vlm", "world_2d", "world_3d"],
                            "action": ["vlm", "world_3d", "action"],
                        }
                    }
                    if enable_2d and enable_3d
                    else {}
                )
                policy = _bare_cached_magicw0_policy(
                    enable_2d=enable_2d,
                    enable_3d=enable_3d,
                    **visibility_override,
                )
                policy.eval()
                prepared = _install_fake_magicw0_prefix(policy)
                states = prepared[STATE]
                batch_size = states.shape[0]
                actions = torch.randn(
                    batch_size, policy.config.chunk_size, policy.config.action_dim
                )
                timesteps = torch.rand(batch_size)
                action_mask = torch.ones(
                    batch_size, policy.config.chunk_size, dtype=torch.bool
                )
                with mock.patch.dict(
                    sys.modules,
                    {
                        "transformers": transformers_module,
                        "transformers.masking_utils": masking_module,
                    },
                ):
                    reference, *_ = policy._run_joint_trunks(
                        prepared["vlm_inputs"],
                        actions,
                        timesteps,
                        states,
                        action_mask,
                        prepared["prefix_len"],
                        None,
                        prepared[VIEW_VALID_MASK] if enable_2d else None,
                        prepared[WORLD_3D_VALID_MASK] if enable_3d else None,
                    )
                    cache = policy._prepare_flow_inference_cache(prepared)
                    cached = policy._run_cached_joint_expert_trunks(
                        actions,
                        timesteps,
                        states,
                        action_mask,
                        cache,
                    )

                torch.testing.assert_close(cached, reference, rtol=1e-5, atol=1e-6)

    def test_flow_sampling_runs_vlm_once_and_reuses_prefix_kv(self) -> None:
        torch.manual_seed(8)
        policy = _bare_cached_magicw0_policy()
        prepared = _install_fake_magicw0_prefix(policy)
        batch_size = prepared[STATE].shape[0]
        masking_module = types.ModuleType("transformers.masking_utils")
        masking_module.create_causal_mask = lambda **_kwargs: None
        transformers_module = types.ModuleType("transformers")

        with mock.patch.dict(
            sys.modules,
            {
                "transformers": transformers_module,
                "transformers.masking_utils": masking_module,
            },
        ):
            output = policy.generate_actions(
                prepared,
                num_steps=3,
                initial_noise=torch.randn(
                    batch_size,
                    policy.config.chunk_size,
                    policy.config.action_dim,
                ),
            )

        self.assertEqual(
            tuple(output.shape),
            (
                batch_size,
                policy.config.chunk_size,
                policy.config.action_dim,
            ),
        )
        self.assertEqual(
            [layer.calls for layer in policy.vlm.model.language_model.layers],
            [1, 1, 1, 1],
        )

    def test_prepare_inference_batch_accepts_new_base_prefix_contract(self) -> None:
        config = _tiny_config(
            enable_2d_world_expert=False,
            enable_3d_world_expert=False,
            enable_world_teacher_supervision=False,
        )
        policy = MagicW0TrainingPolicy.__new__(MagicW0TrainingPolicy)
        nn.Module.__init__(policy)
        policy.config = config
        policy.world_2d_expert = None
        policy.world_3d_expert = None

        def fake_prepare_prefix(_self, *_args, **_kwargs):
            return (
                {"input_ids": torch.ones(2, 7, dtype=torch.long)},
                7,
                None,
                None,
            )

        policy._prepare_robotics_prefix = types.MethodType(fake_prepare_prefix, policy)
        prepared = policy.prepare_inference_batch(
            {
                STATE: torch.zeros(2, config.state_dim),
                TASK: ["first", "second"],
                CAMERA_VALID: torch.zeros(2, len(config.camera_keys), dtype=torch.bool),
            },
            None,
        )

        self.assertEqual(prepared["prefix_len"], 7)
        self.assertEqual(tuple(prepared["vlm_inputs"]["input_ids"].shape), (2, 7))

    def test_generate_actions_uses_supplied_noise_without_world_teachers(self) -> None:
        config = _tiny_config(
            enable_world_teacher_supervision=False,
            num_inference_steps=4,
        )
        policy = MagicW0TrainingPolicy.__new__(MagicW0TrainingPolicy)
        nn.Module.__init__(policy)
        policy.config = config
        policy.dinov3_teacher = None
        policy.world_2d_expert = None
        policy.world_3d_expert = None
        calls = []

        def fake_cached_trunks(
            _self,
            actions,
            timesteps,
            states,
            action_mask,
            cache,
        ):
            self.assertEqual(cache, {"cached": True})
            calls.append(timesteps.clone())
            return torch.ones_like(actions)

        policy._prepare_flow_inference_cache = lambda _batch: {"cached": True}
        policy._run_cached_joint_expert_trunks = types.MethodType(
            fake_cached_trunks, policy
        )
        initial_noise = torch.zeros(2, config.chunk_size, config.action_dim)
        generated = policy.generate_actions(
            {
                "vlm_inputs": {"input_ids": torch.ones(2, 1, dtype=torch.long)},
                "prefix_len": 1,
                STATE: torch.zeros(2, config.state_dim),
            },
            initial_noise=initial_noise,
        )
        self.assertEqual(len(calls), 4)
        self.assertTrue(torch.equal(calls[0], torch.ones(2)))
        self.assertTrue(torch.allclose(generated, torch.full_like(generated, -1.0)))
        self.assertIsNone(policy.dinov3_teacher)
        self.assertIsNone(getattr(policy, "_track4world_teacher", None))

    def test_generate_actions_keeps_masked_dimensions_zero(self) -> None:
        config = _tiny_config(
            enable_world_teacher_supervision=False,
            num_inference_steps=4,
        )
        policy = MagicW0TrainingPolicy.__new__(MagicW0TrainingPolicy)
        nn.Module.__init__(policy)
        policy.config = config
        policy.dinov3_teacher = None
        policy.world_2d_expert = None
        policy.world_3d_expert = None
        seen_actions = []

        def fake_cached_trunks(
            _self,
            actions,
            timesteps,
            states,
            action_mask,
            cache,
        ):
            self.assertEqual(cache, {"cached": True})
            seen_actions.append(actions.clone())
            return torch.ones_like(actions)

        policy._prepare_flow_inference_cache = lambda _batch: {"cached": True}
        policy._run_cached_joint_expert_trunks = types.MethodType(
            fake_cached_trunks, policy
        )
        initial_noise = torch.zeros(2, config.chunk_size, config.action_dim)
        dim_mask = torch.zeros(config.action_dim, dtype=torch.bool)
        dim_mask[:14] = True
        generated = policy.generate_actions(
            {
                "vlm_inputs": {"input_ids": torch.ones(2, 1, dtype=torch.long)},
                "prefix_len": 1,
                STATE: torch.zeros(2, config.state_dim),
            },
            initial_noise=initial_noise,
            action_dim_mask=dim_mask,
        )
        self.assertTrue(
            torch.allclose(
                generated[..., :14], torch.full_like(generated[..., :14], -1.0)
            )
        )
        self.assertTrue(
            torch.equal(generated[..., 14:], torch.zeros_like(generated[..., 14:]))
        )
        self.assertTrue(
            all(
                torch.equal(actions[..., 14:], torch.zeros_like(actions[..., 14:]))
                for actions in seen_actions
            )
        )

    def test_track4world_guard_rejects_inference_mode(self) -> None:
        policy = MagicW0TrainingPolicy.__new__(MagicW0TrainingPolicy)
        nn.Module.__init__(policy)
        policy.config = _tiny_config(enable_world_teacher_supervision=False)
        with self.assertRaisesRegex(RuntimeError, "disabled for inference"):
            policy._get_track4world_teacher(torch.device("cpu"))

    def test_track4world_factory_selects_tensorrt_backend(self) -> None:
        policy = MagicW0TrainingPolicy.__new__(MagicW0TrainingPolicy)
        nn.Module.__init__(policy)
        policy.config = _tiny_config(
            track4world_inference_backend="tensorrt",
            track4world_tensorrt_engine_dir="/engines",
            track4world_project_root="/track4world",
        )
        sentinel = object()
        with patch(
            "magic_w0.models.world_action.track4world_tensorrt.Track4WorldTensorRT",
            return_value=sentinel,
        ) as constructor:
            teacher = policy._get_track4world_teacher(torch.device("cuda"))

        self.assertIs(teacher, sentinel)
        constructor.assert_called_once_with(
            "/engines",
            "/track4world",
            keep_execution_memory=False,
            expected_engine_batch_sizes=None,
        )


if __name__ == "__main__":
    unittest.main()
