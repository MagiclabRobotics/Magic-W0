"""MagicW0 visual history for magicw0.

CPU-only except the last class, which uses the production Qwen3.5-2B processor
when it is mounted and is skipped otherwise.
"""

from __future__ import annotations

import sys
import os
import types
import unittest
import unittest.mock
from pathlib import Path
from typing import ClassVar

import torch

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from magic_w0.models.action_backbone.modeling_magicw0_base import MagicW0BasePolicy
from magic_w0.models.world_action import MagicW0TrainingConfig, MagicW0TrainingPolicy
from magic_w0.models.world_action import history_frames as history
from magic_w0.models.world_action.processor_magicw0 import _MagicW0TrainingDataset

QWEN = Path(os.environ.get("MAGIC_W0_QWEN_ASSETS", "checkpoints/qwen3.5-2b-assets"))
HIGH = "observation.images.cam_high"
LEFT = "observation.images.cam_left_wrist"
RIGHT = "observation.images.cam_right_wrist"


def config(**overrides) -> MagicW0TrainingConfig:
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
        "action_dim": 4,
        "state_dim": 3,
        "chunk_size": 4,
        "camera_keys": (HIGH, LEFT, RIGHT),
        "gen_2d_camera_keys": (HIGH,),
        "gen_2d_grid_size": (2, 2),
        "gen_2d_num_query_tokens_per_view": 4,
        "gen_3d_grid_size": (2, 2),
        "gen_3d_num_query_tokens": 4,
        "gen_3d_camera_key": HIGH,
        "use_fast_action_tokens": False,
        "state_prompt_mode": "text",
    }
    values.update(overrides)
    return MagicW0TrainingConfig(**values)


def owner_for(cfg, processor):
    """A policy stand-in exposing exactly the methods the prefix path uses."""
    owner = types.SimpleNamespace(config=cfg, processor=processor)
    for name in ("_prefix_text_parts", "_finalize_robotics_prefix"):
        setattr(owner, name, types.MethodType(getattr(MagicW0BasePolicy, name), owner))
    owner._history_placeholder_id = types.MethodType(
        MagicW0TrainingPolicy._history_placeholder_id, owner
    )
    return owner


class FakeTokenizer:
    padding_side = "right"


class FakeImageProcessor:
    def __call__(self, images, return_tensors):
        return {
            "pixel_values": torch.stack([image.float() for image in images]),
            "image_grid_thw": torch.tensor([[1, 16, 16]] * len(images)),
        }


class FakeProcessor:
    def __init__(self):
        self.tokenizer = FakeTokenizer()
        self.image_processor = FakeImageProcessor()
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs, self.tokenizer.padding_side))
        return {"input_ids": torch.ones(len(messages), 5, dtype=torch.long)}


class SamplingTests(unittest.TestCase):
    def test_interval_is_in_action_steps(self):
        self.assertEqual(history.history_frame_stride(20, 1), 20)
        self.assertEqual(history.history_frame_stride(20, 2), 10)
        with self.assertRaisesRegex(ValueError, "whole number"):
            history.history_frame_stride(5, 2)

    def test_past_frames_are_strictly_past_and_dropped_before_start(self):
        self.assertEqual(history.past_frame_indices(0, 25, 20), [])
        self.assertEqual(history.past_frame_indices(30, 25, 20), [10])
        self.assertEqual(
            history.past_frame_indices(500, 25, 20), list(range(0, 500, 20))
        )
        self.assertEqual(len(history.past_frame_indices(10_000, 25, 20)), 25)

    def test_adaptive_pool_is_two_by_two_mean_on_the_256_canvas(self):
        feature = torch.arange(64, dtype=torch.float32)[:, None].repeat(1, 3)
        pooled = history.pool_frame_features(
            [feature], torch.tensor([[1, 16, 16]]), merge=2, pool_size=4
        )
        grid = torch.arange(64, dtype=torch.float32).reshape(8, 8)
        expected = grid.reshape(4, 2, 4, 2).mean((1, 3)).reshape(16)
        torch.testing.assert_close(pooled[:, 0], expected)
        odd = history.pool_frame_features(
            [feature], torch.tensor([[1, 16, 16]]), merge=2, pool_size=3
        )
        self.assertEqual(tuple(odd.shape), (9, 3))

    def test_block_text(self):
        run = history.HISTORY_PLACEHOLDER_TOKEN * 16
        self.assertEqual(history.history_block_text(0, 16), "History images: \n")
        self.assertEqual(
            history.history_block_text(2, 16),
            "History images: " + run + "\n" + run + "\n",
        )


class ConfigTests(unittest.TestCase):
    def test_defaults_are_off_and_history_when_on(self):
        self.assertFalse(config().history_enabled)
        cfg = config(history_enabled=True)
        self.assertEqual((cfg.history_frames, cfg.history_interval), (25, 20))
        self.assertEqual(cfg.history_tokens_per_frame, 16)

    def test_from_dict_accepts_history_keys(self):
        cfg = MagicW0TrainingConfig.from_dict(
            {"history_enabled": True, "history_frames": 10, "history_interval": 5}
        )
        self.assertEqual((cfg.history_frames, cfg.history_interval), (10, 5))

    def test_invalid_values_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "history_camera_key"):
            config(history_enabled=True, history_camera_key="nope")
        with self.assertRaisesRegex(ValueError, "history_pool_size"):
            config(history_enabled=True, history_pool_size=9)
        with self.assertRaisesRegex(ValueError, ">= 1"):
            config(history_enabled=True, history_interval=0)


class RawDataset:
    camera_keys: ClassVar[tuple[str, ...]] = (HIGH,)
    state_key = "observation.state"
    action_key = "action"
    chunk_size = 1
    action_upsample_rate = 1
    supervise_episode_end_padding = True
    _cum: ClassVar[tuple[int, ...]] = (0, 30)

    def __init__(self):
        self.meta_stats = {self.state_key: {}, self.action_key: {}}
        self.history_calls = []

    def __len__(self):
        return 30

    def __getitem__(self, index):
        return {
            self.state_key: torch.zeros(3),
            self.action_key: torch.zeros(1, 4),
            f"{self.action_key}_is_pad": torch.zeros(1, dtype=torch.bool),
            "task": "move",
            "episode_index": torch.tensor(0),
            "frame_index": torch.tensor(index),
            HIGH: torch.zeros(3, 8, 8, dtype=torch.uint8),
        }

    def get_episode_length(self, episode):
        return 30

    def get_episode_frame_images(self, episode, frame, camera_keys):
        return {key: torch.zeros(3, 8, 8, dtype=torch.uint8) for key in camera_keys}

    def get_episode_frames(self, episode, frames, camera_keys):
        self.history_calls.append(tuple(frames))
        clip = torch.stack(
            [torch.full((3, 8, 8), 1 + f, dtype=torch.uint8) for f in frames]
        )
        return {key: clip for key in camera_keys}


class DatasetTests(unittest.TestCase):
    def wrapped(self, raw=None, **overrides):
        cfg = config(
            normalization="none",
            history_enabled=True,
            history_frames=4,
            history_interval=5,
            **overrides,
        )
        return _MagicW0TrainingDataset(raw or RawDataset(), cfg)

    def test_disabled_history_adds_no_keys(self):
        ds = _MagicW0TrainingDataset(RawDataset(), config(normalization="none"))
        self.assertNotIn(history.HISTORY_FRAMES, ds[5])

    def test_clip_holds_only_past_frames_left_padded(self):
        ds = self.wrapped()
        first = ds[0]
        self.assertEqual(int(first[history.HISTORY_LEN]), 0)
        self.assertEqual(ds.dataset.history_calls, [])
        early = ds[12]  # slots -8, -3, 2, 7 -> frames 2 and 7
        clip = early[history.HISTORY_FRAMES]
        self.assertEqual(tuple(clip.shape), (4, 3, 256, 256))
        self.assertEqual(int(early[history.HISTORY_LEN]), 2)
        self.assertEqual(float(clip[:2].abs().sum()), 0.0)
        self.assertEqual(clip[2:, 0, 128, 128].tolist(), [3, 8])
        full = ds[20]
        self.assertEqual(
            full[history.HISTORY_FRAMES][:, 0, 128, 128].tolist(), [1, 6, 11, 16]
        )
        self.assertEqual(ds.dataset.history_calls[-1], (0, 5, 10, 15))

    def test_default_collation_stacks_mixed_lengths(self):
        ds = self.wrapped()
        batch = torch.utils.data.default_collate([ds[0], ds[20]])
        self.assertEqual(batch[history.HISTORY_LEN].tolist(), [0, 4])

    def test_upsampled_source_converts_action_steps_to_frames(self):
        raw = RawDataset()
        raw.action_upsample_rate = 2
        cfg = config(normalization="none", history_enabled=True, history_interval=20)
        self.assertEqual(_MagicW0TrainingDataset(raw, cfg).history_stride, 10)
        bad = config(normalization="none", history_enabled=True, history_interval=5)
        with self.assertRaisesRegex(ValueError, "whole number"):
            _MagicW0TrainingDataset(raw, bad)


class DropoutTests(unittest.TestCase):
    def wrapped(self, source_root="/data/gen__stack_bowls", **overrides):
        cfg = config(
            normalization="none",
            history_enabled=True,
            history_frames=4,
            history_interval=5,
            history_keep_prob=0.5,
            history_partial_drop_prob=0.3,
            history_full_drop_prob=0.2,
            history_dropout_exempt_sources=("memory__",),
            **overrides,
        )
        return _MagicW0TrainingDataset(RawDataset(), cfg, source_root=source_root)

    def sample(self, ds, draw, keep=None):
        with unittest.mock.patch("torch.rand", return_value=torch.tensor(draw)):
            if keep is None:
                return ds[20]
            with unittest.mock.patch("torch.randint", return_value=torch.tensor(keep)):
                return ds[20]

    def centre(self, item):
        n = int(item[history.HISTORY_LEN])
        return item[history.HISTORY_FRAMES][4 - n :, 0, 128, 128].tolist() if n else []

    def test_full_partial_keep_follow_the_draw(self):
        ds = self.wrapped()
        full = self.sample(ds, 0.1)  # < 0.2
        self.assertEqual(int(full[history.HISTORY_MODE]), history.HISTORY_MODE_FULL)
        self.assertEqual(int(full[history.HISTORY_LEN]), 0)
        partial = self.sample(ds, 0.3, keep=2)  # 0.2 <= u < 0.5
        self.assertEqual(
            int(partial[history.HISTORY_MODE]), history.HISTORY_MODE_PARTIAL
        )
        self.assertEqual(self.centre(partial), [11, 16])  # the most recent two
        kept = self.sample(ds, 0.9)
        self.assertEqual(int(kept[history.HISTORY_MODE]), history.HISTORY_MODE_KEEP)
        self.assertEqual(self.centre(kept), [1, 6, 11, 16])

    def test_dropped_frames_are_never_decoded(self):
        ds = self.wrapped()
        self.sample(ds, 0.1)
        self.assertEqual(ds.dataset.history_calls, [])
        self.sample(ds, 0.3, keep=1)
        self.assertEqual(ds.dataset.history_calls, [(15,)])

    def test_memory_sources_are_exempt(self):
        ds = self.wrapped(source_root="/data/memory__cover_block_337__teleop")
        item = self.sample(ds, 0.0)  # would be a full drop elsewhere
        self.assertEqual(int(item[history.HISTORY_MODE]), history.HISTORY_MODE_EXEMPT)
        self.assertEqual(self.centre(item), [1, 6, 11, 16])

    def test_ratios_over_many_draws(self):
        torch.manual_seed(0)
        ds = self.wrapped()
        modes = [int(ds[20][history.HISTORY_MODE]) for _ in range(2000)]
        share = [modes.count(m) / len(modes) for m in (0, 1, 2)]
        for got, want in zip(share, (0.5, 0.3, 0.2)):
            self.assertAlmostEqual(got, want, delta=0.04)

    def test_probabilities_must_sum_to_one(self):
        with self.assertRaisesRegex(ValueError, "sum to 1"):
            config(
                history_enabled=True,
                history_keep_prob=0.5,
                history_partial_drop_prob=0.3,
                history_full_drop_prob=0.3,
            )
        cfg = MagicW0TrainingConfig.from_dict(
            {"history_enabled": True, "history_dropout_exempt_sources": ["memory__"]}
        )
        self.assertEqual(cfg.history_dropout_exempt_sources, ("memory__",))


class PrefixTests(unittest.TestCase):
    def batch(self, lengths):
        size = len(lengths)
        batch = {
            key: torch.zeros(size, 3, 8, 8, dtype=torch.uint8)
            for key in (HIGH, LEFT, RIGHT)
        }
        clip = torch.zeros(size, 25, 3, 8, 8, dtype=torch.uint8)
        clip[:, :, 0] = torch.arange(25, dtype=torch.uint8)[None, :, None, None]
        batch[history.HISTORY_FRAMES] = clip
        batch[history.HISTORY_LEN] = torch.tensor(lengths)
        return batch

    def test_base_text_is_byte_identical_after_the_split(self):
        cfg = config(robot_prompt_metadata="meta {action_space}")
        fake = FakeProcessor()
        owner = owner_for(cfg, fake)
        MagicW0BasePolicy._prepare_robotics_prefix(
            owner,
            self.batch([0]),
            torch.tensor([[0.5, 0.25, 0.0]]),
            ["stack"],
            torch.ones(1, 3, dtype=torch.bool),
            ["arx_x5_sim"],
        )
        text = fake.calls[0][0][0][0]["content"][-1]["text"]
        self.assertIn("Robot task: stack\n", text)
        self.assertTrue(text.endswith("Predict the next robot action chunk."))
        # Legacy layout: images first, one text block last.
        kinds = [item["type"] for item in fake.calls[0][0][0][0]["content"]]
        self.assertEqual(kinds, ["image", "image", "image", "text"])

    def test_disabled_history_delegates_to_base(self):
        policy = MagicW0TrainingPolicy.__new__(MagicW0TrainingPolicy)
        torch.nn.Module.__init__(policy)
        policy.config = config()
        policy.processor = FakeProcessor()
        policy._prepare_robotics_prefix(
            self.batch([3]),
            torch.zeros(1, 3),
            ["x"],
            torch.ones(1, 3, dtype=torch.bool),
        )
        content = policy.processor.calls[0][0][0][0]["content"]
        self.assertEqual(
            [item["type"] for item in content], ["image", "image", "image", "text"]
        )
        self.assertNotIn("History images", content[-1]["text"])

    def test_history_prefix_uses_history_order(self):
        cfg = config(history_enabled=True)
        fake = FakeProcessor()
        owner = owner_for(cfg, fake)
        inputs, _, _, _ = MagicW0TrainingPolicy._prepare_robotics_prefix(
            owner,
            self.batch([3, 0]),
            torch.zeros(2, 3),
            ["x", "x"],
            torch.ones(2, 3, dtype=torch.bool),
        )
        messages, kwargs, side = fake.calls[0]
        self.assertTrue(kwargs["tokenize"])
        self.assertEqual(side, "left")
        first = messages[0][0]["content"]
        self.assertEqual(
            [item["type"] for item in first],
            ["text", "text", "image", "text", "image", "text", "image", "text"],
        )
        run = history.HISTORY_PLACEHOLDER_TOKEN * 16
        self.assertEqual(
            first[0]["text"], "Robot task: x\nHistory images: " + (run + "\n") * 3
        )
        self.assertEqual(
            [item["text"] for item in first[1:7:2]],
            ["Head image: ", "Left wrist image: ", "Right wrist image: "],
        )
        self.assertTrue(first[-1]["text"].startswith("Current normalized robot state"))
        self.assertEqual(
            messages[1][0]["content"][0]["text"], "Robot task: x\nHistory images: \n"
        )
        pixels = inputs[history.HISTORY_IMAGES]["pixel_values"]
        self.assertEqual(pixels[:, 0, 0, 0].tolist(), [22, 23, 24])

    def test_missing_clip_fails_loudly(self):
        owner = owner_for(config(history_enabled=True), FakeProcessor())
        with self.assertRaisesRegex(KeyError, "past_frame_indices"):
            MagicW0TrainingPolicy._prepare_robotics_prefix(
                owner, {}, torch.zeros(1, 3), ["x"], torch.ones(1, 3, dtype=torch.bool)
            )


class ScatterTests(unittest.TestCase):
    def test_pooled_features_land_on_placeholders_in_frame_order(self):
        slot = 7
        features = [torch.full((64, 4), float(i)) for i in range(2)]
        policy = MagicW0TrainingPolicy.__new__(MagicW0TrainingPolicy)
        torch.nn.Module.__init__(policy)
        policy.config = config(history_enabled=True)
        policy._history_placeholder_cache = slot
        policy.vlm = types.SimpleNamespace(
            config=types.SimpleNamespace(
                vision_config=types.SimpleNamespace(spatial_merge_size=2)
            ),
            model=types.SimpleNamespace(
                get_image_features=lambda pixels, grid, return_dict: (
                    types.SimpleNamespace(pooler_output=features)
                )
            ),
        )
        ids = torch.tensor([[1] + [slot] * 32 + [2]])
        base_out = (torch.zeros(1, 34, 4), torch.ones(1, 34, dtype=torch.bool), None)
        frames = {
            "pixel_values": torch.zeros(2, 1),
            "image_grid_thw": torch.tensor([[1, 16, 16]] * 2),
        }
        with unittest.mock.patch.object(
            MagicW0BasePolicy, "_embed_vlm_inputs", return_value=base_out
        ):
            out, _, _ = policy._embed_vlm_inputs(
                {"input_ids": ids, history.HISTORY_IMAGES: frames}
            )
            self.assertEqual(out[0, 1:17, 0].tolist(), [0.0] * 16)
            self.assertEqual(out[0, 17:33, 0].tolist(), [1.0] * 16)
            self.assertEqual(float(out[0, [0, 33]].abs().sum()), 0.0)
            with self.assertRaisesRegex(ValueError, "placeholders"):
                policy._embed_vlm_inputs({"input_ids": ids})


@unittest.skipUnless(QWEN.is_dir(), "production Qwen3.5-2B processor not mounted")
class NativeQwenTests(unittest.TestCase):
    def test_real_processor_renders_history_prompt(self):
        from transformers import AutoProcessor

        qwen = AutoProcessor.from_pretrained(QWEN, local_files_only=True)
        owner = owner_for(config(history_enabled=True), qwen)
        batch = {
            key: torch.randint(0, 256, (2, 3, 256, 256), dtype=torch.uint8)
            for key in (HIGH, LEFT, RIGHT)
        }
        batch[history.HISTORY_FRAMES] = torch.randint(
            0, 256, (2, 25, 3, 256, 256), dtype=torch.uint8
        )
        batch[history.HISTORY_LEN] = torch.tensor([25, 0])
        inputs, _, _, _ = MagicW0TrainingPolicy._prepare_robotics_prefix(
            owner,
            batch,
            torch.zeros(2, 3),
            ["stack the bowls"] * 2,
            torch.ones(2, 3, dtype=torch.bool),
        )
        slots = inputs["input_ids"] == owner._history_placeholder_id()
        self.assertEqual(slots.sum(1).tolist(), [400, 0])
        self.assertEqual(
            inputs[history.HISTORY_IMAGES]["image_grid_thw"].tolist(),
            [[1, 16, 16]] * 25,
        )
        self.assertEqual(inputs["image_grid_thw"].shape[0], 6)
        text = qwen.tokenizer.decode(inputs["input_ids"][0])
        self.assertIn(
            "Robot task: stack the bowls\nHistory images: "
            + history.HISTORY_PLACEHOLDER_TOKEN,
            text,
        )
        self.assertIn("Head image: <|vision_start|>", text)
        self.assertIn("Right wrist image: <|vision_start|>", text)

    def test_left_padding_never_reaches_the_prefix(self):
        """A short-history row reads the same alone or beside a long one."""
        from transformers import AutoProcessor, Qwen3_5Config
        from transformers import Qwen3_5ForConditionalGeneration

        qwen = AutoProcessor.from_pretrained(QWEN, local_files_only=True)
        torch.manual_seed(0)
        hf = Qwen3_5Config(
            text_config={
                "hidden_size": 64,
                "intermediate_size": 96,
                "vocab_size": 248320,
                "num_hidden_layers": 8,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 32,
                "layer_types": (["linear_attention"] * 3 + ["full_attention"]) * 2,
                "linear_num_key_heads": 2,
                "linear_num_value_heads": 2,
                "linear_key_head_dim": 16,
                "linear_value_head_dim": 16,
                "linear_conv_kernel_dim": 4,
                "rope_parameters": {
                    "rope_type": "default",
                    "rope_theta": 10000000,
                    "partial_rotary_factor": 0.25,
                    "mrope_section": [2, 1, 1],
                    "mrope_interleaved": True,
                },
            },
            vision_config={
                "depth": 1,
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_heads": 2,
                "out_hidden_size": 64,
                "patch_size": 16,
                "spatial_merge_size": 2,
                "temporal_patch_size": 2,
            },
            image_token_id=248056,
            video_token_id=248057,
            vision_start_token_id=248053,
            vision_end_token_id=248054,
        )
        hf._attn_implementation = "sdpa"
        hf.text_config._attn_implementation = "sdpa"
        policy = MagicW0TrainingPolicy.__new__(MagicW0TrainingPolicy)
        torch.nn.Module.__init__(policy)
        policy.config = config(history_enabled=True, history_frames=6)
        policy.processor = qwen
        policy.vlm = Qwen3_5ForConditionalGeneration(hf).eval()
        policy.vlm_layer_types = tuple(hf.text_config.layer_types)
        policy.state_prompt_projection = None
        generator = torch.Generator().manual_seed(1)
        views = {
            key: torch.randint(
                0, 256, (2, 3, 256, 256), dtype=torch.uint8, generator=generator
            )
            for key in (HIGH, LEFT, RIGHT)
        }
        clips = torch.randint(
            0, 256, (2, 6, 3, 256, 256), dtype=torch.uint8, generator=generator
        )

        def prefix_hidden(rows, lengths, clip=clips):
            batch = {key: value[rows] for key, value in views.items()}
            batch[history.HISTORY_FRAMES] = clip[rows]
            batch[history.HISTORY_LEN] = torch.tensor(lengths)
            inputs, _, _, _ = policy._prepare_robotics_prefix(
                batch,
                torch.zeros(len(rows), 3),
                ["stack the bowls"] * len(rows),
                torch.ones(len(rows), 3, dtype=torch.bool),
            )
            with torch.no_grad():
                hidden = policy._vlm_prefix_hidden(dict(inputs))
            return hidden, inputs["attention_mask"].bool()

        alone, alone_mask = prefix_hidden([1], [2])
        mixed, mixed_mask = prefix_hidden([0, 1], [6, 2])
        self.assertTrue(bool(alone_mask.all()))
        pad = int((~mixed_mask[1]).sum())
        self.assertGreater(pad, 0)
        torch.testing.assert_close(mixed[1, pad:], alone[0], atol=2e-5, rtol=1e-4)
        edited = clips.clone()
        edited[1, -2] = 255 - edited[1, -2]
        changed, _ = prefix_hidden([1], [2], clip=edited)
        self.assertGreater(float((changed[0, -1] - alone[0, -1]).abs().max()), 1e-4)


if __name__ == "__main__":
    unittest.main()
