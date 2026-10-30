"""Tests for step-level checkpointing in ``DistributedTrainer``.

Long EIP runs only saved at epoch end, so a preempted job lost everything. These
tests drive the real ``train_loop`` on CPU with an in-memory model and assert that
mid-epoch checkpoints land, rotate, and are written atomically.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
import weakref
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

try:
    import torch  # noqa: F401

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

requires_torch = unittest.skipUnless(
    HAS_TORCH, "requires torch (pip install -e '.[train]')"
)


class _Logger:
    def __init__(self):
        self.events = []

    def log(self, event):
        self.events.append(event)


def _trainer(
    output_dir,
    *,
    steps,
    save_every_steps,
    keep_last_n_checkpoints=None,
    save_milestone_every_steps=None,
):
    import torch
    from torch import nn

    from magic_w0.models.base import ModelForwardOutput
    from magic_w0.training.dist_trainer import DistributedTrainer, TrainerState
    from magic_w0.training.hooks import HookRunner
    from magic_w0.training.loss_monitor import LossMonitor

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(4, 1)
            self.training_progress = []

        def set_training_progress(self, *, global_step, max_steps):
            self.training_progress.append((global_step, max_steps))

        def forward(self, batch):
            loss = ((self.linear(batch["x"]) - batch["target"]) ** 2).mean()
            return ModelForwardOutput(
                loss=loss,
                metrics={"modality": batch["modality"]},
                loss_name="mse",
            )

    model = _Model()
    batch = {"x": torch.ones(2, 4), "target": torch.zeros(2, 1), "modality": "robotics"}

    trainer = DistributedTrainer.__new__(DistributedTrainer)
    trainer.model = model
    trainer.initial_model = model
    trainer.train_dataloader = [batch] * steps
    trainer.initial_train_dataloader = trainer.train_dataloader
    trainer.val_dataloader = None
    trainer.initial_val_dataloader = None
    trainer.device = torch.device("cpu")
    trainer.use_amp = False
    trainer.grad_scaler = None
    trainer.grad_clip_norm = 1.0
    trainer.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    trainer.lr_scheduler = None
    trainer.state = TrainerState()
    trainer._num_epochs = 1
    trainer.max_steps = steps
    trainer.log_interval = 0
    trainer.metrics_logger = _Logger()
    trainer.loss_monitor = LossMonitor()
    trainer.hook_runner = HookRunner([], sync_decision=lambda fired: fired)
    trainer.output_dir = Path(output_dir)
    trainer.save_every_steps = save_every_steps
    trainer.keep_last_n_checkpoints = keep_last_n_checkpoints
    trainer.save_milestone_every_steps = save_milestone_every_steps
    return trainer


@requires_torch
class StepCheckpointTests(unittest.TestCase):
    def test_model_receives_progress_before_every_forward(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = _trainer(tmp, steps=3, save_every_steps=None)

            trainer.train_loop(epoch=0)

            self.assertEqual(
                trainer.model.training_progress,
                [(0, 3), (1, 3), (2, 3)],
            )

    def test_saves_on_the_step_interval(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = _trainer(tmp, steps=6, save_every_steps=2)

            trainer.train_loop(epoch=0)

            names = sorted(p.name for p in Path(tmp).glob("*.pt"))
            self.assertEqual(
                names,
                [
                    "checkpoint-epoch-0-step-2.pt",
                    "checkpoint-epoch-0-step-4.pt",
                    "checkpoint-epoch-0-step-6.pt",
                ],
            )

    def test_no_step_checkpoints_when_disabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = _trainer(tmp, steps=6, save_every_steps=None)

            trainer.train_loop(epoch=0)

            self.assertEqual(list(Path(tmp).glob("*.pt")), [])

    def test_rotation_keeps_only_the_newest(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = _trainer(
                tmp, steps=10, save_every_steps=2, keep_last_n_checkpoints=2
            )

            trainer.train_loop(epoch=0)

            names = sorted(p.name for p in Path(tmp).glob("*.pt"))
            self.assertEqual(
                names,
                ["checkpoint-epoch-0-step-10.pt", "checkpoint-epoch-0-step-8.pt"],
            )

    def test_rotation_orders_numerically_not_lexicographically(self):
        """step-9 must not outrank step-10 the way string sorting would have it."""
        from magic_w0.training.dist_trainer import DistributedTrainer

        paths = [
            Path("checkpoint-epoch-0-step-9.pt"),
            Path("checkpoint-epoch-0-step-10.pt"),
            Path("checkpoint-epoch-1-step-2.pt"),
        ]
        self.assertEqual(
            [
                p.name
                for p in sorted(paths, key=DistributedTrainer._checkpoint_sort_key)
            ],
            [
                "checkpoint-epoch-0-step-9.pt",
                "checkpoint-epoch-0-step-10.pt",
                "checkpoint-epoch-1-step-2.pt",
            ],
        )

    def test_milestones_survive_rolling_checkpoint_rotation(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = _trainer(
                tmp,
                steps=12,
                save_every_steps=2,
                keep_last_n_checkpoints=2,
                save_milestone_every_steps=5,
            )

            trainer.train_loop(epoch=0)

            names = sorted(p.name for p in Path(tmp).glob("*.pt"))
            self.assertEqual(
                names,
                [
                    "checkpoint-epoch-0-step-10.pt",
                    "checkpoint-epoch-0-step-12.pt",
                    "checkpoint-epoch-0-step-5.pt",
                    "checkpoint-epoch-0-step-8.pt",
                ],
            )
            events = [
                event["checkpoint_kind"]
                for event in trainer.metrics_logger.events
                if event["event"] == "checkpoint"
            ]
            self.assertEqual(
                events,
                [
                    "rolling",
                    "rolling",
                    "milestone",
                    "rolling",
                    "rolling",
                    "milestone",
                    "rolling",
                ],
            )

    def test_milestones_are_saved_even_when_they_miss_recovery_cadence(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = _trainer(
                tmp,
                steps=12,
                save_every_steps=4,
                keep_last_n_checkpoints=2,
                save_milestone_every_steps=3,
            )

            trainer.train_loop(epoch=0)

            self.assertEqual(
                sorted(path.name for path in Path(tmp).glob("*.pt")),
                [
                    "checkpoint-epoch-0-step-12.pt",
                    "checkpoint-epoch-0-step-3.pt",
                    "checkpoint-epoch-0-step-4.pt",
                    "checkpoint-epoch-0-step-6.pt",
                    "checkpoint-epoch-0-step-8.pt",
                    "checkpoint-epoch-0-step-9.pt",
                ],
            )

    def test_checkpoint_payload_carries_step_and_optimizer(self):
        with tempfile.TemporaryDirectory() as tmp:
            import torch

            trainer = _trainer(tmp, steps=4, save_every_steps=4)

            trainer.train_loop(epoch=0)

            payload = torch.load(
                Path(tmp) / "checkpoint-epoch-0-step-4.pt", weights_only=False
            )
            self.assertEqual(payload["step"], 4)
            self.assertIn("optimizer", payload)
            self.assertIn("model", payload)
            self.assertEqual(payload["extra_state"]["epoch"], 0)

    def test_checkpoint_carries_consumed_data_cursors(self):
        with tempfile.TemporaryDirectory() as tmp:
            import torch

            class _StatefulBatches(list):
                def state_dict(self):
                    return {"cursor": 4}

            trainer = _trainer(tmp, steps=4, save_every_steps=4)
            batches = _StatefulBatches(trainer.train_dataloader)
            trainer.train_dataloader = batches
            trainer.initial_train_dataloader = batches

            trainer.train_loop(epoch=0)

            payload = torch.load(
                Path(tmp) / "checkpoint-epoch-0-step-4.pt", weights_only=False
            )
            self.assertEqual(payload["data_state"]["version"], 1)
            self.assertEqual(payload["data_state"]["world_size"], 1)
            self.assertEqual(payload["data_state"]["ranks"][0]["train"], {"cursor": 4})

    def test_checkpoint_carries_the_normalization_contract(self):
        """A checkpoint must be usable without rebuilding the datasets.

        Actions are trained normalized and partly delta'd against the anchor
        state.  Recovering a command from a prediction therefore needs the
        per-source statistics and action_delta_indices -- both data-side facts the
        model never sees.  If they are not in the payload, inference has to
        reconstruct every reader and rescan it to get back a few kilobytes.
        """
        with tempfile.TemporaryDirectory() as tmp:
            import numpy as np
            import torch

            class _Reader:
                root = "/data/robot_a"
                state_key = "observation.state"
                action_key = "action"
                action_representation = "chunk_delta"
                action_delta_indices = tuple(i for i in range(32) if i not in (6, 13))
                action_rotation_groups = (
                    (17, 18, 19, 20, 21, 22),
                    (26, 27, 28, 29, 30, 31),
                )
                action_relative_rotation_groups = action_rotation_groups
                action_rotation_delta = "compose_v1"
                chunk_size = 50
                meta_stats = {
                    "action": {"q01": np.zeros(32), "q99": np.ones(32)},
                    "observation.state": {"q01": np.zeros(32), "q99": np.ones(32)},
                }

            class _Loader:
                dataset = _Reader()

            class _Iter:
                _loader = _Loader()

            trainer = _trainer(tmp, steps=4, save_every_steps=4)
            trainer.initial_train_dataloader = type("L", (), {"_iters": [_Iter()]})()

            trainer.train_loop(epoch=0)

            payload = torch.load(
                Path(tmp) / "checkpoint-epoch-0-step-4.pt", weights_only=False
            )
            self.assertIn("normalization", payload)
            entry = payload["normalization"]["/data/robot_a"]
            self.assertEqual(entry["action_representation"], "chunk_delta")
            self.assertNotIn(6, entry["action_delta_indices"])
            self.assertNotIn(13, entry["action_delta_indices"])
            self.assertEqual(len(entry["action_delta_indices"]), 30)
            self.assertEqual(
                entry["action_rotation_groups"],
                [[17, 18, 19, 20, 21, 22], [26, 27, 28, 29, 30, 31]],
            )
            self.assertEqual(entry["action_rotation_delta"], "compose_v1")
            self.assertEqual(entry["stats"]["action"]["q99"], [1.0] * 32)

    def test_checkpoint_saves_even_when_norm_metadata_is_unavailable(self):
        """Metadata collection must never be able to kill a run that is otherwise fine."""
        with tempfile.TemporaryDirectory() as tmp:
            import torch

            trainer = _trainer(tmp, steps=4, save_every_steps=4)
            trainer.initial_train_dataloader = object()

            trainer.train_loop(epoch=0)

            payload = torch.load(
                Path(tmp) / "checkpoint-epoch-0-step-4.pt", weights_only=False
            )
            self.assertEqual(payload["step"], 4)
            self.assertNotIn("normalization", payload)

    def test_deployable_model_rejects_checkpoint_without_normalization(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = _trainer(tmp, steps=4, save_every_steps=4)
            trainer.initial_model.requires_normalization_metadata = True
            trainer.initial_train_dataloader = object()

            with self.assertRaisesRegex(
                RuntimeError, "did not expose any dataset statistics"
            ):
                trainer.train_loop(epoch=0)

            self.assertEqual(list(Path(tmp).glob("*.pt")), [])

    def test_save_leaves_no_staging_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = _trainer(tmp, steps=4, save_every_steps=2)

            trainer.train_loop(epoch=0)

            self.assertEqual(list(Path(tmp).glob("*.tmp")), [])

    def test_checkpoint_event_is_logged(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = _trainer(tmp, steps=4, save_every_steps=2)

            trainer.train_loop(epoch=0)

            steps = [
                e["global_step"]
                for e in trainer.metrics_logger.events
                if e["event"] == "checkpoint"
            ]
            self.assertEqual(steps, [2, 4])


@requires_torch
class CheckpointTransportTests(unittest.TestCase):
    def test_checkpoint_releases_host_memory_after_impl_exits_and_on_failure(self):
        from magic_w0.training.dist_trainer import DistributedTrainer

        trainer = DistributedTrainer.__new__(DistributedTrainer)
        trainer.output_dir = Path("unused")
        trainer.model = object()
        trainer.optimizer = object()

        class _CheckpointAllocation:
            pass

        allocation_ref = None
        events = []

        def save_impl(_epoch, _step):
            nonlocal allocation_ref
            allocation = _CheckpointAllocation()
            allocation_ref = weakref.ref(allocation)
            events.append("saved")

        def release():
            self.assertIsNone(allocation_ref())
            events.append("released")

        with (
            patch.object(trainer, "_save_checkpoint_impl", side_effect=save_impl),
            patch.object(
                trainer, "_release_checkpoint_host_memory", side_effect=release
            ),
        ):
            trainer.save_checkpoint(epoch=0, step=1)

        self.assertEqual(events, ["saved", "released"])

        events.clear()
        with (
            patch.object(
                trainer,
                "_save_checkpoint_impl",
                side_effect=RuntimeError("write failed"),
            ),
            patch.object(
                trainer,
                "_release_checkpoint_host_memory",
                side_effect=lambda: events.append("released"),
            ),
            self.assertRaisesRegex(RuntimeError, "write failed"),
        ):
            trainer.save_checkpoint(epoch=0, step=2)

        self.assertEqual(events, ["released"])

    def test_zero2_cpu_payload_is_prepared_before_transfer_and_released_after_send(
        self,
    ):
        import torch.distributed as dist

        import magic_w0.training.dist_trainer.dist_trainer as trainer_module
        from magic_w0.training.dist_trainer import DistributedTrainer

        events = []

        class _LocalOptimizer:
            def state_dict(self):
                events.append("build")
                return {"state": {}, "param_groups": []}

        trainer = DistributedTrainer.__new__(DistributedTrainer)
        trainer.optimizer = SimpleNamespace(
            optim=_LocalOptimizer(),
            param_groups=[],
        )
        checkpoint_group = object()

        def send_object_list(payload, *, dst, group):
            self.assertEqual(dst, 0)
            self.assertIs(group, checkpoint_group)
            self.assertEqual(payload[0]["optimizer"]["state"], {})
            events.append("send")

        with (
            patch.object(trainer_module, "get_world_size", return_value=4),
            patch.object(trainer_module, "get_rank", return_value=2),
            patch.object(dist, "is_available", return_value=True),
            patch.object(dist, "is_initialized", return_value=True),
            patch.object(
                trainer, "_checkpoint_cpu_group", return_value=checkpoint_group
            ),
            patch.object(trainer, "_optimizer_parameter_names", return_value=[]),
            patch.object(dist, "send_object_list", side_effect=send_object_list),
            patch.object(
                dist, "barrier", side_effect=lambda **_: events.append("barrier")
            ),
            patch.object(
                trainer,
                "_release_checkpoint_host_memory",
                side_effect=lambda: events.append("release"),
            ),
        ):
            state = trainer._consolidate_zero2_optimizer_state_cpu()

        self.assertIsNone(state)
        self.assertEqual(
            events,
            ["build", "barrier", "send", "release", "barrier", "barrier"],
        )

    def test_zero2_installs_owner_reduce_overlap_without_ddp(self):
        import torch

        from magic_w0.training.dist_trainer import DistributedTrainer
        from magic_w0.training.optimizer import MuonConfig
        import magic_w0.training.dist_trainer.dist_trainer as trainer_module

        model = torch.nn.Linear(2, 1)
        trainer = DistributedTrainer.__new__(DistributedTrainer)
        trainer.initial_model = model
        trainer.device = torch.device("cpu")
        trainer.rank = 0
        trainer.world_size = 2
        trainer.strategy_config = SimpleNamespace(
            strategy="zero2",
            zero2_overlap_backward=True,
        )
        trainer.optimizer_config = MuonConfig(
            shard_optimizer_state=True,
            shard_ns=False,
        )
        trainer.learning_rate = 1e-3
        trainer.weight_decay = 0.0
        trainer.scheduler_config = None
        trainer.find_unused_parameters = True
        trainer._read_resume_payload = lambda: None
        optimizer = SimpleNamespace(
            param_groups=[{"params": list(model.parameters())}],
            optim=SimpleNamespace(defaults={"shard_bucket_mib": 1}),
            _param_to_rank={parameter: 0 for parameter in model.parameters()},
        )
        trainer.configure_optimizer = lambda **_kwargs: setattr(
            trainer, "optimizer", optimizer
        )

        ddp_calls = []

        def wrap_ddp(wrapped_model, **kwargs):
            ddp_calls.append((wrapped_model, kwargs))
            return SimpleNamespace(module=wrapped_model)

        with (
            patch.object(trainer_module, "get_world_size", return_value=2),
            patch.object(torch.nn.parallel, "DistributedDataParallel", wrap_ddp),
            patch.object(torch.distributed, "broadcast"),
        ):
            trainer.load_model()

        self.assertEqual(ddp_calls, [])
        self.assertIs(trainer.model, model)
        self.assertIs(
            trainer._zero2_gradient_overlap.optimizer,
            optimizer,
        )

    def test_zero2_overlap_disabled_uses_manual_reduction(self):
        import torch

        from magic_w0.training.dist_trainer import DistributedTrainer
        from magic_w0.training.optimizer import MuonConfig
        import magic_w0.training.dist_trainer.dist_trainer as trainer_module

        model = torch.nn.Linear(2, 1)
        trainer = DistributedTrainer.__new__(DistributedTrainer)
        trainer.initial_model = model
        trainer.device = torch.device("cpu")
        trainer.strategy_config = SimpleNamespace(
            strategy="zero2",
            zero2_overlap_backward=False,
        )
        trainer.optimizer_config = MuonConfig(
            shard_optimizer_state=True,
            shard_ns=False,
        )
        trainer.learning_rate = 1e-3
        trainer.weight_decay = 0.0
        trainer.scheduler_config = None
        trainer._read_resume_payload = lambda: None
        trainer.configure_optimizer = lambda **_kwargs: None
        grad_modes = []

        with (
            patch.object(trainer_module, "get_world_size", return_value=2),
            patch.object(
                torch.distributed,
                "broadcast",
                side_effect=lambda *_args, **_kwargs: grad_modes.append(
                    torch.is_grad_enabled()
                ),
            ),
        ):
            trainer.load_model()

        self.assertIs(trainer.model, model)
        self.assertIsNone(trainer._zero2_gradient_overlap)
        self.assertEqual(grad_modes, [False] * len(list(model.parameters())))

    def test_zero2_checkpoint_writes_and_restores_consolidated_optimizer_file(self):
        import torch

        from magic_w0.training.dist_trainer import DistributedTrainer, TrainerState

        with tempfile.TemporaryDirectory() as tmp:
            model = torch.nn.Linear(2, 1)
            local_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            loss = model(torch.ones(1, 2)).sum()
            loss.backward()
            local_optimizer.step()

            class _Zero2Optimizer:
                def __init__(self, optim):
                    self.optim = optim
                    self.param_groups = optim.param_groups

                @staticmethod
                def _sync_param_groups(source, destination):
                    for source_group, destination_group in zip(
                        source, destination, strict=True
                    ):
                        for key, value in source_group.items():
                            if key != "params":
                                destination_group[key] = value

                def consolidate_state_dict(self, to=0):
                    raise AssertionError("checkpoint consolidation must stay on CPU")

                def state_dict(self):
                    raise AssertionError("zero2 wrapper state_dict must not be used")

                def load_state_dict(self, state):
                    self.optim.load_state_dict(state)

            trainer = DistributedTrainer.__new__(DistributedTrainer)
            trainer.output_dir = Path(tmp)
            trainer.model = model
            trainer.initial_model = model
            trainer.optimizer = _Zero2Optimizer(local_optimizer)
            trainer.strategy_config = SimpleNamespace(strategy="zero2")
            trainer.device = torch.device("cpu")
            trainer.state = TrainerState(global_step=2)
            trainer.lr_scheduler = None
            trainer.initial_train_dataloader = None
            trainer.initial_val_dataloader = None
            trainer.metrics_logger = _Logger()
            trainer.keep_last_n_checkpoints = None
            trainer.save_milestone_every_steps = None

            trainer.save_checkpoint(epoch=0, step=2)

            checkpoint = Path(tmp) / "checkpoint-epoch-0-step-2.pt"
            optimizer_file = Path(tmp) / "checkpoint-epoch-0-step-2.optimizer.pt"
            payload = torch.load(checkpoint, weights_only=False)
            optimizer_payload = torch.load(optimizer_file, weights_only=False)
            self.assertNotIn("optimizer", payload)
            self.assertEqual(payload["optimizer_file"], optimizer_file.name)
            self.assertTrue(optimizer_file.is_file())
            self.assertTrue(
                all(
                    not torch.is_tensor(value) or value.device.type == "cpu"
                    for state in optimizer_payload["state"].values()
                    for value in state.values()
                )
            )

            restored_model = torch.nn.Linear(2, 1)
            restored_local_optimizer = torch.optim.AdamW(
                restored_model.parameters(), lr=9e-3
            )
            restored = DistributedTrainer.__new__(DistributedTrainer)
            restored.optimizer = _Zero2Optimizer(restored_local_optimizer)
            restored.lr_scheduler = None
            restored.state = TrainerState()
            restored._resume_path = checkpoint
            restored.initial_train_dataloader = None
            restored.initial_val_dataloader = None
            restored.metrics_logger = _Logger()

            restored._restore_training_state(payload)

            self.assertEqual(restored.state.global_step, 2)
            self.assertEqual(len(restored_local_optimizer.state), 2)
            self.assertEqual(restored_local_optimizer.param_groups[0]["lr"], 1e-3)

    def test_zero2_milestone_embeds_optimizer_in_single_ddp_compatible_file(self):
        import torch

        from magic_w0.training.dist_trainer import DistributedTrainer, TrainerState

        model = torch.nn.Linear(2, 1)
        local_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        model(torch.ones(1, 2)).sum().backward()
        local_optimizer.step()

        class _Zero2Optimizer:
            def __init__(self, optim):
                self.optim = optim
                self.param_groups = optim.param_groups

            def consolidate_state_dict(self, to=0):
                raise AssertionError("checkpoint consolidation must stay on CPU")

            def state_dict(self):
                raise AssertionError("zero2 wrapper state_dict must not be used")

        with tempfile.TemporaryDirectory() as tmp:
            trainer = DistributedTrainer.__new__(DistributedTrainer)
            trainer.output_dir = Path(tmp)
            trainer.model = model
            trainer.initial_model = model
            trainer.optimizer = _Zero2Optimizer(local_optimizer)
            trainer.strategy_config = SimpleNamespace(strategy="zero2")
            trainer.device = torch.device("cpu")
            trainer.state = TrainerState(global_step=10)
            trainer.lr_scheduler = None
            trainer.initial_train_dataloader = None
            trainer.initial_val_dataloader = None
            trainer.metrics_logger = _Logger()
            trainer.keep_last_n_checkpoints = None
            trainer.save_milestone_every_steps = 10

            stale_optimizer_file = Path(tmp) / "checkpoint-epoch-0-step-10.optimizer.pt"
            stale_optimizer_file.touch()
            stale_shard_dir = Path(tmp) / "checkpoint-epoch-0-step-10.optimizer"
            stale_shard_dir.mkdir()
            (stale_shard_dir / "rank-0.pt").touch()

            trainer.save_checkpoint(epoch=0, step=10)

            checkpoint = Path(tmp) / "checkpoint-epoch-0-step-10.pt"
            payload = torch.load(checkpoint, weights_only=False)
            self.assertIn("model", payload)
            self.assertIn("optimizer", payload)
            self.assertFalse(stale_optimizer_file.exists())
            self.assertFalse(stale_shard_dir.exists())

    def test_zero2_rolling_overwrite_never_leaves_a_mixed_checkpoint_pair(self):
        import torch

        import magic_w0.training.dist_trainer.dist_trainer as trainer_module
        from magic_w0.training.dist_trainer import DistributedTrainer, TrainerState

        model = torch.nn.Linear(2, 1)
        local_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        class _Zero2Optimizer:
            def __init__(self, optim):
                self.optim = optim
                self.param_groups = optim.param_groups

            def consolidate_state_dict(self, to=0):
                raise AssertionError("checkpoint consolidation must stay on CPU")

            def state_dict(self):
                raise AssertionError("zero2 wrapper state_dict must not be used")

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "checkpoint-epoch-0-step-2.pt"
            torch.save({"old": True}, checkpoint)
            trainer = DistributedTrainer.__new__(DistributedTrainer)
            trainer.output_dir = Path(tmp)
            trainer.model = model
            trainer.initial_model = model
            trainer.optimizer = _Zero2Optimizer(local_optimizer)
            trainer.strategy_config = SimpleNamespace(strategy="zero2")
            trainer.device = torch.device("cpu")
            trainer.state = TrainerState(global_step=2)
            trainer.lr_scheduler = None
            trainer.initial_train_dataloader = None
            trainer.initial_val_dataloader = None
            trainer.metrics_logger = _Logger()
            trainer.keep_last_n_checkpoints = None
            trainer.save_milestone_every_steps = None
            real_save = trainer_module.save_checkpoint

            def fail_main_write(path, payload):
                if Path(path) == checkpoint:
                    raise RuntimeError("interrupted main checkpoint write")
                real_save(path, payload)

            with (
                patch.object(
                    trainer_module, "save_checkpoint", side_effect=fail_main_write
                ),
                self.assertRaisesRegex(RuntimeError, "interrupted"),
            ):
                trainer.save_checkpoint(epoch=0, step=2)

            self.assertFalse(checkpoint.exists())
            self.assertTrue(
                (Path(tmp) / "checkpoint-epoch-0-step-2.optimizer.pt").is_file()
            )

    def test_checkpoint_rotation_removes_zero2_optimizer_artifacts(self):
        from magic_w0.training.dist_trainer import DistributedTrainer

        with tempfile.TemporaryDirectory() as tmp:
            trainer = DistributedTrainer.__new__(DistributedTrainer)
            trainer.output_dir = Path(tmp)
            trainer.keep_last_n_checkpoints = 1
            trainer.save_milestone_every_steps = None
            for step in (2, 4):
                checkpoint = Path(tmp) / f"checkpoint-epoch-0-step-{step}.pt"
                checkpoint.touch()
                shard_dir = Path(tmp) / f"checkpoint-epoch-0-step-{step}.optimizer"
                shard_dir.mkdir()
                (shard_dir / "rank-0.pt").touch()
                (Path(tmp) / f"checkpoint-epoch-0-step-{step}.optimizer.pt").touch()

            trainer._prune_checkpoints()

            self.assertFalse(
                (Path(tmp) / "checkpoint-epoch-0-step-2.optimizer").exists()
            )
            self.assertTrue(
                (Path(tmp) / "checkpoint-epoch-0-step-4.optimizer").is_dir()
            )
            self.assertFalse(
                (Path(tmp) / "checkpoint-epoch-0-step-2.optimizer.pt").exists()
            )
            self.assertTrue(
                (Path(tmp) / "checkpoint-epoch-0-step-4.optimizer.pt").is_file()
            )

    def test_rank_local_cursors_use_one_cached_gloo_group(self):
        import torch.distributed as dist

        import magic_w0.training.dist_trainer.dist_trainer as trainer_module
        from magic_w0.training.dist_trainer import DistributedTrainer

        class _Loader:
            def state_dict(self):
                return {"cursor": 17}

        trainer = DistributedTrainer.__new__(DistributedTrainer)
        trainer.initial_train_dataloader = _Loader()
        trainer.initial_val_dataloader = None
        cpu_group = object()

        def gather_object(local_state, gathered, *, dst, group):
            self.assertEqual(dst, 0)
            self.assertIs(group, cpu_group)
            gathered[:] = [local_state, {"rank": 1, "train": {"cursor": 23}}]

        with (
            patch.object(trainer_module, "get_world_size", return_value=2),
            patch.object(trainer_module, "get_rank", return_value=0),
            patch.object(trainer_module, "is_main_process", return_value=True),
            patch.object(dist, "is_available", return_value=True),
            patch.object(dist, "is_initialized", return_value=True),
            patch.object(dist, "is_gloo_available", return_value=True),
            patch.object(dist, "new_group", return_value=cpu_group) as new_group,
            patch.object(dist, "gather_object", side_effect=gather_object) as gather,
        ):
            first = trainer._gather_data_state()
            second = trainer._gather_data_state()

        new_group.assert_called_once_with(backend="gloo")
        self.assertEqual(gather.call_count, 2)
        self.assertIs(trainer._checkpoint_gloo_group, cpu_group)
        self.assertEqual(first["ranks"][0]["train"], {"cursor": 17})
        self.assertEqual(first["ranks"][1]["train"], {"cursor": 23})
        self.assertEqual(second, first)

    def test_checkpoint_releases_cuda_cache_before_data_state_gather(self):
        import torch

        import magic_w0.training.dist_trainer.dist_trainer as trainer_module
        from magic_w0.training.dist_trainer import DistributedTrainer

        trainer = DistributedTrainer.__new__(DistributedTrainer)
        trainer.output_dir = Path("unused")
        trainer.model = object()
        trainer.optimizer = object()
        trainer.device = torch.device("cuda", 0)
        order = []

        with (
            patch.object(torch.cuda, "is_available", return_value=True),
            patch.object(
                torch.cuda,
                "synchronize",
                side_effect=lambda device: order.append("sync"),
            ) as synchronize,
            patch.object(
                torch.cuda, "empty_cache", side_effect=lambda: order.append("empty")
            ) as empty_cache,
            patch.object(
                trainer,
                "_gather_data_state",
                side_effect=lambda: order.append("gather"),
            ),
            patch.object(trainer_module, "is_main_process", return_value=False),
        ):
            trainer.save_checkpoint(epoch=0, step=2)

        synchronize.assert_called_once_with(trainer.device)
        empty_cache.assert_called_once_with()
        self.assertEqual(order, ["sync", "empty", "gather"])


@requires_torch
class CheckpointConfigValidationTests(unittest.TestCase):
    @staticmethod
    def _build(**kwargs):
        from torch import nn

        from magic_w0.training.dist_trainer import DistributedTrainer

        return DistributedTrainer(nn.Linear(2, 2), train_dataloader=[], **kwargs)

    def test_rejects_non_positive_save_every_steps(self):
        with self.assertRaises(ValueError):
            self._build(save_every_steps=0)

    def test_rejects_non_positive_keep_last_n(self):
        with self.assertRaises(ValueError):
            self._build(keep_last_n_checkpoints=0)

    def test_rejects_non_positive_milestone_interval(self):
        with self.assertRaises(ValueError):
            self._build(save_milestone_every_steps=0)


if __name__ == "__main__":
    unittest.main()
