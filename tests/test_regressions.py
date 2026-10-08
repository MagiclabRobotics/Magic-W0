"""Regressions for camera assignment, readiness identity and process trees."""

import importlib
import itertools
import json
import os
import signal
import socket
import subprocess
import sys
import time
from types import ModuleType, SimpleNamespace
import numpy as np
import pytest
from eval.robodojo.run import check_port_available, wait_ready, stop_process

CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")


@pytest.fixture
def adapter_model(monkeypatch):
    template = ModuleType("XPolicyLab.model_template")
    template.ModelTemplate = object
    monkeypatch.setitem(sys.modules, template.__name__, template)
    torch = ModuleType("torch")

    class Tensor:
        def __init__(self, array):
            self.array = array

        def to(self, **kwargs):
            return self

    torch.from_numpy = Tensor
    monkeypatch.setitem(sys.modules, "torch", torch)
    import magic_w0

    monkeypatch.setitem(magic_w0.__dict__, "_prepare_image", lambda value, size: value)
    model = importlib.import_module("eval.robodojo.policy").Model.__new__(
        importlib.import_module("eval.robodojo.policy").Model
    )
    model.robot_action_dim = model.state_dim = model.action_dim = 20
    model.chunk_size = 50
    model.state_active_robot_indices = model.action_active_robot_indices = tuple(
        range(20)
    )
    model.stats = SimpleNamespace(normalize_state=lambda value: value)
    model.embodiment = "arx_x5_sim"
    model.device = "cpu"
    model.history = None
    model.save_model_inputs = False
    model.policy = SimpleNamespace(prepare_inference_batch=lambda value, device: value)
    return model


@pytest.mark.parametrize(
    "cameras", [*itertools.permutations(CAMERAS), (CAMERAS[0],), CAMERAS[:2]]
)
def test_camera_identity_is_independent_of_order_or_count(adapter_model, cameras):
    model = adapter_model
    keys = ["observation.images." + camera for camera in cameras]
    model.policy.config = SimpleNamespace(camera_keys=keys, image_size=256)
    observations = [
        {
            "state": np.zeros(20),
            "task": "stack bowls",
            "images": {
                name: np.full((3, 2, 2), i + env * 10, np.uint8)
                for i, name in enumerate(CAMERAS)
            },
        }
        for env in range(2)
    ]
    batch = model._prepare_model_batch([0, 1], observations)
    for key in keys:
        expected = CAMERAS.index(key.removeprefix("observation.images."))
        assert batch[key].array[0].mean() == expected
        assert batch[key].array[1].mean() == expected + 10


def test_unknown_camera_fails_before_model_execution(adapter_model):
    adapter_model.policy.config = SimpleNamespace(
        camera_keys=["observation.images.unknown"], image_size=256
    )
    with pytest.raises(ValueError, match="unsupported model camera"):
        adapter_model._prepare_model_batch(
            [0], [{"state": np.zeros(20), "task": "test", "images": {}}]
        )


def test_unrelated_listener_is_rejected_without_ready_record(tmp_path):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]
        with pytest.raises(RuntimeError, match="occupied"):
            check_port_available("127.0.0.1", port)
        process = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            start_new_session=True,
        )
        try:
            with pytest.raises(TimeoutError):
                wait_ready(
                    process,
                    "127.0.0.1",
                    port,
                    0.2,
                    ready_file=tmp_path / "ready.json",
                    run_id="current",
                )
        finally:
            stop_process(process)


def test_stale_readiness_record_is_rejected(tmp_path):
    record = tmp_path / "ready.json"
    record.write_text(json.dumps({"run_id": "old", "url": "ws://127.0.0.1:19000"}))
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True
    )
    try:
        with pytest.raises(RuntimeError, match="does not match"):
            wait_ready(
                process, "127.0.0.1", 19000, 1, ready_file=record, run_id="current"
            )
    finally:
        stop_process(process)


def test_cleanup_kills_term_ignoring_descendant_after_leader_exits(tmp_path):
    done = tmp_path / "survived"
    child = f"import signal,time; from pathlib import Path; signal.signal(signal.SIGTERM,signal.SIG_IGN); print('ready',flush=True); time.sleep(1); Path({str(done)!r}).touch(); time.sleep(30)"
    parent = f"import subprocess,sys,time; child=subprocess.Popen([sys.executable,'-c',{child!r}]); print(child.pid,flush=True); time.sleep(30)"
    process = subprocess.Popen(
        [sys.executable, "-c", parent],
        stdout=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        int(process.stdout.readline())
        assert process.stdout.readline().strip() == "ready"
        stop_process(process, grace_seconds=0.2)
        time.sleep(1)
        assert not done.exists(), "Descendant survived cleanup"
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
