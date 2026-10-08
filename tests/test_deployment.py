"""Exercise the simulator boundary with two asynchronous episode lifetimes."""

import importlib.util
from pathlib import Path
import sys
from types import ModuleType

import numpy as np
import pytest


@pytest.fixture
def boundary(monkeypatch):
    # The external helpers are replaced by an identity-rotation camera whose
    # translation changes over time. This isolates the action-chunk contract.
    camera = ModuleType("XPolicyLab.utils.camera_extrinsics")
    camera._as_numpy = np.asarray
    camera.camera_to_world_matrix = lambda value: value.copy()
    pose = ModuleType("XPolicyLab.utils.pose_transform")

    def shift(value, transform, sign):
        result = np.asarray(value).copy()
        result[:3] += sign * transform[:3, 3]
        return result

    pose.world_pose_to_camera_pose = lambda value, transform: shift(
        value, transform, -1
    )
    pose.camera_pose_to_world_pose = lambda value, transform: shift(value, transform, 1)
    for name, module in [(camera.__name__, camera), (pose.__name__, pose)]:
        monkeypatch.setitem(sys.modules, name, module)
    root = Path(__file__).resolve().parents[1] / "eval/robodojo"

    def load(name, file):
        spec = importlib.util.spec_from_file_location(name, root / file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    adapter = load("eval.robodojo._adapter_test", "robodojo_adapter.py")
    monkeypatch.setitem(sys.modules, "eval.robodojo.robodojo_adapter", adapter)
    deploy = load("eval.robodojo._deploy_test", "deploy.py")
    return adapter, deploy


class Environment:
    def __init__(self):
        self.steps = {0: 0, 1: 0}
        self.actions = []
        self.env_origins = np.zeros((2, 3))
        self.camera_manager = type("Manager", (), {})()
        self.camera_manager.camera_names = [["cam_head"], ["cam_head"]]
        self.camera_manager.cameras = [[np.eye(4)], [np.eye(4)]]

    def is_episode_end(self):
        return not self.get_running_env_idx_list()

    def get_running_env_idx_list(self):
        return [idx for idx in (0, 1) if self.steps[idx] < idx + 1]

    def get_obs_batch(self, indices):
        result = []
        for idx in indices:
            self.camera_manager.cameras[idx][0][0, 3] = 10 * idx + self.steps[idx]
            result.append(
                {
                    "env_idx": idx,
                    "state": {"left_ee_pose": np.array([30.0, 0, 0, 1, 0, 0, 0])},
                }
            )
        return result

    def take_action_batch(self, actions, indices):
        for idx, action in zip(indices, actions, strict=True):
            self.actions.append((idx, action["left_ee_pose"].copy()))
            self.steps[idx] += 1


class Client:
    def __init__(self):
        self.observation_batches = []

    def call(self, *, func_name, obs=None):
        if func_name == "update_obs_batch":
            self.observation_batches.append(obs)
        if func_name == "get_action_batch":
            return [
                [
                    {"left_camera_ee_pose": np.array([1.0, 0, 0, 1, 0, 0, 0])}
                    for _ in range(2)
                ]
                for _ in obs
            ]


def test_batch_removes_completed_environments_and_freezes_camera(boundary):
    _, deploy = boundary
    env, client = Environment(), Client()
    deploy.eval_one_episode_batch(env, client)
    assert [idx for idx, _ in env.actions] == [0, 1, 1]
    # Env 1's camera moves from x=10 to x=11 between actions. Both targets
    # still use the planning transform at x=10, rather than the latest pose.
    assert [float(pose[0]) for _, pose in env.actions] == [1, 11, 11]
    assert [obs["env_idx"] for obs in client.observation_batches[-1]] == [1]


def test_boundary_preserves_observation_and_requires_planning_transform(boundary):
    adapter, _ = boundary
    env = Environment()
    env.env_origins[0, 0] = 2
    observation = env.get_obs_batch([0])[0]
    prepared = adapter.prepare_observation(env, observation)
    assert "left_camera_ee_pose" not in observation["state"]
    assert prepared["state"]["left_camera_ee_pose"][0] == 32
    action = {"left_camera_ee_pose": np.array([1.0, 0, 0, 1, 0, 0, 0])}
    with pytest.raises(ValueError, match="planning"):
        adapter.prepare_action(env, action)
    with pytest.raises(ValueError, match="Conflicting"):
        adapter.prepare_action(
            env,
            {**action, "left_ee_pose": action["left_camera_ee_pose"]},
            camera_to_env=np.eye(4),
        )
