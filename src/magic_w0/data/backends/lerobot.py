# Copyright 2026 MagicVLA authors. (Apache-2.0.)
"""LeRobot data backend (source ``type="lerobot"``) -- fully self-contained.

Dispatches on the per-source ``SubDatasetConfig.lerobot_version`` to our own
standalone readers (no lerobot package import in the data layer):

  * ``"v2.1"`` -> :class:`data.backends.lerobot_v21.LeRobotV21Dataset`
  * ``"v3.x"`` -> :class:`data.backends.lerobot_v30.LeRobotV30Dataset`

Both readers yield the same sample schema (uint8 CHW cameras, float32 state,
``[chunk, dim]`` action stack + ``*_is_pad``, ``task`` string), so mixed-version
sources interleave transparently.
"""

from __future__ import annotations

from magic_w0.data.backends.base import DataBackend
from magic_w0.data.backends.lerobot_v21 import LeRobotV21Dataset
from magic_w0.data.backends.lerobot_v30 import LeRobotV30Dataset
from magic_w0.data.config import SubDatasetConfig


class LeRobotBackend(DataBackend):
    """Source ``type="lerobot"``; dispatches on per-source ``lerobot_version``."""

    version = "v2.1/v3.0"

    def make_dataset(self, sub: SubDatasetConfig):
        version = sub.lerobot_version or "v2.1"
        params = sub.params or {}
        if version.startswith("v2"):
            return LeRobotV21Dataset(
                root=sub.root,
                chunk_size=int(params.get("chunk_size", 50)),
                action_key=params.get("action_key"),
                task_override=params.get("task_override"),
                embodiment=params.get("embodiment"),
                episodes=sub.episodes,
                task_filter=sub.task_filter,
                obs_steps=int(params.get("obs_steps", sub.obs_steps)),
                obs_interval_seconds=float(
                    params.get("obs_interval_seconds", sub.obs_interval_seconds)
                ),
                video_backend=params.get("video_backend", sub.video_backend or "auto"),
                camera_keys=params.get("camera_keys"),
                camera_key_map=params.get("camera_key_map"),
                camera_rotations=params.get("camera_rotations"),
                state_indices=params.get("state_indices"),
                action_indices=params.get("action_indices"),
                preserve_selected_slots=bool(
                    params.get("preserve_selected_slots", False)
                ),
                state_target_indices=params.get("state_target_indices"),
                action_target_indices=params.get("action_target_indices"),
                target_state_dim=params.get("target_state_dim"),
                target_action_dim=params.get("target_action_dim"),
                state_active_indices=params.get("state_active_indices"),
                action_active_indices=params.get("action_active_indices"),
                state_mask_key=params.get("state_mask_key"),
                action_mask_key=params.get("action_mask_key"),
                action_upsample_rate=int(params.get("action_upsample_rate", 1)),
                action_upsample_mode=params.get("action_upsample_mode", "semantic"),
                action_interpolation=params.get("action_interpolation"),
                action_representation=params.get("action_representation", "abs"),
                action_delta_indices=params.get("action_delta_indices"),
                action_supervision_indices=params.get("action_supervision_indices"),
                action_rotation_groups=params.get("action_rotation_groups"),
                action_stats_samples=int(params.get("action_stats_samples", 10_000)),
                allow_norm_stats_dataset_mismatch=bool(
                    params.get("allow_norm_stats_dataset_mismatch", False)
                ),
                quality=params.get("quality"),
                anchor_sampling=params.get("anchor_sampling"),
                norm_stats_path=params.get("norm_stats_path"),
                index_cache_path=params.get("index_cache_path"),
                estimate_action_stats=bool(params.get("estimate_action_stats", True)),
                load_metadata_stats=bool(params.get("load_metadata_stats", True)),
                supervise_episode_end_padding=bool(
                    params.get("supervise_episode_end_padding", False)
                ),
                action_stats_per_horizon=bool(
                    params.get("action_stats_per_horizon", False)
                ),
                load_subtasks=bool(params.get("load_subtasks", False)),
            )
        elif version.startswith("v3"):
            if sub.task_filter is not None:
                raise ValueError(
                    "task_filter is currently supported only by the LeRobot v2.x "
                    "backend; select v3 episodes explicitly instead"
                )
            if params.get("anchor_sampling") is not None:
                raise ValueError(
                    "anchor_sampling is currently supported only for lerobot v2.1; "
                    "v3.x reader still exposes every frame"
                )
            return LeRobotV30Dataset(
                root=sub.root,
                chunk_size=int(params.get("chunk_size", 50)),
                action_key=params.get("action_key"),
                obs_steps=int(params.get("obs_steps", sub.obs_steps)),
                obs_interval_seconds=float(
                    params.get("obs_interval_seconds", sub.obs_interval_seconds)
                ),
                video_backend=params.get("video_backend", sub.video_backend or "auto"),
                supervise_episode_end_padding=bool(
                    params.get("supervise_episode_end_padding", False)
                ),
            )
        raise ValueError(
            f"unsupported lerobot_version {version!r} (expected v2.x / v3.x)"
        )
