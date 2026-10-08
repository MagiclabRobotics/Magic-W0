"""RoboDojo inference adapter components."""

from __future__ import annotations
from typing import Any
import numpy as np
from .rotation6d import matrix_to_rot6d, rot6d_to_matrix


class DeltaNormalizationStats:
    @classmethod
    def from_checkpoint(
        cls, payload: dict[str, Any], *, source_hint: str = "RoboDojo"
    ) -> DeltaNormalizationStats:
        """Load the exact per-source stats embedded by the training checkpoint."""
        sources = payload.get("normalization") or {}
        if not isinstance(sources, dict) or not sources:
            raise KeyError("checkpoint does not contain normalization metadata")
        matches = [
            (name, entry)
            for name, entry in sources.items()
            if source_hint.lower() in str(name).lower()
        ]
        if not matches:
            if len(sources) != 1:
                raise KeyError(
                    f"no normalization source matching {source_hint!r}; "
                    f"available={list(sources)}"
                )
            matches = list(sources.items())
        # Prefer a real dataset path over wrapper-class fallback entries.
        _, entry = min(matches, key=lambda item: str(item[0]).startswith("<"))
        return cls.from_entry(entry)

    @classmethod
    def from_entry(cls, entry: dict[str, Any]) -> DeltaNormalizationStats:
        """Build the normalization contract from one checkpoint source entry."""
        stats = entry.get("stats", entry)
        instance = cls.__new__(cls)
        representation = str(entry.get("action_representation", ""))
        if representation != "chunk_delta":
            raise ValueError(
                "delta-only inference requires checkpoint "
                f"action_representation='chunk_delta', got {representation!r}"
            )
        instance.action_delta_indices = tuple(
            int(index) for index in entry.get("action_delta_indices", ())
        )
        instance.action_rotation_groups = tuple(
            tuple(int(index) for index in group)
            for group in entry.get("action_rotation_groups", ())
        )
        instance.action_stats_layout = str(entry.get("action_stats_layout", "per_dim"))
        instance.chunk_size = (
            int(entry["chunk_size"]) if entry.get("chunk_size") is not None else None
        )
        instance.target_state_dim = (
            int(entry["target_state_dim"])
            if entry.get("target_state_dim") is not None
            else None
        )
        instance.target_action_dim = (
            int(entry["target_action_dim"])
            if entry.get("target_action_dim") is not None
            else None
        )
        instance._load(stats)
        return instance

    def _load(self, stats: dict[str, Any]) -> None:
        self.state_low, self.state_high = self._bounds(stats["observation.state"])
        self.action_low, self.action_high = self._bounds(stats["action"])

    @staticmethod
    def _bounds(stats: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        low = np.asarray(
            stats["q01"] if "q01" in stats else stats["min"], dtype=np.float32
        )
        high = np.asarray(
            stats["q99"] if "q99" in stats else stats["max"], dtype=np.float32
        )
        if low.shape != high.shape or np.any(high - low < 1e-6):
            raise ValueError("invalid normalization bounds")
        return low, high

    def normalize_state(self, state: np.ndarray) -> np.ndarray:
        if state.shape[-1] != self.state_low.shape[0]:
            raise ValueError(
                f"state dim mismatch: expected {self.state_low.shape[0]}, "
                f"got {state.shape[-1]}"
            )
        scale = np.maximum(self.state_high - self.state_low, 1e-6)
        return np.clip((state - self.state_low) / scale * 2.0 - 1.0, -1.0, 1.0).astype(
            np.float32
        )

    def denormalize_action(
        self, action: np.ndarray, *, clip: bool = False
    ) -> np.ndarray:
        if action.shape[-1] != self.action_low.shape[0]:
            raise ValueError(
                f"action dim mismatch: expected {self.action_low.shape[0]}, "
                f"got {action.shape[-1]}"
            )
        if clip:
            action = np.clip(action, -1.0, 1.0)
        return (
            (action + 1.0) * 0.5 * (self.action_high - self.action_low)
            + self.action_low
        ).astype(np.float32)

    def reconstruct_joint_targets(
        self, action: np.ndarray, current_state: np.ndarray
    ) -> np.ndarray:
        """Reconstruct checkpoint-declared deltas in the shared action space.

        The policy uses a 32-D union schema, while RoboDojo joint observations
        provide 14 measured dimensions. The measured state is therefore padded
        with zero anchors before reconstruction; the unmeasured union tail is
        discarded when the server returns the robot's 14-D joint command.
        """
        values = np.asarray(action, dtype=np.float32).copy()
        state = np.asarray(current_state, dtype=np.float32).reshape(-1)
        if self.action_rotation_groups:
            raise ValueError(
                "RoboDojo joint inference does not support relative rotation groups"
            )
        indices = self.action_delta_indices
        if not indices:
            raise ValueError("chunk_delta checkpoint has no action_delta_indices")
        if min(indices) < 0 or max(indices) >= values.shape[-1]:
            raise ValueError(
                f"delta indices {indices} exceed action dimension {values.shape[-1]}"
            )
        if state.shape[0] > values.shape[-1]:
            raise ValueError(
                f"state dimension {state.shape[0]} exceeds action dimension "
                f"{values.shape[-1]}"
            )
        anchor = np.zeros(values.shape[-1], dtype=np.float32)
        anchor[: state.shape[0]] = state
        values[..., list(indices)] += anchor[list(indices)]
        return values

    def reconstruct_camera_eef_targets(
        self, action: np.ndarray, current_state: np.ndarray
    ) -> np.ndarray:
        """Reconstruct camera EEF deltas, composing rotations on SO(3)."""
        values = np.asarray(action, dtype=np.float32).copy()
        state = np.asarray(current_state, dtype=np.float32).reshape(-1)
        if values.ndim != 2 or state.shape[0] > values.shape[-1]:
            raise ValueError(
                f"expected camera EEF state no wider than actions, got "
                f"{state.shape}/{values.shape}"
            )
        if (
            not self.action_delta_indices
            or max(self.action_delta_indices) >= state.shape[0]
        ):
            raise ValueError(
                f"camera EEF delta indices {self.action_delta_indices} exceed "
                f"state dimension {state.shape[0]}"
            )
        delta_set = set(self.action_delta_indices)
        rotated = {index for group in self.action_rotation_groups for index in group}
        vector_indices = [
            index for index in self.action_delta_indices if index not in rotated
        ]
        values[:, vector_indices] += state[vector_indices]
        for group in self.action_rotation_groups:
            if not delta_set.issuperset(group):
                raise ValueError(f"rotation group {group} is not fully delta encoded")
            columns = list(group)
            relative = rot6d_to_matrix(values[:, columns])
            reference = rot6d_to_matrix(state[columns])
            values[:, columns] = matrix_to_rot6d(relative @ reference).astype(
                np.float32
            )
        return values
