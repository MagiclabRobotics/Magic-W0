# Copyright 2026 MagicVLA authors. (Apache-2.0.)
"""Normalization metadata must identify real sources, not transparent wrappers."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from magic_w0.data.norm_export import collect_norm_metadata  # noqa: E402


def _stats() -> dict[str, dict[str, np.ndarray]]:
    return {
        "observation.state": {
            "q01": np.zeros(2, dtype=np.float32),
            "q99": np.ones(2, dtype=np.float32),
        },
        "action": {
            "q01": np.zeros(2, dtype=np.float32),
            "q99": np.ones(2, dtype=np.float32),
        },
    }


class _Reader:
    def __init__(self) -> None:
        self.root = Path("/datasets/robodojo")
        self.meta_stats = _stats()
        self.state_key = "observation.state"
        self.action_key = "action"
        self.action_representation = "chunk_delta"
        self.action_delta_indices = (0, 1)
        self.chunk_size = 50
        self.target_state_dim = 2
        self.target_action_dim = 2


class _MetadataOnlyWrapper:
    """Match MagicW0's wrapper: copied stats, but no source/action contract."""

    def __init__(self, dataset) -> None:
        self.dataset = dataset
        self.meta_stats = dataset.meta_stats


class _ContractOwningWrapper:
    """A transforming wrapper is authoritative and shadows its child."""

    def __init__(self, dataset) -> None:
        self.dataset = dataset
        self.meta_stats = _stats()
        self.state_key = "observation.state"
        self.action_key = "action"
        self.action_representation = "chunk_delta"
        self.action_delta_indices = (1,)


class _Loader:
    def __init__(self, dataset) -> None:
        self.dataset = dataset


class NormalizationExportTests(unittest.TestCase):
    def test_metadata_only_wrapper_does_not_create_a_fake_source(self) -> None:
        metadata = collect_norm_metadata(_Loader(_MetadataOnlyWrapper(_Reader())))

        self.assertEqual(list(metadata), ["/datasets/robodojo"])
        self.assertNotIn("<_MetadataOnlyWrapper>", metadata)
        self.assertEqual(
            metadata["/datasets/robodojo"]["action_delta_indices"],
            [0, 1],
        )

    def test_contract_owning_wrapper_shadows_its_child(self) -> None:
        metadata = collect_norm_metadata(_Loader(_ContractOwningWrapper(_Reader())))

        self.assertEqual(list(metadata), ["<_ContractOwningWrapper>"])
        self.assertEqual(
            metadata["<_ContractOwningWrapper>"]["action_delta_indices"],
            [1],
        )


if __name__ == "__main__":
    unittest.main()
