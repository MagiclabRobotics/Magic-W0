"""Training and deployment must use identical quantile scales, even for constants."""

import numpy as np
import pytest
import torch

from eval.robodojo.normalization import DeltaNormalizationStats
from magic_w0.models._normalization import normalize_tensor


def test_explicit_training_epsilon_matches_runtime_and_legacy_remains_strict():
    stats = {"q01": [1.0, 0.0], "q99": [1.0, 2.0]}
    entry = {
        "action_representation": "chunk_delta",
        "action_delta_indices": [0, 1],
        "stats": {"observation.state": stats, "action": stats},
    }
    with pytest.raises(ValueError, match="bounds"):
        DeltaNormalizationStats.from_entry(entry)
    runtime = DeltaNormalizationStats.from_entry(
        {**entry, "quantile_scale_epsilon": 1e-6}
    )
    raw = torch.tensor([1.0, 1.0])
    np.testing.assert_allclose(
        runtime.normalize_state(raw.numpy()),
        normalize_tensor(raw, stats, "quantile").numpy(),
    )
    predictions = torch.tensor([[0.0, 0.25], [-1.0, 1.0]])
    expected = (predictions + 1) * 0.5 * (
        torch.tensor(stats["q99"]) - torch.tensor(stats["q01"])
    ).clamp_min(1e-6) + torch.tensor(stats["q01"])
    np.testing.assert_allclose(
        runtime.denormalize_action(predictions.numpy()), expected.numpy(), rtol=1e-6
    )
