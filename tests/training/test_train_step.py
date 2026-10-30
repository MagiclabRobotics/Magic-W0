"""Joint MagicW0 optimization on CPU/CUDA, using small deterministic teacher stubs.

This verifies trainable expert computation and optimizer updates. It does not
claim to validate external teacher models or the production-sized backbone.
"""

import pytest
import torch

from .test_magicw0 import _bare_cached_magicw0_policy, _install_fake_magicw0_prefix
from magic_w0.models.action_backbone.processor_magicw0_base import (
    ACTIONS,
    ACTION_DIM_MASK,
    ACTION_IS_PAD,
)
from magic_w0.models.world_action.processor_magicw0 import (
    FUTURE_2D_PIXEL_VALUES,
    FUTURE_3D_IMAGES,
    GEOMETRY_3D_TARGET,
    MOTION_3D_TARGET,
    WORLD_3D_CURRENT_IMAGES,
    WORLD_3D_VALID_MASK,
)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_joint_losses_backward_update_and_state_reload(device, tmp_path):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    torch.manual_seed(19)
    policy = _bare_cached_magicw0_policy().to(device)
    from transformers import PretrainedConfig

    policy.vlm.model.language_model.config = PretrainedConfig(is_decoder=True)
    policy.vlm.model.language_model.config._attn_implementation = "sdpa"
    policy.config.enable_world_teacher_supervision = True
    policy._training_global_step = 0
    batch = _install_fake_magicw0_prefix(policy)
    original_prefix = policy._embed_vlm_inputs
    policy._embed_vlm_inputs = lambda *args: tuple(
        t.to(device) for t in original_prefix(*args)
    )
    batch = {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }
    batch.update(
        {
            ACTIONS: torch.randn(2, 4, 4, device=device),
            ACTION_DIM_MASK: torch.ones(2, 4, 4, dtype=torch.bool, device=device),
            ACTION_IS_PAD: torch.zeros(2, 4, dtype=torch.bool, device=device),
            FUTURE_2D_PIXEL_VALUES: torch.empty(0, device=device),
            WORLD_3D_CURRENT_IMAGES: torch.empty(0, device=device),
            FUTURE_3D_IMAGES: torch.empty(0, device=device),
        }
    )
    geometry = torch.randn(2, 4, 1024, device=device)
    motion = torch.randn(2, 4, 1024, device=device)
    dino = torch.randn(2, 4, 12, device=device)
    policy._track4world_teacher_targets = lambda *_args: {
        GEOMETRY_3D_TARGET: geometry,
        MOTION_3D_TARGET: motion,
        WORLD_3D_VALID_MASK: batch[WORLD_3D_VALID_MASK],
    }
    policy._dino_patch_targets = lambda *_args: dino
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-3)
    before = policy.action_expert.output_projection.weight.detach().clone()
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        output = policy(batch)
        assert torch.isfinite(output.loss)
        assert {
            "flow_loss",
            "gen_2d_loss",
            "geometry_3d_loss",
            "motion_3d_loss",
        } <= output.metrics.keys()
        output.loss.backward()
        for expert in [
            policy.action_expert,
            policy.world_2d_expert,
            policy.world_3d_expert,
        ]:
            grads = [p.grad for p in expert.parameters() if p.grad is not None]
            assert grads and all(torch.isfinite(g).all() for g in grads)
            assert sum(float(g.abs().sum()) for g in grads) > 0
        optimizer.step()
    assert not torch.equal(before, policy.action_expert.output_projection.weight)
    checkpoint = tmp_path / "state.pt"
    torch.save(
        {"model": policy.state_dict(), "optimizer": optimizer.state_dict()}, checkpoint
    )
    loaded = torch.load(checkpoint, map_location=device, weights_only=True)
    policy.load_state_dict(loaded["model"], strict=True)
    optimizer.load_state_dict(loaded["optimizer"])
