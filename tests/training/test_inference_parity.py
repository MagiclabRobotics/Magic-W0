"""Compare training-side sampling with the existing public inference runtime."""

import copy
import torch
from torch import nn
from transformers import PretrainedConfig

from magic_w0.modeling import MagicW0
from magic_w0.modules import (
    HybridActionExpert,
    Hybrid2DWorldExpert,
    Hybrid3DWorldExpert,
)
from magic_w0.config import (
    STATE as PUBLIC_STATE,
    VIEW_VALID_MASK as PUBLIC_VIEWS,
    WORLD_3D_VALID_MASK as PUBLIC_WORLD,
)
from magic_w0.models.action_backbone.processor_magicw0_base import STATE
from magic_w0.models.world_action.processor_magicw0 import (
    VIEW_VALID_MASK,
    WORLD_3D_VALID_MASK,
)
from .test_magicw0 import _bare_cached_magicw0_policy, _install_fake_magicw0_prefix


def test_same_weights_prefix_and_noise_produce_same_actions():
    torch.manual_seed(31)
    training = _bare_cached_magicw0_policy().eval()
    training.vlm.model.language_model.config = PretrainedConfig(is_decoder=True)
    training.vlm.model.language_model.config._attn_implementation = "sdpa"
    batch = _install_fake_magicw0_prefix(training)
    runtime = MagicW0.__new__(MagicW0)
    nn.Module.__init__(runtime)
    runtime.config = training.config
    runtime.vlm = copy.deepcopy(training.vlm)
    runtime.action_residual_dropout = nn.Identity()
    experts = {
        "action_expert": HybridActionExpert(training.config),
        "world_2d_expert": Hybrid2DWorldExpert(training.config, 12),
        "world_3d_expert": Hybrid3DWorldExpert(training.config),
    }
    for name, module in experts.items():
        module.load_state_dict(getattr(training, name).state_dict(), strict=True)
        setattr(runtime, name, module)
    runtime._embed_vlm_inputs = training._embed_vlm_inputs
    runtime.eval()
    public_batch = {
        **batch,
        PUBLIC_STATE: batch[STATE],
        PUBLIC_VIEWS: batch[VIEW_VALID_MASK],
        PUBLIC_WORLD: batch[WORLD_3D_VALID_MASK],
    }
    noise = torch.randn(2, 4, 4)
    with torch.no_grad():
        expected = training.generate_actions(batch, num_steps=3, initial_noise=noise)
        actual = runtime.generate_actions(
            public_batch, num_steps=3, initial_noise=noise
        )
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
