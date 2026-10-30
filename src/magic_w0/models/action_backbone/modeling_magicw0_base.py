"""Qwen3.5-2B + a 460M hybrid action expert with MoT joint attention.

Both trunks use the Qwen3.5 schedule ``3 x Gated DeltaNet + 1 x full
attention``, repeated six times.  Linear-attention layers are independent.  At
each full-attention layer the action queries attend to the VLM prefix and the
whole action chunk, while VLM queries retain their original causal attention
and never read action tokens.

Optional VLM co-training batches bypass the action expert and train the shared
Qwen language model with next-token CE; robotics batches retain the joint
flow-matching/FAST path.
"""

from __future__ import annotations

import json
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from magic_w0.models.base import ModelForwardOutput, StandalonePolicy
from magic_w0.modules import (
    TimestepEmbedding,
    RMSNorm as QwenRMSNorm,
    GatedRMSNorm as QwenGatedRMSNorm,
    SwiGLU as QwenSwiGLU,
)
from magic_w0.data.vlm_chat import build_vlm_chat_messages
from magic_w0.models.commons.nn import gated_delta_rule, gated_delta_rule_sequential
from magic_w0.models.action_loss_weighting import action_loss_time_weights

from .configuration_magicw0_base import MagicW0BaseConfig
from magic_w0.data.embodiment import describe

from .processor_magicw0_base import (
    ACTIONS,
    ACTION_DIM_MASK,
    ACTION_IS_PAD,
    CAMERA_VALID,
    STATE,
    EMBODIMENT,
    STATE_DIM_MASK,
    SUBTASK,
    TASK,
    TEXT,
    VQA_CHAT,
    VQA_IMAGES,
    VQA_N_IMAGES,
)

logger = logging.getLogger(__name__)


def _maybe_to(value, device):
    if device is None or not isinstance(value, torch.Tensor):
        return value
    return value.to(device, non_blocking=True)


def _load_fast_action_tokenizer(path: str):
    """Load FAST locally, or resolve an official Hugging Face repo id once."""
    from .fast_action_tokenizer import load_fast_action_tokenizer

    requested = str(path)
    local_path = Path(requested)
    if not local_path.is_dir() and not local_path.is_absolute():
        repository_path = Path(__file__).resolve().parents[4] / local_path
        if repository_path.is_dir():
            local_path = repository_path
    if not local_path.is_dir():
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise ImportError(
                "FAST tokenizer is configured as a Hugging Face repo id, but "
                "huggingface_hub is not installed"
            ) from exc
        local_path = Path(
            snapshot_download(
                repo_id=requested,
                allow_patterns=[
                    "processor_config.json",
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "special_tokens_map.json",
                ],
            )
        )

    return load_fast_action_tokenizer(local_path)


def _l2_normalize(value: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return value * torch.rsqrt(value.square().sum(dim=-1, keepdim=True) + eps)


class QwenGatedDeltaNet(nn.Module):
    """Torch implementation of the Qwen3.5 gated delta-rule mixer.

    The recurrence is solved per chunk rather than per step: the whole action
    chunk is one triangular solve plus two masked matmuls instead of ``T``
    sequential state updates.  The pretrained VLM keeps using Transformers'
    native FLA/causal-conv fast path; this class is only the expert's own
    mixer, which that path never reaches.

    ``_delta_rule_sequential`` is kept as the reference recurrence the parity
    tests check against, and as the arm the A/B bench swaps in.
    """

    _delta_rule_sequential = staticmethod(gated_delta_rule_sequential)
    _delta_rule = staticmethod(gated_delta_rule)

    def __init__(self, config: MagicW0BaseConfig) -> None:
        super().__init__()
        hidden = config.action_expert_hidden_size
        self.num_key_heads = config.deltanet_num_key_heads
        self.num_value_heads = config.deltanet_num_value_heads
        self.key_head_dim = config.deltanet_key_head_dim
        self.value_head_dim = config.deltanet_value_head_dim
        self.key_dim = self.num_key_heads * self.key_head_dim
        self.value_dim = self.num_value_heads * self.value_head_dim
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv_kernel = config.deltanet_conv_kernel

        self.in_proj_qkv = nn.Linear(hidden, self.conv_dim, bias=False)
        self.in_proj_z = nn.Linear(hidden, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(hidden, self.num_value_heads, bias=False)
        self.in_proj_a = nn.Linear(hidden, self.num_value_heads, bias=False)
        self.conv1d = nn.Conv1d(
            self.conv_dim,
            self.conv_dim,
            self.conv_kernel,
            groups=self.conv_dim,
            padding=self.conv_kernel - 1,
            bias=False,
        )
        self.dt_bias = nn.Parameter(torch.ones(self.num_value_heads))
        self.A_log = nn.Parameter(torch.empty(self.num_value_heads))
        self.norm = QwenGatedRMSNorm(
            config.deltanet_value_head_dim, config.rms_norm_eps
        )
        self.out_proj = nn.Linear(self.value_dim, hidden, bias=False)

    def reset_parameters(self) -> None:
        nn.init.normal_(self.in_proj_qkv.weight, std=0.02)
        nn.init.normal_(self.in_proj_z.weight, std=0.02)
        nn.init.normal_(self.in_proj_b.weight, std=0.02)
        nn.init.normal_(self.in_proj_a.weight, std=0.02)
        nn.init.normal_(self.conv1d.weight, std=0.02)
        nn.init.normal_(self.out_proj.weight, std=0.02)
        nn.init.ones_(self.dt_bias)
        with torch.no_grad():
            self.A_log.copy_(torch.empty_like(self.A_log).uniform_(0.001, 16.0).log_())
            self.norm.weight.fill_(1.0)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if attention_mask is not None:
            hidden_states = hidden_states * attention_mask.to(
                hidden_states.dtype
            ).unsqueeze(-1)
        batch, sequence, _ = hidden_states.shape
        mixed = self.in_proj_qkv(hidden_states).transpose(1, 2)
        mixed = F.silu(self.conv1d(mixed)[..., :sequence]).transpose(1, 2)
        query, key, value = torch.split(
            mixed, (self.key_dim, self.key_dim, self.value_dim), dim=-1
        )
        query = query.view(batch, sequence, self.num_key_heads, self.key_head_dim)
        key = key.view(batch, sequence, self.num_key_heads, self.key_head_dim)
        value = value.view(batch, sequence, self.num_value_heads, self.value_head_dim)
        if self.num_value_heads != self.num_key_heads:
            repeats = self.num_value_heads // self.num_key_heads
            query = query.repeat_interleave(repeats, dim=2)
            key = key.repeat_interleave(repeats, dim=2)

        query = _l2_normalize(query.float()).transpose(1, 2)
        key = _l2_normalize(key.float()).transpose(1, 2)
        value = value.float().transpose(1, 2)
        query = query * (self.key_head_dim**-0.5)
        beta = torch.sigmoid(self.in_proj_b(hidden_states)).float().transpose(1, 2)
        decay = -self.A_log.float().exp() * F.softplus(
            self.in_proj_a(hidden_states).float() + self.dt_bias
        )
        decay = decay.transpose(1, 2)

        # The .float() calls above are not enough on their own.  The trainer wraps
        # the whole forward in torch.autocast(bfloat16), and matmul is on
        # autocast's lower-precision list -- it casts fp32 inputs DOWN, so all
        # three matmuls inside the rule ran in bf16 despite the explicit upcast.
        # Worse, the results are multiplied by the fp32 decay matrix immediately
        # after, which promotes them back, so `interaction.dtype` read float32
        # while carrying 8 mantissa bits.  Nothing about the dtypes revealed it.
        #
        # This matters here specifically because the rule solves a triangular
        # system: the Gram matrix K K^T feeds forward substitution over the whole
        # chunk, so a ~0.2% error per inner product accumulates over T terms
        # rather than staying local.  The tests could not catch it either -- they
        # run without autocast, so the reference path stayed fp32 while the
        # shipped path did not.
        with torch.autocast(device_type=query.device.type, enabled=False):
            output = self._delta_rule(query, key, value, beta, decay).transpose(1, 2)
        output = output.to(hidden_states.dtype)
        gate = self.in_proj_z(hidden_states).view(
            batch, sequence, self.num_value_heads, self.value_head_dim
        )
        output = self.norm(
            output.reshape(-1, self.value_head_dim),
            gate.reshape(-1, self.value_head_dim),
        ).view(batch, sequence, self.value_dim)
        return self.out_proj(output)


def _reset_expert_deltanet(module: nn.Module) -> None:
    """Initialize either DeltaNet backend with the existing expert contract."""
    for name in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a"):
        nn.init.normal_(getattr(module, name).weight, std=0.02)
    nn.init.normal_(module.conv1d.weight, std=0.02)
    nn.init.normal_(module.out_proj.weight, std=0.02)
    nn.init.ones_(module.dt_bias)
    with torch.no_grad():
        module.A_log.copy_(torch.empty_like(module.A_log).uniform_(0.001, 16.0).log_())
        module.norm.weight.fill_(1.0)


def _build_transformers_fla_deltanet(
    config: MagicW0BaseConfig,
    layer_idx: int,
) -> nn.Module:
    """Build HF Qwen3.5's FLA mixer without changing checkpoint contracts."""
    try:
        import causal_conv1d  # noqa: F401
        import fla  # noqa: F401
        import transformers
        from transformers.models.auto import CONFIG_MAPPING
        from transformers.models.qwen3_5.modeling_qwen3_5 import (
            Qwen3_5GatedDeltaNet,
        )
    except ImportError as exc:
        raise ImportError(
            "expert_deltanet_backend='transformers_fla' requires transformers, "
            "flash-linear-attention, and causal-conv1d"
        ) from exc

    if transformers.__version__ != "5.5.4":
        raise RuntimeError(
            "transformers_fla expert backend is pinned to transformers==5.5.4; "
            f"found {transformers.__version__}"
        )

    native_config = CONFIG_MAPPING["qwen3_5_text"]()
    native_config.hidden_size = config.action_expert_hidden_size
    native_config.intermediate_size = config.action_expert_intermediate_size
    native_config.num_hidden_layers = config.action_expert_num_layers
    native_config.layer_types = [
        "full_attention"
        if (index + 1) % config.full_attention_interval == 0
        else "linear_attention"
        for index in range(config.action_expert_num_layers)
    ]
    native_config.linear_num_key_heads = config.deltanet_num_key_heads
    native_config.linear_num_value_heads = config.deltanet_num_value_heads
    native_config.linear_key_head_dim = config.deltanet_key_head_dim
    native_config.linear_value_head_dim = config.deltanet_value_head_dim
    native_config.linear_conv_kernel_dim = config.deltanet_conv_kernel
    native_config.rms_norm_eps = config.rms_norm_eps
    native_config.hidden_act = "silu"
    native_config.dtype = (
        torch.bfloat16 if config.dtype == "bfloat16" else torch.float32
    )

    module = Qwen3_5GatedDeltaNet(native_config, layer_idx)
    causal_conv = getattr(module, "causal_conv1d_fn", None)
    chunk_rule = getattr(module, "chunk_gated_delta_rule", None)
    if (
        causal_conv is None
        or chunk_rule is None
        or not getattr(chunk_rule, "__module__", "").startswith("fla.")
    ):
        raise RuntimeError(
            "Transformers Qwen3.5 DeltaNet did not activate its FLA/causal-conv "
            "fast path; check the flash-linear-attention and causal-conv1d builds"
        )
    expected_stateful_modules = (
        "in_proj_qkv",
        "in_proj_z",
        "in_proj_b",
        "in_proj_a",
        "conv1d",
        "norm",
        "out_proj",
    )
    actual_modules = set(module._modules)
    missing_modules = set(expected_stateful_modules) - actual_modules
    auxiliary_modules = actual_modules - set(expected_stateful_modules)
    if missing_modules or auxiliary_modules != {"act"}:
        raise RuntimeError(
            "Transformers Qwen3.5 DeltaNet module contract changed: "
            f"modules={tuple(module._modules)}, missing={tuple(sorted(missing_modules))}, "
            f"auxiliary={tuple(sorted(auxiliary_modules))}"
        )
    activation = module._modules["act"]
    activation_state = activation.state_dict()
    activation_parameters = tuple(activation.named_parameters(recurse=True))
    activation_buffers = tuple(activation.named_buffers(recurse=True))
    if activation_state or activation_parameters or activation_buffers:
        raise RuntimeError(
            "Transformers Qwen3.5 DeltaNet activation unexpectedly carries state: "
            f"type={type(activation).__name__}, state={tuple(activation_state)}, "
            f"parameters={tuple(name for name, _ in activation_parameters)}, "
            f"buffers={tuple(name for name, _ in activation_buffers)}"
        )
    # Optimizer checkpoints associate state with parameter-group position, not
    # parameter names. Transformers registers these submodules in a different
    # order, so restore the existing MagicW0 order before DDP/optimizer setup.
    # Keep Transformers' stateless activation registered after all stateful
    # modules: its forward may access ``self.act``, while its position cannot
    # affect state_dict keys or optimizer parameter positions.
    module._modules = {
        name: module._modules[name] for name in expected_stateful_modules
    }
    module._modules["act"] = activation

    expected_parameter_names = (
        "dt_bias",
        "A_log",
        "in_proj_qkv.weight",
        "in_proj_z.weight",
        "in_proj_b.weight",
        "in_proj_a.weight",
        "conv1d.weight",
        "norm.weight",
        "out_proj.weight",
    )
    actual_parameter_names = tuple(name for name, _ in module.named_parameters())
    if actual_parameter_names != expected_parameter_names:
        raise RuntimeError(
            "Transformers FLA parameter order is not optimizer-checkpoint compatible: "
            f"expected={expected_parameter_names}, actual={actual_parameter_names}"
        )

    hidden = config.action_expert_hidden_size
    key_dim = config.deltanet_num_key_heads * config.deltanet_key_head_dim
    value_dim = config.deltanet_num_value_heads * config.deltanet_value_head_dim
    conv_dim = key_dim * 2 + value_dim
    expected_shapes = {
        "dt_bias": (config.deltanet_num_value_heads,),
        "A_log": (config.deltanet_num_value_heads,),
        "in_proj_qkv.weight": (conv_dim, hidden),
        "in_proj_z.weight": (value_dim, hidden),
        "in_proj_b.weight": (config.deltanet_num_value_heads, hidden),
        "in_proj_a.weight": (config.deltanet_num_value_heads, hidden),
        "conv1d.weight": (conv_dim, 1, config.deltanet_conv_kernel),
        "norm.weight": (config.deltanet_value_head_dim,),
        "out_proj.weight": (hidden, value_dim),
    }
    actual_shapes = {
        name: tuple(value.shape) for name, value in module.state_dict().items()
    }
    if actual_shapes != expected_shapes:
        raise RuntimeError(
            "Transformers FLA state_dict is not checkpoint compatible: "
            f"expected={expected_shapes}, actual={actual_shapes}"
        )
    module._magicvla_transformers_fla = True
    return module


def _run_expert_deltanet(
    module: nn.Module,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    if getattr(module, "_magicvla_transformers_fla", False):
        # Transformers 5.5.4 only applies its padding mask when batch_size > 1.
        # Preserve the baseline contract for batch_size=1 as well, then avoid a
        # second multiply inside the native implementation.
        if attention_mask is not None:
            hidden_states = hidden_states * attention_mask.to(
                hidden_states.dtype
            ).unsqueeze(-1)
        return module(
            hidden_states=hidden_states,
            cache_params=None,
            attention_mask=None,
        )
    return module(hidden_states, attention_mask)


def _rotate_half(value: torch.Tensor) -> torch.Tensor:
    first, second = value.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def _apply_rope(
    value: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    cos, sin = position_embeddings
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    rotary_dim = cos.shape[-1]
    rotated, passthrough = value[..., :rotary_dim], value[..., rotary_dim:]
    rotated = rotated * cos + _rotate_half(rotated) * sin
    return torch.cat((rotated, passthrough), dim=-1)


def _repeat_kv(value: torch.Tensor, repeats: int) -> torch.Tensor:
    if repeats == 1:
        return value
    return value.repeat_interleave(repeats, dim=1)


@lru_cache(maxsize=1)
def _load_external_flash_attention_2() -> tuple[Any, str]:
    """Load flash-attn's varlen entrypoint or fail before training starts."""
    try:
        import flash_attn
        from flash_attn import flash_attn_varlen_func
    except ImportError as exc:
        raise ImportError(
            "expert_full_attention_backend='flash_attention_2' requires the "
            "external flash-attn package"
        ) from exc
    if not callable(flash_attn_varlen_func):
        raise RuntimeError("flash_attn.flash_attn_varlen_func is not callable")
    return flash_attn_varlen_func, str(getattr(flash_attn, "__version__", "unknown"))


def _pack_variable_length_kv(
    key: torch.Tensor,
    value: torch.Tensor,
    key_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack padded [B,H,S,D] K/V into FlashAttention's varlen layout."""
    key_sequence = key.transpose(1, 2).contiguous()
    value_sequence = value.transpose(1, 2).contiguous()
    packed_key = key_sequence[key_mask].contiguous()
    packed_value = value_sequence[key_mask].contiguous()
    key_lengths = key_mask.sum(dim=-1, dtype=torch.int32)
    cu_seqlens_k = F.pad(
        key_lengths.cumsum(dim=0, dtype=torch.int32), (1, 0)
    ).contiguous()
    return packed_key, packed_value, cu_seqlens_k


class QwenJointFullAttention(nn.Module):
    """Expert projections participating in asymmetric VLM/action attention."""

    def __init__(self, config: MagicW0BaseConfig) -> None:
        super().__init__()
        hidden = config.action_expert_hidden_size
        self.num_heads = config.full_attn_num_heads
        self.num_kv_heads = config.full_attn_num_kv_heads
        self.head_dim = config.full_attn_head_dim
        self.kv_repeats = self.num_heads // self.num_kv_heads
        self.dropout = config.dropout
        self.full_attention_backend = config.expert_full_attention_backend
        self._flash_attn_varlen_func = None
        self.flash_attn_version: str | None = None
        if self.full_attention_backend == "flash_attention_2":
            (
                self._flash_attn_varlen_func,
                self.flash_attn_version,
            ) = _load_external_flash_attention_2()
        self.q_proj = nn.Linear(hidden, self.num_heads * self.head_dim * 2, bias=False)
        self.k_proj = nn.Linear(hidden, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, hidden, bias=False)
        self.q_norm = QwenRMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = QwenRMSNorm(self.head_dim, config.rms_norm_eps)
        self.efficient_sdpa = config.action_expert_efficient_sdpa

    def reset_parameters(self) -> None:
        for projection in (self.q_proj, self.k_proj, self.v_proj, self.o_proj):
            nn.init.normal_(projection.weight, std=0.02)
        nn.init.zeros_(self.q_norm.weight)
        nn.init.zeros_(self.k_norm.weight)

    def _expert_qkv(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, sequence, _ = hidden_states.shape
        query, gate = (
            self.q_proj(hidden_states)
            .view(batch, sequence, self.num_heads, self.head_dim * 2)
            .chunk(2, dim=-1)
        )
        query = self.q_norm(query).transpose(1, 2)
        key = self.k_norm(
            self.k_proj(hidden_states).view(
                batch, sequence, self.num_kv_heads, self.head_dim
            )
        ).transpose(1, 2)
        value = (
            self.v_proj(hidden_states)
            .view(batch, sequence, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
        )
        return (
            _apply_rope(query, position_embeddings),
            _apply_rope(key, position_embeddings),
            value,
            gate.reshape(batch, sequence, -1),
        )

    def _vlm_kv(
        self,
        hidden_states: torch.Tensor,
        attention_module: nn.Module,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        *,
        detach_parameters: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if int(attention_module.head_dim) != self.head_dim:
            raise RuntimeError("VLM and expert full-attention head dimensions differ")
        batch, sequence, _ = hidden_states.shape
        if detach_parameters:
            # KI needs more than ``hidden_states.detach()``: the expert reuses
            # the VLM K/V projections and their RMSNorm weights.  Stateless
            # projections with detached parameters keep the expert context
            # numerically identical while preventing flow loss gradients from
            # updating any VLM parameter through this branch.
            k_proj = F.linear(
                hidden_states,
                attention_module.k_proj.weight.detach(),
                None
                if attention_module.k_proj.bias is None
                else attention_module.k_proj.bias.detach(),
            )
            v_proj = F.linear(
                hidden_states,
                attention_module.v_proj.weight.detach(),
                None
                if attention_module.v_proj.bias is None
                else attention_module.v_proj.bias.detach(),
            )
            key = k_proj.view(batch, sequence, self.num_kv_heads, self.head_dim)
            detached_norm_state = {
                name: tensor.detach()
                for name, tensor in (
                    *attention_module.k_norm.named_parameters(),
                    *attention_module.k_norm.named_buffers(),
                )
            }
            key = torch.func.functional_call(
                attention_module.k_norm,
                detached_norm_state,
                (key,),
            )
            key = key.transpose(1, 2)
            value = v_proj.view(
                batch, sequence, self.num_kv_heads, self.head_dim
            ).transpose(1, 2)
        else:
            key = attention_module.k_norm(
                attention_module.k_proj(hidden_states).view(
                    batch, sequence, self.num_kv_heads, self.head_dim
                )
            ).transpose(1, 2)
            value = (
                attention_module.v_proj(hidden_states)
                .view(batch, sequence, self.num_kv_heads, self.head_dim)
                .transpose(1, 2)
            )
        return _apply_rope(key, position_embeddings), value

    def prepare_attention_kv(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        key_mask: torch.Tensor,
    ) -> tuple[Any, ...]:
        """Prepare K/V once so MagicW0's three expert queries can share the work."""
        if key.ndim != 4 or value.shape != key.shape:
            raise ValueError("expert attention K/V must be aligned [B,H,S,D] tensors")
        if key_mask.shape != (key.shape[0], key.shape[2]):
            raise ValueError(
                "expert attention key mask must be [B,S], got "
                f"{tuple(key_mask.shape)} for K={tuple(key.shape)}"
            )
        key_mask = key_mask.to(device=key.device, dtype=torch.bool)

        if self.full_attention_backend == "flash_attention_2":
            if not key.is_cuda:
                raise RuntimeError(
                    "external FlashAttention-2 expert attention requires CUDA"
                )
            if key.dtype not in {torch.float16, torch.bfloat16}:
                raise RuntimeError(
                    "external FlashAttention-2 expert attention requires fp16 or bf16"
                )
            # flash_attn_varlen_func consumes packed [total_tokens,H,D] K/V.  The
            # boolean gather preserves the exact non-causal key-padding semantics
            # of the former SDPA mask without materializing a dense attention mask.
            packed_key, packed_value, cu_seqlens_k = _pack_variable_length_kv(
                key, value, key_mask
            )
            return (
                "flash_attention_2",
                packed_key,
                packed_value,
                cu_seqlens_k,
                int(key.shape[2]),
            )

        repeated_key = _repeat_kv(key, self.kv_repeats)
        repeated_value = _repeat_kv(value, self.kv_repeats)
        return (
            "sdpa",
            repeated_key,
            repeated_value,
            key_mask[:, None, None, :],
        )

    def attend_prepared(
        self,
        query: torch.Tensor,
        prepared_kv: tuple[Any, ...],
        *,
        dropout_p: float,
    ) -> torch.Tensor:
        """Attend one expert query stream to shared prepared K/V."""
        backend = str(prepared_kv[0])
        scale = self.head_dim**-0.5
        if backend == "flash_attention_2":
            if self._flash_attn_varlen_func is None:
                raise RuntimeError("FlashAttention-2 function was not initialized")
            _, packed_key, packed_value, cu_seqlens_k, max_seqlen_k = prepared_kv
            batch, heads, query_length, head_dim = query.shape
            if heads != self.num_heads or head_dim != self.head_dim:
                raise ValueError("expert FlashAttention-2 query shape is incompatible")
            query_sequence = query.transpose(1, 2).contiguous()
            packed_query = query_sequence.reshape(batch * query_length, heads, head_dim)
            cu_seqlens_q = torch.arange(
                batch + 1,
                dtype=torch.int32,
                device=query.device,
            ).mul_(query_length)
            attended = self._flash_attn_varlen_func(
                packed_query,
                packed_key,
                packed_value,
                cu_seqlens_q,
                cu_seqlens_k,
                query_length,
                max_seqlen_k,
                dropout_p,
                softmax_scale=scale,
                causal=False,
            )
            return attended.view(batch, query_length, heads, head_dim).transpose(1, 2)

        if backend != "sdpa":
            raise RuntimeError(f"unknown prepared expert-attention backend {backend!r}")
        _, key, value, allowed = prepared_kv
        return F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=allowed,
            dropout_p=dropout_p,
            scale=scale,
        )

    def forward(
        self,
        expert_hidden_states: torch.Tensor,
        expert_position_embeddings: tuple[torch.Tensor, torch.Tensor],
        expert_attention_mask: torch.Tensor,
        *,
        vlm_hidden_states: torch.Tensor | None = None,
        vlm_attention_module: nn.Module | None = None,
        vlm_position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        vlm_attention_mask: torch.Tensor | None = None,
        detach_vlm_parameters: bool = False,
        cached_vlm_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if vlm_hidden_states is not None and cached_vlm_key_value is not None:
            raise ValueError(
                "pass either VLM hidden states or cached VLM key/value tensors, not both"
            )
        query, expert_key, expert_value, gate = self._expert_qkv(
            expert_hidden_states, expert_position_embeddings
        )
        keys = [expert_key]
        values = [expert_value]
        masks = [expert_attention_mask.bool()]
        if vlm_hidden_states is not None or cached_vlm_key_value is not None:
            if cached_vlm_key_value is None:
                if vlm_attention_module is None or vlm_position_embeddings is None:
                    raise ValueError(
                        "joint attention requires the VLM attention module and RoPE"
                    )
                vlm_key, vlm_value = self._vlm_kv(
                    vlm_hidden_states,
                    vlm_attention_module,
                    vlm_position_embeddings,
                    detach_parameters=detach_vlm_parameters,
                )
            else:
                vlm_key, vlm_value = cached_vlm_key_value
            keys.insert(0, vlm_key)
            values.insert(0, vlm_value)
            if vlm_attention_mask is None:
                if vlm_hidden_states is None:
                    raise ValueError(
                        "cached joint attention requires vlm_attention_mask"
                    )
                vlm_attention_mask = torch.ones(
                    vlm_hidden_states.shape[:2],
                    dtype=torch.bool,
                    device=vlm_hidden_states.device,
                )
            masks.insert(0, vlm_attention_mask.bool())

        key = torch.cat(keys, dim=2)
        value = torch.cat(values, dim=2)
        key_mask = torch.cat(masks, dim=1)
        prepared_kv = self.prepare_attention_kv(key, value, key_mask)
        attended = self.attend_prepared(
            query,
            dropout_p=self.dropout if self.training else 0.0,
            prepared_kv=prepared_kv,
        )
        attended = attended.transpose(1, 2).reshape(
            expert_hidden_states.shape[0], expert_hidden_states.shape[1], -1
        )
        return self.o_proj(attended * torch.sigmoid(gate))


class QwenHybridExpertLayer(nn.Module):
    def __init__(
        self,
        config: MagicW0BaseConfig,
        *,
        use_full_attention: bool,
        layer_idx: int = 0,
    ) -> None:
        super().__init__()
        hidden = config.action_expert_hidden_size
        self.use_full_attention = use_full_attention
        self.input_layernorm = QwenRMSNorm(hidden, config.rms_norm_eps)
        self.post_attention_layernorm = QwenRMSNorm(hidden, config.rms_norm_eps)
        self.mixer: nn.Module = (
            QwenJointFullAttention(config)
            if use_full_attention
            else (
                QwenGatedDeltaNet(config)
                if config.expert_deltanet_backend == "torch_parallel"
                else _build_transformers_fla_deltanet(config, layer_idx)
            )
        )
        self.mlp = QwenSwiGLU(hidden, config.action_expert_intermediate_size)

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.input_layernorm.weight)
        nn.init.zeros_(self.post_attention_layernorm.weight)
        if self.use_full_attention:
            self.mixer.reset_parameters()
        else:
            _reset_expert_deltanet(self.mixer)
        for projection in (self.mlp.gate_proj, self.mlp.up_proj, self.mlp.down_proj):
            nn.init.normal_(projection.weight, std=0.02)

    def forward_linear(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        if self.use_full_attention:
            raise RuntimeError("forward_linear called for a full-attention layer")
        hidden_states = hidden_states + _run_expert_deltanet(
            self.mixer,
            self.input_layernorm(hidden_states),
            attention_mask,
        )
        return hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))

    def forward_full(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor,
        *,
        vlm_hidden_states: torch.Tensor | None = None,
        vlm_attention_module: nn.Module | None = None,
        vlm_position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        vlm_attention_mask: torch.Tensor | None = None,
        detach_vlm_parameters: bool = False,
        cached_vlm_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if not self.use_full_attention:
            raise RuntimeError("forward_full called for a DeltaNet layer")
        mixed = self.mixer(
            self.input_layernorm(hidden_states),
            position_embeddings,
            attention_mask,
            vlm_hidden_states=vlm_hidden_states,
            vlm_attention_module=vlm_attention_module,
            vlm_position_embeddings=vlm_position_embeddings,
            vlm_attention_mask=vlm_attention_mask,
            detach_vlm_parameters=detach_vlm_parameters,
            cached_vlm_key_value=cached_vlm_key_value,
        )
        hidden_states = hidden_states + mixed
        return hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))


def _text_rope(
    positions: torch.Tensor,
    *,
    head_dim: int,
    dtype: torch.dtype,
    theta: float = 10_000_000.0,
    partial_rotary_factor: float = 0.25,
) -> tuple[torch.Tensor, torch.Tensor]:
    rotary_dim = int(head_dim * partial_rotary_factor)
    inv_freq = 1.0 / (
        theta
        ** (
            torch.arange(0, rotary_dim, 2, device=positions.device, dtype=torch.float32)
            / rotary_dim
        )
    )
    frequencies = positions.float().unsqueeze(-1) * inv_freq
    embedding = torch.cat((frequencies, frequencies), dim=-1)
    return embedding.cos().to(dtype), embedding.sin().to(dtype)


class QwenHybridActionExpert(nn.Module):
    """Vocabulary-free Qwen3.5-Text trunk for flow-matching action tokens."""

    def __init__(self, config: MagicW0BaseConfig) -> None:
        super().__init__()
        self.config = config
        hidden = config.action_expert_hidden_size
        self.noisy_action_projection = nn.Linear(config.action_dim, hidden)
        self.action_position_embedding = nn.Parameter(
            torch.empty(1, config.chunk_size, hidden)
        )
        self.state_projection = nn.Sequential(
            nn.Linear(config.state_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.time_embedding = TimestepEmbedding(hidden)
        self.blocks = nn.ModuleList(
            QwenHybridExpertLayer(
                config,
                use_full_attention=((index + 1) % config.full_attention_interval == 0),
                layer_idx=index,
            )
            for index in range(config.action_expert_num_layers)
        )
        self.output_norm = QwenRMSNorm(hidden, config.rms_norm_eps)
        self.output_projection = nn.Linear(hidden, config.action_dim)
        self.reset_parameters()

    @property
    def num_full_attention_layers(self) -> int:
        return sum(int(block.use_full_attention) for block in self.blocks)

    @property
    def trunk_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.blocks.parameters())

    @staticmethod
    def expected_trunk_parameter_count(config: MagicW0BaseConfig) -> int:
        """Analytical count for the vocabulary-free 24-layer LLM trunk."""
        hidden = config.action_expert_hidden_size
        intermediate = config.action_expert_intermediate_size
        mlp = hidden * intermediate * 3
        layer_norms = hidden * 2

        key_dim = config.deltanet_num_key_heads * config.deltanet_key_head_dim
        value_dim = config.deltanet_num_value_heads * config.deltanet_value_head_dim
        conv_dim = key_dim * 2 + value_dim
        deltanet = (
            hidden * conv_dim
            + hidden * value_dim
            + hidden * config.deltanet_num_value_heads * 2
            + conv_dim * config.deltanet_conv_kernel
            + config.deltanet_num_value_heads * 2
            + config.deltanet_value_head_dim
            + value_dim * hidden
        )
        full_attention = (
            hidden * config.full_attn_num_heads * config.full_attn_head_dim * 2
            + hidden * config.full_attn_num_kv_heads * config.full_attn_head_dim * 2
            + config.full_attn_num_heads * config.full_attn_head_dim * hidden
            + config.full_attn_head_dim * 2
        )
        full_layers = config.action_expert_num_layers // config.full_attention_interval
        linear_layers = config.action_expert_num_layers - full_layers
        return linear_layers * (layer_norms + deltanet + mlp) + full_layers * (
            layer_norms + full_attention + mlp
        )

    def reset_parameters(self) -> None:
        nn.init.normal_(self.noisy_action_projection.weight, std=0.02)
        nn.init.zeros_(self.noisy_action_projection.bias)
        nn.init.normal_(self.action_position_embedding, std=0.02)
        for module in self.state_projection:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                nn.init.zeros_(module.bias)
        for module in self.time_embedding.mlp:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                nn.init.zeros_(module.bias)
        for block in self.blocks:
            block.reset_parameters()
        nn.init.zeros_(self.output_norm.weight)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def embed_inputs(
        self,
        noisy_actions: torch.Tensor,
        timesteps: torch.Tensor,
        states: torch.Tensor,
    ) -> torch.Tensor:
        dtype = self.noisy_action_projection.weight.dtype
        noisy_actions = noisy_actions.to(dtype)
        states = states.to(dtype)
        hidden_states = self.noisy_action_projection(noisy_actions)
        hidden_states = (
            hidden_states + self.action_position_embedding[:, : noisy_actions.shape[1]]
        )
        condition = self.time_embedding(timesteps) + self.state_projection(states)
        return hidden_states + condition.unsqueeze(1)

    def project_output(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.output_projection(self.output_norm(hidden_states))

    def forward(
        self,
        noisy_actions: torch.Tensor,
        timesteps: torch.Tensor,
        states: torch.Tensor,
        action_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Standalone expert-only path used by CPU tests and diagnostics."""
        hidden_states = self.embed_inputs(noisy_actions, timesteps, states)
        if action_attention_mask is None:
            action_attention_mask = torch.ones(
                hidden_states.shape[:2], dtype=torch.bool, device=hidden_states.device
            )
        positions = (
            torch.arange(hidden_states.shape[1], device=hidden_states.device)
            .unsqueeze(0)
            .expand(hidden_states.shape[0], -1)
        )
        position_embeddings = _text_rope(
            positions,
            head_dim=self.config.full_attn_head_dim,
            dtype=hidden_states.dtype,
        )
        for block in self.blocks:
            hidden_states = (
                block.forward_full(
                    hidden_states, position_embeddings, action_attention_mask
                )
                if block.use_full_attention
                else block.forward_linear(hidden_states, action_attention_mask)
            )
        return self.project_output(hidden_states)


#: The 32-dimension state schema shared by every current training source, split
#: into the groups the mask actually switches on and off.  Verified against
#: ``meta/info.json`` ``names`` for all eight sources, and verified to be the
#: right granularity: across 6 sources carrying ``state_mask``, no frame has a
#: partially-valid group, so a group is always wholly present or wholly absent.
STATE_GROUPS: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("Left arm joints", tuple(range(0, 6))),
    ("Left gripper", (6,)),
    ("Right arm joints", tuple(range(7, 13))),
    ("Right gripper", (13,)),
    ("Left end-effector position", (14, 15, 16)),
    ("Left end-effector rotation", tuple(range(17, 23))),
    ("Right end-effector position", (23, 24, 25)),
    ("Right end-effector rotation", tuple(range(26, 32))),
)


def _render_state_text(
    state: torch.Tensor,
    dim_mask: torch.Tensor | None,
    decimals: int = 5,
) -> str:
    """Name each state group, and omit the ones this frame does not have.

    The flat rendering had to emit something for every dimension, and a masked
    dimension had been zeroed -- but the state is quantile-normalized to
    [-1, 1], so 0.0 is the middle of the range.  On RoboTwin2.0 the right end
    effector is valid in 41.3% of frames, so that single 0.0 meant "median pose"
    and "not measured" in the same source with nothing to separate them.

    Absence is unambiguous where a zero is not: a group that was not measured
    simply does not appear.  Naming the groups is what makes that readable --
    without labels the model could not tell which dimensions were dropped.

    Falls back to the flat form for any state width other than the shared
    32-dimension schema, so a differently-shaped source degrades instead of
    being mislabelled.
    """
    values = state.reshape(-1)
    if values.shape[0] != 32:
        body = ", ".join(f"{float(v):.{decimals}f}" for v in values)
        return f"Current normalized robot state: [{body}]\n"

    mask = None if dim_mask is None else dim_mask.reshape(-1).bool()
    lines = []
    for label, indices in STATE_GROUPS:
        if mask is not None and not bool(mask[list(indices)].all()):
            continue
        body = ", ".join(f"{float(values[i]):.{decimals}f}" for i in indices)
        lines.append(f"{label}: {body}")
    if not lines:
        return "Current robot state: not measured\n"
    return "Current normalized robot state --\n" + "\n".join(lines) + "\n"


def _render_robot_prompt_metadata(
    template: str,
    state_dim_mask: torch.Tensor | None,
    action_dim_mask: torch.Tensor | None = None,
) -> str:
    """Render the configured robot metadata line from the effective dimension mask.

    The 34-D target layout reserves slots ``0..6`` and ``8..14`` for the left
    and right joint groups.  Legacy 32-D sources leave slots 6 and 14 absent
    after remapping, which distinguishes 6-DoF arms from native 7-DoF data
    without adding source-specific metadata to the prompt configuration.
    """
    configured = str(template).strip()
    if not configured:
        return ""

    def reduce_mask(raw_mask: torch.Tensor | None) -> torch.Tensor | None:
        if raw_mask is None:
            return None
        mask = raw_mask if torch.is_tensor(raw_mask) else torch.as_tensor(raw_mask)
        if mask.ndim == 0 or mask.shape[-1] <= 14:
            return None
        # Action masks may carry a horizon dimension.  A slot is part of the
        # robot schema when it is valid in at least one frame of the sample.
        return mask.bool().reshape(-1, mask.shape[-1]).any(dim=0)

    state_mask = reduce_mask(state_dim_mask)
    action_mask = reduce_mask(action_dim_mask)
    if state_mask is None:
        mask = action_mask
    elif action_mask is None or state_mask.shape != action_mask.shape:
        mask = state_mask
    else:
        # End-of-episode action padding can make an entire action chunk invalid;
        # state validity keeps that structural information available.
        mask = state_mask | action_mask
    if mask is None:
        return ""

    joint_slots = (*range(0, 7), *range(8, 15))
    has_joint = bool(mask[list(joint_slots)].any())
    action_space = "joint+EEF" if has_joint else "EEF"
    joint_dimensions = ""
    if has_joint:
        dof = 7 if bool(mask[6]) and bool(mask[14]) else 6
        joint_dimensions = f" Joint dimensions: {dof}-DoF per arm"

    return configured.replace("{action_space}", action_space).replace(
        "{joint_dimensions}", joint_dimensions
    )


#: The two halves of the inline state line.  Kept as module constants because
#: :meth:`MagicW0BasePolicy._splice_state_placeholders` locates the insertion
#: column as a fixed offset from the end of the closing sentence and then
#: asserts the marker precedes it -- editing one string without the other turns
#: a silent misalignment into a loud failure.
_STATE_INLINE_MARKER = "\nRobot state:"
_STATE_INLINE_CLOSING = "\nPredict the next robot action chunk."


class MagicW0BasePolicy(StandalonePolicy):
    """Qwen3.5-2B with a 24-layer, width-1024 hybrid action expert."""

    config_class = MagicW0BaseConfig
    name = "magicw0_base"
    loss_name = "flow_matching_action_mse"
    supports_cpu_prepare = True
    # Deployment needs the source-specific action/state normalization contract.
    # A checkpoint without it cannot reconstruct commands from model outputs.
    requires_normalization_metadata = True

    def __init__(
        self,
        config: MagicW0BaseConfig,
        *,
        vlm_state_dict: dict[str, torch.Tensor] | None = None,
    ) -> None:
        super().__init__(config)
        checkpoint = config.validate_checkpoint_layout()
        self._validate_runtime_version()

        from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

        hf_config = AutoConfig.from_pretrained(
            checkpoint,
            local_files_only=config.local_files_only,
            trust_remote_code=config.trust_remote_code,
        )
        if hf_config.model_type != config.vlm_model_type:
            raise ValueError(
                f"magicw0_base requires model_type={config.vlm_model_type!r}, "
                f"got {hf_config.model_type!r} from {checkpoint}"
            )
        text_config = getattr(hf_config, "text_config", hf_config)
        self._validate_hybrid_layout(config, text_config)
        # The schedule comes from the checkpoint's config, not from the layer
        # objects: transformers 5.5.4 names the attribute ``layer_type``, not
        # ``block_type``, so reading it off the layer raised AttributeError on
        # the first real forward.  The config key is the stable one, and
        # _validate_hybrid_layout has just checked it against the expert.
        self.vlm_layer_types = tuple(text_config.layer_types)

        self.processor = AutoProcessor.from_pretrained(
            checkpoint,
            local_files_only=config.local_files_only,
            trust_remote_code=config.trust_remote_code,
        )
        load_kwargs: dict[str, Any] = {
            "local_files_only": config.local_files_only,
            "trust_remote_code": config.trust_remote_code,
            "dtype": self._torch_dtype(config.dtype),
        }
        if config.attention_implementation:
            load_kwargs["attn_implementation"] = config.attention_implementation
        if vlm_state_dict is not None:
            # Deployment checkpoints contain the complete fine-tuned VLM. Build
            # its structure on meta and materialize it from those tensors once;
            # do not read or allocate the base model weights first.
            construction_kwargs = {"dtype": load_kwargs["dtype"]}
            if config.attention_implementation:
                construction_kwargs["attn_implementation"] = (
                    config.attention_implementation
                )
            with torch.device("meta"):
                self.vlm = AutoModelForImageTextToText.from_config(
                    hf_config, **construction_kwargs
                )
            self.vlm.load_state_dict(vlm_state_dict, strict=True, assign=True)
            # Rotary frequencies are deliberately non-persistent HF buffers, so
            # they are absent from state_dict. Recreate only these tiny modules
            # on CPU after the meta construction.
            visual_rotary = self.vlm.model.visual.rotary_pos_emb
            self.vlm.model.visual.rotary_pos_emb = type(visual_rotary)(
                visual_rotary.dim, visual_rotary.theta
            )
            text_rotary = self.vlm.model.language_model.rotary_emb
            self.vlm.model.language_model.rotary_emb = type(text_rotary)(text_config)
            unmaterialized = [
                name
                for name, value in list(self.vlm.named_parameters())
                + list(self.vlm.named_buffers())
                if value.is_meta
            ]
            if unmaterialized:
                raise RuntimeError(
                    f"VLM checkpoint left meta tensors unmaterialized: {unmaterialized}"
                )
        else:
            self.vlm = AutoModelForImageTextToText.from_pretrained(
                checkpoint, **load_kwargs
            )
        self.vlm.config.use_cache = False
        self._lm_head = self.vlm.get_output_embeddings()
        if config.train_action_expert_only:
            self.vlm.requires_grad_(False)
            self.vlm.eval()
        else:
            if config.gradient_checkpointing:
                self.vlm.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            if config.freeze_vision_encoder:
                self._freeze_vision_encoder()

        self.action_expert = QwenHybridActionExpert(config)
        self.action_expert.to(dtype=self._torch_dtype(config.dtype))

        # State enters the VLM prefix the way images do: as continuous embeddings
        # scattered into placeholder positions, not as digits.  The mask is an
        # input rather than a gate, because a masked dimension is zeroed and the
        # state is quantile-normalized -- 0.0 is the middle of the range, so
        # without the mask "not measured" and "median pose" are the same vector.
        # Measured on RoboTwin2.0 the right end effector is valid in 41.3% of
        # frames, so this is not a corner case.
        if config.state_prompt_mode == "embedding":
            vlm_hidden = int(self.vlm.config.text_config.hidden_size)
            self.state_prompt_projection = nn.Sequential(
                nn.Linear(config.state_dim * 2, vlm_hidden),
                nn.SiLU(),
                nn.Linear(vlm_hidden, vlm_hidden * config.state_prompt_tokens),
            )
            self.state_prompt_projection.to(dtype=self._torch_dtype(config.dtype))
            for module in self.state_prompt_projection:
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, std=0.02)
                    nn.init.zeros_(module.bias)
        else:
            self.state_prompt_projection = None
        if config.compile_model:
            # Compile the trunk loop, not `forward`: forward also builds the FAST
            # labels and the loss dict, which is Python bookkeeping that would only
            # add graph breaks.  `_run_joint_trunks` is the pure-tensor part and is
            # where essentially the whole step lives.
            #
            # dynamic=True because prefix_len is the padded length of a batch of
            # chat templates and moves every step; without it each new length is a
            # fresh compile.  Under gradient_checkpointing the loop calls
            # torch.utils.checkpoint, which compile handles but does not always
            # fuse across -- if this turns out slower, that interaction is the
            # first place to look.
            self._run_joint_trunks = torch.compile(
                self._run_joint_trunks,
                mode=config.compile_mode,
                dynamic=config.compile_dynamic,
            )
            logger.info(
                "magicw0_base compiled _run_joint_trunks (mode=%s, dynamic=%s)",
                config.compile_mode,
                config.compile_dynamic,
            )
        self.fast_tokenizer = None
        if config.use_fast_action_tokens:
            self.fast_tokenizer = _load_fast_action_tokenizer(
                config.fast_tokenizer_path
            )
            logger.info(
                "FAST action tokens enabled: %s (fast_vocab=%d, skip=%d, max_tokens=%d)",
                config.fast_tokenizer_path,
                self.fast_tokenizer.vocab_size,
                config.fast_skip_tokens,
                config.fast_max_action_tokens,
            )

    @staticmethod
    def _validate_hybrid_layout(config: MagicW0BaseConfig, text_config: Any) -> None:
        layer_types = list(text_config.layer_types)
        expected = [
            "full_attention"
            if (index + 1) % config.full_attention_interval == 0
            else "linear_attention"
            for index in range(config.action_expert_num_layers)
        ]
        if int(text_config.num_hidden_layers) != config.action_expert_num_layers:
            raise ValueError(
                "VLM and action expert must have equal depth: "
                f"{text_config.num_hidden_layers} != {config.action_expert_num_layers}"
            )
        if layer_types != expected:
            raise ValueError(
                "VLM layer schedule does not match the configured 3:1 hybrid expert"
            )
        checks = {
            "num_attention_heads": config.full_attn_num_heads,
            "num_key_value_heads": config.full_attn_num_kv_heads,
            "head_dim": config.full_attn_head_dim,
            "linear_num_key_heads": config.deltanet_num_key_heads,
            "linear_num_value_heads": config.deltanet_num_value_heads,
            "linear_key_head_dim": config.deltanet_key_head_dim,
            "linear_value_head_dim": config.deltanet_value_head_dim,
            "linear_conv_kernel_dim": config.deltanet_conv_kernel,
        }
        mismatches = {
            name: (int(getattr(text_config, name)), expected_value)
            for name, expected_value in checks.items()
            if int(getattr(text_config, name)) != expected_value
        }
        if mismatches:
            raise ValueError(
                "expert attention dimensions must mirror Qwen3.5 for joint attention; "
                f"mismatches={mismatches}"
            )

    @staticmethod
    def _validate_runtime_version() -> None:
        try:
            import transformers
            from packaging.version import Version
        except ImportError as exc:
            raise ImportError("magicw0_base requires transformers>=5.1.0") from exc
        if Version(transformers.__version__) < Version("5.1.0"):
            raise RuntimeError(
                f"Qwen3.5 requires transformers>=5.1.0; found {transformers.__version__}"
            )

    @staticmethod
    def _torch_dtype(name: str) -> torch.dtype:
        return torch.bfloat16 if name == "bfloat16" else torch.float32

    def _freeze_vision_encoder(self) -> None:
        frozen = 0
        for module_name, module in self.vlm.named_modules():
            lname = module_name.lower()
            if any(
                hint in lname for hint in ("visual", "vision_tower", "vision_model")
            ):
                for parameter in module.parameters(recurse=True):
                    parameter.requires_grad = False
                    frozen += 1
        if frozen:
            logger.info("magicw0_base froze %d vision-encoder params", frozen)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.config.train_action_expert_only:
            self.vlm.eval()
        return self

    def prepare_batch(
        self, batch: dict[str, Any], device: torch.device | None
    ) -> dict[str, Any]:
        """Prepare either a robotics batch or a VLM co-training batch on CPU."""
        modality = str(
            batch.get(
                self.config.cotrain_modality_key, self.config.cotrain_native_modality
            )
        )
        if modality in self.config.vlm_modalities:
            return self._prepare_vlm_batch(batch, device, modality)

        states = batch[STATE]
        actions = batch[ACTIONS]
        pad_mask = batch[ACTION_IS_PAD]
        action_dim_mask = batch[ACTION_DIM_MASK]
        camera_valid = batch.get(CAMERA_VALID)
        if camera_valid is None:
            camera_valid = torch.ones(
                (states.shape[0], len(self.config.camera_keys)), dtype=torch.bool
            )
        tasks = batch[TASK]
        vlm_inputs, prefix_len, subtask_ids, subtask_valid = (
            self._prepare_robotics_prefix(
                batch,
                states,
                tasks,
                camera_valid,
                batch.get(EMBODIMENT),
                batch.get(STATE_DIM_MASK),
                batch.get(SUBTASK),
                action_dim_masks=batch.get(ACTION_DIM_MASK),
            )
        )
        prepared: dict[str, Any] = {
            self.config.cotrain_modality_key: modality,
            "prefix_len": prefix_len,
        }
        if "__sample_metadata__" in batch:
            prepared["__sample_metadata__"] = batch["__sample_metadata__"]
        if subtask_ids is not None:
            prepared["subtask_ids"] = _maybe_to(subtask_ids, device)
            prepared["subtask_valid"] = _maybe_to(subtask_valid, device)
        if self.config.use_fast_action_tokens:
            fast_ids, fast_valid = self._encode_fast_actions(actions, pad_mask)
            n_fast = int(fast_ids.shape[1])
            for key, value in list(vlm_inputs.items()):
                if not (
                    torch.is_tensor(value)
                    and value.ndim == 2
                    and value.shape[1] == prefix_len
                ):
                    continue
                if key == "input_ids":
                    tail = fast_ids
                elif key == "attention_mask":
                    tail = fast_valid.to(value.dtype)
                else:
                    # FAST ids are text tokens, not multimodal image tokens.
                    tail = torch.zeros(value.shape[0], n_fast, dtype=value.dtype)
                vlm_inputs[key] = torch.cat([value, tail], dim=1)
            prepared["fast_ids"] = _maybe_to(fast_ids, device)
            prepared["fast_valid"] = _maybe_to(fast_valid, device)

        prepared["vlm_inputs"] = {
            key: _maybe_to(value, device) if isinstance(value, torch.Tensor) else value
            for key, value in vlm_inputs.items()
        }
        prepared[STATE] = _maybe_to(states, device)
        prepared[ACTIONS] = _maybe_to(actions, device)
        prepared[ACTION_IS_PAD] = _maybe_to(pad_mask, device)
        prepared[ACTION_DIM_MASK] = _maybe_to(action_dim_mask, device)
        state_dim_mask = batch.get(STATE_DIM_MASK)
        if state_dim_mask is not None:
            prepared[STATE_DIM_MASK] = _maybe_to(state_dim_mask, device)
        return prepared

    def prepare_inference_batch(
        self,
        batch: dict[str, Any],
        device: torch.device | None,
        subtasks: list[str] | None = None,
    ) -> dict[str, Any]:
        """Prepare online observations without requiring future action targets.

        Stage one of pi_0.5's two-stage inference happens here when the policy
        is asked to produce its own subtask.  An externally supplied ``subtasks``
        wins over anything the model would say -- a caller with a real high-level
        planner, or an oracle segment during evaluation, should not be overruled
        -- and skips the decode entirely.  Either way the two paths meet at the
        same completed prefix, so everything downstream is identical.
        """
        states = batch[STATE]
        camera_valid = batch.get(CAMERA_VALID)
        if camera_valid is None:
            camera_valid = torch.ones(
                (states.shape[0], len(self.config.camera_keys)), dtype=torch.bool
            )
        state_dim_mask = batch.get(STATE_DIM_MASK)
        if self.config.subtask_generation and subtasks is None:
            # A prefix with no subtask turn: the decode appends the header and
            # grows it one token at a time from there.
            base_inputs, _, _, _ = self._prepare_robotics_prefix(
                batch,
                states,
                batch[TASK],
                camera_valid,
                batch.get(EMBODIMENT),
                state_dim_mask,
                None,
                action_dim_masks=batch.get(ACTION_DIM_MASK),
            )
            subtasks = self.generate_subtask(base_inputs, states, state_dim_mask)
        vlm_inputs, prefix_len, subtask_ids, subtask_valid = (
            self._prepare_robotics_prefix(
                batch,
                states,
                batch[TASK],
                camera_valid,
                batch.get(EMBODIMENT),
                state_dim_mask,
                subtasks if subtasks is not None else batch.get(SUBTASK),
                action_dim_masks=batch.get(ACTION_DIM_MASK),
            )
        )
        prepared: dict[str, Any] = {
            self.config.cotrain_modality_key: str(
                batch.get(
                    self.config.cotrain_modality_key,
                    self.config.cotrain_native_modality,
                )
            ),
            "prefix_len": prefix_len,
            "subtask": list(subtasks) if subtasks is not None else None,
            "vlm_inputs": {
                key: _maybe_to(value, device)
                if isinstance(value, torch.Tensor)
                else value
                for key, value in vlm_inputs.items()
            },
            STATE: _maybe_to(states, device),
        }
        if state_dim_mask is not None:
            prepared[STATE_DIM_MASK] = _maybe_to(state_dim_mask, device)
        action_dim_mask = batch.get(ACTION_DIM_MASK)
        if action_dim_mask is not None:
            prepared[ACTION_DIM_MASK] = _maybe_to(action_dim_mask, device)
        return prepared

    def _prepare_vlm_batch(self, batch, device, modality):
        """Tokenize text VLM data or build supervised image-VQA conversations."""
        if VQA_IMAGES in batch:
            return self._prepare_vlm_image_batch(batch, device, modality)

        texts = batch[TEXT]
        if isinstance(texts, str):
            texts = [texts]
        encoded = self.processor.tokenizer(
            list(texts), padding=True, truncation=True, return_tensors="pt"
        )
        vlm_inputs = {
            key: _maybe_to(value, device) if isinstance(value, torch.Tensor) else value
            for key, value in encoded.items()
        }
        return {self.config.cotrain_modality_key: modality, "vlm_inputs": vlm_inputs}

    def _prepare_vlm_image_batch(self, batch, device, modality):
        """Apply Qwen's multimodal chat template and mask non-assistant tokens."""
        images = batch[VQA_IMAGES]
        n_images = batch[VQA_N_IMAGES]
        chats = batch[VQA_CHAT]
        if isinstance(chats, str):
            images, n_images, chats = images.unsqueeze(0), [n_images], [chats]
        batch_size = len(chats)

        all_messages = [
            build_vlm_chat_messages(
                json.loads(chats[index]),
                int(n_images[index]),
                image_at=lambda image_index, values=images[index]: (
                    values[image_index].detach().cpu()
                ),
            )
            for index in range(batch_size)
        ]
        tokenizer = getattr(self.processor, "tokenizer", None)
        previous_padding_side = getattr(tokenizer, "padding_side", None)
        if tokenizer is not None:
            tokenizer.padding_side = "right"
        try:
            full_enc = self.processor.apply_chat_template(
                all_messages,
                tokenize=True,
                add_generation_prompt=False,
                return_dict=True,
                return_tensors="pt",
                processor_kwargs={"padding": True},
            )
        finally:
            if tokenizer is not None and previous_padding_side is not None:
                tokenizer.padding_side = previous_padding_side

        labels = self._assistant_only_labels(
            full_enc["input_ids"], full_enc["attention_mask"]
        )
        vlm_inputs = {
            key: _maybe_to(value, device) if isinstance(value, torch.Tensor) else value
            for key, value in full_enc.items()
        }
        return {
            self.config.cotrain_modality_key: modality,
            "vlm_inputs": vlm_inputs,
            "labels": _maybe_to(labels, device),
        }

    def _assistant_only_labels(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Supervise assistant turns while ignoring prompts, images, and padding."""
        tokenizer = self.processor.tokenizer
        header = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
        im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
        labels = torch.full_like(input_ids, self.config.cotrain_ignore_index)
        header_len = len(header)
        for index in range(input_ids.shape[0]):
            row = input_ids[index].tolist()
            valid_len = int(attention_mask[index].sum())
            position = 0
            while position + header_len <= valid_len:
                if row[position : position + header_len] == header:
                    start = position + header_len
                    end = start
                    while end < valid_len and row[end] != im_end:
                        end += 1
                    end = min(end + 1, valid_len)
                    labels[index, start:end] = input_ids[index, start:end]
                    position = end
                else:
                    position += 1
        return labels

    def _state_placeholder_id(self) -> int:
        """An id from the block the FAST encoder already reserves at the top.

        ``fast_skip_tokens`` exists because the last ids of the Qwen vocabulary
        are not real text tokens; FAST allocates *below* that block, so taking
        the very top id cannot collide with FAST ids or with anything the
        tokenizer emits.  Reusing the existing convention beats resizing the
        embedding table, which would invalidate the pretrained checkpoint.
        """
        vocab = int(self.vlm.get_input_embeddings().weight.shape[0])
        return vocab - 1

    def _append_state_placeholders(self, vlm_inputs: dict[str, Any]) -> dict[str, Any]:
        """Reserve prefix positions for the state embedding.

        Appended after the chat template rather than spliced into it: the prefix
        is causal, so the state sits last and everything the action expert reads
        can attend to it, which is the arrangement the prompt wording assumes.
        Mirrors how FAST ids are concatenated, including the zero fill for the
        non-text 2-D tensors.
        """
        count = int(self.config.state_prompt_tokens)
        input_ids = vlm_inputs["input_ids"]
        prefix_len = int(input_ids.shape[1])
        batch_size = int(input_ids.shape[0])
        placeholder = torch.full(
            (batch_size, count),
            self._state_placeholder_id(),
            dtype=input_ids.dtype,
        )
        updated = dict(vlm_inputs)
        for key, value in vlm_inputs.items():
            if not (
                torch.is_tensor(value)
                and value.ndim == 2
                and value.shape[1] == prefix_len
            ):
                continue
            if key == "input_ids":
                tail = placeholder
            elif key == "attention_mask":
                tail = torch.ones(batch_size, count, dtype=value.dtype)
            else:
                tail = torch.zeros(batch_size, count, dtype=value.dtype)
            updated[key] = torch.cat([value, tail], dim=1)
        return updated

    def _splice_state_placeholders(self, vlm_inputs: dict[str, Any]) -> dict[str, Any]:
        """Insert the state columns inside the user turn, before the closing line.

        Located by searching for the marker rather than by a fixed offset from
        the right edge.  The offset version was tried first and was wrong on
        contact with the real processor: ``apply_chat_template`` closes the turn
        with its own ``<|im_end|>`` and newline, so the closing sentence is not
        what the prefix ends with.  Searching depends on nothing the template
        does.

        The scan runs backwards, so a marker string appearing earlier in a task
        description cannot win over the real one, and it matches across the whole
        batch at once -- rows that disagree therefore raise instead of scattering
        the state into different places per sample.
        """
        tokenizer = self.processor.tokenizer
        marker = tokenizer.encode(_STATE_INLINE_MARKER, add_special_tokens=False)
        input_ids = vlm_inputs["input_ids"]
        prefix_len = int(input_ids.shape[1])
        expected = torch.tensor(marker, dtype=input_ids.dtype, device=input_ids.device)
        column = -1
        for start in range(prefix_len - len(marker), -1, -1):
            if bool((input_ids[:, start : start + len(marker)] == expected).all()):
                column = start + len(marker)
                break
        if column < 0:
            raise RuntimeError(
                "inline state placement could not find "
                f"{_STATE_INLINE_MARKER!r} at the same column in every row of "
                "the prefix, so there is nowhere well-defined to splice the "
                f"state (the closing sentence should be {_STATE_INLINE_CLOSING!r})"
            )

        count = int(self.config.state_prompt_tokens)
        batch_size = int(input_ids.shape[0])
        device = input_ids.device
        updated = dict(vlm_inputs)
        for key, value in vlm_inputs.items():
            if not (
                torch.is_tensor(value)
                and value.ndim == 2
                and value.shape[1] == prefix_len
            ):
                continue
            if key == "input_ids":
                block = torch.full(
                    (batch_size, count),
                    self._state_placeholder_id(),
                    dtype=value.dtype,
                    device=device,
                )
            elif key == "attention_mask":
                block = torch.ones(batch_size, count, dtype=value.dtype, device=device)
            else:
                block = torch.zeros(batch_size, count, dtype=value.dtype, device=device)
            updated[key] = torch.cat(
                [value[:, :column], block, value[:, column:]], dim=1
            )
        return updated

    def _subtask_span(
        self, prefix_len: int, subtask_ids: torch.Tensor | None
    ) -> tuple[int, int] | None:
        """The columns dropout must hide from the action expert.

        Covers the prefilled header as well as the slot.  Dropout exists to make
        a sample behave like one from a source that has no annotation at all,
        and such a source contributes no assistant turn -- so leaving the seven
        header tokens visible would give the expert a third state to learn
        ("asked for a subtask, told nothing") that no real source produces.
        """
        if subtask_ids is None:
            return None
        turn_len = int(subtask_ids.shape[1]) + len(self._subtask_header_ids())
        return (prefix_len - turn_len, prefix_len)

    def _expert_prefix_mask(
        self,
        vlm_padding_mask: torch.Tensor,
        prefix_len: int,
        subtask_span: tuple[int, int] | None,
    ) -> torch.Tensor:
        """The prefix mask the action expert sees, with subtask dropout applied.

        Withholding the subtask from the expert and withholding supervision from
        the VLM are two different things, and they happen to be governed by two
        different masks: the VLM's causal mask decides what the CE may look at,
        this one decides what the expert may condition on.  Dropping here alone
        therefore keeps training the generation half on exactly the samples whose
        action half is being taught to cope without a subtask -- which the older
        "omit the line" dropout did not, since a withheld line was neither
        conditioned on nor supervised.

        Both conditioning and generation use this mask. The VLM is causal, so
        the hidden states before the slot do not depend on it. The action RoPE
        origin also follows this masked prefix, making a dropped turn equivalent
        to an absent one rather than leaking its length to the expert.

        Gated on ``self.training`` so evaluation and deployment can never sample
        it: a policy whose conditioning changes between two eval runs is not
        measurable.  Drawn per sample, so one batch teaches both conditionings,
        and ``prefix_len`` stays a scalar because the span is masked, not removed.
        """
        mask = vlm_padding_mask[:, :prefix_len]
        if (
            subtask_span is None
            or not self.training
            or self.config.subtask_dropout <= 0.0
        ):
            return mask
        start, end = subtask_span
        drop = (
            torch.rand(mask.shape[0], device=mask.device) < self.config.subtask_dropout
        )
        mask = mask.clone()
        mask[:, start:end] &= ~drop.unsqueeze(1)
        return mask

    def _prepare_robotics_prefix(
        self,
        batch,
        states,
        tasks,
        camera_valid,
        embodiments=None,
        state_dim_masks=None,
        subtasks=None,
        action_dim_masks=None,
    ):
        messages = []
        for sample_index in range(int(states.shape[0])):
            content: list[dict[str, Any]] = []
            for camera_index, camera_key in enumerate(self.config.camera_keys):
                if not bool(camera_valid[sample_index, camera_index]):
                    continue
                image = batch[camera_key][sample_index]
                if image.ndim == 4:
                    image = image[-1]
                content.append({"type": "image", "image": image.detach().cpu()})
            task_text, tail_text = self._prefix_text_parts(
                sample_index,
                states,
                tasks,
                embodiments,
                state_dim_masks,
                action_dim_masks,
            )
            content.append({"type": "text", "text": f"{task_text}{tail_text}"})
            messages.append([{"role": "user", "content": content}])
        return self._finalize_robotics_prefix(messages, subtasks)

    def _prefix_text_parts(
        self,
        sample_index,
        states,
        tasks,
        embodiments=None,
        state_dim_masks=None,
        action_dim_masks=None,
    ) -> tuple[str, str]:
        """The user turn's text, split where a subclass may insert content.

        Returns ``(embodiment + metadata + task line, state text + closing)``;
        their concatenation is the legacy text block byte for byte.
        """
        if self.config.state_prompt_mode == "embedding":
            if self.config.state_prompt_position == "inline":
                # The embedding is spliced in after this label, so the
                # closing sentence no longer points forward at it and drops
                # the trailing colon.  Same shape the text mode already has.
                state_text = _STATE_INLINE_MARKER.lstrip("\n")
                closing = _STATE_INLINE_CLOSING
            else:
                # The numbers follow as embeddings appended to this prefix,
                # so the sentence ends by naming them and stops.
                state_text = ""
                closing = (
                    "Predict the next robot action chunk given the current robot state:"
                )
        else:
            state_text = _render_state_text(
                states[sample_index],
                None if state_dim_masks is None else state_dim_masks[sample_index],
                decimals=self.config.state_prompt_decimals,
            )
            closing = "Predict the next robot action chunk."
        # The embodiment goes in as language rather than as an id embedding.
        # It is a category with 9 members whose frequencies span two orders of
        # magnitude, so a learned embedding would be starved for the rare ones,
        # while "AgileX PiPER dual-arm" is something the VLM already has priors
        # about.  A source that cannot name its robot contributes no line at
        # all -- an omission the model can average over, rather than a wrong
        # robot it would have to unlearn.
        description = describe(embodiments[sample_index]) if embodiments else None
        embodiment_text = f"Robot embodiment: {description}\n" if description else ""
        prompt_metadata = _render_robot_prompt_metadata(
            self.config.robot_prompt_metadata,
            None if state_dim_masks is None else state_dim_masks[sample_index],
            None if action_dim_masks is None else action_dim_masks[sample_index],
        )
        metadata_text = f"{prompt_metadata}\n" if prompt_metadata else ""
        # Supplied and generated subtasks share the assistant span after
        # the observation/state prefix. The user turn never contains the
        # answer; subtask_generation controls CE and decoding, not layout.
        return (
            f"{embodiment_text}{metadata_text}Robot task: {tasks[sample_index]}\n",
            f"{state_text}{closing}",
        )

    def _finalize_robotics_prefix(self, messages, subtasks=None):
        """Tokenize user turns left-padded, then add state and subtask tails."""
        tokenizer = getattr(self.processor, "tokenizer", None)
        previous_padding_side = getattr(tokenizer, "padding_side", None)
        if tokenizer is not None:
            tokenizer.padding_side = "left"
        try:
            vlm_inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
                return_dict=True,
                return_tensors="pt",
                processor_kwargs={"padding": True},
            )
        finally:
            if tokenizer is not None and previous_padding_side is not None:
                tokenizer.padding_side = previous_padding_side
        if self.config.state_prompt_mode == "embedding":
            if self.config.state_prompt_position == "inline":
                vlm_inputs = self._splice_state_placeholders(vlm_inputs)
            else:
                vlm_inputs = self._append_state_placeholders(vlm_inputs)
        subtask_ids = subtask_valid = None
        if subtasks is not None:
            vlm_inputs, subtask_ids, subtask_valid = self._append_subtask_turn(
                vlm_inputs, subtasks
            )
        return (
            vlm_inputs,
            int(vlm_inputs["input_ids"].shape[1]),
            subtask_ids,
            subtask_valid,
        )

    def _subtask_header_ids(self) -> list[int]:
        """The fixed, unsupervised prefill that opens the subtask turn.

        A constant, so it is prefilled rather than generated: supervising it
        would only dilute the mean CE and let the exact-match metric count a
        span that is always right, and decoding it would spend tokens on
        something already known.  The label has to be here rather than in the
        user turn because the user turn is frozen by the byte-for-byte
        backwards-compatibility contract, and its closing sentence asks for an
        action chunk, not a subtask.
        """
        cached = getattr(self, "_subtask_header_cache", None)
        if cached is None:
            cached = self.processor.tokenizer.encode(
                "<|im_start|>assistant\nSubtask: ", add_special_tokens=False
            )
            self._subtask_header_cache = cached
        return cached

    def _append_subtask_turn(
        self, vlm_inputs: dict[str, Any], subtasks
    ) -> tuple[dict[str, Any], "torch.Tensor | None", "torch.Tensor | None"]:
        """Append the assistant turn carrying the subtask, inside ``prefix_len``.

        Shared by conditioning and generation. Right-padded to the batch maximum:
        the prompt is left-padded so real tokens are right-aligned, which puts
        every sample's slot at the same columns and keeps ``prefix_len`` a
        scalar. A sample with no annotation masks the entire turn, including the
        header. An all-empty batch leaves the original prefix unchanged.

        The terminator is part of the slot, not of the header: the model has to
        learn where to stop, and a budget-truncated decode is a different
        failure from one that never learned to end.
        """
        tokenizer = self.processor.tokenizer
        header = self._subtask_header_ids()
        im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
        encoded = []
        for text in subtasks:
            text = str(text).strip()
            encoded.append(
                tokenizer.encode(text, add_special_tokens=False) + [im_end]
                if text
                else []
            )
        slot_len = max((len(row) for row in encoded), default=0)
        if slot_len == 0:
            return vlm_inputs, None, None

        input_ids = vlm_inputs["input_ids"]
        batch_size = int(input_ids.shape[0])
        prefix_len = int(input_ids.shape[1])
        device = input_ids.device
        subtask_ids = torch.zeros(
            batch_size, slot_len, dtype=input_ids.dtype, device=device
        )
        subtask_valid = torch.zeros(
            batch_size, slot_len, dtype=torch.bool, device=device
        )
        for index, row in enumerate(encoded):
            if not row:
                continue
            subtask_ids[index, : len(row)] = torch.tensor(
                row, dtype=input_ids.dtype, device=device
            )
            subtask_valid[index, : len(row)] = True

        header_block = (
            torch.tensor(header, dtype=input_ids.dtype, device=device)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )
        appended = torch.cat([header_block, subtask_ids], dim=1)
        appended_valid = torch.cat(
            [
                subtask_valid.any(dim=1, keepdim=True).expand(-1, len(header)),
                subtask_valid,
            ],
            dim=1,
        )
        updated = dict(vlm_inputs)
        # A processor that returned no attention mask has, until now, had every
        # prefix position real.  The slot is the first thing to carry padding
        # inside the prefix, so the mask has to exist from here on.
        if "attention_mask" not in updated:
            updated["attention_mask"] = torch.ones(
                batch_size, prefix_len, dtype=torch.long, device=device
            )
            vlm_inputs = dict(updated)
        for key, value in vlm_inputs.items():
            if not (
                torch.is_tensor(value)
                and value.ndim == 2
                and value.shape[1] == prefix_len
            ):
                continue
            if key == "input_ids":
                tail = appended
            elif key == "attention_mask":
                tail = appended_valid.to(value.dtype)
            else:
                tail = torch.zeros(
                    batch_size, appended.shape[1], dtype=value.dtype, device=device
                )
            updated[key] = torch.cat([value, tail], dim=1)
        return updated, subtask_ids, subtask_valid

    def _encode_fast_actions(
        self, actions: torch.Tensor, pad_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode normalized action chunks into reserved Qwen vocabulary ids.

        Padded chunk rows are dropped before tokenization rather than handed to
        the tokenizer as zeros.  A short chunk is right-padded by the reader with
        ``is_pad`` set, and the processor then zeroes those rows -- but the
        actions are quantile-normalized, so zero is the MIDDLE of the range, not
        "no motion" and not "hold the last pose".  Feeding them in makes the CE
        target a fabricated snap to mid-range for the whole tail.

        This is not a rare edge: RoboTwin2.0 averages 180-frame episodes against a
        50-row chunk, so ~31% of its eligible anchors carry at least one padded
        row and ~15% of all its supervised rows are padding.  The flow loss
        already excludes them (``valid_steps`` in :meth:`forward`); only this
        branch did not, which left the two losses disagreeing about what the
        target even is.

        Dropping rows shortens the token sequence per sample, which the encoder
        already handles -- ``ids``/``valid`` are padded to ``fast_max_action_tokens``
        and the loss reads ``valid``.
        """
        if self.fast_tokenizer is None:
            raise RuntimeError("FAST action tokenizer is not loaded")
        vocab = int(self.vlm.get_input_embeddings().weight.shape[0])
        high = vocab - 1 - self.config.fast_skip_tokens
        eos_id = high - self.fast_tokenizer.vocab_size
        if eos_id < 0:
            raise ValueError(
                "Qwen vocabulary is too small for the configured FAST reserved block: "
                f"vocab={vocab}, fast_vocab={self.fast_tokenizer.vocab_size}, "
                f"skip={self.config.fast_skip_tokens}"
            )
        actions_np = actions.detach().float().cpu().numpy()
        batch_size = actions.shape[0]
        if pad_mask is None:
            token_lists = self.fast_tokenizer(actions_np)
        else:
            keep = ~pad_mask.detach().bool().cpu().numpy()
            # is_pad carries TWO meanings, and they need opposite treatment.
            #
            # lerobot_v21.py:972-974 builds it as `row_valid[:produced_rows] =
            # quality_valid`, so a row is marked invalid either because the chunk
            # ran past the end of the episode -- a trailing run -- or because the
            # per-frame quality predicate rejected it, which for `eef_visual`
            # flickers and lands anywhere INSIDE the chunk.
            #
            # Dropping the trailing run is what this branch exists for: those rows
            # are zeros after the processor's masked_fill, and actions are
            # quantile-normalised, so zero is the MIDDLE of the range rather than
            # "no motion" -- tokenising it teaches a snap to mid-range that never
            # happened. Dropping an INTERIOR row is a different mistake and a worse
            # one: FAST takes a DCT along time, so deleting a row shifts every
            # later row earlier and changes the frequency content of the WHOLE
            # chunk, not just its neighbourhood. The target becomes a trajectory
            # that teleports, while the flow loss -- which masks the same rows in
            # place (see `valid` in forward) -- keeps the correct one. The two
            # action losses would be fitting different trajectories.
            #
            # So: truncate the trailing run, and hold the last good value across
            # interior gaps. Holding is what the reader itself does for length
            # padding (np.repeat of the last row), it keeps the time axis intact,
            # and a held pose is a real pose rather than the midpoint of the
            # action range. Those rows are still excluded from the flow loss by
            # ACTION_DIM_MASK; this only decides what the FAST branch sees.
            if keep.all():
                token_lists = self.fast_tokenizer(actions_np)
            else:
                token_lists = []
                for index in range(batch_size):
                    row_keep = keep[index]
                    valid_rows = np.flatnonzero(row_keep)
                    if valid_rows.size == 0:
                        # Every row invalid. Nothing to encode; the sample still
                        # gets its EOS below, so the CE sees a well-formed empty
                        # sequence instead of a fabricated one.
                        token_lists.append([])
                        continue
                    rows = actions_np[index][: valid_rows[-1] + 1].copy()
                    interior = ~row_keep[: valid_rows[-1] + 1]
                    if interior.any():
                        # Forward-fill: each invalid row takes the most recent
                        # valid row before it. valid_rows[0] is the anchor, which
                        # the current_frame predicate guarantees is valid, so
                        # there is always a predecessor.
                        source = np.maximum.accumulate(
                            np.where(
                                row_keep[: rows.shape[0]], np.arange(rows.shape[0]), 0
                            )
                        )
                        rows[interior] = rows[source[interior]]
                    token_lists.append(self.fast_tokenizer(rows[None, ...])[0])
        max_len = self.config.fast_max_action_tokens
        ids = torch.zeros((batch_size, max_len), dtype=torch.long)
        valid = torch.zeros((batch_size, max_len), dtype=torch.bool)
        for index, token_list in enumerate(token_lists):
            mapped = [high - int(token) for token in list(token_list)[: max_len - 1]]
            mapped.append(eos_id)
            ids[index, : len(mapped)] = torch.as_tensor(mapped, dtype=torch.long)
            valid[index, : len(mapped)] = True
        return ids, valid

    def _embed_vlm_inputs(
        self,
        vlm_inputs: dict[str, Any],
        states: torch.Tensor | None = None,
        state_dim_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        multimodal_model = self.vlm.model
        input_ids = vlm_inputs["input_ids"]
        attention_mask = vlm_inputs.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        inputs_embeds = multimodal_model.get_input_embeddings()(input_ids)

        if self.state_prompt_projection is not None and states is not None:
            if state_dim_mask is None:
                state_dim_mask = torch.ones_like(states, dtype=torch.bool)
            features = torch.cat((states, state_dim_mask.to(states.dtype)), dim=-1).to(
                inputs_embeds.dtype
            )
            state_embeds = self.state_prompt_projection(features).reshape(
                states.shape[0], int(self.config.state_prompt_tokens), -1
            )
            state_mask = (input_ids == self._state_placeholder_id()).unsqueeze(-1)
            inputs_embeds = inputs_embeds.masked_scatter(
                state_mask, state_embeds.to(inputs_embeds.dtype)
            )

        pixel_values = vlm_inputs.get("pixel_values")
        image_grid_thw = vlm_inputs.get("image_grid_thw")
        if pixel_values is not None:
            image_outputs = multimodal_model.get_image_features(
                pixel_values, image_grid_thw, return_dict=True
            )
            image_embeds = torch.cat(image_outputs.pooler_output, dim=0).to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            image_mask, _ = multimodal_model.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        pixel_values_videos = vlm_inputs.get("pixel_values_videos")
        video_grid_thw = vlm_inputs.get("video_grid_thw")
        if pixel_values_videos is not None:
            video_outputs = multimodal_model.get_video_features(
                pixel_values_videos, video_grid_thw, return_dict=True
            )
            video_embeds = torch.cat(video_outputs.pooler_output, dim=0).to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            _, video_mask = multimodal_model.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        position_ids = multimodal_model.compute_3d_position_ids(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            attention_mask=attention_mask,
            past_key_values=None,
            mm_token_type_ids=vlm_inputs.get("mm_token_type_ids"),
        )
        if position_ids is None:
            text_positions = attention_mask.long().cumsum(dim=-1).sub(1).clamp_min(0)
            position_ids = text_positions.unsqueeze(0).expand(3, -1, -1)
        return inputs_embeds, attention_mask, position_ids

    def _vlm_prefix_hidden(
        self,
        vlm_inputs: dict[str, Any],
        states: torch.Tensor | None = None,
        state_dim_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Final hidden states over a robotics prefix, VLM side only.

        Deliberately the same sequence of calls :meth:`_run_joint_trunks` makes
        on its VLM branch -- same causal mask, same rotary embeddings, same
        per-layer-type attention argument -- with the action expert left out.
        Written this way rather than by calling ``language_model(...)`` because
        the 3-D mrope positions are built here and handing a language model a
        different position convention is exactly the kind of divergence that
        would not show up until a checkpoint quietly underperformed.
        """
        vlm_hidden, vlm_padding_mask, vlm_position_ids = self._embed_vlm_inputs(
            vlm_inputs, states, state_dim_mask
        )
        language_model = self.vlm.model.language_model
        vlm_position_embeddings = language_model.rotary_emb(
            vlm_hidden, vlm_position_ids
        )
        from transformers.masking_utils import create_causal_mask

        vlm_causal_mask = create_causal_mask(
            config=language_model.config,
            inputs_embeds=vlm_hidden,
            attention_mask=vlm_padding_mask,
            past_key_values=None,
            position_ids=vlm_position_ids[0],
        )
        for index, vlm_layer in enumerate(language_model.layers):
            attention_mask = (
                vlm_causal_mask
                if self.vlm_layer_types[index] == "full_attention"
                else vlm_padding_mask
            )
            vlm_hidden = vlm_layer(
                vlm_hidden,
                position_embeddings=vlm_position_embeddings,
                attention_mask=attention_mask,
                position_ids=vlm_position_ids[0],
                past_key_values=None,
                use_cache=False,
            )
        return language_model.norm(vlm_hidden)

    @torch.no_grad()
    def generate_subtask(
        self,
        vlm_inputs: dict[str, Any],
        states: torch.Tensor | None = None,
        state_dim_mask: torch.Tensor | None = None,
    ) -> list[str]:
        """Stage one: decode the subtask the policy will then condition on.

        Greedy, because the subtask is not meant to be creative: RoboDojo's
        annotation is a closed set of 509 strings and the action half wants the
        modal one, not a sample from the tail.

        ``vlm_inputs`` must be a prefix built WITHOUT a subtask turn; the header
        is appended here, so step k sees
        ``[images][task][state][assistant header][subtask_<k]`` -- byte for byte
        the context the CE trained against.  That correspondence is why no
        second forward pass is needed to align training with inference.
        """
        tokenizer = self.processor.tokenizer
        im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
        header = torch.tensor(
            self._subtask_header_ids(), dtype=vlm_inputs["input_ids"].dtype
        )
        batch_size = int(vlm_inputs["input_ids"].shape[0])
        working = self._extend_prefix(
            dict(vlm_inputs), header.unsqueeze(0).expand(batch_size, -1)
        )

        generated: list[list[int]] = [[] for _ in range(batch_size)]
        finished = torch.zeros(batch_size, dtype=torch.bool)
        for _ in range(int(self.config.subtask_max_new_tokens)):
            hidden = self._vlm_prefix_hidden(working, states, state_dim_mask)
            next_ids = self._lm_head(hidden[:, -1]).float().argmax(dim=-1)
            # A finished row keeps emitting the terminator so the batch stays
            # rectangular; its tail is dropped when the text is assembled.
            next_ids = torch.where(
                finished.to(next_ids.device),
                torch.full_like(next_ids, im_end),
                next_ids,
            )
            for index in range(batch_size):
                if not bool(finished[index]) and int(next_ids[index]) != im_end:
                    generated[index].append(int(next_ids[index]))
            finished |= (next_ids == im_end).cpu()
            if bool(finished.all()):
                break
            working = self._extend_prefix(
                working, next_ids.unsqueeze(1).to(working["input_ids"].dtype)
            )
        return [tokenizer.decode(row) for row in generated]

    def _extend_prefix(
        self, vlm_inputs: dict[str, Any], new_ids: torch.Tensor
    ) -> dict[str, Any]:
        """Append token ids to every 2-D prefix tensor, zero-filling non-text ones.

        Same shape discipline as :meth:`_append_state_placeholders` and
        :meth:`_append_subtask_turn`: only tensors whose second dimension is the
        current prefix length are extended, so image grids and the like are left
        alone.
        """
        prefix_len = int(vlm_inputs["input_ids"].shape[1])
        count = int(new_ids.shape[1])
        batch_size = int(new_ids.shape[0])
        updated = dict(vlm_inputs)
        for key, value in vlm_inputs.items():
            if not (
                torch.is_tensor(value)
                and value.ndim == 2
                and value.shape[1] == prefix_len
            ):
                continue
            if key == "input_ids":
                tail = new_ids.to(value.dtype)
            elif key == "attention_mask":
                tail = torch.ones(
                    batch_size, count, dtype=value.dtype, device=value.device
                )
            else:
                tail = torch.zeros(
                    batch_size, count, dtype=value.dtype, device=value.device
                )
            updated[key] = torch.cat([value, tail], dim=1)
        return updated

    def _run_joint_trunks(
        self,
        vlm_inputs: dict[str, Any],
        noisy_actions: torch.Tensor,
        timesteps: torch.Tensor,
        states: torch.Tensor,
        action_attention_mask: torch.Tensor,
        prefix_len: int,
        state_dim_mask: torch.Tensor | None = None,
        subtask_span: tuple[int, int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        vlm_hidden, vlm_padding_mask, vlm_position_ids = self._embed_vlm_inputs(
            vlm_inputs, states, state_dim_mask
        )
        language_model = self.vlm.model.language_model
        vlm_position_embeddings = language_model.rotary_emb(
            vlm_hidden, vlm_position_ids
        )
        from transformers.masking_utils import create_causal_mask

        vlm_causal_mask = create_causal_mask(
            config=language_model.config,
            inputs_embeds=vlm_hidden,
            attention_mask=vlm_padding_mask,
            past_key_values=None,
            position_ids=vlm_position_ids[0],
        )

        prefix_padding_mask = self._expert_prefix_mask(
            vlm_padding_mask, prefix_len, subtask_span
        )
        prefix_position_ids = vlm_position_ids[:, :, :prefix_len]
        prefix_position_embeddings = tuple(
            value[:, :prefix_len] for value in vlm_position_embeddings
        )
        action_hidden = self.action_expert.embed_inputs(
            noisy_actions, timesteps, states
        )
        # Dropped/absent subtasks must affect neither keys nor the action RoPE
        # origin. Otherwise their hidden token count still conditions the action.
        valid_positions = prefix_position_ids.masked_fill(
            ~prefix_padding_mask.unsqueeze(0), 0
        )
        action_start = valid_positions.amax(dim=(0, 2)).add(1)
        action_positions = action_start[:, None] + torch.arange(
            action_hidden.shape[1], device=action_hidden.device
        )
        action_position_ids = action_positions.unsqueeze(0).expand(3, -1, -1)
        action_position_embeddings = language_model.rotary_emb(
            action_hidden, action_position_ids
        )

        checkpoint_expert = self.config.gradient_checkpointing and self.training
        if checkpoint_expert:
            from torch.utils.checkpoint import checkpoint

        for index, (vlm_layer, expert_layer) in enumerate(
            zip(language_model.layers, self.action_expert.blocks, strict=True)
        ):
            if expert_layer.use_full_attention:
                if self.vlm_layer_types[index] != "full_attention":
                    raise RuntimeError("VLM/expert full-attention schedules diverged")
                vlm_normalized = vlm_layer.input_layernorm(vlm_hidden)
                expert_context = vlm_normalized[:, :prefix_len]
                if self.config.knowledge_insulation:
                    expert_context = expert_context.detach()
                if checkpoint_expert:

                    def run_joint(
                        action_value,
                        context_value,
                        *,
                        current_expert_layer=expert_layer,
                        current_vlm_layer=vlm_layer,
                    ):
                        return current_expert_layer.forward_full(
                            action_value,
                            action_position_embeddings,
                            action_attention_mask,
                            vlm_hidden_states=context_value,
                            vlm_attention_module=current_vlm_layer.self_attn,
                            vlm_position_embeddings=prefix_position_embeddings,
                            vlm_attention_mask=prefix_padding_mask,
                            detach_vlm_parameters=self.config.knowledge_insulation,
                        )

                    action_hidden = checkpoint(
                        run_joint,
                        action_hidden,
                        expert_context,
                        use_reentrant=False,
                    )
                else:
                    action_hidden = expert_layer.forward_full(
                        action_hidden,
                        action_position_embeddings,
                        action_attention_mask,
                        vlm_hidden_states=expert_context,
                        vlm_attention_module=vlm_layer.self_attn,
                        vlm_position_embeddings=prefix_position_embeddings,
                        vlm_attention_mask=prefix_padding_mask,
                        detach_vlm_parameters=self.config.knowledge_insulation,
                    )
                vlm_hidden = vlm_layer(
                    vlm_hidden,
                    position_embeddings=vlm_position_embeddings,
                    attention_mask=vlm_causal_mask,
                    position_ids=vlm_position_ids[0],
                    past_key_values=None,
                    use_cache=False,
                )
            else:
                if self.vlm_layer_types[index] != "linear_attention":
                    raise RuntimeError("VLM/expert DeltaNet schedules diverged")
                if checkpoint_expert:
                    action_hidden = checkpoint(
                        expert_layer.forward_linear,
                        action_hidden,
                        action_attention_mask,
                        use_reentrant=False,
                    )
                else:
                    action_hidden = expert_layer.forward_linear(
                        action_hidden, action_attention_mask
                    )
                vlm_hidden = vlm_layer(
                    vlm_hidden,
                    position_embeddings=vlm_position_embeddings,
                    attention_mask=vlm_padding_mask,
                    position_ids=vlm_position_ids[0],
                    past_key_values=None,
                    use_cache=False,
                )
        vlm_hidden = language_model.norm(vlm_hidden)
        return self.action_expert.project_output(action_hidden), vlm_hidden

    def _prepare_flow_inference_cache(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Run the action-independent VLM prefix once for flow sampling.

        The VLM branch is asymmetric: action tokens read its prefix, while the
        VLM never reads the noisy action trajectory.  Its hidden states and the
        projected K/V tensors are therefore identical at every reverse-flow
        step and can be cached for the whole generated chunk.
        """
        states = batch[STATE]
        vlm_hidden, vlm_padding_mask, vlm_position_ids = self._embed_vlm_inputs(
            batch["vlm_inputs"], states, batch.get(STATE_DIM_MASK)
        )
        language_model = self.vlm.model.language_model
        vlm_position_embeddings = language_model.rotary_emb(
            vlm_hidden, vlm_position_ids
        )
        from transformers.masking_utils import create_causal_mask

        vlm_causal_mask = create_causal_mask(
            config=language_model.config,
            inputs_embeds=vlm_hidden,
            attention_mask=vlm_padding_mask,
            past_key_values=None,
            position_ids=vlm_position_ids[0],
        )
        prefix_len = int(batch["prefix_len"])
        prefix_padding_mask = vlm_padding_mask[:, :prefix_len]
        prefix_position_ids = vlm_position_ids[:, :, :prefix_len]
        prefix_position_embeddings = tuple(
            value[:, :prefix_len] for value in vlm_position_embeddings
        )
        valid_positions = prefix_position_ids.masked_fill(
            ~prefix_padding_mask.unsqueeze(0), 0
        )
        action_start = valid_positions.amax(dim=(0, 2)).add(1)
        action_positions = action_start[:, None] + torch.arange(
            self.config.chunk_size, device=states.device
        )
        action_position_ids = action_positions.unsqueeze(0).expand(3, -1, -1)
        action_rope_reference = vlm_hidden.new_empty(
            (
                states.shape[0],
                self.config.chunk_size,
                self.config.action_expert_hidden_size,
            )
        )
        action_position_embeddings = language_model.rotary_emb(
            action_rope_reference, action_position_ids
        )

        vlm_key_values: list[tuple[torch.Tensor, torch.Tensor] | None] = []
        for index, (vlm_layer, expert_layer) in enumerate(
            zip(language_model.layers, self.action_expert.blocks, strict=True)
        ):
            if expert_layer.use_full_attention:
                if self.vlm_layer_types[index] != "full_attention":
                    raise RuntimeError("VLM/expert full-attention schedules diverged")
                expert_context = vlm_layer.input_layernorm(vlm_hidden)[:, :prefix_len]
                vlm_key_values.append(
                    expert_layer.mixer._vlm_kv(
                        expert_context,
                        vlm_layer.self_attn,
                        prefix_position_embeddings,
                    )
                )
                vlm_hidden = vlm_layer(
                    vlm_hidden,
                    position_embeddings=vlm_position_embeddings,
                    attention_mask=vlm_causal_mask,
                    position_ids=vlm_position_ids[0],
                    past_key_values=None,
                    use_cache=False,
                )
            else:
                if self.vlm_layer_types[index] != "linear_attention":
                    raise RuntimeError("VLM/expert DeltaNet schedules diverged")
                vlm_key_values.append(None)
                vlm_hidden = vlm_layer(
                    vlm_hidden,
                    position_embeddings=vlm_position_embeddings,
                    attention_mask=vlm_padding_mask,
                    position_ids=vlm_position_ids[0],
                    past_key_values=None,
                    use_cache=False,
                )
        return {
            "action_position_embeddings": action_position_embeddings,
            "prefix_padding_mask": prefix_padding_mask,
            "vlm_key_values": tuple(vlm_key_values),
        }

    def _run_cached_action_trunk(
        self,
        noisy_actions: torch.Tensor,
        timesteps: torch.Tensor,
        states: torch.Tensor,
        action_attention_mask: torch.Tensor,
        cache: dict[str, Any],
    ) -> torch.Tensor:
        """Run only the Action Expert using a precomputed VLM prefix cache."""
        action_hidden = self.action_expert.embed_inputs(
            noisy_actions, timesteps, states
        )
        for expert_layer, cached_vlm_key_value in zip(
            self.action_expert.blocks, cache["vlm_key_values"], strict=True
        ):
            if expert_layer.use_full_attention:
                action_hidden = expert_layer.forward_full(
                    action_hidden,
                    cache["action_position_embeddings"],
                    action_attention_mask,
                    vlm_attention_mask=cache["prefix_padding_mask"],
                    cached_vlm_key_value=cached_vlm_key_value,
                )
            else:
                action_hidden = expert_layer.forward_linear(
                    action_hidden, action_attention_mask
                )
        return self.action_expert.project_output(action_hidden)

    def _sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        alpha = torch.tensor(
            self.config.time_sampling_beta_alpha, device=device, dtype=torch.float32
        )
        beta = torch.tensor(
            self.config.time_sampling_beta_beta, device=device, dtype=torch.float32
        )
        samples = torch.distributions.Beta(alpha, beta).sample((batch_size,))
        return self.config.time_min + samples * (
            self.config.time_max - self.config.time_min
        )

    @torch.no_grad()
    def generate_actions(
        self,
        batch: dict[str, Any],
        num_steps: int | None = None,
        initial_noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Sample a normalized action chunk by reversing the flow trajectory."""
        if "vlm_inputs" not in batch:
            raise ValueError(
                "generate_actions needs a prepared batch; call "
                "prepare_inference_batch first"
            )
        states = batch[STATE]
        batch_size = int(states.shape[0])
        device = states.device
        n_steps = int(num_steps or self.config.num_inference_steps)
        if n_steps <= 0:
            raise ValueError("num_steps must be positive")

        expected_shape = (
            batch_size,
            self.config.chunk_size,
            self.config.action_dim,
        )
        if initial_noise is None:
            actions = torch.randn(*expected_shape, device=device, dtype=torch.float32)
        else:
            if tuple(initial_noise.shape) != expected_shape:
                raise ValueError(
                    "initial_noise shape mismatch: expected "
                    f"{expected_shape}, got {tuple(initial_noise.shape)}"
                )
            if not torch.isfinite(initial_noise).all():
                raise ValueError("initial_noise contains non-finite values")
            actions = initial_noise.to(device=device, dtype=torch.float32).clone()

        action_dim_mask = batch.get(ACTION_DIM_MASK)
        if action_dim_mask is None:
            action_dim_mask = torch.ones(
                expected_shape, device=device, dtype=torch.bool
            )
        else:
            if tuple(action_dim_mask.shape) != expected_shape:
                raise ValueError(
                    "action dimension mask shape mismatch: expected "
                    f"{expected_shape}, got {tuple(action_dim_mask.shape)}"
                )
            action_dim_mask = action_dim_mask.to(device=device, dtype=torch.bool)
        # Training never presents noise on invalid embodiment dimensions. Keep
        # the same contract throughout reverse flow, not only at the final slice.
        actions = actions.masked_fill(~action_dim_mask, 0.0)
        valid_steps = action_dim_mask.any(dim=-1)
        was_training = self.training
        self.eval()
        try:
            flow_cache = self._prepare_flow_inference_cache(batch)
            for step in range(n_steps):
                time = 1.0 - step / n_steps
                timesteps = torch.full(
                    (batch_size,), time, device=device, dtype=torch.float32
                )
                velocity = self._run_cached_action_trunk(
                    actions,
                    timesteps,
                    states,
                    valid_steps,
                    flow_cache,
                )
                velocity = velocity.float().masked_fill(~action_dim_mask, 0.0)
                actions = (actions - velocity / n_steps).masked_fill(
                    ~action_dim_mask, 0.0
                )
            return actions
        finally:
            self.train(was_training)

    @torch.no_grad()
    def evaluate_from_noise(
        self, batch: dict[str, Any], num_steps: int | None = None
    ) -> dict[str, Any]:
        """Open-loop error of a chunk sampled from pure noise, in normalized units.

        The training objective scores one velocity at one random flow time; this
        integrates the whole trajectory from ``t=1`` to ``t=0`` and compares the
        chunk that actually comes out against the targets.  It is the quantity
        deployment cares about, and the two can move in opposite directions --
        integration error accumulates over the horizon, which a per-timestep
        velocity MSE never charges for.

        The ``per_horizon`` vector is the point of the exercise: the horizon
        normalizer work found end-of-chunk error a quarter high while the
        aggregate looked fine, and only the curve shows that.

        Runs on a *training* batch (targets included), which is safe in both of
        the ways it could leak:

        * FAST action tokens encode the targets, but ``prepare_batch`` appends
          them past ``prefix_len`` and ``_prepare_flow_inference_cache`` keys the
          expert off ``[:prefix_len]`` only.  A causal VLM cannot see them from
          inside the prefix either.
        * ``subtask_dropout`` is gated on ``self.training``, so conditioning does
          not vary between two eval passes.

        With ``subtask_generation`` on, the prefix carries the *ground-truth*
        subtask, so this is open-loop error under an oracle subtask -- a lower
        bound on what the same policy does when it has to say the subtask itself.
        """
        targets = batch[ACTIONS].float()
        valid = (~batch[ACTION_IS_PAD].bool()).unsqueeze(-1) & batch[
            ACTION_DIM_MASK
        ].bool()
        valid = valid.expand_as(targets)

        predicted = self.generate_actions(batch, num_steps=num_steps).float()
        squared_error = (predicted - targets).square() * valid

        total = valid.sum()
        # Sum over batch and action dim, leaving one entry per horizon position.
        per_horizon_error = squared_error.sum(dim=(0, 2))
        per_horizon_count = valid.sum(dim=(0, 2))
        return {
            "from_noise_mse": float(squared_error.sum() / total.clamp_min(1)),
            "from_noise_elements": int(total),
            "from_noise_steps": int(num_steps or self.config.num_inference_steps),
            # Kept as tensors: the trainer reduces them across ranks before any
            # of it becomes a number worth reading.
            "from_noise_per_horizon_error": per_horizon_error,
            "from_noise_per_horizon_count": per_horizon_count,
        }

    def forward(self, batch: dict[str, Any]) -> ModelForwardOutput:
        if "vlm_inputs" not in batch:
            raise ValueError(
                "magicw0_base received an unprepared batch; the trainer must call "
                "policy.prepare_batch before forward"
            )
        modality = str(
            batch.get(
                self.config.cotrain_modality_key, self.config.cotrain_native_modality
            )
        )
        if modality in self.config.vlm_modalities:
            loss, metrics = self._vlm_loss(batch)
            return ModelForwardOutput(
                loss=loss,
                metrics=metrics,
                loss_name="magicw0_base_vlm_cross_entropy",
            )

        actions = batch[ACTIONS]
        states = batch[STATE]
        valid_steps = ~batch[ACTION_IS_PAD].bool()
        valid = valid_steps.unsqueeze(-1) & batch[ACTION_DIM_MASK].bool()
        timesteps = self._sample_timesteps(actions.shape[0], actions.device)
        noise = torch.randn_like(actions).masked_fill(~valid, 0.0)
        time_view = timesteps[:, None, None].to(dtype=actions.dtype)
        noisy_actions = (1.0 - time_view) * actions + time_view * noise
        target_velocity = noise - actions

        prefix_len = int(
            batch.get("prefix_len", batch["vlm_inputs"]["input_ids"].shape[1])
        )
        subtask_ids = batch.get("subtask_ids")
        subtask_span = self._subtask_span(prefix_len, subtask_ids)
        predicted_velocity, vlm_hidden = self._run_joint_trunks(
            batch["vlm_inputs"],
            noisy_actions,
            timesteps,
            states,
            valid_steps,
            prefix_len,
            batch.get(STATE_DIM_MASK),
            subtask_span,
        )
        mask = valid.expand_as(predicted_velocity)
        squared_error = (predicted_velocity.float() - target_velocity.float()).square()
        time_weights = action_loss_time_weights(
            self.config.action_loss_weighting,
            predicted_velocity.shape[1],
            device=predicted_velocity.device,
            dtype=torch.float32,
        )
        weighted_mask = mask.float() * time_weights[None, :, None]
        denominator = weighted_mask.sum().clamp_min(1.0)
        weighted_error = squared_error * weighted_mask
        flow_loss = weighted_error.sum() / denominator
        per_sample_flow_loss = weighted_error.sum(dim=(1, 2)) / weighted_mask.sum(
            dim=(1, 2)
        ).clamp_min(1.0)
        loss = self.config.flow_matching_loss_weight * flow_loss
        per_sample_loss = self.config.flow_matching_loss_weight * per_sample_flow_loss
        metrics: dict[str, Any] = {
            "modality": modality,
            "flow_loss": float(flow_loss.detach().cpu()),
            "valid_action_steps": int(valid_steps.sum().detach().cpu()),
            "valid_action_elements": int(valid.sum().detach().cpu()),
            "action_loss_weight_mean": float(time_weights.mean().detach().cpu()),
            "expert_full_attention_layers": self.action_expert.num_full_attention_layers,
        }
        if self.config.use_fast_action_tokens:
            fast_loss, n_fast_tokens, per_sample_fast_loss = self._fast_ce_loss(
                vlm_hidden,
                prefix_len,
                batch["fast_ids"],
                batch["fast_valid"],
                batch.get("subtask_valid"),
                return_per_sample=True,
            )
            loss = loss + self.config.fast_ce_loss_weight * fast_loss
            per_sample_loss = (
                per_sample_loss + self.config.fast_ce_loss_weight * per_sample_fast_loss
            )
            metrics["fast_loss"] = float(fast_loss.detach().cpu())
            metrics["n_fast_tokens"] = n_fast_tokens
        if (
            self.config.subtask_generation
            and batch.get("subtask_ids") is not None
            and self._subtask_ce_is_supervised(batch["subtask_valid"])
        ):
            subtask_loss, n_subtask_tokens, exact = self._subtask_ce_loss(
                vlm_hidden,
                prefix_len,
                batch["subtask_ids"],
                batch["subtask_valid"],
            )
            loss = loss + self.config.subtask_ce_loss_weight * subtask_loss
            metrics["subtask_ce"] = float(subtask_loss.detach().cpu())
            metrics["subtask_exact_match"] = exact
            metrics["n_subtask_tokens"] = n_subtask_tokens
        metrics["loss"] = float(loss.detach().cpu())
        metrics["per_sample_loss"] = per_sample_loss.detach().float().cpu().tolist()
        return ModelForwardOutput(
            loss=loss,
            metrics=metrics,
            loss_name=self.loss_name,
        )

    def _vlm_hidden(self, vlm_inputs: dict[str, Any]) -> torch.Tensor:
        outputs = self.vlm.model(**vlm_inputs, use_cache=False, return_dict=True)
        hidden = outputs.last_hidden_state
        if hidden is None:
            raise RuntimeError(
                "Qwen3.5 did not return last_hidden_state for magicw0_base"
            )
        return hidden

    def _vlm_loss(self, batch: dict[str, Any]) -> tuple[torch.Tensor, dict[str, Any]]:
        # Opt-in CUDA-event timings make dynamic-shape/Triton stalls observable
        # without synchronizing every normal training phase. The scalar metrics
        # below already materialize the completed forward on CPU, so reading the
        # events does not add another synchronization point.
        diagnose = (
            os.environ.get("MAGIC_W0_VLM_DIAGNOSTICS") == "1"
            and torch.cuda.is_available()
            and batch["vlm_inputs"]["input_ids"].is_cuda
        )
        events = None
        if diagnose:
            events = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
            events[0].record()
        hidden = self._vlm_hidden(batch["vlm_inputs"])
        if events is not None:
            events[1].record()
        logits = self._lm_head(hidden).float()
        if events is not None:
            events[2].record()
        vlm_inputs = batch["vlm_inputs"]
        labels = batch.get("labels")
        if labels is None:
            labels = vlm_inputs["input_ids"].clone()
            labels[~vlm_inputs["attention_mask"].bool()] = (
                self.config.cotrain_ignore_index
            )
        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        per_token_ce = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.shape[-1]),
            shift_labels.reshape(-1),
            ignore_index=self.config.cotrain_ignore_index,
            reduction="none",
        ).view_as(shift_labels)
        supervised_mask = shift_labels != self.config.cotrain_ignore_index
        ce = (per_token_ce * supervised_mask).sum() / supervised_mask.sum().clamp_min(1)
        per_sample_ce = (per_token_ce * supervised_mask).sum(
            dim=1
        ) / supervised_mask.sum(dim=1).clamp_min(1)
        if events is not None:
            events[3].record()
        loss = self.config.vlm_loss_weight * ce
        supervised = int(
            (shift_labels != self.config.cotrain_ignore_index).sum().detach().cpu()
        )
        metrics = {
            "modality": "vlm",
            "loss": float(loss.detach().cpu()),
            "vlm_ce": float(ce.detach().cpu()),
            "vlm_loss_weight": self.config.vlm_loss_weight,
            "supervised_tokens": supervised,
            "per_sample_loss": (self.config.vlm_loss_weight * per_sample_ce)
            .detach()
            .float()
            .cpu()
            .tolist(),
            # Actual post-processor shape. This is deliberately logged next to the
            # sampler's budget so a run can verify that no unexpected template or
            # image expansion made it past the scan-time cost model.
            "vlm_batch_samples": int(vlm_inputs["input_ids"].shape[0]),
            "vlm_sequence_tokens": int(vlm_inputs["input_ids"].shape[1]),
            "vlm_valid_input_tokens": int(
                vlm_inputs["attention_mask"].sum().detach().cpu()
            ),
            "vlm_padding_tokens": int(
                vlm_inputs["input_ids"].numel()
                - vlm_inputs["attention_mask"].sum().detach().cpu()
            ),
        }
        if events is not None:
            pixel_values = vlm_inputs.get("pixel_values")
            image_grid = vlm_inputs.get("image_grid_thw")
            metrics.update(
                {
                    "vlm_diag_hidden_ms": float(events[0].elapsed_time(events[1])),
                    "vlm_diag_lm_head_ms": float(events[1].elapsed_time(events[2])),
                    "vlm_diag_ce_ms": float(events[2].elapsed_time(events[3])),
                    "vlm_diag_logits_gel": float(logits.numel()) / 1.0e9,
                    "vlm_diag_logits_gb": (
                        float(logits.numel() * logits.element_size()) / 1.0e9
                    ),
                    "vlm_diag_pixel_elements": (
                        int(pixel_values.numel())
                        if torch.is_tensor(pixel_values)
                        else 0
                    ),
                    "vlm_diag_image_grid_rows": (
                        int(image_grid.shape[0]) if torch.is_tensor(image_grid) else 0
                    ),
                }
            )
        return loss, metrics

    @staticmethod
    def _subtask_ce_is_supervised(subtask_valid: torch.Tensor) -> bool:
        """Whether any target survives the ignore index.

        ``F.cross_entropy`` with mean reduction divides by the number of
        unignored targets, so a batch in which nothing is annotated yields
        0/0 = NaN rather than 0 and would poison the whole loss.  Before the
        turn was made unconditional this could not happen -- there were no slot
        tensors to reach the loss with -- so the guard arrives with the change
        that made it reachable.
        """
        return bool(subtask_valid.any())

    def _subtask_ce_loss(
        self,
        hidden_states: torch.Tensor,
        prefix_len: int,
        subtask_ids: torch.Tensor,
        subtask_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, float, float]:
        """Next-token CE over the subtask slot, plus a whole-string match rate.

        The slot sits at the tail of the prefix, so the window is the standard
        shift by one: the hidden at ``slot_start - 1 + k`` predicts token ``k``.
        Its context under the causal mask is
        ``[images][task][state][assistant header][subtask_<k]`` -- byte for byte
        what step k of the two-stage decode will see, which is why training the
        subtask inside the long action sequence and decoding it on its own are
        the same distribution and no second forward pass is needed to align them.

        Exact match is reported next to the CE because this annotation is a
        closed set of 509 strings: a rate over whole strings says more about the
        task than a per-token perplexity does.  Samples with no subtask are left
        out of the denominator rather than scored as correct -- an unlabelled
        neighbour must not read as a free win.
        """
        slot_len = int(subtask_ids.shape[1])
        slot_start = prefix_len - slot_len
        window = hidden_states[:, slot_start - 1 : slot_start - 1 + slot_len]
        logits = self._lm_head(window).float()
        labels = subtask_ids.masked_fill(~subtask_valid, -100)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
            ignore_index=-100,
        )
        predicted = logits.argmax(dim=-1)
        correct = ((predicted == subtask_ids) | ~subtask_valid).all(dim=1)
        has_subtask = subtask_valid.any(dim=1)
        exact = float(
            (correct & has_subtask)
            .sum()
            .float()
            .div(has_subtask.sum().clamp_min(1))
            .detach()
            .cpu()
        )
        n_tokens = float(subtask_valid.sum(dim=1).float().mean().detach().cpu())
        return loss, n_tokens, exact

    def _fast_ce_loss(
        self,
        hidden_states: torch.Tensor,
        prefix_len: int,
        fast_ids: torch.Tensor,
        fast_valid: torch.Tensor,
        subtask_valid: torch.Tensor | None = None,
        return_per_sample: bool = False,
    ):
        """Next-token CE over the appended FAST window.

        ``prefix_len - 1`` is the right window start only while every sample's
        prefix ends on a real token.  A right-padded subtask slot breaks that: a
        sample whose annotation is shorter than the batch maximum ends on
        padding, and reading a pad position's hidden state to predict the first
        FAST id is not what the alignment means. Only the first prediction reads
        each sample's last real prefix position; later predictions read the
        physical FAST positions after prefix_len, skipping any subtask padding.
        """
        length = int(fast_ids.shape[1])
        batch_size = int(fast_ids.shape[0])
        device = hidden_states.device
        if subtask_valid is None:
            start = torch.full(
                (batch_size,), prefix_len - 1, device=device, dtype=torch.long
            )
        else:
            slot_start = prefix_len - int(subtask_valid.shape[1])
            counts = subtask_valid.sum(dim=1).to(torch.long)
            start = torch.where(
                counts > 0,
                slot_start + counts - 1,
                torch.full_like(
                    counts, slot_start - len(self._subtask_header_ids()) - 1
                ),
            )
        following = torch.arange(prefix_len, prefix_len + length - 1, device=device)
        index = torch.cat(
            [start[:, None], following[None, :].expand(batch_size, -1)], dim=1
        )
        window = hidden_states.gather(
            1, index.unsqueeze(-1).expand(-1, -1, hidden_states.shape[-1])
        )
        logits = self._lm_head(window).float()
        labels = fast_ids.masked_fill(~fast_valid, -100)
        per_token = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
            ignore_index=-100,
            reduction="none",
        ).view_as(labels)
        loss = (per_token * fast_valid).sum() / fast_valid.sum().clamp_min(1)
        n_tokens = float(fast_valid.sum(dim=1).float().mean().detach().cpu())
        if return_per_sample:
            per_sample = (per_token * fast_valid).sum(dim=1) / fast_valid.sum(
                dim=1
            ).clamp_min(1)
            return loss, n_tokens, per_sample
        return loss, n_tokens

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str | Path,
        *,
        config: MagicW0BaseConfig,
        local_files_only: bool = True,
        strict: bool = True,
    ) -> "MagicW0BasePolicy":
        payload = torch.load(
            checkpoint, map_location="cpu", weights_only=True, mmap=True
        )
        state = payload.get("model", payload)
        state = {key.removeprefix("module."): value for key, value in state.items()}
        vlm_state = {
            key.removeprefix("vlm."): value
            for key, value in state.items()
            if key.startswith("vlm.")
        }
        if not vlm_state:
            raise ValueError(
                "Initialization checkpoint must include complete vlm.* weights"
            )
        model = cls(config, vlm_state_dict=vlm_state)
        # Public inference checkpoints omit this duplicate module registration.
        # Reconstruct only the alias from the VLM tensors already strictly
        # loaded above; do not relax validation of any policy parameter.
        if model._lm_head is not None:
            for key, value in model._lm_head.state_dict().items():
                alias = f"_lm_head.{key}"
                if alias in state and not torch.equal(state[alias], value):
                    raise ValueError(
                        f"Checkpoint output-head alias conflicts with VLM: {alias}"
                    )
                state[alias] = value
        model.load_state_dict(state, strict=strict)
        return model


MagicW0Base = MagicW0BasePolicy
