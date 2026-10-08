"""Magic-W0 inference components; tensor names and computations are preserved."""

from __future__ import annotations
from typing import Any
import math
import torch
from torch import nn
from torch.nn import functional as F
from .config import MagicW0Config


def gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    beta: torch.Tensor,
    decay: torch.Tensor,
) -> torch.Tensor:
    """The same rule, solved for the whole chunk in one shot.

    Unrolling the recurrence gives ``S_t = sum_{i<=t} (A_t / A_i) k_i u_i^T`` for
    cumulative decay ``A_t = prod_{j<=t} a_j``. Substituting that back leaves the
    ``u`` rows satisfying one unit-lower-triangular system,

        ``u_t = b_t v_t - b_t sum_{i<t} (A_t / A_i) (k_t . k_i) u_i``

    so the chunk is a triangular solve instead of ``T`` dependent steps, and the
    outputs are ``o_t = sum_{i<=t} (A_t / A_i) (q_t . k_i) u_i`` -- one masked
    matmul. Decays enter only as ratios ``A_t / A_i`` with ``i <= t``, bounded by
    1; the raw ``A_t`` underflows over a long chunk and must not appear alone.

    Stepping this in Python costs ~6 kernel launches per timestep per layer, and
    both experts stack it 9-18 deep, which is why the closed form is the default.
    """
    sequence = query.shape[2]
    log_decay = decay.cumsum(dim=-1)
    relative = log_decay.unsqueeze(-1) - log_decay.unsqueeze(-2)
    causal = torch.ones(
        sequence, sequence, dtype=torch.bool, device=query.device
    ).tril()
    decay_matrix = relative.masked_fill(~causal, -float("inf")).exp()
    interaction = key @ key.transpose(-1, -2) * decay_matrix
    system = beta.unsqueeze(-1) * interaction.tril(-1)
    updates = torch.linalg.solve_triangular(
        system, beta.unsqueeze(-1) * value, upper=False, unitriangular=True
    )
    attention = (query @ key.transpose(-1, -2) * decay_matrix).tril()
    return attention @ updates


class TimestepEmbedding(nn.Module):
    """Sinusoidal flow time followed by the standard two-layer MLP."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.SiLU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.hidden_size // 2
        frequencies = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=timesteps.device, dtype=torch.float32)
            / max(half - 1, 1)
        )
        angles = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
        embedding = torch.cat((angles.cos(), angles.sin()), dim=-1)
        if embedding.shape[-1] < self.hidden_size:
            embedding = F.pad(embedding, (0, self.hidden_size - embedding.shape[-1]))
        return self.mlp(embedding.to(dtype=self.mlp[0].weight.dtype))


class RMSNorm(nn.Module):
    """Qwen3.5 RMSNorm, whose learned weight is centered around one."""

    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        normalized = value.float() * torch.rsqrt(
            value.float().square().mean(dim=-1, keepdim=True) + self.eps
        )
        return (normalized * (1.0 + self.weight.float())).to(value.dtype)


class GatedRMSNorm(nn.Module):
    """Per-value-head gated RMSNorm used at the DeltaNet output."""

    def __init__(self, head_dim: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(head_dim))

    def forward(self, value: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        dtype = value.dtype
        normalized = value.float() * torch.rsqrt(
            value.float().square().mean(dim=-1, keepdim=True) + self.eps
        )
        normalized = normalized.to(dtype) * self.weight
        return (normalized.float() * F.silu(gate.float())).to(dtype)


class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(value)) * self.up_proj(value))


def _l2_normalize(value: torch.Tensor, eps: float = 1e-06) -> torch.Tensor:
    return value * torch.rsqrt(value.square().sum(dim=-1, keepdim=True) + eps)


class GatedDeltaNet(nn.Module):
    """Parallel PyTorch gated DeltaNet mixer for the three expert streams."""

    _delta_rule = staticmethod(gated_delta_rule)

    def __init__(self, config: MagicW0Config) -> None:
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
        self.norm = GatedRMSNorm(config.deltanet_value_head_dim, config.rms_norm_eps)
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
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None
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
        query = query * self.key_head_dim ** (-0.5)
        beta = torch.sigmoid(self.in_proj_b(hidden_states)).float().transpose(1, 2)
        decay = -self.A_log.float().exp() * F.softplus(
            self.in_proj_a(hidden_states).float() + self.dt_bias
        )
        decay = decay.transpose(1, 2)
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
    """Initialize expert parameters before strict checkpoint loading."""
    for name in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a"):
        nn.init.normal_(getattr(module, name).weight, std=0.02)
    nn.init.normal_(module.conv1d.weight, std=0.02)
    nn.init.normal_(module.out_proj.weight, std=0.02)
    nn.init.ones_(module.dt_bias)
    with torch.no_grad():
        module.A_log.copy_(torch.empty_like(module.A_log).uniform_(0.001, 16.0).log_())
        module.norm.weight.fill_(1.0)


def _rotate_half(value: torch.Tensor) -> torch.Tensor:
    first, second = value.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def _apply_rope(
    value: torch.Tensor, position_embeddings: tuple[torch.Tensor, torch.Tensor]
) -> torch.Tensor:
    cos, sin = position_embeddings
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    rotary_dim = cos.shape[-1]
    rotated, passthrough = (value[..., :rotary_dim], value[..., rotary_dim:])
    rotated = rotated * cos + _rotate_half(rotated) * sin
    return torch.cat((rotated, passthrough), dim=-1)


def _repeat_kv(value: torch.Tensor, repeats: int) -> torch.Tensor:
    if repeats == 1:
        return value
    return value.repeat_interleave(repeats, dim=1)


class JointFullAttention(nn.Module):
    """Expert projections participating in asymmetric VLM/action attention."""

    def __init__(self, config: MagicW0Config) -> None:
        super().__init__()
        hidden = config.action_expert_hidden_size
        self.num_heads = config.full_attn_num_heads
        self.num_kv_heads = config.full_attn_num_kv_heads
        self.head_dim = config.full_attn_head_dim
        self.kv_repeats = self.num_heads // self.num_kv_heads
        self.q_proj = nn.Linear(hidden, self.num_heads * self.head_dim * 2, bias=False)
        self.k_proj = nn.Linear(hidden, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, hidden, bias=False)
        self.q_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, config.rms_norm_eps)

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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if int(attention_module.head_dim) != self.head_dim:
            raise RuntimeError("VLM and expert full-attention head dimensions differ")
        batch, sequence, _ = hidden_states.shape
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
        return (_apply_rope(key, position_embeddings), value)

    def prepare_attention_kv(
        self, key: torch.Tensor, value: torch.Tensor, key_mask: torch.Tensor
    ) -> tuple[Any, ...]:
        """Prepare K/V once so joint's three expert queries can share the work."""
        if key.ndim != 4 or value.shape != key.shape:
            raise ValueError("expert attention K/V must be aligned [B,H,S,D] tensors")
        if key_mask.shape != (key.shape[0], key.shape[2]):
            raise ValueError(
                f"expert attention key mask must be [B,S], got {tuple(key_mask.shape)} for K={tuple(key.shape)}"
            )
        key_mask = key_mask.to(device=key.device, dtype=torch.bool)
        repeated_key = _repeat_kv(key, self.kv_repeats)
        repeated_value = _repeat_kv(value, self.kv_repeats)
        return ("sdpa", repeated_key, repeated_value, key_mask[:, None, None, :])

    def attend_prepared(self, query, prepared_kv):
        _, key, value, allowed = prepared_kv
        return F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=allowed,
            dropout_p=0.0,
            scale=self.head_dim ** (-0.5),
        )


class HybridExpertLayer(nn.Module):
    def __init__(
        self, config: MagicW0Config, *, use_full_attention: bool, layer_idx: int = 0
    ) -> None:
        super().__init__()
        hidden = config.action_expert_hidden_size
        self.use_full_attention = use_full_attention
        self.input_layernorm = RMSNorm(hidden, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(hidden, config.rms_norm_eps)
        self.mixer: nn.Module = (
            JointFullAttention(config) if use_full_attention else GatedDeltaNet(config)
        )
        self.mlp = SwiGLU(hidden, config.action_expert_intermediate_size)

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.input_layernorm.weight)
        nn.init.zeros_(self.post_attention_layernorm.weight)
        if self.use_full_attention:
            self.mixer.reset_parameters()
        else:
            _reset_expert_deltanet(self.mixer)
        for projection in (self.mlp.gate_proj, self.mlp.up_proj, self.mlp.down_proj):
            nn.init.normal_(projection.weight, std=0.02)


class HybridActionExpert(nn.Module):
    """Vocabulary-free Qwen3.5-Text trunk for flow-matching action tokens."""

    def __init__(self, config: MagicW0Config) -> None:
        super().__init__()
        self.config = config
        hidden = config.action_expert_hidden_size
        self.noisy_action_projection = nn.Linear(config.action_dim, hidden)
        self.action_position_embedding = nn.Parameter(
            torch.empty(1, config.chunk_size, hidden)
        )
        self.state_projection = nn.Sequential(
            nn.Linear(config.state_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )
        self.time_embedding = TimestepEmbedding(hidden)
        self.blocks = nn.ModuleList(
            (
                HybridExpertLayer(
                    config,
                    use_full_attention=(index + 1) % config.full_attention_interval
                    == 0,
                    layer_idx=index,
                )
                for index in range(config.action_expert_num_layers)
            )
        )
        self.output_norm = RMSNorm(hidden, config.rms_norm_eps)
        self.output_projection = nn.Linear(hidden, config.action_dim)
        self.reset_parameters()

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
        self, noisy_actions: torch.Tensor, timesteps: torch.Tensor, states: torch.Tensor
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


class Hybrid2DWorldExpert(nn.Module):
    """Hybrid DeltaNet/full-attention queries supervised by frozen DINOv3."""

    def __init__(self, config: MagicW0Config, dino_hidden_size: int) -> None:
        super().__init__()
        self.config = config
        hidden = config.action_expert_hidden_size
        grid_h, grid_w = config.gen_2d_grid_size
        self.query_tokens = nn.Parameter(
            torch.empty(1, config.gen_2d_num_query_tokens_per_view, hidden)
        )
        self.row_embedding = nn.Parameter(torch.empty(grid_h, hidden))
        self.col_embedding = nn.Parameter(torch.empty(grid_w, hidden))
        self.camera_embedding = nn.Parameter(
            torch.empty(config.gen_2d_max_views, hidden)
        )
        self.blocks = nn.ModuleList(
            (
                HybridExpertLayer(
                    config,
                    use_full_attention=(index + 1) % config.full_attention_interval
                    == 0,
                    layer_idx=index,
                )
                for index in range(config.action_expert_num_layers)
            )
        )
        # Stored projection tensors keep full-checkpoint loading strict. Only
        # query embeddings and blocks are executed during action inference.
        self.output_norm = RMSNorm(hidden, config.rms_norm_eps)
        self.projector = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, dino_hidden_size)
        )
        self.residual_dropout = nn.Identity()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.query_tokens, std=0.02)
        nn.init.normal_(self.row_embedding, std=0.02)
        nn.init.normal_(self.col_embedding, std=0.02)
        nn.init.normal_(self.camera_embedding, std=0.02)
        for block in self.blocks:
            block.reset_parameters()
        nn.init.zeros_(self.output_norm.weight)
        nn.init.ones_(self.projector[0].weight)
        nn.init.zeros_(self.projector[0].bias)
        nn.init.normal_(self.projector[1].weight, std=0.02)
        nn.init.zeros_(self.projector[1].bias)

    def embed_queries(self, batch_size: int, device: torch.device) -> torch.Tensor:
        grid_h, grid_w = self.config.gen_2d_grid_size
        spatial = (self.row_embedding[:, None] + self.col_embedding[None]).reshape(
            grid_h * grid_w, -1
        )
        per_view = self.query_tokens + spatial.unsqueeze(0)
        per_view = per_view.expand(self.config.gen_2d_max_views, -1, -1)
        per_view = per_view + self.camera_embedding[:, None]
        tokens = per_view.reshape(-1, per_view.shape[-1]).to(device)
        return tokens.unsqueeze(0).expand(batch_size, -1, -1)


class Hybrid3DWorldExpert(nn.Module):
    """Hybrid world queries supervised by Track4World geometry and motion latents."""

    def __init__(self, config: MagicW0Config) -> None:
        super().__init__()
        self.config = config
        hidden = config.action_expert_hidden_size
        grid_h, grid_w = config.gen_3d_grid_size
        self.query_tokens = nn.Parameter(
            torch.empty(1, config.gen_3d_num_query_tokens, hidden)
        )
        self.row_embedding = nn.Parameter(torch.empty(grid_h, hidden))
        self.col_embedding = nn.Parameter(torch.empty(grid_w, hidden))
        self.blocks = nn.ModuleList(
            (
                HybridExpertLayer(
                    config,
                    use_full_attention=(index + 1) % config.full_attention_interval
                    == 0,
                    layer_idx=index,
                )
                for index in range(config.action_expert_num_layers)
            )
        )
        self.output_norm = RMSNorm(hidden, config.rms_norm_eps)
        # Keep these stored heads for checkpoint compatibility, without any
        # teacher execution or geometry/motion prediction API.
        self.geometry_projector = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1024),
        )
        self.motion_projector = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1024),
        )
        self.residual_dropout = nn.Identity()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.query_tokens, std=0.02)
        nn.init.normal_(self.row_embedding, std=0.02)
        nn.init.normal_(self.col_embedding, std=0.02)
        for block in self.blocks:
            block.reset_parameters()
        nn.init.zeros_(self.output_norm.weight)
        for projector in (self.geometry_projector, self.motion_projector):
            nn.init.ones_(projector[0].weight)
            nn.init.zeros_(projector[0].bias)
            for module in projector:
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, std=0.02)
                    nn.init.zeros_(module.bias)

    def embed_queries(self, batch_size: int, device: torch.device) -> torch.Tensor:
        grid_h, grid_w = self.config.gen_3d_grid_size
        spatial = (self.row_embedding[:, None] + self.col_embedding[None]).reshape(
            grid_h * grid_w, -1
        )
        tokens = (self.query_tokens + spatial.unsqueeze(0)).to(device)
        return tokens.expand(batch_size, -1, -1)
