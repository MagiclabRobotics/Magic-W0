"""The gated delta rule, stepped and solved.

Both MagicW0 action experts run this recurrence over the action chunk, and both
mirror a Qwen3.5 Gated DeltaNet layer, so the arithmetic lives here once rather
than in two files that would drift apart in ways no shape error would catch.

Inputs are ``[B, H, T, D]`` for the projections and ``[B, H, T]`` for the two
scalars, all float32. ``decay`` is the LOG of the per-step decay -- the callers
build it as ``-exp(A_log) * softplus(...)``, which is already a log, so nobody
has to take one.
"""

from __future__ import annotations

import torch


def gated_delta_rule_sequential(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    beta: torch.Tensor,
    decay: torch.Tensor,
) -> torch.Tensor:
    """The rule, stepped. Reference only -- see :func:`gated_delta_rule`.

    ``S_t = a_t (I - b_t k_t k_t^T) S_{t-1} + b_t k_t v_t^T``, ``o_t = S_t^T q_t``,
    with ``a_t = exp(decay_t)``. This is the definition the fast path is held to.
    """
    batch, heads, sequence, _ = query.shape
    state = torch.zeros(
        batch,
        heads,
        key.shape[-1],
        value.shape[-1],
        device=query.device,
        dtype=query.dtype,
    )
    outputs: list[torch.Tensor] = []
    for step in range(sequence):
        q_step = query[:, :, step]
        k_step = key[:, :, step]
        v_step = value[:, :, step]
        state = state * decay[:, :, step].exp()[..., None, None]
        memory = (state * k_step.unsqueeze(-1)).sum(dim=-2)
        delta = (v_step - memory) * beta[:, :, step].unsqueeze(-1)
        state = state + k_step.unsqueeze(-1) * delta.unsqueeze(-2)
        outputs.append((state * q_step.unsqueeze(-1)).sum(dim=-2))
    return torch.stack(outputs, dim=2)


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
    # [t, i] = log(A_t / A_i): <= 0 on and below the diagonal, and filled with a
    # value that exponentiates to a hard zero above it. Masking after exp would
    # first have to form exp(+large) = inf, and inf * 0 is a NaN.
    relative = log_decay.unsqueeze(-1) - log_decay.unsqueeze(-2)
    causal = torch.ones(
        sequence, sequence, dtype=torch.bool, device=query.device
    ).tril()
    decay_matrix = relative.masked_fill(~causal, -float("inf")).exp()

    # (I + diag(b) L) u = diag(b) v, with L strictly lower. solve_triangular is
    # told the diagonal is unit, so the identity is never materialized.
    interaction = (key @ key.transpose(-1, -2)) * decay_matrix
    system = beta.unsqueeze(-1) * interaction.tril(-1)
    updates = torch.linalg.solve_triangular(
        system,
        beta.unsqueeze(-1) * value,
        upper=False,
        unitriangular=True,
    )
    attention = ((query @ key.transpose(-1, -2)) * decay_matrix).tril()
    return attention @ updates
