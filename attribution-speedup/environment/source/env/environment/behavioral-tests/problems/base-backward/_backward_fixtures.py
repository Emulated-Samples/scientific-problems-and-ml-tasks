"""Deterministic fixtures for the base-backward adjoint anchors.

Builds seeded ``NormSnapshot`` / ``LinearizedAttention`` inputs and the
covectors ``d_out`` to feed the backward primitives. Same seeds in the
agent-backward probe and the pristine-forward-vjp probe, so the parent can
compare tensor-for-tensor. float64 throughout for tight tolerances.
"""

import torch

from lat.ir import LinearizedAttention, NormSnapshot

_DT = torch.float64
_EPS = 1e-6


def _rope_tables(n_pos, rotary_dim):
    half = rotary_dim // 2
    freq = 1.0 / (100.0 ** (torch.arange(half, dtype=_DT) / half))
    ang = torch.arange(n_pos, dtype=_DT)[:, None] * freq[None, :]
    ang = torch.cat([ang, ang], dim=-1)
    return ang.cos(), ang.sin()


def norm_snapshot(kind, n_pos=6, d=12, seed=0):
    """kind in {'ln', 'rms', 'radial'} (radial = jacobian-mode LN)."""
    g = torch.Generator().manual_seed(seed)
    scale = torch.rand(n_pos, generator=g, dtype=_DT) + 0.5
    gamma = torch.randn(d, generator=g, dtype=_DT) * 0.2 + 1.0
    mean = torch.randn(n_pos, generator=g, dtype=_DT) * 0.01 if kind != "rms" else None
    beta = torch.randn(d, generator=g, dtype=_DT) * 0.05 if kind != "rms" else None
    radial = None
    if kind == "radial":
        u = torch.randn(n_pos, d, generator=g, dtype=_DT)
        radial = torch.nn.functional.normalize(u, dim=-1) / (d ** 0.5)
    return NormSnapshot(scale=scale, gamma=gamma, mean=mean, beta=beta, radial=radial)


def attention(kind, n_pos=6, d=12, heads=3, dh=8, seed=0):
    """kind in {'mha', 'qk'} — 'qk' populates the jacobian-mode QK fields
    (per-head QK-RMSNorm + half-split RoPE + softmax formation term)."""
    g = torch.Generator().manual_seed(seed)
    pat = torch.tril(torch.ones(n_pos, n_pos, dtype=_DT))
    pat = (pat / pat.sum(-1, keepdim=True)).unsqueeze(0).repeat(heads, 1, 1)
    base = dict(
        pattern=pat * 0.4,
        W_V=torch.randn(heads, d, dh, generator=g, dtype=_DT) * 0.25,
        W_O=torch.randn(heads, dh, d, generator=g, dtype=_DT) * 0.25,
        b_V=torch.randn(heads, dh, generator=g, dtype=_DT) * 0.05,
        b_O=torch.randn(d, generator=g, dtype=_DT) * 0.05,
        pre_norm=norm_snapshot("ln", n_pos, d, seed + 1),
        is_causal=True,
    )
    if kind == "mha":
        return LinearizedAttention(**base)
    cos, sin = _rope_tables(n_pos, dh)
    base.update(
        q0=torch.randn(n_pos, heads, dh, generator=g, dtype=_DT) * 0.3,
        k0=torch.randn(n_pos, heads, dh, generator=g, dtype=_DT) * 0.3,
        v0=torch.randn(n_pos, heads, dh, generator=g, dtype=_DT) * 0.3,
        W_Q=torch.randn(heads, d, dh, generator=g, dtype=_DT) * 0.3,
        W_K=torch.randn(heads, d, dh, generator=g, dtype=_DT) * 0.3,
        q_pre0=torch.randn(n_pos, heads, dh, generator=g, dtype=_DT) * 0.4,
        k_pre0=torch.randn(n_pos, heads, dh, generator=g, dtype=_DT) * 0.4,
        q_norm_gamma=torch.rand(dh, generator=g, dtype=_DT) + 0.5,
        k_norm_gamma=torch.rand(dh, generator=g, dtype=_DT) + 0.5,
        qk_norm_eps=_EPS,
        rope_cos=cos,
        rope_sin=sin,
        attn_scale=float(dh) ** 0.5,
    )
    return LinearizedAttention(**base)


def d_out(n_pos=6, d=12, seed=99, batch=None):
    g = torch.Generator().manual_seed(seed)
    shape = (n_pos, d) if batch is None else (batch, n_pos, d)
    return torch.randn(*shape, generator=g, dtype=_DT)


# The anchor set: (name, category, builder-kwargs). Category picks which
# primitive the probe/parent exercise.
NORM_CASES = {
    "norm_ln": dict(kind="ln"),
    "norm_rms": dict(kind="rms"),
    "norm_radial": dict(kind="radial"),
}
ATTN_CASES = {
    "attn_mha": dict(kind="mha"),
    "attn_qk": dict(kind="qk"),
}

# Deterministic per-anchor covector seed (hash() is per-process randomized).
DOUT_SEED = {
    "norm_ln": 1101, "norm_rms": 1102, "norm_radial": 1103,
    "attn_mha": 2201, "attn_qk": 2202,
}
