"""Tests for the forward-sweep backend pointers and their reference impls.

Mirrors ``test_feature_dot_backend.py`` for the two FORWARD hot ops:

- ``_pattern_apply_forward_ref`` — the ``pattern @ v`` contraction in
  ``apply_attention_linear``, taking UNEXPANDED (KV-head-width) ``v`` and
  owning the GQA expansion.
- ``_feature_readout_apply_ref`` — the encoder-side grouped readout in
  ``forward_inject``, with suffix (``pos_start``) trimming.

CPU tests pin the reference math (including the adjoint identity against the
backward twin) and the ``set_backend`` pointer swap with CPU fallback. CUDA
parity for the kernels themselves runs under the ``cuda`` skipif on a GPU box
(the same box that runs ``test_oracle.py``'s qwen marker).
"""

import pytest
import torch

import lat
from lat import primitives
from lat.ir import LinearizedAttention, LinearizedMLP, NormSnapshot
from lat.primitives import (
    _feature_readout_apply_ref,
    _pattern_apply_forward_ref,
    _pattern_apply_ref,
    apply_attention_linear,
)


def _norm_snapshot(*, n_pos: int, d_model: int) -> NormSnapshot:
    return NormSnapshot(
        scale=torch.ones(n_pos),
        gamma=torch.ones(d_model),
        mean=None,
        beta=None,
    )


def _mlp_with_groups(*, n_pos: int = 6, d_model: int = 8) -> LinearizedMLP:
    positions = torch.tensor([0, 0, 2, 2, 2, 5], dtype=torch.long)
    position_groups = ((0, 0, 2), (2, 2, 3), (5, 5, 1))
    n_active = positions.numel()
    return LinearizedMLP(
        active_positions=positions,
        active_feature_idx=torch.arange(n_active, dtype=torch.long),
        active_activations=torch.ones(n_active),
        W_enc_active=torch.randn(n_active, d_model),
        W_dec_active=torch.zeros(n_active, d_model),
        b_enc_active=torch.zeros(n_active),
        b_dec=torch.zeros(d_model),
        d_transcoder=64,
        pre_norm=_norm_snapshot(n_pos=n_pos, d_model=d_model),
        error_vec=torch.zeros(n_pos, d_model),
        position_groups=position_groups,
        group_positions=torch.tensor([0, 2, 5], dtype=torch.long),
        group_starts=torch.tensor([0, 2, 5], dtype=torch.long),
        group_lengths=torch.tensor([2, 3, 1], dtype=torch.long),
    )


def _causal_pattern(n_heads: int, n_pos: int) -> torch.Tensor:
    pattern = torch.rand(n_heads, n_pos, n_pos)
    return pattern.tril()


@pytest.fixture(autouse=True)
def _reset_backend():
    """Each test starts and ends on the default ``"ref"`` backend."""
    lat.set_backend("ref")
    yield
    lat.set_backend("ref")


# ─── Reference impls ────────────────────────────────────────────────────────


def test_reference_forward_pattern_matches_expanded_einsum():
    torch.manual_seed(0)
    n_heads, n_kv_groups, n_pos, d_head = 4, 2, 6, 3
    pattern = _causal_pattern(n_heads, n_pos)
    v = torch.randn(2, n_pos, n_heads // n_kv_groups, d_head)

    got = _pattern_apply_forward_ref(pattern, v, is_causal=True, n_kv_groups=n_kv_groups)
    expanded = v.repeat_interleave(n_kv_groups, dim=-2)
    expected = torch.einsum("hqk, ...khe -> ...qhe", pattern, expanded)

    assert torch.allclose(got, expected, atol=1e-6)


def test_forward_pattern_is_adjoint_of_backward_pattern():
    """<u, P v> == <P^T u, v> — the forward contraction must stay the exact
    transpose of ``_pattern_apply_ref`` or the forward/backward edge duality
    breaks."""
    torch.manual_seed(1)
    n_heads, n_pos, d_head = 3, 5, 4
    pattern = _causal_pattern(n_heads, n_pos)
    v = torch.randn(2, n_pos, n_heads, d_head)
    u = torch.randn(2, n_pos, n_heads, d_head)

    forward = _pattern_apply_forward_ref(pattern, v, is_causal=True)
    backward = _pattern_apply_ref(pattern, u, is_causal=True)

    lhs = (u * forward).sum()
    rhs = (backward * v).sum()
    assert torch.allclose(lhs, rhs, atol=1e-5)


def test_reference_feature_readout_matches_manual_gather():
    torch.manual_seed(0)
    mlp = _mlp_with_groups()
    normed = torch.randn(4, 6, 8)
    out = torch.zeros(4, mlp.n_active)

    _feature_readout_apply_ref(normed, mlp, out, pos_start=0)
    expected = (normed[:, mlp.active_positions] * mlp.W_enc_active.unsqueeze(0)).sum(dim=-1)

    assert torch.allclose(out, expected, atol=1e-6)


def test_reference_feature_readout_suffix_skips_early_groups():
    """With ``pos_start=2`` the suffix tangent covers absolute positions
    [2, 6); the position-0 group is unreachable and its slots stay zero."""
    torch.manual_seed(0)
    mlp = _mlp_with_groups()
    pos_start = 2
    normed = torch.randn(4, 6 - pos_start, 8)
    out = torch.zeros(4, mlp.n_active)

    _feature_readout_apply_ref(normed, mlp, out, pos_start=pos_start)

    expected = torch.zeros_like(out)
    keep = mlp.active_positions >= pos_start
    local = mlp.active_positions[keep] - pos_start
    expected[:, keep] = (normed[:, local] * mlp.W_enc_active[keep]).sum(dim=-1)

    assert torch.allclose(out, expected, atol=1e-6)


# ─── set_backend switch ─────────────────────────────────────────────────────


def test_default_backend_binds_forward_refs():
    assert primitives.pattern_apply_forward is primitives._pattern_apply_forward_ref
    assert primitives.feature_readout_apply is primitives._feature_readout_apply_ref


def test_set_backend_triton_rebinds_forward_pointers_and_falls_back_on_cpu():
    pytest.importorskip("triton")

    torch.manual_seed(0)
    n_heads, n_pos, d_head = 4, 6, 3
    pattern = _causal_pattern(n_heads, n_pos)
    v = torch.randn(2, n_pos, n_heads, d_head)
    expected_z = _pattern_apply_forward_ref(pattern, v, is_causal=True)

    mlp = _mlp_with_groups()
    normed = torch.randn(3, 6, 8)
    expected_out = torch.zeros(3, mlp.n_active)
    _feature_readout_apply_ref(normed, mlp, expected_out, pos_start=0)

    lat.set_backend("triton")
    assert primitives.pattern_apply_forward.__module__ == "lat._triton"
    assert primitives.feature_readout_apply.__module__ == "lat._triton"

    got_z = primitives.pattern_apply_forward(pattern, v, is_causal=True)
    assert torch.equal(got_z, expected_z)

    got_out = torch.zeros(3, mlp.n_active)
    primitives.feature_readout_apply(normed, mlp, got_out, pos_start=0)
    assert torch.equal(got_out, expected_out)


def test_set_backend_ref_restores_forward_pointers():
    pytest.importorskip("triton")
    lat.set_backend("triton")
    lat.set_backend("ref")
    assert primitives.pattern_apply_forward is primitives._pattern_apply_forward_ref
    assert primitives.feature_readout_apply is primitives._feature_readout_apply_ref


# ─── CUDA kernel parity ─────────────────────────────────────────────────────


cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _assert_close_scaled(got: torch.Tensor, expected: torch.Tensor, tol: float) -> None:
    """Parity check with tolerance scaled to the output's magnitude.

    fp32 ``tl.dot`` runs TF32 on Ampere while torch matmul runs true fp32, so
    the kernel's error is ~1e-3 RELATIVE TO THE ACCUMULATED ROW MAGNITUDE and
    lands on every element, including near-zero ones — elementwise rtol can't
    express that. A real indexing bug produces O(row-magnitude) differences,
    which this still catches by orders of magnitude.
    """
    scale = float(expected.float().abs().max().clamp(min=1.0))
    diff = float((got.float() - expected.float()).abs().max())
    assert diff <= tol * scale, f"max diff {diff} > {tol} * scale {scale}"


@cuda
@pytest.mark.parametrize("n_kv_groups", [1, 4])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_triton_forward_pattern_matches_ref_cuda(n_kv_groups: int, dtype: torch.dtype):
    pytest.importorskip("triton")
    from lat._triton import _pattern_apply_forward_triton

    torch.manual_seed(0)
    n_heads, n_pos, d_head = 8, 97, 40  # deliberately not tile-aligned
    pattern = _causal_pattern(n_heads, n_pos).to("cuda", dtype)
    v = torch.randn(3, n_pos, n_heads // n_kv_groups, d_head, device="cuda", dtype=dtype)

    got = _pattern_apply_forward_triton(pattern, v, is_causal=True, n_kv_groups=n_kv_groups)
    expected = _pattern_apply_forward_ref(pattern, v, is_causal=True, n_kv_groups=n_kv_groups)

    tol = 5e-3 if dtype is torch.float32 else 2e-2
    _assert_close_scaled(got, expected, tol)


@cuda
@pytest.mark.parametrize("pos_start", [0, 3])
def test_triton_forward_pattern_on_sliced_pattern_cuda(pos_start: int):
    """apply_attention_linear feeds a diagonal SLICE of the pattern (a
    non-contiguous view); the kernel must handle its strides."""
    pytest.importorskip("triton")
    from lat._triton import _pattern_apply_forward_triton

    torch.manual_seed(1)
    n_heads, n_pos, d_head = 4, 33, 16
    full = _causal_pattern(n_heads, n_pos).to("cuda", torch.float32)
    n = n_pos - pos_start
    pattern = full[:, pos_start : pos_start + n, pos_start : pos_start + n]
    v = torch.randn(2, n, n_heads, d_head, device="cuda")

    got = _pattern_apply_forward_triton(pattern, v, is_causal=True)
    expected = _pattern_apply_forward_ref(pattern, v, is_causal=True)

    _assert_close_scaled(got, expected, 5e-3)


@cuda
@pytest.mark.parametrize("pos_start", [0, 2])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_triton_feature_readout_matches_ref_cuda(pos_start: int, dtype: torch.dtype):
    pytest.importorskip("triton")
    from lat._triton import _feature_readout_apply_triton

    torch.manual_seed(0)
    d_model = 64
    mlp = _mlp_with_groups(d_model=d_model)
    mlp = LinearizedMLP(
        active_positions=mlp.active_positions.cuda(),
        active_feature_idx=mlp.active_feature_idx.cuda(),
        active_activations=mlp.active_activations.to("cuda", dtype),
        W_enc_active=torch.randn(mlp.n_active, d_model, device="cuda", dtype=dtype),
        W_dec_active=mlp.W_dec_active.to("cuda", dtype),
        b_enc_active=mlp.b_enc_active.to("cuda", dtype),
        b_dec=mlp.b_dec.to("cuda", dtype),
        d_transcoder=mlp.d_transcoder,
        pre_norm=mlp.pre_norm,
        error_vec=mlp.error_vec.to("cuda", dtype),
        position_groups=mlp.position_groups,
        group_positions=mlp.group_positions.cuda(),
        group_starts=mlp.group_starts.cuda(),
        group_lengths=mlp.group_lengths.cuda(),
    )
    normed = torch.randn(5, 6 - pos_start, d_model, device="cuda", dtype=dtype)

    # Write into a NARROW view (the forward_inject call shape) so the
    # kernel's stride handling is what's under test.
    edges = torch.zeros(5, mlp.n_active + 7, device="cuda", dtype=dtype)
    got = edges[:, 3 : 3 + mlp.n_active]
    _feature_readout_apply_triton(normed, mlp, got, pos_start=pos_start)

    expected = torch.zeros(5, mlp.n_active, device="cuda", dtype=dtype)
    _feature_readout_apply_ref(normed, mlp, expected, pos_start=pos_start)

    tol = 5e-3 if dtype is torch.float32 else 2e-2
    _assert_close_scaled(got, expected, tol)


@cuda
@pytest.mark.parametrize("n_kv_groups", [1, 2])
def test_apply_attention_linear_triton_matches_ref_cuda(n_kv_groups: int):
    """The composite forward step (norm + V + pattern + O) through the triton
    backend must match the ref backend — the end-to-end gate for the pattern
    kernel inside its real call site, GQA included."""
    pytest.importorskip("triton")

    torch.manual_seed(0)
    n_heads, n_pos, d_head, d_model = 4, 21, 16, 32  # d_head >= 16: tl.dot tile minimum
    n_kv_heads = n_heads // n_kv_groups
    attn = LinearizedAttention(
        pattern=_causal_pattern(n_heads, n_pos).cuda(),
        W_V=torch.randn(n_kv_heads, d_model, d_head, device="cuda"),
        W_O=torch.randn(n_heads, d_head, d_model, device="cuda"),
        b_V=None,
        b_O=None,
        pre_norm=NormSnapshot(
            scale=torch.ones(n_pos, device="cuda"),
            gamma=torch.ones(d_model, device="cuda"),
            mean=None,
            beta=None,
        ),
        is_causal=True,
        n_kv_groups=n_kv_groups,
    )
    resid = torch.randn(3, n_pos, d_model, device="cuda")

    lat.set_backend("ref")
    expected = apply_attention_linear(attn, resid)
    lat.set_backend("triton")
    got = apply_attention_linear(attn, resid)

    _assert_close_scaled(got, expected, 5e-3)
