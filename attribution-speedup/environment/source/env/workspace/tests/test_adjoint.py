"""Every backward primitive is the exact adjoint of its forward's linear part.

The load-bearing identity is ``<L(x), y> == <x, Lᵀ(y)>`` for random ``x``,
``y``: it pins the backward against the forward with no reference
implementation in the loop. Norm and attention are checked separately, then
``backward_inject_batch``'s API contract (shapes, validation, batching
semantics, linearity) on top.
"""

import pytest
import torch

from lat.ir import LinearizedAttention, NormSnapshot
from lat.primitives import (
    apply_attention,
    apply_attention_linear,
    apply_norm,
    apply_norm_linear,
    backward_attention,
    backward_inject_batch,
    backward_norm,
)

_N_POS = 5
_D_MODEL = 8
_N_HEADS = 2
_D_HEAD = 4


def _gen(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def _norm_snapshot(kind: str, g: torch.Generator) -> NormSnapshot:
    layernorm = kind == "layernorm"
    return NormSnapshot(
        scale=torch.rand(_N_POS, generator=g) + 0.5,
        gamma=torch.randn(_D_MODEL, generator=g),
        mean=torch.randn(_N_POS, generator=g) if layernorm else None,
        beta=torch.randn(_D_MODEL, generator=g) if layernorm else None,
    )


def _attention(g: torch.Generator, *, with_bias: bool = True) -> LinearizedAttention:
    return LinearizedAttention(
        pattern=torch.randn(_N_HEADS, _N_POS, _N_POS, generator=g),
        W_V=torch.randn(_N_HEADS, _D_MODEL, _D_HEAD, generator=g),
        W_O=torch.randn(_N_HEADS, _D_HEAD, _D_MODEL, generator=g),
        b_V=torch.randn(_N_HEADS, _D_HEAD, generator=g) if with_bias else None,
        b_O=torch.randn(_D_MODEL, generator=g) if with_bias else None,
        pre_norm=_norm_snapshot("layernorm", g),
    )


# ─── Norm adjoints ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("kind", ["layernorm", "rmsnorm"])
def test_backward_norm_adjoint(kind):
    g = _gen(0)
    snapshot = _norm_snapshot(kind, g)
    x = torch.randn(_N_POS, _D_MODEL, generator=g, dtype=torch.float64)
    y = torch.randn(_N_POS, _D_MODEL, generator=g, dtype=torch.float64)
    snapshot64 = NormSnapshot(
        scale=snapshot.scale.double(),
        gamma=snapshot.gamma.double(),
        mean=None if snapshot.mean is None else snapshot.mean.double(),
        beta=None if snapshot.beta is None else snapshot.beta.double(),
    )
    lhs = (apply_norm_linear(snapshot64, x) * y).sum()
    rhs = (x * backward_norm(snapshot64, y)).sum()
    assert torch.allclose(lhs, rhs, atol=1e-12)


@pytest.mark.parametrize("kind", ["layernorm", "rmsnorm"])
def test_apply_norm_linear_is_backward_norm_transpose(kind):
    """Materialize both operators as matrices; one must be the other's T."""
    g = _gen(1)
    snapshot = _norm_snapshot(kind, g)
    n = _N_POS * _D_MODEL
    eye = torch.eye(n, dtype=torch.float64)
    snapshot64 = NormSnapshot(
        scale=snapshot.scale.double(),
        gamma=snapshot.gamma.double(),
        mean=None if snapshot.mean is None else snapshot.mean.double(),
        beta=None if snapshot.beta is None else snapshot.beta.double(),
    )
    fwd_mat = torch.stack(
        [apply_norm_linear(snapshot64, e.reshape(_N_POS, _D_MODEL)).flatten() for e in eye]
    )
    bwd_mat = torch.stack(
        [backward_norm(snapshot64, e.reshape(_N_POS, _D_MODEL)).flatten() for e in eye]
    )
    assert torch.allclose(fwd_mat, bwd_mat.T, atol=1e-12)


def test_apply_norm_linear_equals_apply_norm_minus_beta():
    g = _gen(2)
    snapshot = _norm_snapshot("layernorm", g)
    x = torch.randn(_N_POS, _D_MODEL, generator=g)
    expected = apply_norm(snapshot, x) - snapshot.beta
    assert torch.allclose(apply_norm_linear(snapshot, x), expected, atol=1e-6)


def test_backward_norm_known_value_rmsnorm():
    """RMS adjoint is a plain diagonal: gamma multiply then scale divide."""
    snapshot = NormSnapshot(
        scale=torch.tensor([2.0, 4.0]),
        gamma=torch.full((_D_MODEL,), 3.0),
        mean=None,
        beta=None,
    )
    d_out = torch.ones(2, _D_MODEL)
    out = backward_norm(snapshot, d_out)
    assert torch.allclose(out[0], torch.full((_D_MODEL,), 1.5))
    assert torch.allclose(out[1], torch.full((_D_MODEL,), 0.75))


def test_backward_norm_layernorm_subtracts_grad_mean():
    """The LN adjoint's output rows are mean-zero (adjoint of centering)."""
    g = _gen(3)
    snapshot = _norm_snapshot("layernorm", g)
    d_out = torch.randn(_N_POS, _D_MODEL, generator=g)
    out = backward_norm(snapshot, d_out)
    assert torch.allclose(out.mean(dim=-1), torch.zeros(_N_POS), atol=1e-6)


# ─── Attention adjoints ─────────────────────────────────────────────────────


def test_backward_attention_adjoint():
    g = _gen(4)
    attn = _attention(g)
    x = torch.randn(_N_POS, _D_MODEL, generator=g)
    y = torch.randn(_N_POS, _D_MODEL, generator=g)
    lhs = (apply_attention_linear(attn, x) * y).sum()
    rhs = (x * backward_attention(attn, y)).sum()
    assert torch.allclose(lhs, rhs, atol=1e-4)


def test_apply_attention_linear_is_backward_attention_transpose():
    g = _gen(5)
    attn = _attention(g)
    n = _N_POS * _D_MODEL
    eye = torch.eye(n)
    fwd_mat = torch.stack(
        [apply_attention_linear(attn, e.reshape(_N_POS, _D_MODEL)).flatten() for e in eye]
    )
    bwd_mat = torch.stack(
        [backward_attention(attn, e.reshape(_N_POS, _D_MODEL)).flatten() for e in eye]
    )
    assert torch.allclose(fwd_mat, bwd_mat.T, atol=1e-5)


def test_apply_attention_linear_equals_apply_attention_minus_bias():
    """The linear part is the affine map with its constant offset removed."""
    g = _gen(6)
    attn = _attention(g, with_bias=True)
    x = torch.randn(_N_POS, _D_MODEL, generator=g)
    offset = apply_attention(attn, torch.zeros(_N_POS, _D_MODEL))
    expected = apply_attention(attn, x) - offset
    assert torch.allclose(apply_attention_linear(attn, x), expected, atol=1e-5)


def test_backward_attention_zero_pattern_yields_zero():
    g = _gen(7)
    attn = _attention(g)
    attn = LinearizedAttention(
        pattern=torch.zeros_like(attn.pattern),
        W_V=attn.W_V,
        W_O=attn.W_O,
        b_V=attn.b_V,
        b_O=attn.b_O,
        pre_norm=attn.pre_norm,
    )
    d_out = torch.randn(_N_POS, _D_MODEL, generator=g)
    assert torch.equal(backward_attention(attn, d_out), torch.zeros(_N_POS, _D_MODEL))


def test_backward_attention_batched_matches_unbatched():
    g = _gen(8)
    attn = _attention(g)
    d_out = torch.randn(3, _N_POS, _D_MODEL, generator=g)
    batched = backward_attention(attn, d_out)
    for b in range(3):
        assert torch.allclose(batched[b], backward_attention(attn, d_out[b]), atol=1e-6)


# ─── backward_inject_batch: API contract ────────────────────────────────────


def _tiny_ir():
    from lat.ir import LinearizedAttribution, LinearizedLayer, LinearizedMLP

    g = _gen(9)
    n_active = 6
    positions = torch.tensor([0, 0, 1, 2, 3, 4], dtype=torch.long)
    layers = []
    for i in range(3):
        w = torch.randn(n_active, _D_MODEL, generator=g) * 0.4
        mlp = LinearizedMLP(
            active_positions=positions,
            active_feature_idx=torch.arange(n_active, dtype=torch.long),
            active_activations=torch.rand(n_active, generator=g) + 0.1,
            W_enc_active=w,
            W_dec_active=torch.randn(n_active, _D_MODEL, generator=g) * 0.4,
            b_enc_active=torch.zeros(n_active),
            b_dec=torch.zeros(_D_MODEL),
            d_transcoder=16,
            pre_norm=_norm_snapshot("layernorm", _gen(20 + i)),
            error_vec=torch.randn(_N_POS, _D_MODEL, generator=g) * 0.1,
        )
        layers.append(
            LinearizedLayer(layer_idx=i, attention=_attention(_gen(30 + i)), mlp=mlp)
        )

    return LinearizedAttribution(
        input_tokens=torch.arange(_N_POS, dtype=torch.long),
        resid_0=torch.randn(_N_POS, _D_MODEL, generator=g),
        embed_vecs=torch.randn(_N_POS, _D_MODEL, generator=g),
        layers=tuple(layers),
        final_norm=_norm_snapshot("layernorm", _gen(40)),
        W_unembed=torch.randn(_D_MODEL, 12, generator=g),
        b_unembed=None,
        logits=torch.randn(_N_POS, 12, generator=g),
    )


def test_backward_inject_batch_shape():
    ir = _tiny_ir()
    rows = backward_inject_batch(
        ir,
        inject_layers=torch.tensor([3, 1, 0]),
        inject_positions=torch.tensor([4, 2, 0]),
        inject_vectors=torch.randn(3, _D_MODEL),
    )
    assert rows.features.shape == (3, ir.total_active_features)
    assert rows.errors.shape == (3, ir.n_layers, _N_POS)
    assert rows.tokens.shape == (3, _N_POS)


def test_backward_inject_batch_rejects_mismatched_shapes():
    ir = _tiny_ir()
    with pytest.raises(ValueError, match="same shape"):
        backward_inject_batch(
            ir,
            inject_layers=torch.tensor([1, 2]),
            inject_positions=torch.tensor([0]),
            inject_vectors=torch.randn(2, _D_MODEL),
        )


def test_backward_inject_batch_rejects_wrong_vector_shape():
    ir = _tiny_ir()
    with pytest.raises(ValueError, match="inject_vectors"):
        backward_inject_batch(
            ir,
            inject_layers=torch.tensor([1]),
            inject_positions=torch.tensor([0]),
            inject_vectors=torch.randn(1, _D_MODEL + 1),
        )


def test_backward_inject_batch_rejects_non_1d_layers():
    ir = _tiny_ir()
    with pytest.raises(ValueError, match="1-D"):
        backward_inject_batch(
            ir,
            inject_layers=torch.zeros(1, 1, dtype=torch.long),
            inject_positions=torch.zeros(1, 1, dtype=torch.long),
            inject_vectors=torch.randn(1, _D_MODEL),
        )


def test_backward_inject_batch_batched_equals_serial():
    """A batch of B injections equals the B single-injection calls stacked."""
    ir = _tiny_ir()
    g = _gen(10)
    inject_layers = torch.tensor([3, 2, 1, 0])
    inject_positions = torch.tensor([4, 1, 3, 0])
    inject_vectors = torch.randn(4, _D_MODEL, generator=g)

    batched = backward_inject_batch(
        ir,
        inject_layers=inject_layers,
        inject_positions=inject_positions,
        inject_vectors=inject_vectors,
    )
    for b in range(4):
        single = backward_inject_batch(
            ir,
            inject_layers=inject_layers[b : b + 1],
            inject_positions=inject_positions[b : b + 1],
            inject_vectors=inject_vectors[b : b + 1],
        )
        assert torch.allclose(batched.features[b], single.features[0], atol=1e-5)
        assert torch.allclose(batched.errors[b], single.errors[0], atol=1e-5)
        assert torch.allclose(batched.tokens[b], single.tokens[0], atol=1e-5)


def test_backward_inject_batch_is_linear_in_inject_vector():
    ir = _tiny_ir()
    g = _gen(11)
    layers = torch.tensor([2])
    positions = torch.tensor([3])
    v1 = torch.randn(1, _D_MODEL, generator=g)
    v2 = torch.randn(1, _D_MODEL, generator=g)

    def run(v):
        return backward_inject_batch(
            ir, inject_layers=layers, inject_positions=positions, inject_vectors=v
        )

    combined = run(2.0 * v1 + 3.0 * v2)
    a, b = run(v1), run(v2)
    assert torch.allclose(combined.features, 2.0 * a.features + 3.0 * b.features, atol=1e-5)
    assert torch.allclose(combined.errors, 2.0 * a.errors + 3.0 * b.errors, atol=1e-5)
    assert torch.allclose(combined.tokens, 2.0 * a.tokens + 3.0 * b.tokens, atol=1e-5)


def test_own_layer_features_receive_no_contribution():
    """A feature injected at layer L never attributes to features AT layer L
    (records happen before the splice)."""
    ir = _tiny_ir()
    rows = backward_inject_batch(
        ir,
        inject_layers=torch.tensor([1]),
        inject_positions=torch.tensor([2]),
        inject_vectors=torch.randn(1, _D_MODEL, generator=_gen(12)),
    )
    n_per_layer = ir.layers[0].mlp.n_active
    own_layer_block = rows.features[0, n_per_layer : 2 * n_per_layer]
    assert torch.equal(own_layer_block, torch.zeros_like(own_layer_block))
