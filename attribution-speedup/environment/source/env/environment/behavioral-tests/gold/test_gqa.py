"""GQA-aware attention: grouped KV-head storage vs the expanded reference.

The IR stores ``W_V``/``b_V`` per KV head with ``n_kv_groups`` recording the
query-heads-per-KV-head fan-out (``repeat_interleave`` convention). These
tests pin the load-bearing claims:

- the grouped path is the *same operator* as running the expanded weights
  through the MHA path (forward and backward, exactly — GQA dedup is not an
  approximation);
- backward stays the adjoint of forward under grouping;
- ``_dedupe_kv_heads`` only collapses tensors that really are
  ``repeat_interleave`` expansions, and falls back to the expanded path
  otherwise.
"""

import pytest
import torch

from lat.ir import LinearizedAttention, NormSnapshot
from lat.linearize import _dedupe_kv_heads
from lat.primitives import apply_attention, backward_attention

_N_POS = 4
_D_MODEL = 8
_N_HEADS = 4
_N_KV_HEADS = 2
_GROUPS = _N_HEADS // _N_KV_HEADS
_D_HEAD = 4


def _seeded(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def _norm(g: torch.Generator) -> NormSnapshot:
    return NormSnapshot(
        scale=torch.rand(_N_POS, generator=g) + 0.5,
        gamma=torch.randn(_D_MODEL, generator=g),
        mean=torch.zeros(_N_POS),
        beta=torch.randn(_D_MODEL, generator=g),
    )


def _flat(W_V: torch.Tensor) -> torch.Tensor:
    return W_V.permute(0, 2, 1).reshape(-1, W_V.shape[1]).contiguous()


def _paired_attentions(seed: int) -> tuple[LinearizedAttention, LinearizedAttention]:
    """(expanded-MHA-path, grouped-GQA-path) views of the same operator."""
    g = _seeded(seed)
    W_V_kv = torch.randn(_N_KV_HEADS, _D_MODEL, _D_HEAD, generator=g)
    b_V_kv = torch.randn(_N_KV_HEADS, _D_HEAD, generator=g)
    common = dict(
        pattern=torch.randn(_N_HEADS, _N_POS, _N_POS, generator=g),
        W_O=torch.randn(_N_HEADS, _D_HEAD, _D_MODEL, generator=g),
        b_O=torch.randn(_D_MODEL, generator=g),
        pre_norm=_norm(g),
    )
    W_V_exp = torch.repeat_interleave(W_V_kv, _GROUPS, dim=0)
    b_V_exp = torch.repeat_interleave(b_V_kv, _GROUPS, dim=0)
    expanded = LinearizedAttention(
        W_V=W_V_exp, b_V=b_V_exp, W_V_flat=_flat(W_V_exp), n_kv_groups=1, **common
    )
    grouped = LinearizedAttention(
        W_V=W_V_kv, b_V=b_V_kv, W_V_flat=_flat(W_V_kv), n_kv_groups=_GROUPS, **common
    )
    return expanded, grouped


def test_grouped_forward_matches_expanded():
    expanded, grouped = _paired_attentions(seed=0)
    g = _seeded(1)
    x = torch.randn(_N_POS, _D_MODEL, generator=g)
    x_batched = torch.randn(3, _N_POS, _D_MODEL, generator=g)

    assert torch.allclose(apply_attention(grouped, x), apply_attention(expanded, x), atol=1e-5)
    assert torch.allclose(
        apply_attention(grouped, x_batched), apply_attention(expanded, x_batched), atol=1e-5
    )


def test_grouped_backward_matches_expanded():
    expanded, grouped = _paired_attentions(seed=2)
    g = _seeded(3)
    d_out = torch.randn(_N_POS, _D_MODEL, generator=g)
    d_batched = torch.randn(3, _N_POS, _D_MODEL, generator=g)

    assert torch.allclose(
        backward_attention(grouped, d_out), backward_attention(expanded, d_out), atol=1e-5
    )
    assert torch.allclose(
        backward_attention(grouped, d_batched), backward_attention(expanded, d_batched), atol=1e-5
    )


def test_grouped_backward_is_adjoint_of_grouped_forward():
    """``<u, L(v)> == <Lᵀ(u), v>`` with the bias stripped, on the GQA path."""
    _, grouped = _paired_attentions(seed=4)
    g = _seeded(5)
    v = torch.randn(_N_POS, _D_MODEL, generator=g)
    u = torch.randn(_N_POS, _D_MODEL, generator=g)

    Lv = apply_attention(grouped, v) - apply_attention(grouped, torch.zeros_like(v))
    Lt_u = backward_attention(grouped, u)

    assert abs((u * Lv).sum().item() - (Lt_u * v).sum().item()) < 1e-4


def test_grouped_einsum_fallback_matches_flat():
    """W_V_flat=None takes the einsum path; same operator under grouping."""
    from dataclasses import replace

    _, grouped = _paired_attentions(seed=6)
    no_flat = replace(grouped, W_V_flat=None)
    g = _seeded(7)
    x = torch.randn(_N_POS, _D_MODEL, generator=g)

    assert torch.allclose(apply_attention(grouped, x), apply_attention(no_flat, x), atol=1e-5)
    assert torch.allclose(backward_attention(grouped, x), backward_attention(no_flat, x), atol=1e-5)


# ─── _dedupe_kv_heads ───────────────────────────────────────────────────────


def test_dedupe_detects_repeat_interleave_expansion():
    g = _seeded(8)
    W_kv = torch.randn(_N_KV_HEADS, _D_MODEL, _D_HEAD, generator=g)
    b_kv = torch.randn(_N_KV_HEADS, _D_HEAD, generator=g)

    W, b, groups = _dedupe_kv_heads(
        torch.repeat_interleave(W_kv, _GROUPS, dim=0),
        torch.repeat_interleave(b_kv, _GROUPS, dim=0),
        _N_KV_HEADS,
    )
    assert groups == _GROUPS
    assert torch.equal(W, W_kv)
    assert b is not None
    assert torch.equal(b, b_kv)


def test_dedupe_rejects_non_repeated_weights():
    """Distinct per-head weights with a (wrong) KV-head hint stay expanded."""
    g = _seeded(9)
    W = torch.randn(_N_HEADS, _D_MODEL, _D_HEAD, generator=g)

    W_out, b_out, groups = _dedupe_kv_heads(W, None, _N_KV_HEADS)
    assert groups == 1
    assert W_out is W
    assert b_out is None


def test_dedupe_rejects_non_repeated_bias():
    """Repeated W_V but per-query-head bias must stay expanded as a unit."""
    g = _seeded(10)
    W_kv = torch.randn(_N_KV_HEADS, _D_MODEL, _D_HEAD, generator=g)
    W = torch.repeat_interleave(W_kv, _GROUPS, dim=0)
    b = torch.randn(_N_HEADS, _D_HEAD, generator=g)

    W_out, b_out, groups = _dedupe_kv_heads(W, b, _N_KV_HEADS)
    assert groups == 1
    assert W_out is W
    assert b_out is b


@pytest.mark.parametrize("n_kv_heads", [None, 0, _N_HEADS, 3])
def test_dedupe_is_identity_for_non_gqa_configs(n_kv_heads):
    """MHA (kv == heads), missing config, and non-divisible counts pass through."""
    g = _seeded(11)
    W = torch.randn(_N_HEADS, _D_MODEL, _D_HEAD, generator=g)

    W_out, _b, groups = _dedupe_kv_heads(W, None, n_kv_heads)
    assert groups == 1
    assert W_out is W


def test_ir_rejects_inconsistent_group_count():
    g = _seeded(12)
    with pytest.raises(ValueError, match="KV heads"):
        LinearizedAttention(
            pattern=torch.randn(_N_HEADS, _N_POS, _N_POS, generator=g),
            W_V=torch.randn(_N_KV_HEADS, _D_MODEL, _D_HEAD, generator=g),
            W_O=torch.randn(_N_HEADS, _D_HEAD, _D_MODEL, generator=g),
            b_V=None,
            b_O=None,
            pre_norm=_norm(g),
            n_kv_groups=3,  # 2 KV heads * 3 != 4 query heads
        )
