"""Tests for the per-feature decoder-dot reference impl and the backend switch.

Two things under test:

- The reference ``_feature_dot_apply_ref`` produces the right per-feature
  outputs across both full and causal-prefix evaluations.
- ``lat.set_backend("triton")`` rebinds ``primitives.feature_dot_apply`` to
  the triton wrapper, which is required to gracefully fall back to the
  reference on CPU inputs. Triton import is guarded; the test is skipped
  when triton isn't installed.

The triton kernel itself (compile + correctness on real CUDA shapes) is
exercised by ``test_oracle.py`` under the ``qwen`` marker on a CUDA box.
"""

import pytest
import torch

import lat
from lat import primitives
from lat.ir import LinearizedMLP, NormSnapshot
from lat.primitives import _feature_dot_apply_ref


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
        W_enc_active=torch.zeros(n_active, d_model),
        W_dec_active=torch.randn(n_active, d_model),
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


@pytest.fixture(autouse=True)
def _reset_backend():
    """Each test starts and ends on the default ``"ref"`` backend.

    The triton tests below flip to ``"triton"`` and the fixture restores
    ``"ref"`` afterwards so backend state never leaks between tests in the
    same process.
    """
    lat.set_backend("ref")
    yield
    lat.set_backend("ref")


# ─── Reference impl ─────────────────────────────────────────────────────────


def test_reference_feature_dot_matches_manual_gather():
    torch.manual_seed(0)
    mlp = _mlp_with_groups()
    grad = torch.randn(4, 6, 8)

    got = _feature_dot_apply_ref(grad, mlp)
    expected = (grad[:, mlp.active_positions] * mlp.W_dec_active.unsqueeze(0)).sum(dim=-1)

    assert torch.allclose(got, expected, atol=1e-6)


def test_reference_feature_dot_skips_future_position_groups():
    torch.manual_seed(0)
    mlp = _mlp_with_groups()
    grad = torch.randn(4, 3, 8)

    got = _feature_dot_apply_ref(grad, mlp)
    expected = torch.zeros_like(got)
    keep = mlp.active_positions < grad.shape[1]
    expected[:, keep] = (grad[:, mlp.active_positions[keep]] * mlp.W_dec_active[keep]).sum(dim=-1)

    assert torch.allclose(got, expected, atol=1e-6)


# ─── set_backend switch ─────────────────────────────────────────────────────


def test_default_backend_is_ref():
    """Right after ``import lat``, the pointers are the ref impls."""
    assert primitives.pattern_apply is primitives._pattern_apply_ref
    assert primitives.feature_dot_apply is primitives._feature_dot_apply_ref


def test_set_backend_invalid_raises():
    with pytest.raises(ValueError, match="must be one of"):
        lat.set_backend("nope")  # type: ignore[arg-type]


def test_set_backend_triton_rebinds_pointers_and_falls_back_on_cpu():
    """``set_backend("triton")`` swaps the pointers, but the triton wrappers
    transparently fall back to ref for non-CUDA inputs so calls keep working.
    """
    pytest.importorskip("triton")

    torch.manual_seed(0)
    mlp = _mlp_with_groups()
    grad = torch.randn(3, 6, 8)
    expected = _feature_dot_apply_ref(grad, mlp)

    lat.set_backend("triton")

    # Pointers are now the triton wrappers (different identity than the ref).
    assert primitives.feature_dot_apply is not primitives._feature_dot_apply_ref
    assert primitives.feature_dot_apply.__module__ == "lat._triton"

    # Calling on CPU should fall back to ref per-call and produce the same result.
    got = primitives.feature_dot_apply(grad, mlp)
    assert torch.equal(got, expected)


def test_set_backend_ref_after_triton_restores_ref():
    """Round-trip: triton → ref restores the original pointer identities."""
    pytest.importorskip("triton")

    lat.set_backend("triton")
    lat.set_backend("ref")

    assert primitives.pattern_apply is primitives._pattern_apply_ref
    assert primitives.feature_dot_apply is primitives._feature_dot_apply_ref
