"""Correctness for the LN-Jacobian norm mode (``norm_mode="jacobian"``).

Gates:

- The primitive with a ``radial`` snapshot must equal the AUTOGRAD Jacobian-
  vector product of the true norm function at the captured point, for both
  RMSNorm and LayerNorm, including the ``eps`` inside the scale. This is the
  whole point of the mode: the linear part IS the local Jacobian.
- ``backward_norm`` must be the exact adjoint of ``apply_norm_linear``
  (the radial projection is symmetric, so the transpose duality that the
  forward/backward engines rely on survives).
- ``_add_inject_rows`` (the inlined per-row adjoint used for seeding) must
  match the dense ``backward_norm`` path row for row.
- ``linearize(norm_mode="jacobian")`` must capture the radial that reproduces
  the true model's norm JVP; ``"frozen"`` (default) must capture no radial,
  keeping every existing behavior bit-identical.
"""

import math

import pytest
import torch

from lat.ir import NormSnapshot
from lat.linearize import linearize
from lat.primitives import _add_inject_rows, apply_norm_linear, backward_norm

_N_POS, _D = 5, 16
_EPS = 1e-5


def _true_norm_fn(gamma: torch.Tensor, *, rms: bool):
    def fn(x: torch.Tensor) -> torch.Tensor:
        centered = x if rms else x - x.mean(dim=-1, keepdim=True)
        scale = (centered.pow(2).mean(dim=-1, keepdim=True) + _EPS).sqrt()
        return gamma * centered / scale

    return fn


def _snapshot_at(x0: torch.Tensor, gamma: torch.Tensor, *, rms: bool) -> NormSnapshot:
    """Build the jacobian-mode snapshot exactly as ``_snapshot_norm`` does."""
    centered = x0 if rms else x0 - x0.mean(dim=-1, keepdim=True)
    scale = (centered.pow(2).mean(dim=-1) + _EPS).sqrt()
    return NormSnapshot(
        scale=scale,
        gamma=gamma,
        mean=None if rms else x0.mean(dim=-1),
        beta=None,
        radial=centered / (scale.unsqueeze(-1) * math.sqrt(x0.shape[-1])),
    )


@pytest.mark.parametrize("rms", [True, False], ids=["rmsnorm", "layernorm"])
def test_jacobian_mode_equals_autograd_jvp(rms):
    torch.manual_seed(0)
    x0 = torch.randn(_N_POS, _D, dtype=torch.float64)
    gamma = torch.rand(_D, dtype=torch.float64) + 0.5
    tangent = torch.randn(_N_POS, _D, dtype=torch.float64)

    snapshot = _snapshot_at(x0, gamma, rms=rms)
    got = apply_norm_linear(snapshot, tangent)

    _, expected = torch.autograd.functional.jvp(
        _true_norm_fn(gamma, rms=rms), x0, tangent, strict=True
    )
    torch.testing.assert_close(got, expected, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("rms", [True, False], ids=["rmsnorm", "layernorm"])
def test_backward_is_exact_adjoint(rms):
    torch.manual_seed(1)
    x0 = torch.randn(_N_POS, _D, dtype=torch.float64)
    gamma = torch.rand(_D, dtype=torch.float64) + 0.5
    snapshot = _snapshot_at(x0, gamma, rms=rms)

    x = torch.randn(_N_POS, _D, dtype=torch.float64)
    y = torch.randn(_N_POS, _D, dtype=torch.float64)
    lhs = (apply_norm_linear(snapshot, x) * y).sum()
    rhs = (x * backward_norm(snapshot, y)).sum()
    torch.testing.assert_close(lhs, rhs, atol=1e-10, rtol=1e-10)


def test_inject_rows_match_dense_backward():
    torch.manual_seed(2)
    x0 = torch.randn(_N_POS, _D, dtype=torch.float64)
    gamma = torch.rand(_D, dtype=torch.float64) + 0.5
    snapshot = _snapshot_at(x0, gamma, rms=False)

    batch = 3
    positions = torch.tensor([0, 2, 4])
    vectors = torch.randn(batch, _D, dtype=torch.float64)

    got = torch.zeros(batch, _N_POS, _D, dtype=torch.float64)
    _add_inject_rows(
        got,
        snapshot,
        mask=torch.ones(batch, dtype=torch.bool),
        positions=positions,
        vectors=vectors,
    )

    dense_seed = torch.zeros(batch, _N_POS, _D, dtype=torch.float64)
    for b in range(batch):
        dense_seed[b, positions[b]] = vectors[b]
    expected = backward_norm(snapshot, dense_seed)
    torch.testing.assert_close(got, expected, atol=1e-12, rtol=1e-12)


@pytest.fixture(scope="module")
def irs(faithful_attribution_model, sample_prompt):
    frozen = linearize(faithful_attribution_model, sample_prompt, show_progress=False)
    jacobian = linearize(
        faithful_attribution_model, sample_prompt, show_progress=False, norm_mode="jacobian"
    )
    return frozen, jacobian


def test_frozen_mode_captures_no_radial(irs):
    frozen, _ = irs
    assert frozen.final_norm.radial is None
    for layer in frozen.layers:
        assert layer.attention.pre_norm.radial is None
        assert layer.mlp.pre_norm.radial is None


def test_jacobian_capture_matches_true_model_jvp(irs, faithful_attribution_model):
    """The captured radial must make ``apply_norm_linear`` the JVP of the
    model's OWN norm at the captured input — reconstructed from the snapshot
    (``x0_centered = u * sqrt(d) * scale``), so this pins capture + primitive
    together against autograd."""
    _, jacobian = irs
    eps = float(faithful_attribution_model.transformer.cfg.eps)
    rms = jacobian.final_norm.mean is None

    for snapshot in (
        jacobian.layers[0].attention.pre_norm,
        jacobian.layers[-1].mlp.pre_norm,
        jacobian.final_norm,
    ):
        assert snapshot.radial is not None
        d_model = snapshot.radial.shape[-1]
        centered = (
            snapshot.radial.double() * math.sqrt(d_model) * snapshot.scale.double().unsqueeze(-1)
        )
        x0 = centered if rms else centered + snapshot.mean.double().unsqueeze(-1)
        gamma = snapshot.gamma.double()

        def fn(x: torch.Tensor, *, _gamma: torch.Tensor = gamma) -> torch.Tensor:
            c = x if rms else x - x.mean(dim=-1, keepdim=True)
            scale = (c.pow(2).mean(dim=-1, keepdim=True) + eps).sqrt()
            return _gamma * c / scale

        # The reconstructed x0 must reproduce the captured scale (sanity that
        # the reconstruction is self-consistent before differentiating there).
        recon_scale = (
            ((x0 if rms else x0 - x0.mean(dim=-1, keepdim=True)).pow(2).mean(dim=-1) + eps)
            .sqrt()
        )
        torch.testing.assert_close(
            recon_scale, snapshot.scale.double(), atol=1e-4, rtol=1e-4
        )

        tangent = torch.randn_like(x0)
        _, expected = torch.autograd.functional.jvp(fn, x0, tangent, strict=True)
        got = apply_norm_linear(snapshot, tangent.to(snapshot.scale.dtype)).double()
        torch.testing.assert_close(got, expected, atol=1e-4, rtol=1e-3)
