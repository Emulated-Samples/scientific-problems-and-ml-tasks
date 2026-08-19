"""``norm_mode="jacobian"``: the norm's full Jacobian as a linear operator.

With ``radial`` captured, ``apply_norm_linear`` must equal the autograd JVP
of the TRUE norm function (dynamic scale included) at the captured point,
and ``backward_norm`` must be its exact adjoint. Frozen mode captures no
radial and keeps the frozen-scale convention.
"""

import pytest
import torch

import lat
from lat.linearize import linearize
from lat.ir import NormSnapshot
from lat.primitives import apply_norm_linear, backward_norm

pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")

_N_POS = 5
_D_MODEL = 16
_EPS = 1e-6


def _true_norm_fn(kind: str, gamma: torch.Tensor):
    """The real (nonlinear) norm: dynamic mean AND dynamic scale."""

    def layernorm(x: torch.Tensor) -> torch.Tensor:
        centered = x - x.mean(dim=-1, keepdim=True)
        scale = (centered.pow(2).mean(dim=-1, keepdim=True) + _EPS).sqrt()
        return centered / scale * gamma

    def rmsnorm(x: torch.Tensor) -> torch.Tensor:
        scale = (x.pow(2).mean(dim=-1, keepdim=True) + _EPS).sqrt()
        return x / scale * gamma

    return layernorm if kind == "layernorm" else rmsnorm


def _snapshot_at(kind: str, x0: torch.Tensor, gamma: torch.Tensor) -> NormSnapshot:
    """Capture scale/mean/radial at x0 — what linearize(jacobian) stores."""
    d = x0.shape[-1]
    if kind == "layernorm":
        mean = x0.mean(dim=-1)
        centered = x0 - mean.unsqueeze(-1)
    else:
        mean = None
        centered = x0
    scale = (centered.pow(2).mean(dim=-1) + _EPS).sqrt()
    radial = centered / (scale.unsqueeze(-1) * d**0.5)
    return NormSnapshot(
        scale=scale,
        gamma=gamma,
        mean=mean,
        beta=None,
        radial=radial,
    )


@pytest.mark.parametrize("kind", ["layernorm", "rmsnorm"])
def test_jacobian_mode_equals_autograd_jvp(kind):
    torch.manual_seed(0)
    gamma = torch.rand(_D_MODEL, dtype=torch.float64) + 0.5
    x0 = torch.randn(_N_POS, _D_MODEL, dtype=torch.float64)
    tangent = torch.randn(_N_POS, _D_MODEL, dtype=torch.float64)

    fn = _true_norm_fn(kind, gamma)
    snapshot = _snapshot_at(kind, x0, gamma)

    _, jvp = torch.func.jvp(fn, (x0,), (tangent,))
    ours = apply_norm_linear(snapshot, tangent)
    assert torch.allclose(ours, jvp, atol=1e-10), (ours - jvp).abs().max()


@pytest.mark.parametrize("kind", ["layernorm", "rmsnorm"])
def test_backward_is_exact_adjoint(kind):
    torch.manual_seed(1)
    gamma = torch.rand(_D_MODEL, dtype=torch.float64) + 0.5
    x0 = torch.randn(_N_POS, _D_MODEL, dtype=torch.float64)
    snapshot = _snapshot_at(kind, x0, gamma)

    x = torch.randn(_N_POS, _D_MODEL, dtype=torch.float64)
    y = torch.randn(_N_POS, _D_MODEL, dtype=torch.float64)
    lhs = (apply_norm_linear(snapshot, x) * y).sum()
    rhs = (x * backward_norm(snapshot, y)).sum()
    assert torch.allclose(lhs, rhs, atol=1e-12)


def test_frozen_mode_captures_no_radial(tiny_attribution_model, sample_prompt):
    ir = linearize(tiny_attribution_model, sample_prompt, show_progress=False)
    for layer in ir.layers:
        assert layer.attention.pre_norm.radial is None
        assert layer.mlp.pre_norm.radial is None
    assert ir.final_norm.radial is None


def test_jacobian_mode_captures_radial_everywhere(tiny_attribution_model, sample_prompt):
    ir = linearize(
        tiny_attribution_model, sample_prompt, show_progress=False, norm_mode="jacobian"
    )
    n_pos, d_model = ir.n_pos, ir.d_model
    for layer in ir.layers:
        for snapshot in (layer.attention.pre_norm, layer.mlp.pre_norm):
            assert snapshot.radial is not None
            assert snapshot.radial.shape == (n_pos, d_model)
            # ||u|| < 1, approaching 1 as eps becomes negligible.
            norms = snapshot.radial.norm(dim=-1)
            assert bool((norms < 1.0 + 1e-4).all())
    assert ir.final_norm.radial is not None


def test_jacobian_capture_matches_true_model_jvp(tiny_attribution_model, sample_prompt):
    """The captured radial reproduces the model's own LN Jacobian: autograd
    JVP through the true (fully dynamic) LayerNorm built from the model's
    parameters at the captured input equals apply_norm_linear on the
    jacobian-mode snapshot. (The replacement model's ln1 module itself is
    not usable as the reference: its scale is detached for attribution, so
    differentiating through it gives the frozen-scale operator.)"""
    ir = linearize(
        tiny_attribution_model, sample_prompt, show_progress=False, norm_mode="jacobian"
    )
    transformer = tiny_attribution_model.transformer
    with torch.no_grad():
        _, cache = transformer.run_with_cache(
            ir.input_tokens.unsqueeze(0),
            names_filter=lambda n: n == "blocks.0.hook_resid_pre",
            return_cache_object=False,
        )
    x0 = cache["blocks.0.hook_resid_pre"].squeeze(0)
    ln1 = transformer.blocks[0].ln1
    tangent = torch.randn_like(x0) * 0.1

    def true_ln(x):
        centered = x - x.mean(dim=-1, keepdim=True)
        scale = (centered.pow(2).mean(dim=-1, keepdim=True) + ln1.eps).sqrt()
        return centered / scale * ln1.w + ln1.b

    _, jvp = torch.func.jvp(true_ln, (x0,), (tangent,))
    ours = apply_norm_linear(ir.layers[0].attention.pre_norm, tangent)
    assert torch.allclose(ours, jvp, atol=1e-4), (ours - jvp).abs().max()


def test_frozen_mode_matches_replacement_model_gradient_convention(
    tiny_attribution_model, sample_prompt
):
    """The frozen-scale operator IS the convention the replacement model's
    (scale-detached) LayerNorm exposes to autograd — the compatibility
    contract with circuit_tracer's attribution."""
    ir = linearize(tiny_attribution_model, sample_prompt, show_progress=False)
    transformer = tiny_attribution_model.transformer
    with torch.no_grad():
        _, cache = transformer.run_with_cache(
            ir.input_tokens.unsqueeze(0),
            names_filter=lambda n: n == "blocks.0.hook_resid_pre",
            return_cache_object=False,
        )
    x0 = cache["blocks.0.hook_resid_pre"].squeeze(0)
    ln1 = transformer.blocks[0].ln1
    tangent = torch.randn_like(x0) * 0.1

    def fn(x):
        return ln1(x.unsqueeze(0)).squeeze(0)

    _, jvp = torch.func.jvp(fn, (x0,), (tangent,))
    ours = apply_norm_linear(ir.layers[0].attention.pre_norm, tangent)
    assert torch.allclose(ours, jvp, atol=1e-4), (ours - jvp).abs().max()
