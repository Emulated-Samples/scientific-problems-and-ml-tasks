"""Ground truth for MLP-substitution faithfulness on skip-connection transcoders.

The faithfulness contract (``forward(ir)`` reproduces the model on the
snapshot prompt — the error term absorbs whatever the transcoder doesn't
reconstruct) must hold for EVERY transcoder satisfying the Protocol,
including ones carrying an affine skip path (``W_skip``/``compute_skip``,
the circuit-tracer skip-transcoder convention). The plain-transcoder
fixtures elsewhere in the suites cannot see a skip-path defect: this file
is the only place the contract is exercised with a skip term present.

Public-API only (linearize → forward), so any internally-restructured but
behaviorally-correct handling of the skip term passes — absorbing it into
``error_vec`` and carrying it as an explicit IR field are both valid.

Gated on LAT_BENCH_BASE_FAMILY like the rest of base/.
"""

import os

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(
    not os.environ.get("LAT_BENCH_BASE_FAMILY"),
    reason="base-family skip-transcoder ground truth (LAT_BENCH_BASE_FAMILY not set)",
)

_D_TRANSCODER = 64
_SEED = 11


@pytest.fixture(scope="module")
def skip_attribution_model():
    """gelu-2l + random transcoders WITH an affine skip on every layer.

    Same construction as the shared ``tiny_attribution_model`` fixture, but
    ``skip_connection=True`` and a non-trivial ``W_skip`` — zero-initialized
    skips would make the skip path invisible and the test vacuous.
    ``zero_positions=None`` so the faithfulness contract is asserted at
    every position (same reasoning as ``faithful_attribution_model``).
    """
    pytest.importorskip("circuit_tracer")

    from dataclasses import replace

    from circuit_tracer.replacement_model.replacement_model_transformerlens import (
        TransformerLensReplacementModel,
    )
    from circuit_tracer.transcoder.single_layer_transcoder import (
        SingleLayerTranscoder,
        TranscoderSet,
    )

    from lat.linearize import AttributionModel

    torch.manual_seed(_SEED)

    transcoders: dict[int, SingleLayerTranscoder] = {}
    for layer in range(2):
        t = SingleLayerTranscoder(
            d_model=512,
            d_transcoder=_D_TRANSCODER,
            activation_function=F.relu,
            layer_idx=layer,
            device=torch.device("cpu"),
            dtype=torch.float32,
            skip_connection=True,
        )
        with torch.no_grad():
            t.W_enc.copy_(torch.randn(_D_TRANSCODER, 512) * (512**-0.5))
            t.W_dec.copy_(torch.randn(_D_TRANSCODER, 512) * (_D_TRANSCODER**-0.5))
            t.W_skip.copy_(torch.randn(512, 512) * (512**-0.5))
        transcoders[layer] = t

    transcoder_set = TranscoderSet(
        transcoders,
        feature_input_hook="mlp.hook_in",
        feature_output_hook="mlp.hook_out",
        scan_name="lat-test-skip",
    )
    rm = TransformerLensReplacementModel.from_pretrained_and_transcoders(
        "gelu-2l", transcoder_set, device="cpu"
    )
    return replace(AttributionModel.from_circuit_tracer(rm), zero_positions=None)


def test_forward_is_faithful_with_skip_transcoders(skip_attribution_model):
    """``forward(ir)`` must reproduce the model's logits when the
    transcoders carry an affine skip path. The decomposition contract says
    the error term absorbs everything the feature reconstruction doesn't
    carry — a linearization that folds the skip into the error computation
    without the forward re-adding it silently drops the skip contribution
    from every MLP output."""
    import lat
    from lat.linearize import linearize

    ir = linearize(
        skip_attribution_model, "The capital of France is", show_progress=False
    )
    with torch.no_grad():
        expected = skip_attribution_model.transformer(
            ir.input_tokens.unsqueeze(0)
        ).squeeze(0)

    deviation = (lat.forward(ir) - expected).abs().max().item()
    assert deviation < 2e-2, (
        f"forward(ir) does not reproduce the model on a skip-transcoder "
        f"configuration (max abs logit deviation {deviation:.4f}) — the MLP "
        f"substitution is dropping the transcoders' skip contribution"
    )
