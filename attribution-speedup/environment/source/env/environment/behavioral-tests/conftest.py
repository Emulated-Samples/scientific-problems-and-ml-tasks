"""Shared pytest fixtures for lat tests.

The ``tiny_attribution_model`` fixture is session-scoped so the (slow) gelu-2l
download from HuggingFace amortizes across tests in one run.

``circuit_tracer`` is imported lazily inside the fixture body — tests that
don't request it (e.g. ``test_ir.py``) run cleanly even when it's not
installed. Tests that do request it are skipped if it's missing.
"""

import sys
from pathlib import Path

# Tests live in audience subdirectories (core/, gold/, base/, problems/<id>/)
# and import shared helpers from _lib/; make the suite root importable no
# matter which subdirectory a module is loaded from.
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Imported FIRST — before pytest imports any test module, which is before
# anything can import the agent's lat: pops the sensitive grading env vars
# (metrics sidecar path, pristine-baseline path) out of os.environ into
# module state, so code imported later in this process and every
# test-spawned subprocess can never discover them. See _lib/grade_env.py.
from _lib import grade_env as _grade_env  # noqa: E402,F401

# Offline-verifier support: when the verifier runs with HF_HUB_OFFLINE=1 (see
# task.toml [verifier.env]) against the image's baked HF cache, teach
# TransformerLens's NeelNanda loader to answer its list_repo_files() call from
# the local snapshot instead of the live hub. Inert when not offline.
from _lib import hf_offline as _hf_offline  # noqa: E402,F401

import pytest
import torch
import torch.nn.functional as F

# gelu-2l: 2 layers, d_model=512, d_mlp=2048. Standard TransformerLens alias
# for NeelNanda/gelu-2l. Small enough to run on CPU in seconds, big enough to
# exercise multi-layer attention + MLP substitution.
_TINY_MODEL_NAME = "gelu-2l"
_TINY_D_MODEL = 512
_TINY_N_LAYERS = 2
_TINY_D_TRANSCODER = 64
_FIXTURE_SEED = 0


@pytest.fixture(scope="session")
def tiny_attribution_model():
    """Tiny TL model + random transcoders, bundled as an AttributionModel.

    Random transcoder init means the resulting attribution is content-noisy
    but exercises the same algorithmic path as a real attribution — we care
    that two code paths produce the same numbers on the same inputs, not
    that the numbers mean anything.
    """
    pytest.importorskip("circuit_tracer")

    from circuit_tracer.replacement_model.replacement_model_transformerlens import (
        TransformerLensReplacementModel,
    )
    from circuit_tracer.transcoder.single_layer_transcoder import (
        SingleLayerTranscoder,
        TranscoderSet,
    )

    from lat.linearize import AttributionModel

    torch.manual_seed(_FIXTURE_SEED)

    transcoders: dict[int, SingleLayerTranscoder] = {}
    for layer in range(_TINY_N_LAYERS):
        t = SingleLayerTranscoder(
            d_model=_TINY_D_MODEL,
            d_transcoder=_TINY_D_TRANSCODER,
            activation_function=F.relu,
            layer_idx=layer,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        # Small init so ReLU fires on a reasonable fraction of features.
        with torch.no_grad():
            t.W_enc.copy_(torch.randn(_TINY_D_TRANSCODER, _TINY_D_MODEL) * (_TINY_D_MODEL**-0.5))
            t.W_dec.copy_(
                torch.randn(_TINY_D_TRANSCODER, _TINY_D_MODEL) * (_TINY_D_TRANSCODER**-0.5)
            )
        transcoders[layer] = t

    transcoder_set = TranscoderSet(
        transcoders,
        feature_input_hook="mlp.hook_in",
        feature_output_hook="mlp.hook_out",
        scan_name="lat-test-tiny",
    )

    # Force CPU placement: TL's auto-device picks MPS on macOS, and the
    # transcoder's `.to_sparse()` path isn't implemented for MPS. CPU is plenty
    # fast for gelu-2l and keeps the fixture portable.
    rm = TransformerLensReplacementModel.from_pretrained_and_transcoders(
        _TINY_MODEL_NAME, transcoder_set, device="cpu"
    )
    return AttributionModel.from_circuit_tracer(rm)


@pytest.fixture(scope="session")
def faithful_attribution_model(tiny_attribution_model):
    """Same fixture as ``tiny_attribution_model`` but with ``zero_positions=None``.

    The bundled fixture inherits ``zero_positions`` from circuit_tracer's
    ReplacementModel (which defaults to filtering BOS); for parity tests like
    ``forward(ir) ≈ model.forward`` we want no filtering so the IR is
    *faithful* to model behavior across every position.
    """
    from dataclasses import replace

    return replace(tiny_attribution_model, zero_positions=None)


@pytest.fixture(scope="session")
def sample_prompt() -> str:
    return "The capital of France is"
