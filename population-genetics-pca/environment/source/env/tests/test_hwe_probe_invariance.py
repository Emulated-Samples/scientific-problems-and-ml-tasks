from pathlib import Path

import numpy as np
import pytest

import grader.gates.probes as probes


REPRESENTATION_KEY = bytes(range(32))


def _keep_canonical_representation(monkeypatch):
    monkeypatch.setattr(
        probes,
        "_keyed_representation",
        lambda G, representation_key, domain: (
            G,
            np.arange(G.shape[1]),
            np.random.default_rng(0),
        ),
    )


@pytest.mark.parametrize("k", [3, 10, 36])
def test_hwe_probe_separates_top_k_subspaces_without_score_scaling(k):
    carrier = probes.ProbeCarrier(384, 20_000, k, "synthetic_plain")
    G, common, rare = probes._build_hwe_probe(np.random.default_rng(17 + k), carrier)
    patterson = probes._patterson_pca(G, k)[0]
    raw = probes._rawcov_scores(G, k)

    def preference(scores):
        return probes._axis_capture(scores, rare) - probes._axis_capture(scores, common)

    assert preference(patterson) > 0.85
    assert preference(raw) < -0.85

    centred = patterson - patterson.mean(axis=0, keepdims=True)
    orthonormal, _ = np.linalg.qr(centred, mode="reduced")
    mixed = orthonormal @ np.diag(np.geomspace(0.1, 10.0, k))
    assert preference(mixed) == pytest.approx(preference(patterson), abs=1e-10)


def test_hwe_gate_accepts_an_exact_orthonormal_basis(monkeypatch, tmp_path: Path):
    k = 10
    carrier = probes.ProbeCarrier(384, 20_000, k, "synthetic_plain")
    G, common, rare = probes._build_hwe_probe(np.random.default_rng(31), carrier)
    monkeypatch.setattr(
        probes, "_build_hwe_probe", lambda rng, requested: (G, common, rare))
    _keep_canonical_representation(monkeypatch)

    def exact_basis(vcf, requested_k, out, sample_ids):
        scores = probes._patterson_pca(G, requested_k)[0]
        scores, _ = np.linalg.qr(scores - scores.mean(axis=0, keepdims=True), mode="reduced")
        return True, scores

    severity, *_ = probes._hwe_severity_once(
        exact_basis, tmp_path, carrier, seed=9, representation_key=REPRESENTATION_KEY,
    )
    assert severity == 0.0


def test_hwe_gate_penalizes_raw_covariance_subspace(monkeypatch, tmp_path: Path):
    k = 10
    carrier = probes.ProbeCarrier(384, 20_000, k, "synthetic_plain")
    G, common, rare = probes._build_hwe_probe(np.random.default_rng(37), carrier)
    monkeypatch.setattr(
        probes, "_build_hwe_probe", lambda rng, requested: (G, common, rare))
    _keep_canonical_representation(monkeypatch)

    def raw_basis(vcf, requested_k, out, sample_ids):
        return True, probes._rawcov_scores(G, requested_k)

    severity, *_ = probes._hwe_severity_once(
        raw_basis, tmp_path, carrier, seed=11, representation_key=REPRESENTATION_KEY,
    )
    assert severity > 0.99
