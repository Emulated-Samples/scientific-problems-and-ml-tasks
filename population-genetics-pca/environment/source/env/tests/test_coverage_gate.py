"""Regression: the coverage gate must survive a degenerate probe draw.

At the probe's low Fst (~0.0022) the full-scan reference occasionally resolves *no* structured
axis (its leading eigenvalue lands on the Marchenko-Pastur edge, ~1% of draws). Before the fix
that made ``subspace_accuracy`` raise "no structured PCs" and crash the whole gate (and
``scripts/validate.py`` with it). Now such a rep is skipped; if every rep degenerates the gate
fails open rather than penalising a genuine submission over pure noise directions.
"""
import tempfile
from pathlib import Path

import numpy as np

import grader.gates.probes as probes


def _never_called_run_one(vcf, k, out, ids):  # pragma: no cover - must not be reached
    raise AssertionError("run_one should not run when the probe draw is degenerate")


def test_all_degenerate_draws_fail_open(monkeypatch):
    # The per-repetition helper owns probe construction and decides whether a draw resolves any
    # structured axes. Stub that boundary directly: this test exercises the aggregate fail-open
    # contract without materialising three scored-scale probe matrices merely to discard them.
    monkeypatch.setattr(probes, "_coverage_severity_once", lambda *args, **kwargs: None)
    with tempfile.TemporaryDirectory() as d:
        res = probes.coverage_gate(
            _never_called_run_one, Path(d),
            [probes.ProbeCarrier(96, 32, 10, "synthetic_plain")] * 3, seed=1,
            representation_key=bytes(range(32)),
        )
    assert res["factor"] == 1.0            # fail-open: no penalty
    assert res["severity"] == 0.0
    assert res["severities"] == []
    assert res["subtle_agreement"] is None
    assert res.get("note") == "no informative probe draw"


def test_degenerate_rep_returns_none(monkeypatch):
    # Force a complete flat spectrum: the rep has no identifiable structured axis and must skip
    # without touching the submission. This remains stable when probe dimensions are recalibrated.
    monkey = {"called": False}

    def run_one(vcf, k, out, ids):
        monkey["called"] = True
        return False, None

    # Keep this branch-level regression small. Probe-size realism is covered by the catalog tests;
    # here only the behavior for an exact flat reference spectrum matters.
    monkeypatch.setattr(
        probes,
        "_build_coverage_probe",
        lambda rng, carrier: np.zeros((32, 96), dtype=np.int8),
    )
    monkeypatch.setattr(
        probes,
        "_patterson_pca",
        lambda G, k: (np.zeros((G.shape[1], k)), np.ones(G.shape[1])),
    )
    res = probes._coverage_severity_once(
        run_one, Path(tempfile.gettempdir()),
        probes.ProbeCarrier(96, 32, 10, "synthetic_plain"), seed=1070,
        representation_key=bytes(range(32)),
    )
    assert res is None
    assert monkey["called"] is False
