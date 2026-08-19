from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from grader import grade
from grader.gates import probes


ROOT = Path(__file__).resolve().parents[1]
CORRECT_SUBMISSIONS = (
    ROOT / "fn_resistance" / "independent_python" / "pca",
    ROOT / "fn_resistance" / "eigsh_local_module" / "pca",
)


def _controlled_representation_probe() -> np.ndarray:
    """Small four-population fixture with three clear spikes and a full noise spectrum."""
    rng = np.random.default_rng(28_731)
    n_samples = 96
    n_variants = 2400
    n_pops = 4
    fst = 0.18
    sample_pop = np.arange(n_samples, dtype=np.int32) % n_pops
    rng.shuffle(sample_pop)
    ancestral = np.clip(rng.beta(0.7, 0.7, size=n_variants), 0.04, 0.96)
    alpha = ancestral * (1.0 - fst) / fst
    beta = (1.0 - ancestral) * (1.0 - fst) / fst
    frequencies = np.stack([rng.beta(alpha, beta) for _ in range(n_pops)])
    genotypes = np.empty((n_variants, n_samples), dtype=np.int8)
    for population in range(n_pops):
        columns = np.flatnonzero(sample_pop == population)
        genotypes[:, columns] = rng.binomial(
            2, frequencies[population, :, None], size=(n_variants, columns.size),
        )
    return genotypes


@pytest.mark.parametrize("entrypoint", CORRECT_SUBMISSIONS, ids=lambda path: path.parent.name)
def test_independent_correct_guards_accept_rich_equivalent_vcfs(
    entrypoint, tmp_path, monkeypatch,
):
    """A correctness guard must satisfy the same visible representation contract as solvers."""

    probe = _controlled_representation_probe()
    carrier = probes.ProbeCarrier(
        probe.shape[1], probe.shape[0], 3, "synthetic_variable_width")
    monkeypatch.setattr(probes, "_build_representation_probe", lambda rng, selected: probe)

    def run_one(vcf: Path, k: int, out: Path, sample_ids: list[str]):
        completed = subprocess.run(
            [sys.executable, str(entrypoint), str(vcf), str(k), str(out)],
            cwd=entrypoint.parent,
            env={**os.environ, "PYTHONPATH": str(entrypoint.parent)},
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=60,
        )
        if completed.returncode != 0:
            return False, None
        scores, _ = grade.load_scores(out, sample_ids, k)
        return scores is not None, scores

    result = probes.representation_equivalence_gate(
        run_one, tmp_path, [carrier], seed=917,
        representation_key=bytes(range(32)),
    )

    assert result["severity"] == 0.0, result
    assert result["factor"] == 1.0, result
    assert result["reference_agreement_a"] >= 0.99, result
    assert result["reference_agreement_b"] >= 0.99, result
