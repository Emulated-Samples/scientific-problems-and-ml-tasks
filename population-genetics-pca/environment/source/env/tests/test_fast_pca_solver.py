from pathlib import Path

import numpy as np
import pytest

from reference import pca_core
from reference.fast_pca import fit, gram_to_scores, matrixfree_scores
from reference.full_scan_pca import fit as full_scan_fit


ROOT = Path(__file__).resolve().parents[1]


def test_standalone_is_exact_copy_of_reference():
    source = (ROOT / "reference" / "fast_pca.py").read_bytes()
    executable = (ROOT / "reference" / "fast_submission" / "pca").read_bytes()
    assert executable == b"#!/usr/bin/env python3\n" + source


def test_deployed_gold_is_the_current_reference():
    """The gold on `main` IS the environment's self-check, so it must not rot.

    Grading `workspace/submission/` without running `setupProblem` is the oracle path: it answers
    "does a known-perfect program still score ~1.0 through the real grader". That answer is only
    meaningful if the deployed gold is the CURRENT reference.

    It was not. The test above pinned `reference/fast_submission/pca` to `fast_pca.py`, so that copy
    tracked every optimization -- but nothing pinned the workspace copy, and it silently froze at the
    env-conversion commit, missing the entire per-fold engine-selection rework (`dc6e0ab`). The env's
    own oracle check was therefore grading a program that commit exists to fix: one measured LOSING
    to a plain full scan on the small folds. A stale gold does not fail loudly -- it quietly reports a
    worse number and reads as a calibration regression in the grader.
    """
    gold = (ROOT / "workspace" / "submission" / "pca").read_bytes()
    reference = (ROOT / "reference" / "fast_submission" / "pca").read_bytes()
    assert gold == reference, (
        "workspace/submission/pca has drifted from reference/fast_submission/pca; "
        "regenerate it so the oracle self-check grades the current reference"
    )


def test_gram_solver_recovers_clustered_top_subspace_exactly():
    rng = np.random.default_rng(42)
    n = 600
    k = 10
    q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    evals = np.concatenate((np.linspace(1.05, 1.041, k), np.ones(n - k)))
    gram = (q * evals) @ q.T

    scores, observed = gram_to_scores(gram, 1, k)
    recovered, _ = np.linalg.qr(scores)
    expected = q[:, :k]
    correlations = np.linalg.svd(expected.T @ recovered, compute_uv=False)

    assert correlations.min() > 1 - 1e-10
    np.testing.assert_allclose(observed, evals[:k], rtol=1e-11, atol=1e-12)


def test_matrixfree_solver_recovers_weak_clustered_subspace():
    rng = np.random.default_rng(7)
    m, n, k = 240, 400, 8
    left, _ = np.linalg.qr(rng.standard_normal((m, m)))
    right, _ = np.linalg.qr(rng.standard_normal((n, m)))
    expected_evals = np.concatenate((np.linspace(1.08, 1.05, k), np.ones(m - k)))
    singular_values = np.sqrt(m * expected_evals)
    design = ((left * singular_values) @ right.T).astype(np.float32)

    scores, observed = matrixfree_scores(design, k, seed=13)
    recovered, _ = np.linalg.qr(scores)
    correlations = np.linalg.svd(right[:, :k].T @ recovered, compute_uv=False)

    assert correlations.min() > 0.9999
    np.testing.assert_allclose(observed, expected_evals[:k], rtol=2e-5, atol=2e-6)


def test_matrixfree_solver_handles_exact_requested_small_variant_side():
    rng = np.random.default_rng(9)
    design = rng.standard_normal((4, 20)).astype(np.float32)
    scores, observed = matrixfree_scores(design, 4)
    _, singular_values, vt = np.linalg.svd(design.astype(np.float64), full_matrices=False)
    expected = vt.T * (singular_values / np.sqrt(design.shape[0]))[None, :]

    recovered, _ = np.linalg.qr(scores)
    target, _ = np.linalg.qr(expected)
    correlations = np.linalg.svd(target.T @ recovered, compute_uv=False)

    assert scores.shape == (20, 4)
    assert correlations.min() > 1 - 1e-10
    np.testing.assert_allclose(
        observed, singular_values ** 2 / design.shape[0], rtol=2e-6, atol=2e-7,
    )


def test_solvers_reject_unavailable_or_null_components():
    rng = np.random.default_rng(10)
    with pytest.raises(ValueError, match="informative variants"):
        matrixfree_scores(rng.standard_normal((4, 20)).astype(np.float32), 5)

    vector = rng.standard_normal(8)
    rank_one_gram = np.outer(vector, vector)
    with pytest.raises(ValueError, match="rank below"):
        gram_to_scores(rank_one_gram, 100, 2)
    with pytest.raises(ValueError, match="rank below"):
        pca_core.gram_to_scores(rank_one_gram, 100, 2)

    duplicated = np.repeat(vector[None, :], 4, axis=0).astype(np.float32)
    with pytest.raises(ValueError, match="rank below"):
        matrixfree_scores(duplicated, 2)


def test_rank_checks_are_invariant_to_covariance_scale():
    diagonal = np.diag(np.linspace(5e-8, 1e-8, 8)).astype(np.float32)
    scores, eigenvalues = gram_to_scores(diagonal, 1, 5)
    assert scores.shape == (8, 5)
    assert eigenvalues[-1] > 0

    rng = np.random.default_rng(12)
    design = (rng.standard_normal((20, 40)) * 1e-4).astype(np.float32)
    scores, eigenvalues = matrixfree_scores(design, 5)
    assert scores.shape == (40, 5)
    assert eigenvalues[-1] > 0


def test_fits_reject_k_not_below_sample_count(tmp_path):
    path = tmp_path / "tiny.vcf"
    path.write_bytes(
        b"##fileformat=VCFv4.2\n"
        b"#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\ts2\ts3\n"
        b"1\t1\t.\tA\tC\t.\tPASS\t.\tGT\t0/0\t0/1\t1/1\n"
    )

    with pytest.raises(ValueError, match="k must satisfy"):
        fit(path, 3)
    with pytest.raises(ValueError, match="k must satisfy"):
        full_scan_fit(path, 3)


def test_auto_matrixfree_fit_matches_full_scan_scores(tmp_path):
    rng = np.random.default_rng(21)
    n_samples = 64
    n_variants = 32
    path = tmp_path / "wide.vcf"
    ids = [f"s{i}".encode() for i in range(n_samples)]
    gt = np.asarray([b"0/0", b"0/1", b"1/1"])
    with open(path, "wb") as fh:
        fh.write(
            b"##fileformat=VCFv4.2\n"
            + b"\t".join([
                b"#CHROM", b"POS", b"ID", b"REF", b"ALT", b"QUAL", b"FILTER", b"INFO",
                b"FORMAT", *ids,
            ])
            + b"\n"
        )
        for variant in range(n_variants):
            p = rng.uniform(0.08, 0.48)
            dosage = rng.binomial(2, p, size=n_samples)
            fh.write(
                b"\t".join([
                    b"1", str(variant + 1).encode(), b".", b"A", b"C", b".", b"PASS",
                    b".", b"GT", *gt[dosage].tolist(),
                ])
                + b"\n"
            )

    _, truth, truth_kept, _ = full_scan_fit(path, 5)
    _, observed, kept, meta = fit(path, 5, n_workers=1, engine="auto")
    truth_basis, _ = np.linalg.qr(truth)
    observed_basis, _ = np.linalg.qr(observed)
    correlations = np.linalg.svd(truth_basis.T @ observed_basis, compute_uv=False)

    assert kept == truth_kept == n_variants
    assert meta["engine"] == "matrixfree"
    assert correlations.min() > 0.99999
    np.testing.assert_allclose(
        np.linalg.norm(observed, axis=0), np.linalg.norm(truth, axis=0), rtol=2e-5, atol=2e-6,
    )


def test_matrixfree_solver_uses_smaller_sample_side_when_forced():
    rng = np.random.default_rng(11)
    design = rng.standard_normal((30, 12)).astype(np.float32)
    scores, observed = matrixfree_scores(design, 5)
    _, singular_values, vt = np.linalg.svd(design.astype(np.float64), full_matrices=False)

    recovered, _ = np.linalg.qr(scores)
    correlations = np.linalg.svd(vt.T[:, :5].T @ recovered, compute_uv=False)

    assert correlations.min() > 1 - 2e-6
    np.testing.assert_allclose(
        observed, singular_values[:5] ** 2 / design.shape[0], rtol=2e-6, atol=2e-7,
    )


_GT_ENCODING = np.array([b"0/0", b"0/1", b"1/1"])


def _write_structured_vcf(path, n_samples, n_variants, n_pops, fst, seed):
    """Write a Balding-Nichols structured genotype VCF: n_pops populations drifted from a common
    ancestral frequency, giving exactly n_pops-1 real structured PC axes."""
    rng = np.random.default_rng(seed)
    pop = rng.integers(0, n_pops, n_samples)
    ancestral = rng.uniform(0.1, 0.9, n_variants)
    a = ancestral * (1.0 - fst) / fst
    b = (1.0 - ancestral) * (1.0 - fst) / fst
    pop_freq = rng.beta(a[None, :], b[None, :], size=(n_pops, n_variants))
    dosage = rng.binomial(2, pop_freq[pop].T)          # (variants x samples), values in {0,1,2}
    encoded = _GT_ENCODING[dosage]
    header = (
        b"##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t"
        + b"\t".join(f"S{i}".encode() for i in range(n_samples)) + b"\n"
    )
    with open(path, "wb") as fh:
        fh.write(header)
        for j in range(n_variants):
            fh.write(b"1\t" + str(j + 1).encode() + b"\t.\tA\tG\t.\t.\t.\tGT\t"
                     + b"\t".join(encoded[j].tolist()) + b"\n")
    return dosage


def _independent_patterson_subspace(dosage, k):
    """A from-scratch Patterson PCA (own standardisation + sample-Gram eigendecomposition) computed
    directly from the raw dosage matrix -- the demigod check the full-scan truth must agree with."""
    X = dosage.astype(np.float64)                      # variants x samples
    p_alt = X.mean(axis=1) / 2.0
    keep = (p_alt > 0) & (p_alt < 1) & (X.var(axis=1) > 0)
    X = X[keep]
    p_alt = p_alt[keep]
    Z = (X - 2.0 * p_alt[:, None]) / np.sqrt(2.0 * p_alt * (1.0 - p_alt))[:, None]
    gram = Z.T @ Z / Z.shape[0]
    _, vectors = np.linalg.eigh(gram)
    return vectors[:, ::-1][:, :k]


def _worst_subspace_cosine(a, b, k):
    qa, _ = np.linalg.qr(np.asarray(a)[:, :k])
    qb, _ = np.linalg.qr(np.asarray(b)[:, :k])
    return float(np.linalg.svd(qa.T @ qb, compute_uv=False).min())


def test_full_scan_truth_equals_an_independent_patterson_pca_end_to_end(tmp_path):
    """Gold-ness of the ACCURACY truth: on a structured genotype VCF the full-scan reference must
    compute the exact same top-k subspace an independently written Patterson PCA computes. If it did
    not, every submission would be scored against a wrong truth. Here all k axes are real structure
    (k = n_pops - 1), so the whole requested subspace must align."""
    vcf = tmp_path / "structured.vcf"
    dosage = _write_structured_vcf(vcf, n_samples=160, n_variants=6000, n_pops=6, fst=0.05, seed=3)
    _, truth_scores, _, _ = full_scan_fit(vcf, 5)
    independent = _independent_patterson_subspace(dosage, 5)
    assert _worst_subspace_cosine(truth_scores, independent, 5) > 0.999


def test_fast_reference_preserves_the_truth_subspace_even_while_subsampling(tmp_path):
    """Gold-ness of the SPEED CEILING: the fast reference sets systems_quality=1.0, so matching its
    speed must not require sacrificing correctness. On a marker-heavy structured VCF it reads only a
    FRACTION of the body (full_coverage is False), yet its recovered subspace must still align with
    the full-scan truth on every structured axis -- otherwise the achievable ceiling would be an
    accuracy-losing program and could not be reached with full reward."""
    vcf = tmp_path / "marker_heavy.vcf"
    _write_structured_vcf(vcf, n_samples=160, n_variants=20000, n_pops=6, fst=0.05, seed=3)
    _, truth_scores, _, _ = full_scan_fit(vcf, 5)
    # Small strata force a genuine subsample rather than full coverage; workers=1 keeps it
    # deterministic. seed is fixed inside fit, so the sampled subspace is reproducible.
    _, fast_scores, kept, meta = fit(
        vcf, 5, n_blocks=400, block_size=32768, n_workers=1,
    )
    assert not meta["full_coverage"], "fixture must exercise the subsampling path, not full coverage"
    assert meta["final_selected_bytes"] < meta["file_size"]
    assert _worst_subspace_cosine(truth_scores, fast_scores, 5) > 0.99


def test_scipy_sparse_linalg_is_imported_lazily_not_at_module_scope():
    """Importing ``scipy.sparse.linalg``'s public API costs ~0.4 s -- more than the entire fit on a
    small VCF -- and it is needed ONLY by the matrix-free branch (the biobank regime,
    n_samples > n_variants). Paying it on every run made the fast path measurably SLOWER than a
    plain full scan on small inputs: the sampler losing to the very thing it exists to beat, on an
    import it never used. Assert structurally that the import stays inside a function.

    (``scipy.linalg`` drags ``scipy.sparse.linalg`` into ``sys.modules`` transitively, so presence
    in ``sys.modules`` proves nothing -- what costs the time is binding its public API here.)
    """
    import ast
    from pathlib import Path

    source = (Path(__file__).resolve().parent.parent / "reference" / "fast_pca.py").read_text()
    tree = ast.parse(source)
    for node in tree.body:                      # module scope only
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("scipy.sparse"):
            raise AssertionError(
                "reference/fast_pca.py imports scipy.sparse at module scope; keep it inside the "
                "matrix-free branch so the common path does not pay ~0.4 s it never uses"
            )
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith("scipy.sparse"), (
                    "reference/fast_pca.py imports scipy.sparse at module scope"
                )
