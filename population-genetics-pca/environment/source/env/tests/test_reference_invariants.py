"""Internal algebra checks for the sealed PCA references.

These tests intentionally target reference code, not submission output.  The benchmark scores a
sample subspace, so signs, rotations inside tied eigenspaces, loadings, and any particular solver
API are deliberately absent from this contract.
"""

from pathlib import Path

import numpy as np
import pytest

from data.generate import DatasetSpec, simulate_genotypes
from grader.metrics.subspace import structure_weights
from reference import pca_core
import reference.full_scan_pca as full_scan


def _structured_dosages(n_variants: int, n_samples: int, seed: int) -> np.ndarray:
    """Generate informative diploid rows with several sample-population contrasts."""
    rng = np.random.default_rng(seed)
    population = np.arange(n_samples) % 4
    base = rng.uniform(0.08, 0.42, size=n_variants)
    contrast = rng.normal(0.0, 0.09, size=(n_variants, 4))
    probability = np.clip(base[:, None] + contrast[:, population], 0.01, 0.49)
    dosage = rng.binomial(2, probability).astype(np.int8)
    dosage[rng.random(dosage.shape) < 0.025] = -1
    return dosage


@pytest.mark.parametrize(
    ("n_variants", "n_samples", "k"),
    [
        (640, 48, 7),   # conventional genomic shape: variants dominate samples
        (24, 96, 7),    # sample-heavy shape: forming an n-sample Gram is not compulsory
    ],
)
def test_reference_scores_obey_pca_algebra_in_both_shape_regimes(
    n_variants: int,
    n_samples: int,
    k: int,
):
    dosage = _structured_dosages(n_variants, n_samples, seed=n_variants + n_samples)
    standardized = pca_core._standardize_kept(dosage)
    assert standardized is not None
    assert standardized.shape[0] >= k

    gram = standardized.T @ standardized
    scores, eigenvalues, spectrum = pca_core.gram_to_scores(
        gram,
        standardized.shape[0],
        k,
    )

    assert scores.shape == (n_samples, k)
    assert eigenvalues.shape == (k,)
    assert spectrum.shape == (n_samples,)
    assert np.isfinite(scores).all()
    assert np.isfinite(eigenvalues).all()
    assert np.all(eigenvalues > 0.0)
    assert np.all(np.diff(spectrum) <= 1e-12)
    np.testing.assert_allclose(scores.mean(axis=0), 0.0, atol=2e-14)
    np.testing.assert_allclose(
        scores.T @ scores,
        np.diag(eigenvalues),
        rtol=2e-12,
        atol=2e-12,
    )

    # Compare against an independent SVD of the standardized variant-by-sample design.  Compare
    # subspaces rather than signed columns: PCA signs are arbitrary, and nearly tied axes may
    # rotate without changing the mathematical answer.
    _, singular_values, sample_axes_t = np.linalg.svd(standardized, full_matrices=False)
    expected_axes = sample_axes_t.T[:, :k]
    observed_axes, _ = np.linalg.qr(scores)
    principal_cosines = np.linalg.svd(expected_axes.T @ observed_axes, compute_uv=False)

    assert principal_cosines.min() > 1.0 - 2e-11
    np.testing.assert_allclose(
        eigenvalues,
        singular_values[:k] ** 2 / standardized.shape[0],
        rtol=2e-12,
        atol=2e-12,
    )


def test_reference_selects_the_leading_k_when_weaker_structured_axes_remain():
    spec = DatasetSpec(
        "reference_spectral_selection", "spectral_selection", 380, 20000, 19, 0.40,
        seed=937, k=12, regime="spectral_selection",
    )
    dosage, _, _, _, _ = simulate_genotypes(spec)
    standardized = pca_core._standardize_kept(dosage)
    assert standardized is not None

    scores, eigenvalues, spectrum = pca_core.gram_to_scores(
        standardized.T @ standardized,
        standardized.shape[0],
        spec.k,
    )
    structured = np.flatnonzero(structure_weights(spectrum, spec.k) > 0.0)

    # The requested answer is a strict leading subset of the identifiable structure. This checks
    # the reference's ordering and algebra without imposing eigenvector signs or a solver API.
    assert structured.tolist() == list(range(spec.n_pops - 1))
    assert structured.size > spec.k
    assert spectrum[spec.k - 1] / spectrum[spec.k] >= 1.18
    assert spectrum[spec.n_pops - 2] / spectrum[spec.n_pops - 1] >= 1.05
    np.testing.assert_allclose(scores.mean(axis=0), 0.0, atol=2e-14)
    np.testing.assert_allclose(
        scores.T @ scores,
        np.diag(eigenvalues),
        rtol=2e-12,
        atol=2e-12,
    )
    np.testing.assert_allclose(eigenvalues, spectrum[:spec.k], rtol=0.0, atol=0.0)


@pytest.mark.parametrize("n_variants", [24, 140])
def test_full_scan_vcf_path_preserves_reference_score_invariants(
    tmp_path: Path,
    n_variants: int,
):
    rng = np.random.default_rng(917)
    n_samples = 36
    k = 6
    sample_ids = [f"sample-{index}" for index in range(n_samples)]
    path = tmp_path / "mixed-format.vcf"
    genotype = np.asarray(["0/0", "0/1", "1/1"], dtype=object)

    with path.open("w", encoding="ascii", newline="\n") as handle:
        handle.write("##fileformat=VCFv4.2\n")
        handle.write(
            "\t".join(
                [
                    "#CHROM",
                    "POS",
                    "ID",
                    "REF",
                    "ALT",
                    "QUAL",
                    "FILTER",
                    "INFO",
                    "FORMAT",
                    *sample_ids,
                ]
            )
            + "\n"
        )
        population = np.arange(n_samples) % 4
        for variant in range(n_variants):
            base = rng.uniform(0.08, 0.42)
            shifts = rng.normal(0.0, 0.08, size=4)
            probability = np.clip(base + shifts[population], 0.01, 0.49)
            calls = genotype[rng.binomial(2, probability, size=n_samples)].tolist()
            for sample in range(n_samples):
                if (variant * 11 + sample * 7) % 97 == 0:
                    calls[sample] = "./."
            rich_format = variant % 3 == 0
            if rich_format:
                calls = [f"{call}:20:99" for call in calls]
            handle.write(
                "\t".join(
                    [
                        "22",
                        str(16_000_000 + variant),
                        ".",
                        "A",
                        "G",
                        ".",
                        "PASS",
                        ".",
                        "GT:DP:GQ" if rich_format else "GT",
                        *calls,
                    ]
                )
                + "\n"
            )

    observed_ids, scores, kept, spectrum = full_scan.fit(path, k, chunk=19)

    assert observed_ids == sample_ids
    assert kept == n_variants
    assert scores.shape == (n_samples, k)
    assert spectrum.shape == (n_samples,)
    assert np.isfinite(scores).all()
    assert np.isfinite(spectrum).all()
    np.testing.assert_allclose(scores.mean(axis=0), 0.0, atol=2e-14)
    np.testing.assert_allclose(
        scores.T @ scores,
        np.diag(spectrum[:k]),
        rtol=2e-12,
        atol=2e-12,
    )
