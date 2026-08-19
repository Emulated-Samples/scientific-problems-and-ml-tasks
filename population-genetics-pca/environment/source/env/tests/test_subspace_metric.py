import numpy as np
import pytest

from grader.metrics.subspace import (
    structure_weights,
    subspace_accuracy,
    _random_projection_mean,
)


def _spiked_spectrum(n=200):
    return np.concatenate(([2.2, 2.1], np.linspace(1.25, 0.75, n - 3), [0.0]))


def _wishart_bulk(p: int, ratio: float, seed: int) -> np.ndarray:
    """A real Marchenko-Pastur noise bulk: eigenvalues of a p x (p/ratio) Gaussian Gram, mean ~1.2.

    ``ratio`` (= p / n_markers) sets the bulk width -- small ratio is the biobank regime (many
    markers per PC, tight bulk); large ratio widens the MP edge to stress the noise-edge fit."""
    m = int(p / ratio)
    X = np.random.default_rng(seed).standard_normal((p, m))
    return np.sort(np.linalg.eigvalsh((X @ X.T) / m))[::-1] * 1.2


def test_full_spectrum_bulk_edge_selects_only_spikes():
    weights = structure_weights(_spiked_spectrum(), 10)
    assert np.array_equal(np.flatnonzero(weights > 0), np.array([0, 1]))


def test_flat_noise_spectrum_has_no_structured_axes():
    spectrum = np.concatenate((np.linspace(1.25, 0.75, 199), [0.0]))
    assert not np.any(structure_weights(spectrum, 10) > 0)


def test_noise_pc_disagreement_is_ignored_but_dropped_structure_is_not():
    rng = np.random.default_rng(12)
    n, k = 200, 10
    basis, _ = np.linalg.qr(rng.standard_normal((n, k)))
    spectrum = _spiked_spectrum(n)
    ref = basis * np.sqrt(spectrum[:k])[None, :]
    weights = structure_weights(spectrum, k)[:k]

    random_noise = rng.standard_normal((n, k - 2))
    submission = np.column_stack((ref[:, :2], random_noise))
    faithful = subspace_accuracy(submission, ref, weights)

    without_pc2 = rng.standard_normal((n, k - 1))
    without_pc2 -= basis[:, :2] @ (basis[:, :2].T @ without_pc2)
    dropped = np.column_stack((ref[:, 0], without_pc2))
    inaccurate = subspace_accuracy(dropped, ref, weights)

    assert faithful["accuracy"] > 0.999
    assert inaccurate["accuracy"] == 0.0


def test_accuracy_is_invariant_to_extreme_finite_output_units():
    rng = np.random.default_rng(120)
    n, k = 200, 10
    basis, _ = np.linalg.qr(rng.standard_normal((n, k)))
    spectrum = _spiked_spectrum(n)
    ref = basis * np.sqrt(spectrum[:k])[None, :]
    weights = structure_weights(spectrum, k)[:k]
    scales = np.geomspace(1e-300, 1e300, k)

    ordinary = subspace_accuracy(ref, ref, weights)
    rescaled = subspace_accuracy(ref * scales[None, :], ref, weights)

    assert ordinary["accuracy"] > 0.999999
    assert rescaled["accuracy"] == pytest.approx(ordinary["accuracy"], abs=1e-12)


def test_malformed_spectra_still_raise():
    with pytest.raises(ValueError):
        structure_weights(np.zeros((3, 3)), 1)              # not 1-D
    with pytest.raises(ValueError):
        structure_weights(np.array([]), 1)                  # empty
    with pytest.raises(ValueError):
        structure_weights(np.array([1.0, np.nan, 0.5]), 1)  # non-finite
    with pytest.raises(ValueError):
        structure_weights(np.array([1.0, -0.5, 0.2]), 1)    # meaningfully negative


@pytest.mark.parametrize("spectrum", [
    np.zeros(100),
    np.concatenate(([5.0], np.zeros(99))),
    np.concatenate((np.ones(50), np.zeros(50))),
    np.array([2.38, 2.11, 1.72, 1.57, 0.84, 0.80, 0.75, 0.72, 0.68, 0.0]),
])
def test_underidentified_spectra_raise(spectrum):
    with pytest.raises(ValueError, match="positive|at least"):
        structure_weights(spectrum, 4)


def test_finite_sample_edge_fluctuations_are_not_structure():
    # Four true spikes over a synthetic bulk matching the grade-scale mean, width, and largest
    # finite-sample eigenvalues. The old asymptotic cutoff promoted one noise direction because
    # 1.0248 was barely above its 1.0229 edge; the Johnstone correction retains exactly four.
    bulk = np.concatenate((
        [1.02479916, 1.02280707],
        0.94317 + 0.0548 * np.cos(np.linspace(0.0, np.pi, 393)),
    ))
    spectrum = np.concatenate((
        [13.4, 13.2, 12.1, 11.3], np.sort(bulk)[::-1], [0.0],
    ))
    weights = structure_weights(spectrum, 10)
    assert np.array_equal(np.flatnonzero(weights > 0), np.arange(4))


def test_extreme_spike_cannot_inflate_its_own_noise_edge():
    spectrum = np.concatenate(([1e6], np.ones(99), [0.0]))
    weights = structure_weights(spectrum, 10)
    assert np.array_equal(np.flatnonzero(weights > 0), np.array([0]))


def test_structure_beyond_scored_axes_cannot_contaminate_noise_fit():
    spectrum = np.concatenate((np.full(11, 1e6), np.ones(189), [0.0]))
    weights = structure_weights(spectrum, 10)
    assert np.array_equal(np.flatnonzero(weights > 0), np.arange(11))


def test_marchenko_pastur_noise_never_fabricates_structure():
    # Complete random-Wishart spectra at identifiable dimensions have no significant spikes.
    rng = np.random.default_rng(7)
    for p, n in [(100, 300), (80, 240), (160, 1000)]:
        X = rng.standard_normal((p, n))
        spectrum = np.sort(np.linalg.eigvalsh((X @ X.T) / n))[::-1]
        assert not np.any(structure_weights(spectrum, 10) > 0), f"false positive at p={p} n={n}"


def test_constant_and_degenerate_submissions_score_zero():
    # A constant (zero-variance) or all-zero submission carries no directions and must not be
    # credited: after centring every column vanishes, so the subspace is empty -> accuracy 0.
    rng = np.random.default_rng(5)
    n, k = 200, 10
    basis, _ = np.linalg.qr(rng.standard_normal((n, k)))
    spectrum = _spiked_spectrum(n)
    ref = basis * np.sqrt(spectrum[:k])[None, :]
    weights = structure_weights(spectrum, k)[:k]

    assert subspace_accuracy(np.ones((n, k)), ref, weights)["accuracy"] == 0.0
    assert subspace_accuracy(np.zeros((n, k)), ref, weights)["accuracy"] == 0.0


def test_partial_recovery_credit_is_monotone_and_fine_grained():
    # Recovering more structured axes must earn strictly more credit, with no cliff: dropping a
    # real axis costs reward (no FP for partial recovery) but partial recovery is not zeroed.
    rng = np.random.default_rng(9)
    n, k = 300, 8
    basis, _ = np.linalg.qr(rng.standard_normal((n, k)))
    spectrum = np.concatenate(([50.0, 40.0, 30.0, 20.0, 10.0, 5.0],
                               np.linspace(1.05, 0.5, n - 7), [0.0]))
    ref = basis * np.sqrt(spectrum[:k])[None, :]
    weights = structure_weights(spectrum, k)[:k]
    assert int((weights > 0).sum()) == 6

    scores = []
    for recovered in range(1, 7):
        sub = np.column_stack((ref[:, :recovered],
                               rng.standard_normal((n, k - recovered))))
        scores.append(subspace_accuracy(sub, ref, weights)["accuracy"])

    assert scores[0] < 0.05                       # recovering only PC1 of 6 -> essentially no credit
    assert scores[-1] > 0.999                      # all six structured axes -> full credit
    assert all(b > a + 0.02 for a, b in zip(scores, scores[1:]))  # strictly, smoothly increasing


def test_biobank_scale_ladder_is_well_behaved_and_non_degenerate():
    """The biobank ladder shape (n ~ 40k samples, k=5, few structured PCs over a large noise
    bulk) must not degrade the accuracy metric. This pins the newly-emphasised regime:

      * ``structure_weights`` isolates exactly the handful of real axes from the large noise
        bulk -- for both a tight (n >> markers) and a wide MP bulk -- with no axis misclassified;
      * the chance formula stays finite and correct at n-1 ~ 40000 (gammaln at large argument);
      * faithful -> ~1, random -> ~0, and extreme output rescaling is still forgiven at scale;
      * dropping a structured axis yields graded partial credit, monotone and not a cliff.
    """
    n, k = 40000, 5
    rng = np.random.default_rng(3)
    basis, _ = np.linalg.qr(rng.standard_normal((n, k)))
    basis -= basis.mean(axis=0, keepdims=True)
    basis, _ = np.linalg.qr(basis)                     # zero-mean orthonormal sample directions
    spikes = np.array([5.4, 5.0, 4.7, 4.4, 4.1])       # five structured PCs (6 populations)
    ref = basis * np.sqrt(spikes)[None, :]

    # Tracy-Widom selection must recover exactly the five spikes, tight bulk AND wide bulk.
    tight = np.sort(np.concatenate((spikes, _wishart_bulk(400, 0.05, 7), [0.0])))[::-1]
    wide = np.sort(np.concatenate((spikes, _wishart_bulk(400, 0.30, 7), [0.0])))[::-1]
    assert int(np.sum(structure_weights(tight, k) > 0)) == 5
    assert int(np.sum(structure_weights(wide, k) > 0)) == 5
    # A pure-noise spectrum at this scale fabricates no structure.
    pure = np.sort(np.concatenate((_wishart_bulk(400, 0.05, 11), [0.0])))[::-1]
    assert not np.any(structure_weights(pure, k) > 0)

    weights = structure_weights(tight, k)[:k]

    # Chance floor is finite and matches the closed-form single-direction random projection.
    chance = _random_projection_mean(k, n - 1)
    assert np.isfinite(chance) and chance == pytest.approx(np.sqrt(k / (n - 1)), rel=0.1)

    assert subspace_accuracy(ref, ref, weights)["accuracy"] == pytest.approx(1.0, abs=1e-6)
    assert subspace_accuracy(rng.standard_normal((n, k)), ref, weights)["accuracy"] < 0.02
    scales = np.geomspace(1e-280, 1e280, k)
    assert subspace_accuracy(ref * scales[None, :], ref, weights)["accuracy"] == pytest.approx(
        1.0, abs=1e-6
    )

    drop1 = subspace_accuracy(
        np.column_stack((ref[:, :4], rng.standard_normal((n, 1)))), ref, weights
    )["accuracy"]
    drop2 = subspace_accuracy(
        np.column_stack((ref[:, :3], rng.standard_normal((n, 2)))), ref, weights
    )["accuracy"]
    assert 0.05 < drop2 < drop1 < 1.0                  # graded, monotone, no cliff to 0 or 1


def test_subspace_accuracy_rejects_reference_without_structure():
    rng = np.random.default_rng(3)
    ref = rng.standard_normal((60, 4))
    with pytest.raises(ValueError, match="no structured PCs"):
        subspace_accuracy(ref.copy(), ref, np.zeros(4))
