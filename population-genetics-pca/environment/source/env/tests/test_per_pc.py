"""Unit tests for grader.metrics.per_pc.

We build a synthetic reference PCA whose score columns are exactly orthonormal-over-samples
directions scaled by prescribed eigenvalues, so we control which PCs are "structured" (above
the Marchenko-Pastur noise floor used by subspace.structure_weights) and can construct
submissions with known per-PC behaviour.
"""

import numpy as np
import pytest

from grader.metrics.per_pc import (
    per_pc_agreement,
    per_pc_subspace_agreement,
    resolved_pc_count,
    per_pc_report,
)
from grader.metrics.subspace import structure_weights, subspace_accuracy


N = 200
# Eight reported PCs backed by a complete 200-value truth spectrum. The first six are spikes;
# the final two reported PCs and the long tail form one continuous noise bulk.
EVALS = np.array([100.0, 90.0, 80.0, 70.0, 60.0, 50.0, 1.05, 1.0])
K = EVALS.size
N_STRUCTURED = 6
SPECTRUM = np.concatenate((EVALS, np.linspace(0.99, 0.4, N - K - 1), [0.0]))


def make_ref(seed: int = 42):
    """Reference scores: zero-mean, mutually orthonormal sample-directions scaled by sqrt(eval)."""
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((N, K))
    M = M - M.mean(axis=0, keepdims=True)
    Q, _ = np.linalg.qr(M)                     # (N, K) column-orthonormal, zero-mean columns
    return Q * np.sqrt(EVALS)[None, :]


def test_identical_scores_all_agree_and_fully_resolved():
    ref = make_ref()
    sub = ref.copy()

    a = per_pc_agreement(sub, ref)
    a_sub = per_pc_subspace_agreement(sub, ref)
    assert np.all(a > 0.999)
    assert np.all(a_sub > 0.999)

    assert resolved_pc_count(sub, ref, ref_evals=SPECTRUM) == N_STRUCTURED

    rep = per_pc_report(sub, ref, ref_evals=SPECTRUM)
    assert rep["n_structured"] == N_STRUCTURED
    assert rep["resolved"] == N_STRUCTURED
    idx, val = rep["weakest_structured_pc"]
    assert 0 <= idx < N_STRUCTURED and val > 0.999


def test_per_pc_metrics_are_invariant_to_extreme_finite_output_units():
    ref = make_ref()
    scales = np.geomspace(1e-300, 1e300, K)
    sub = ref * scales[None, :]

    np.testing.assert_allclose(per_pc_agreement(sub, ref), 1.0, atol=1e-12)
    np.testing.assert_allclose(per_pc_subspace_agreement(sub, ref), 1.0, atol=1e-12)


def test_good_leading_pcs_bad_higher_pcs():
    ref = make_ref()
    rng = np.random.default_rng(7)
    sub = ref.copy()
    sub[:, 2:] = rng.standard_normal((N, K - 2))   # keep PC1/PC2, randomise PC3+

    a = per_pc_agreement(sub, ref)
    assert a[0] > 0.99 and a[1] > 0.99
    assert np.all(a[2:] < 0.5)                     # random directions barely correlate

    # Only the two good structured PCs are resolved.
    assert resolved_pc_count(sub, ref, ref_evals=SPECTRUM) == 2

    rep = per_pc_report(sub, ref, ref_evals=SPECTRUM)
    assert rep["resolved"] == 2
    idx, val = rep["weakest_structured_pc"]
    assert idx >= 2 and val < 0.5                  # the breakdown is in a higher structured PC


def test_sign_flip_and_within_subspace_rotation_robust_for_subspace_variant():
    ref = make_ref()
    sub = ref.copy()
    sub[:, 0] = -ref[:, 0]                          # arbitrary PCA sign flip on PC1

    # Rotate PC2/PC3 (indices 1,2) by 45 deg within their own plane: the 2-D span is unchanged
    # but neither axis lands on the reference axis any more.
    c, s = np.cos(np.pi / 4), np.sin(np.pi / 4)
    sub[:, 1] = c * ref[:, 1] + s * ref[:, 2]
    sub[:, 2] = -s * ref[:, 1] + c * ref[:, 2]

    a = per_pc_agreement(sub, ref)
    a_sub = per_pc_subspace_agreement(sub, ref)

    # Strict measure: sign flip is forgiven, but the in-plane rotation is (correctly) penalised.
    assert a[0] > 0.999
    assert a[1] < 0.9 and a[2] < 0.9

    # Subspace measure: PC1 (sign) fine; once the rotated block is fully inside the leading-k
    # span (k >= 3) the reference direction is recovered, and untouched PCs stay ~1.
    assert a_sub[0] > 0.999
    assert np.all(a_sub[2:] > 0.999)
    # Subspace agreement is never worse than the strict per-axis agreement.
    assert np.all(a_sub + 1e-9 >= a)


def test_strict_per_pc_must_not_gate_reward_it_would_reject_legit_rotation():
    """Pins the design decision that per_pc is DIAGNOSTIC ONLY, never a reward term.

    A legitimate PCA that recovers the full structured subspace but rotates a near-degenerate
    plane is scientifically correct and MUST keep full credit. ``subspace_accuracy`` (the reward
    term) gives it 1.0. The strict per-axis ``resolved_pc_count`` instead reports fewer resolved
    PCs -- so if that count fed the reward it would be a false negative on a correct submission.
    This test fails loudly if anyone wires the strict per-PC measure into scoring.
    """
    ref = make_ref()
    weights = structure_weights(SPECTRUM, K)[:K]

    sub = ref.copy()
    c, s = np.cos(np.pi / 4), np.sin(np.pi / 4)
    # PC5/PC6 (indices 4,5) have the closest eigenvalues (60/50); rotate that plane 45 deg.
    sub[:, 4] = c * ref[:, 4] + s * ref[:, 5]
    sub[:, 5] = -s * ref[:, 4] + c * ref[:, 5]

    # Reward term: rotation forgiven -> full credit (no false negative).
    assert subspace_accuracy(sub, ref, weights)["accuracy"] > 0.999
    # Strict per-axis view: the rotated axes no longer "land", so fewer resolve. Grading on this
    # would wrongly deny credit -- exactly why it stays a diagnostic.
    assert resolved_pc_count(sub, ref, ref_evals=SPECTRUM) < N_STRUCTURED


def test_sample_count_mismatch_raises():
    ref = make_ref()
    sub = ref[:-1].copy()                           # one fewer sample
    with pytest.raises(ValueError, match="sample count mismatch"):
        per_pc_agreement(sub, ref)
    with pytest.raises(ValueError, match="sample count mismatch"):
        per_pc_subspace_agreement(sub, ref)


def test_fewer_submission_columns_are_padded_as_missing():
    ref = make_ref()
    sub = ref[:, :3].copy()                         # only 3 PCs supplied
    a = per_pc_agreement(sub, ref)
    assert a.shape[0] == K
    assert np.all(a[:3] > 0.999)
    assert np.all(a[3:] == 0.0)                     # missing PCs score 0, not an error
