import numpy as np
import pytest

from grader.metrics.structure import (
    label_projection,
    population_separation,
    within_group_label_projection,
)


def _label_scores():
    labels = np.repeat(np.arange(4), 20)
    one_hot = np.column_stack([(labels == value).astype(float) for value in range(4)])
    # Three independent centred population contrasts plus an unrelated direction.
    contrasts = one_hot[:, :3] - one_hot[:, [3]]
    nuisance = np.sin(np.arange(labels.size))[:, None]
    return labels, np.column_stack([contrasts, nuisance])


def test_label_projection_is_invariant_to_score_basis_changes():
    labels, scores = _label_scores()
    mixing = np.array([
        [4.0, 1.0, 0.0, 2.0],
        [0.0, -0.5, 2.0, 0.0],
        [1.0, 0.0, 3.0, -1.0],
        [0.0, 0.0, 0.0, 0.25],
    ])

    original = label_projection(scores, labels)
    transformed = label_projection(scores @ mixing, labels)

    np.testing.assert_allclose(original, 1.0, atol=1e-12)
    np.testing.assert_allclose(transformed, original, atol=1e-12)


def test_label_projection_is_invariant_to_extreme_finite_output_units():
    labels, scores = _label_scores()
    scales = np.geomspace(1e-300, 1e300, scores.shape[1])

    original = label_projection(scores, labels)
    rescaled = label_projection(scores * scales[None, :], labels)

    np.testing.assert_allclose(rescaled, original, atol=1e-12)


def test_label_projection_gives_continuous_credit_for_partial_structure():
    labels, scores = _label_scores()
    full = label_projection(scores[:, :3], labels)
    partial = label_projection(scores[:, :1], labels)
    unrelated = label_projection(np.cos(np.arange(labels.size))[:, None], labels)

    assert full > 0.999
    assert 0.2 < partial < 0.5
    assert unrelated < 0.02


def test_population_diagnostics_ignore_sign_scale_and_rotation():
    labels, scores = _label_scores()
    transformed = scores @ np.array([
        [2.0, 1.0, 0.0, 0.0],
        [0.0, -3.0, 1.0, 0.0],
        [1.0, 0.0, 1.5, 0.0],
        [0.0, 0.0, 0.0, 0.2],
    ])

    left = population_separation(scores, labels)
    right = population_separation(transformed, labels)

    assert left["label_projection"] == right["label_projection"] == 1.0
    assert left["ncc_accuracy"] == right["ncc_accuracy"]
    assert left["mean_eta2"] == right["mean_eta2"]


def test_within_group_projection_requires_nested_contrasts():
    parent = np.repeat([0, 1], 8)
    nested = np.tile(np.repeat([0, 1], 4), 2) + 2 * parent
    parent_only = np.column_stack([parent == 0, parent == 1]).astype(float)
    nested_scores = np.column_stack([
        parent == 0,
        parent == 1,
        nested == 0,
        nested == 2,
    ]).astype(float)

    assert within_group_label_projection(parent_only, parent, nested) == pytest.approx(0.0)
    assert within_group_label_projection(nested_scores, parent, nested) == pytest.approx(1.0)
