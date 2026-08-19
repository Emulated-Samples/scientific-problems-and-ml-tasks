import numpy as np

from reference import fast_pca, pca_core


def test_zero_information_heterozygous_rows_are_not_counted():
    dosage = np.ones((1, 32), dtype=np.int8)

    assert fast_pca._standardize_kept(dosage) is None
    assert pca_core._standardize_kept(dosage) is None

    gram = np.zeros((32, 32), dtype=np.float64)
    counter = {"kept": 0}
    pca_core.standardize_into_gram(dosage, gram, counter)
    assert counter["kept"] == 0
    assert not gram.any()


def test_fast_and_truth_keep_the_same_rows_with_missing_calls():
    dosage = np.asarray(
        [
            [0, 0, 1, 1, 2, 2, -1],
            [1, 1, 1, 1, 1, -1, 1],
            [2, 2, 2, 2, 1, 2, -1],
            [0, 0, 0, 0, 0, 0, -1],
        ],
        dtype=np.int8,
    )

    fast = fast_pca._standardize_kept(dosage)
    truth = pca_core._standardize_kept(dosage)

    assert fast is not None and truth is not None
    assert fast.shape == truth.shape == (2, 7)
    np.testing.assert_allclose(fast, truth, rtol=2e-6, atol=2e-7)


def test_near_fixed_variant_is_oriented_without_float32_cancellation():
    dosage = np.full((1, 5_000_001), 2, dtype=np.int8)
    dosage[0, -1] = 1

    standardized = fast_pca._standardize_kept(dosage)

    assert standardized is not None
    assert standardized.shape == dosage.shape
    assert np.isfinite(standardized).all()
    assert standardized[0, -1] > 1000
    assert standardized[0, 0] < 0
