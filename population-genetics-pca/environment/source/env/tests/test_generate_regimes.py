import numpy as np

from data.generate import (
    DatasetSpec,
    _spectral_axis_strengths,
    grade_specs,
    simulate_genotypes,
    write_vcf,
)
from data.release_key import key_commitment
from grader.metrics.subspace import structure_weights, subspace_accuracy

REPRESENTATION_KEY = bytes(range(32))
MATH_KEY = bytes(reversed(range(32)))
ALTERNATE_MATH_KEY = b"z" * 32
MATH_COMMITMENT = key_commitment(MATH_KEY)


def _population_contrast_rank(pop_freqs: np.ndarray) -> int:
    contrasts = pop_freqs - pop_freqs.mean(axis=0, keepdims=True)
    return int(np.linalg.matrix_rank(contrasts))


def _patterson_subspace(G: np.ndarray, k: int) -> np.ndarray:
    p = G.mean(axis=1, dtype=np.float64) / 2.0
    keep = (p > 0.0) & (p < 1.0)
    Z = (G[keep] - 2.0 * p[keep, None]) / np.sqrt(
        2.0 * p[keep, None] * (1.0 - p[keep, None])
    )
    _, eigenvectors = np.linalg.eigh(Z.T @ Z)
    return eigenvectors[:, -k:]


def _patterson_spectrum(G: np.ndarray) -> np.ndarray:
    p = G.mean(axis=1, dtype=np.float64) / 2.0
    keep = (p > 0.0) & (p < 1.0)
    Z = (G[keep] - 2.0 * p[keep, None]) / np.sqrt(
        2.0 * p[keep, None] * (1.0 - p[keep, None])
    )
    return np.linalg.eigvalsh(Z.T @ Z / Z.shape[0])[::-1]


def _label_subspace(sample_pop: np.ndarray, n_pops: int) -> np.ndarray:
    labels = np.eye(n_pops, dtype=np.float64)[sample_pop]
    labels -= labels.mean(axis=0, keepdims=True)
    q, _ = np.linalg.qr(labels)
    return q[:, :n_pops - 1]


def _captured_fraction(scores: np.ndarray, labels: np.ndarray) -> float:
    return float(np.linalg.norm(scores.T @ labels, ord="fro") ** 2 / labels.shape[1])


def test_high_rank_regime_balances_samples_and_plants_requested_contrasts():
    spec = DatasetSpec(
        "unit_high_rank", "high_rank", 100, 2400, 25, 0.15,
        seed=831, k=24, regime="high_rank",
    )

    G, sample_pop, pop_freqs, p_anc, admix = simulate_genotypes(spec)

    counts = np.bincount(sample_pop, minlength=spec.n_pops)
    assert G.shape == (spec.n_variants, spec.n_samples)
    assert G.dtype == np.int8
    assert set(np.unique(G)) <= {0, 1, 2}
    assert counts.min() > 0
    assert counts.max() - counts.min() <= 1
    assert _population_contrast_rank(pop_freqs) >= spec.k
    assert p_anc.shape == (spec.n_variants,)
    assert admix is None


def test_spectral_selection_has_identifiable_tail_structure_beyond_a_clear_k_gap():
    spec = DatasetSpec(
        "unit_spectral_selection", "spectral_selection", 380, 30000, 19, 0.40,
        seed=471, k=12, regime="spectral_selection",
    )

    strengths = _spectral_axis_strengths(spec)
    G, sample_pop, pop_freqs, _, admix = simulate_genotypes(spec)
    spectrum = _patterson_spectrum(G)
    structured = np.flatnonzero(structure_weights(spectrum, spec.k) > 0.0)

    assert strengths.shape == (spec.n_pops - 1,)
    assert np.all(np.diff(strengths[:spec.k]) < 0.0)
    assert np.all(np.diff(strengths[spec.k:]) < 0.0)
    assert strengths[spec.k - 1] / strengths[spec.k] >= 1.35
    assert structured.tolist() == list(range(spec.n_pops - 1))
    assert spectrum[spec.k - 1] / spectrum[spec.k] >= 1.20
    assert spectrum[spec.n_pops - 2] / spectrum[spec.n_pops - 1] >= 1.08
    assert _population_contrast_rank(pop_freqs) == spec.n_pops - 1
    assert np.unique(np.bincount(sample_pop, minlength=spec.n_pops)).size == 1
    assert admix is None


def test_ill_conditioned_regime_contains_each_pathology_and_retains_structure():
    spec = DatasetSpec(
        "unit_ill_conditioned", "ill_conditioned", 72, 1200, 9, 0.12,
        seed=947, k=8, regime="ill_conditioned",
    )

    G, sample_pop, pop_freqs, _, admix = simulate_genotypes(spec)

    row_sums = G.sum(axis=1)
    _, multiplicities = np.unique(G, axis=0, return_counts=True)
    duplicate_excess = int(np.sum(multiplicities - 1))
    monomorphic = (G == G[:, :1]).all(axis=1)
    near_fixed = ((0 < row_sums) & (row_sums <= 2)) | (
        (2 * spec.n_samples - 2 <= row_sums) & (row_sums < 2 * spec.n_samples)
    )

    assert G.shape == (spec.n_variants, spec.n_samples)
    assert duplicate_excess >= int(0.20 * spec.n_variants)
    assert monomorphic.sum() >= int(0.09 * spec.n_variants)
    assert near_fixed.sum() >= int(0.18 * spec.n_variants)
    assert _population_contrast_rank(pop_freqs) >= spec.k
    assert np.bincount(sample_pop, minlength=spec.n_pops).min() > 0
    assert admix is None


def test_spatial_ld_requires_broad_coverage_for_the_global_target():
    spec = DatasetSpec(
        "unit_spatial", "spatial_ld", 160, 6400, 5, 0.0,
        seed=571, k=4, regime="spatial_ld",
    )

    G, sample_pop, pop_freqs, _, admix = simulate_genotypes(spec)
    block_size = spec.n_variants // (4 * (spec.n_pops - 1))
    labels = _label_subspace(sample_pop, spec.n_pops)
    full_capture = _captured_fraction(_patterson_subspace(G, spec.k), labels)
    local_capture = _captured_fraction(_patterson_subspace(G[:block_size], spec.k), labels)

    # Descendants of local prototypes create strong LD within a block, not between blocks.
    first = G[:block_size].astype(np.float64)
    second = G[block_size:2 * block_size].astype(np.float64)
    first -= first.mean(axis=1, keepdims=True)
    second -= second.mean(axis=1, keepdims=True)
    first /= np.maximum(np.linalg.norm(first, axis=1, keepdims=True), 1e-12)
    second /= np.maximum(np.linalg.norm(second, axis=1, keepdims=True), 1e-12)
    rng = np.random.default_rng(88)
    left = rng.integers(0, block_size, size=1200)
    same = rng.integers(0, block_size, size=1200)
    other = rng.integers(0, block_size, size=1200)
    within_ld = np.mean(np.abs(np.sum(first[left] * first[same], axis=1)))
    between_ld = np.mean(np.abs(np.sum(first[left] * second[other], axis=1)))

    assert G.shape == (spec.n_variants, spec.n_samples)
    assert full_capture >= 0.75
    assert local_capture <= 0.40
    assert within_ld >= 2.0 * between_ld
    assert _population_contrast_rank(pop_freqs) >= spec.k
    assert admix is None


def test_rare_structure_has_singleton_to_low_mac_axes_alongside_common_markers():
    spec = DatasetSpec(
        "unit_rare", "rare_structure", 240, 6400, 5, 0.018,
        seed=613, k=4, regime="rare_structure",
    )

    G, sample_pop, pop_freqs, _, admix = simulate_genotypes(spec)
    mac = G.sum(axis=1)
    rare = (mac >= 1) & (mac <= 20)
    p = mac.astype(np.float64) / (2.0 * spec.n_samples)
    maf_retained = np.minimum(p, 1.0 - p) >= 0.05
    labels = _label_subspace(sample_pop, spec.n_pops)
    rare_capture = _captured_fraction(_patterson_subspace(G[rare], spec.k), labels)
    full_capture = _captured_fraction(_patterson_subspace(G, spec.k), labels)
    maf_filtered_capture = _captured_fraction(
        _patterson_subspace(G[maf_retained], spec.k), labels
    )
    carrier_purity = []
    for row in G[rare]:
        carriers = np.flatnonzero(row)
        carrier_counts = np.bincount(sample_pop[carriers], minlength=spec.n_pops)
        carrier_purity.append(float(carrier_counts.max() / carriers.size))

    assert 0.64 <= rare.mean() <= 0.70
    assert mac[rare].min() == 1
    assert mac[rare].max() == 20
    assert rare_capture >= 0.95
    assert full_capture >= 0.95
    assert maf_filtered_capture <= 0.15
    assert np.mean(carrier_purity) >= 0.90
    assert _population_contrast_rank(pop_freqs) >= spec.k
    assert admix is None


def _decode_contract_gt(token: bytes) -> int:
    gt = token.split(b":", 1)[0].replace(b"|", b"/")
    if gt in {b"0", b"1"}:
        return 2 * int(gt)
    pieces = gt.split(b"/")
    if len(pieces) != 2 or any(allele not in {b"0", b"1"} for allele in pieces):
        return -1
    return int(pieces[0]) + int(pieces[1])


def test_variable_width_representation_preserves_every_dosage(tmp_path):
    spec = DatasetSpec(
        "unit_variable", "variable_width", 40, 120, 4, 0.08,
        seed=719, k=3, representation="variable_width",
    )
    expected, *_ = simulate_genotypes(spec)
    vcf = tmp_path / "variable.vcf"

    write_vcf(
        spec,
        vcf,
        representation_key=REPRESENTATION_KEY,
        math_commitment=MATH_COMMITMENT,
    )

    records = [line for line in vcf.read_bytes().splitlines() if not line.startswith(b"#")]
    formats = set()
    observed = np.empty_like(expected)
    lowercase_records = 0
    genotype_tokens = []
    widths = []
    for row_index, line in enumerate(records):
        fields = line.split(b"\t")
        formats.add(fields[8])
        lowercase_records += int(fields[3].islower() or fields[4].islower())
        widths.append(len(line))
        genotype_tokens.extend(token.split(b":", 1)[0] for token in fields[9:])
        observed[row_index] = [_decode_contract_gt(token) for token in fields[9:]]

    assert formats == {b"GT", b"GT:DP", b"GT:DP:GQ", b"GT:AD:DP:GQ"}
    assert lowercase_records >= 20
    assert max(widths) - min(widths) >= 200
    assert {b"0", b"1"} <= set(genotype_tokens)
    assert b"\r\n" in vcf.read_bytes()
    assert not vcf.read_bytes().endswith((b"\r", b"\n"))
    for observed_row, expected_row in zip(observed, expected, strict=True):
        assert (
            np.array_equal(np.sort(observed_row), np.sort(expected_row))
            or np.array_equal(np.sort(observed_row), np.sort(2 - expected_row))
        )


def test_haploid_representation_carries_identifiable_structure(tmp_path):
    spec = DatasetSpec(
        "unit_haploid", "haploid", 96, 1200, 5, 0.12,
        seed=727, k=4, regime="haploid_structure", representation="haploid_gt",
    )
    expected, *_ = simulate_genotypes(spec)
    assert set(np.unique(expected)) == {0, 2}
    vcf = tmp_path / "haploid.vcf"

    write_vcf(
        spec,
        vcf,
        representation_key=REPRESENTATION_KEY,
        math_commitment=MATH_COMMITMENT,
    )

    genotype_tokens = {
        token
        for line in vcf.read_bytes().splitlines() if line and not line.startswith(b"#")
        for token in line.split(b"\t")[9:]
    }
    assert genotype_tokens == {b"0", b"1"}

    from reference.full_scan_pca import fit

    sample_ids, scores, kept, spectrum = fit(vcf, spec.k)
    assert len(sample_ids) == spec.n_samples
    assert scores.shape == (spec.n_samples, spec.k)
    assert kept > 0
    assert np.all(np.isfinite(scores))
    assert np.all(structure_weights(spectrum, spec.k)[:spec.k] > 0)


def test_variable_width_contract_stress_matches_canonical_scientific_object(tmp_path):
    common = dict(
        name="unit_contract", category="variable_width", n_samples=48, n_variants=180,
        n_pops=4, fst=0.08, seed=811, k=3, missing_frac=0.08,
    )
    canonical_spec = DatasetSpec(**common, representation="plain_gt")
    stressed_spec = DatasetSpec(**common, representation="variable_width")
    canonical = tmp_path / "canonical.vcf"
    stressed = tmp_path / "stressed.vcf"

    write_vcf(
        canonical_spec,
        canonical,
        representation_key=REPRESENTATION_KEY,
        math_commitment=MATH_COMMITMENT,
    )
    write_vcf(
        stressed_spec,
        stressed,
        representation_key=REPRESENTATION_KEY,
        math_commitment=MATH_COMMITMENT,
    )

    from reference.full_scan_pca import fit

    ids_a, scores_a, kept_a, _ = fit(canonical, 3)
    ids_b, scores_b, kept_b, _ = fit(stressed, 3)
    assert ids_a == ids_b
    assert kept_a == kept_b
    assert subspace_accuracy(scores_a, scores_b, np.ones(3))["accuracy"] > 0.999999

    payload = stressed.read_bytes()
    assert b"\r\n" in payload
    assert not payload.endswith((b"\r", b"\n"))
    gt_tokens = {
        token.split(b":", 1)[0]
        for line in payload.splitlines() if line and not line.startswith(b"#")
        for token in line.split(b"\t")[9:]
    }
    assert {
        b"0", b"1", b"0/.", b"./1", b"2/2", b"0x1", b"1x0", b"0/",
        b"1/1/1", b"00", b"11", b"10/0",
    } <= gt_tokens


def test_runtime_representation_key_changes_bytes_not_exact_pca_spectrum(tmp_path):
    spec = DatasetSpec(
        "unit_rekey", "continental", 72, 2000, 5, 0.07,
        seed=913, k=4, missing_frac=0.08,
    )
    first = tmp_path / "first.vcf"
    second = tmp_path / "second.vcf"
    write_vcf(
        spec,
        first,
        representation_key=bytes(range(32)),
        math_commitment=MATH_COMMITMENT,
    )
    write_vcf(
        spec,
        second,
        representation_key=bytes(reversed(range(32))),
        math_commitment=MATH_COMMITMENT,
    )

    from reference.full_scan_pca import fit

    ids_a, _, kept_a, spectrum_a = fit(first, spec.k)
    ids_b, _, kept_b, spectrum_b = fit(second, spec.k)
    assert first.read_bytes() != second.read_bytes()
    assert ids_a != ids_b
    assert kept_a == kept_b
    np.testing.assert_allclose(spectrum_a, spectrum_b, rtol=2e-6, atol=2e-6)


def test_private_catalog_contains_bounded_custom_regimes():
    custom = {
        spec.category: spec for spec in grade_specs(MATH_KEY)
        if spec.regime != "balding_nichols"
    }

    assert set(custom) == {
        "biobank", "biobank_a", "biobank_b",
        "crossover", "haploid", "high_rank", "ill_conditioned", "io_wide",
        "sample_heavy", "spatial_ld", "rare_structure", "spectral_selection", "very_high_rank",
    }
    # The biobank family is a SCALE LADDER that turns memory from one all-or-nothing cliff into a
    # continuous axis. The marker-space decomposition fits at every rung; a sample-space Gram is
    # admitted or killed by the resident cap according to how disciplined it is, so credit grows
    # with the discipline. Pin the whole ladder so it cannot quietly stop biting if someone retunes
    # it. (rss=14 GiB is the aggregate-RSS watchdog; a float64 Gram is N^2*8 bytes, float32 N^2*4.)
    resident_limit = 14 * 1024 ** 3
    rungs = [custom["biobank_a"], custom["biobank_b"], custom["biobank"]]
    for r in rungs:
        # structure resolvable and well-posed at every rung, independent of sample count
        assert r.n_samples > 8 * r.n_variants
        assert r.n_variants * r.n_samples * 8 < resident_limit          # marker space fits easily
        assert r.n_variants * r.fst / r.n_pops >= 15.0                  # structure is resolvable
        assert r.k <= r.n_pops - 1
    # the ladder is strictly increasing in sample count -> strictly increasing Gram pressure
    assert rungs[0].n_samples < rungs[1].n_samples < rungs[2].n_samples
    # rung a: a float64 sample Gram FITS -> even a naive sample-space fit earns this rung
    assert custom["biobank_a"].n_samples ** 2 * 8 < resident_limit
    # rung b: float64 Gram is killed, but a float32 Gram still FITS -> only the float32-or-better
    # sample-space fit (or marker space) passes here
    assert custom["biobank_b"].n_samples ** 2 * 8 > resident_limit
    assert custom["biobank_b"].n_samples ** 2 * 4 < resident_limit
    # rung biobank: float32 Gram is killed too -> only the marker-space fit passes
    assert custom["biobank"].n_samples ** 2 * 4 > resident_limit
    biobank = custom["biobank"]

    assert custom["high_rank"].k == 36
    assert custom["haploid"].representation == "haploid_gt"
    assert custom["haploid"].weight == 0.4
    # Reweighting locked (2026-07-18): weight tracks Opus-discrimination + draw-variance. Up the
    # systems-limited and high-variance folds, down the aced no-headroom haploid; variable_width
    # stays 1.2 (parser axis). biobank ladder keeps a<b<main. Demigod (~1.0 everywhere) is
    # unaffected; only the frontier moves.
    _W = {s.name: s.weight for s in grade_specs(MATH_KEY)}
    assert _W["grade_continental_a"] == 1.8 and _W["grade_continental_b"] == 1.8
    assert _W["grade_subtle_a"] == 1.8 and _W["grade_subtle_b"] == 1.8
    assert _W["grade_scaling"] == 1.8
    assert _W["grade_io_wide"] == 2.5 and _W["grade_sample_heavy"] == 2.5
    assert _W["grade_biobank_a"] == 0.8 < _W["grade_biobank_b"] == 1.6 < _W["grade_biobank"] == 2.5
    assert custom["haploid"].k <= custom["haploid"].n_pops - 1
    assert custom["high_rank"].n_pops - 1 >= custom["high_rank"].k
    assert custom["ill_conditioned"].n_pops - 1 >= custom["ill_conditioned"].k
    assert custom["high_rank"].n_variants <= 80000
    assert 64 <= custom["very_high_rank"].k <= 96
    assert custom["very_high_rank"].n_pops - 1 >= custom["very_high_rank"].k
    assert custom["very_high_rank"].n_variants <= 100000
    assert custom["ill_conditioned"].n_variants <= 60000
    assert custom["spatial_ld"].n_variants <= 70000
    assert custom["rare_structure"].n_variants <= 70000
    assert custom["spectral_selection"].n_pops - 1 >= custom["spectral_selection"].k + 4
    assert custom["spectral_selection"].n_variants <= 70000
    assert custom["io_wide"].n_samples == 96
    assert custom["io_wide"].n_variants == 1_500_000
    assert custom["sample_heavy"].n_samples == 4096
    assert custom["sample_heavy"].n_variants == 1600
    assert custom["sample_heavy"].k == 8
    assert custom["crossover"].n_samples == custom["crossover"].n_variants
    assert custom["crossover"].k == 32

    variable = [
        spec for spec in grade_specs(MATH_KEY)
        if spec.representation == "variable_width"
    ]
    assert len(variable) == 1
    assert variable[0].category == "variable_width"
    assert variable[0].weight == 0.6
    assert variable[0].n_variants <= 70000
    assert variable[0].missing_frac > 0
    io_wide = [spec for spec in grade_specs(MATH_KEY) if spec.category == "io_wide"]
    assert len(io_wide) == 1
    assert io_wide[0].n_samples == 96
    assert io_wide[0].n_variants == 1_500_000
    assert io_wide[0].k == 8
    assert io_wide[0].regime == "spatial_ld"
    assert len(grade_specs(MATH_KEY)) == 21


def test_every_grade_fold_has_a_fair_fast_reference_speed_ceiling():
    """The fast reference sets systems_quality=1.0, so its speed ceiling must be REACHABLE without
    sacrificing accuracy on every fold. fast_pca reads at most ``DEFAULT_TARGET_VARIANTS`` markers
    (folds at or below that are read in full -> identical object to the full scan -> trivially fair).
    A fold that fast SUBSAMPLES is only fair if its structured axes survive that subsampling; the one
    regime where they would not is weak structure with many axes (aggressive subsampling drops the
    weakest high-index PCs -- measured in tests/test_fast_pca_solver.py). So pin the invariant:
    every fold fast subsamples (n_variants > budget) is either strongly structured (fst >= 0.05) or
    low-rank (k <= 8), never weak-and-high-rank. This closes the fast-budget provenance concern
    structurally -- the ceiling cannot become unfair on any deployed fold, and a future fold that
    would break it fails here rather than silently mis-anchoring the speed scale.
    """
    from reference.fast_pca import DEFAULT_TARGET_VARIANTS

    for spec in grade_specs(MATH_KEY):
        if spec.n_variants > DEFAULT_TARGET_VARIANTS:
            assert spec.fst >= 0.05 or spec.k <= 8, (
                f"{spec.name}: fast subsamples ({spec.n_variants} > {DEFAULT_TARGET_VARIANTS}) but it "
                f"is weak-and-high-rank (fst={spec.fst}, k={spec.k}); fast would lose weak axes and "
                f"the speed ceiling would be unfair"
            )


def test_grade_release_math_is_stable_but_private_release_key_changes_it():
    first = grade_specs(MATH_KEY)
    second = grade_specs(MATH_KEY)
    alternate = grade_specs(ALTERNATE_MATH_KEY)

    assert [spec.seed for spec in first] == [spec.seed for spec in second]
    assert [spec.seed for spec in first] != [spec.seed for spec in alternate]
