import math
from pathlib import Path

import numpy as np
import pytest

from grader import grade
from grader.grade import (
    ReferenceCache,
    _SYSTEMS_UNLOCK_FULL,
    _SYSTEMS_UNLOCK_MIN,
    dataset_score,
    systems_unlock,
    effective_gate_product,
    measure_execution_overhead,
    rollup,
    systems_quality,
)


# Real per-fold anchors measured through the grader: (full-scan seconds, fast-reference seconds).
# The first group offers real speedup headroom; in the second the fast reference is at or BELOW
# parity, so a plain full scan is already the optimal strategy there.
_HEADROOM = {"crossover": (6.929, 2.473), "scaling": (15.611, 7.620), "io_wide": (5.933, 3.379)}
_NO_HEADROOM = {"ill_conditioned": (2.191, 2.473), "haploid": (3.220, 4.780),
                "variable_width": (7.502, 10.798)}


def test_a_perfect_solution_earns_the_full_ceiling_on_every_fold():
    """The load-bearing fairness property: matching the best ACHIEVABLE speed scores 1.0 -- on every
    fold, whatever its size. Without this the top of the scale is unearnable and even a flawless
    submission is capped below a reward of 1."""
    for full, fast in {**_HEADROOM, **_NO_HEADROOM}.values():
        best = min(full, fast)
        assert systems_quality(best, full, fast) == pytest.approx(1.0)
        # And a perfect fit then earns a perfect reward.
        assert dataset_score(1.0, systems_quality(best, full, fast), 1.0) == pytest.approx(1.0)


def test_a_large_shared_execution_overhead_cannot_block_the_ceiling():
    """Entering the sandbox costs ~2 s (setpriv/bwrap, the guard runner's seccomp install and eager
    numeric preload, staging) -- often more than the PCA itself on the smaller folds. Because the
    anchors are timed through that SAME path, that cost is common to all three times and must not
    stop a submission from reaching 1.0. (When the anchors were plain subprocesses and only the
    submission paid the tax, the fast reference's own code measured ~2.4x slower as a submission and
    the ceiling was unreachable -- the bug this arrangement exists to kill.)"""
    entry_cost = 2.0
    for full, fast in {**_HEADROOM, **_NO_HEADROOM}.values():
        t_full, t_fast = full + entry_cost, fast + entry_cost
        best = min(t_full, t_fast)
        # A submission whose algorithm matches the best achievable, paying the same entry cost.
        assert systems_quality(best, t_full, t_fast, entry_cost) == pytest.approx(1.0)
        # And it still earns a perfect reward.
        assert dataset_score(
            1.0, systems_quality(best, t_full, t_fast, entry_cost), 1.0,
        ) == pytest.approx(1.0)


def test_full_scan_is_parity_where_speed_is_winnable_and_optimal_where_it_is_not():
    # Where sampling genuinely wins, a correct-but-naive full scan sits at parity: you must skip work.
    for full, fast in _HEADROOM.values():
        assert systems_quality(full, full, fast) == pytest.approx(0.5)
    # Where sampling cannot win, the full scan IS the best strategy -- we must not punish a program
    # for failing to achieve a speedup that does not exist.
    for full, fast in _NO_HEADROOM.values():
        assert systems_quality(full, full, fast) == pytest.approx(1.0)


def test_systems_quality_is_continuous_in_the_measured_headroom():
    """No knife-edge: a hair of timing noise must not move a score across the mastery bar.

    This is the regression for a real defect. While the metric branched on
    ``headroom > 0.3``, a naive full scan scored 0.900 at a measured 1.2305x fold speedup and
    0.500 at 1.2315x -- a 0.4 cliff, straddling the 0.75 systems-mastery bar, decided by a 0.1%
    difference in a wall-clock measurement. The suite could not see it because every fixture in
    ``_NO_HEADROOM`` has the fast reference at or below parity (headroom exactly 0), so nothing
    exercised the 0 < headroom < 0.3 interval where the two branches disagreed.
    """
    full = 10.0
    # Sweep a full scan's score across folds whose achievable speedup crosses the old threshold.
    previous = None
    for speedup in np.linspace(1.0, 1.6, 601):
        value = systems_quality(full, full, full / speedup)
        if previous is not None:
            assert abs(value - previous) < 0.02, (
                f"systems_quality jumps at speedup={speedup:.4f}: {previous:.3f} -> {value:.3f}"
            )
        previous = value


def test_a_fold_offering_almost_no_speedup_still_ranks_the_full_scan_near_the_ceiling():
    """The intent of the old no-headroom branch, kept -- but earned in proportion, not by a jump."""
    full = 10.0
    # Nothing to win: the full scan IS optimal and takes the whole ceiling.
    assert systems_quality(full, full, full) == pytest.approx(1.0)
    # 1.05x available: the forfeit is proportional to the 5% it actually gave up, and a full scan
    # still comfortably clears the 0.75 systems-mastery bar.
    assert 0.93 < systems_quality(full, full, full / 1.05) < 0.98
    # 1.19x available (the measured `rare_structure` fold): a real, if modest, forfeit -- and NOT
    # the near-ceiling free pass the branching rule handed out for landing under the threshold.
    assert 0.55 < systems_quality(full, full, full / 1.19) < 0.70
    # 1.28x available (the measured `spatial_ld` fold): parity credit. Two folds this close in what
    # they actually offer must not be scored 0.4 apart, as they were.
    assert systems_quality(full, full, full / 1.28) == pytest.approx(0.5)
    # The pair that exposed the cliff, stated as the property that matters: neighbouring folds get
    # neighbouring scores.
    assert abs(systems_quality(full, full, full / 1.19)
               - systems_quality(full, full, full / 1.28)) < 0.20


def test_both_documented_calibrations_hold_and_agree_where_they_meet():
    """The metric has exactly two calibrations; neither may be quietly dropped for the other.

    "A full scan earns half" is 0.5/headroom per octave; the historical operational-zero rate is
    1/3 per octave. They agree at 1.5 octaves of headroom, which is what makes interpolating between
    them below that point a join rather than a fudge. The terminal clamp is softened separately.
    """
    full = 10.0
    # They meet: at 1.5 octaves of headroom the parity rule and the slowness rule coincide, so a
    # A submission 8x slower than optimal is an operational near-zero under either reading.
    best = full / 2 ** 1.5
    assert 0.0 < systems_quality(best * 8, full, best) < 0.05
    # The same operational near-zero holds on a fold with NO headroom.
    assert 0.0 < systems_quality(full * 8, full, full) < 0.05
    # And on such a fold, 2x slower than optimal is the documented 0.67 -- not a near-zero.
    assert systems_quality(full * 2, full, full) == pytest.approx(2 / 3, abs=0.02)


def test_a_fixed_shortfall_is_not_punished_hardest_where_the_fold_offers_least():
    """Regression for an inversion introduced while fixing the cliff.

    Flooring the headroom at 0.3 pinned the scale at its STEEPEST exactly where a fold offers the
    least to win, so the same 1.3x shortfall cost ~0.6 on a fold with no headroom and ~0.15 on a
    fold with a 2.3x prize. Measured, that inversion took the gold reference's `variable_width`
    score from 0.87 to 0.40 and cost it that category's mastery -- for being 1.3x slower than a
    plain full scan on the one fold where its sampler does not pay.
    """
    full = 10.0
    behind = math.log2(1.3)   # a 1.3x shortfall against the best achievable, held fixed
    def score_at(headroom_octaves: float) -> float:
        best = full / 2 ** headroom_octaves
        return systems_quality(best * 1.3, full, best)
    no_headroom = score_at(0.0)
    big_headroom = score_at(1.214)      # the measured `scaling` fold
    assert behind > 0
    # The same shortfall on a fold with nothing to win must not cost MORE than on a fold with a
    # large prize.
    assert no_headroom >= big_headroom - 0.05, (
        f"1.3x slow scores {no_headroom:.3f} where there is nothing to win but "
        f"{big_headroom:.3f} where there is a 2.3x prize"
    )
    # Specifically, the gold's real position on `variable_width` keeps its mastery.
    assert no_headroom >= 0.75


def test_the_scoring_scale_bounds_how_far_timing_noise_can_move_a_score():
    """``0.5/headroom`` diverges as headroom vanishes; the floor is what keeps noise survivable.

    The gold reference measured 0.912 rather than 1.0 on `ill_conditioned` purely because its
    submission run was ~5% slower than its own anchor run. That is the noise floor of a burstable
    box, and it must cost a bounded amount on any fold, including one that offers almost nothing.
    """
    full = 10.0
    noise = 1.05  # 5% slower than the best achievable, from measurement noise alone
    for speedup in (1.0, 1.01, 1.1, 1.19, 1.3, 2.0, 6.6):
        best = full / speedup
        loss = 1.0 - systems_quality(best * noise, full, best)
        assert loss <= 0.12, f"5% noise costs {loss:.3f} at speedup={speedup}"


def test_slower_than_optimal_decays_and_is_bounded():
    full, fast = _HEADROOM["scaling"]
    best = min(full, fast)
    # Monotone decreasing in the submission's own time, always in [0, 1].
    previous = 1.01
    for factor in (1.0, 1.5, 2.0, 4.0, 8.0, 64.0):
        value = systems_quality(best * factor, full, fast)
        assert 0.0 <= value <= 1.0
        assert value <= previous
        previous = value
    # The former 8x hard zero is now an operational near-zero, not a flat cliff.
    assert 0.0 < systems_quality(best * 8, full, fast) < 0.05
    assert systems_quality(best * 64, full, fast) < systems_quality(best * 8, full, fast)
    # Beating the reference cannot exceed the ceiling.
    assert systems_quality(best / 10, full, fast) == 1.0


def test_systems_quality_survives_degenerate_and_startup_dominated_inputs():
    # A successfully measured fast reference may honestly tie the full scan (no headroom).
    assert systems_quality(2.0, 2.0, 2.0) == pytest.approx(1.0)
    # Startup baseline larger than the measured times cannot divide by zero or escape [0, 1].
    assert 0.0 <= systems_quality(0.5, 0.5, 0.5, 10.0) <= 1.0
    assert 0.0 <= systems_quality(0.5, 10.0, 0.6, 0.55) <= 1.0


@pytest.mark.parametrize("times", [
    (float("nan"), 2.0, 1.0, 0.0),
    (2.0, float("inf"), 1.0, 0.0),
    (2.0, 1.0, 0.0, 0.0),
    (2.0, 1.0, 0.5, -0.1),
])
def test_systems_quality_rejects_invalid_measurements(times):
    with pytest.raises(ValueError, match="times must be finite"):
        systems_quality(*times)


def test_overhead_calibration_failure_is_infrastructure_error(monkeypatch, tmp_path):
    monkeypatch.setattr(grade, "run_submission", lambda *args, **kwargs: {
        "returncode": 17,
        "stderr": "sealed runner unavailable",
        "seconds": 0.1,
    })

    with pytest.raises(RuntimeError, match="overhead calibration failed with code 17"):
        measure_execution_overhead(tmp_path, isolation="bwrap")


def test_failed_deployment_reports_zero_runtime_quality(monkeypatch, tmp_path):
    source = tmp_path / "case.vcf"
    source.write_text("unused")
    monkeypatch.setattr(grade, "_normalize_vcf", lambda path, _workdir: path)
    monkeypatch.setattr(grade, "run_submission", lambda *args, **kwargs: {
        "returncode": 126,
        "stderr": "runtime policy rejection",
        "seconds": 0.01,
        "peak_pss_bytes": 1024,
        "peak_storage_bytes": 0,
    })

    class Cache:
        def get(self, *args, **kwargs):
            return {
                "sample_ids": ["s0", "s1"],
                "scores": np.array([[1.0], [-1.0]]),
                "weights": np.ones(1),
                "spectrum": np.ones(1),
                "kept": 10,
                "seconds": 5.0,
                "fast_seconds": 1.0,
            }

    result = grade.grade_one_dataset(
        tmp_path, source, {"spec": {"category": "subtle"}}, 1, Cache(), tmp_path,
        lib={"name": "library_scan", "factor": 1.0, "severity": 0.0},
        isolation="bwrap",
    )

    assert result["run"]["returncode"] == 126
    assert result["gates"]["validity"]["factor"] == 0.0
    assert result["gates"]["validity"]["reason"] == "nonzero exit 126"
    assert result["time_quality"] == 0.0
    assert result["reward"] == 0.0


def test_fast_anchor_failure_cannot_be_relabelled_as_no_speedup(monkeypatch, tmp_path):
    vcf = tmp_path / "case.vcf"
    vcf.write_text("placeholder")
    monkeypatch.setattr(grade, "full_scan_fit", lambda path, k: (
        ["s0", "s1"],
        np.array([[1.0], [-1.0]]),
        1,
        np.array([2.0]),
    ))
    monkeypatch.setattr(grade, "structure_weights", lambda spectrum, width: np.ones(width))
    calls = []

    def time_anchor(module, *args, **kwargs):
        calls.append(module)
        if module == "reference.fast_pca":
            raise RuntimeError("fast anchor crashed")
        return 2.0

    monkeypatch.setattr(ReferenceCache, "_time_anchor", staticmethod(time_anchor))

    with pytest.raises(RuntimeError, match="fast anchor crashed"):
        ReferenceCache().get(vcf, 1, tmp_path, isolation="bwrap")
    assert calls == ["reference.full_scan_pca", "reference.fast_pca"]


def test_reference_cache_never_reuses_one_folds_reference_for_another(monkeypatch, tmp_path):
    """The cache key is the physical INPUT FILE (path+dev+ino+size+mtime+ctime) plus k plus
    isolation -- not any structural spec field. The biobank memory ladder (biobank_a/biobank_b/
    biobank) shares regime and structure params and differs only in n_samples, so it is exactly the
    case where a spec-derived key could collide. Distinct folds are distinct files, so they MUST get
    distinct references; reusing one fold's reference scores/anchor times for another would corrupt
    both accuracy AND systems_quality. A repeat lookup of the same (file, k, isolation) must hit the
    cache, and a change in k or isolation must force a fresh reference (both change the truth/anchor).
    """
    monkeypatch.setattr(grade, "full_scan_fit", lambda path, k: (
        ["s0", "s1"], np.array([[1.0], [-1.0]]), 1, np.array([2.0]),
    ))
    monkeypatch.setattr(grade, "structure_weights", lambda spectrum, width: np.ones(width))
    calls = []

    def time_anchor(module, vcf, k, workdir, *, isolation, timeout):
        calls.append((Path(vcf).name, k, isolation))
        return 2.0

    monkeypatch.setattr(ReferenceCache, "_time_anchor", staticmethod(time_anchor))

    # Three ladder folds: same k/regime, different files (different content and size).
    a = tmp_path / "grade_biobank_a.vcf"; a.write_text("A" * 100)
    b = tmp_path / "grade_biobank_b.vcf"; b.write_text("B" * 200)
    c = tmp_path / "grade_biobank.vcf"; c.write_text("C" * 300)
    cache = ReferenceCache()
    refs = {f: cache.get(f, 5, tmp_path, isolation="bwrap") for f in (a, b, c)}

    # Distinct references, not aliases of one another.
    assert len({id(refs[f]["scores"]) for f in (a, b, c)}) == 3
    assert len(cache._store) == 3
    anchors_after_three = len(calls)

    # A repeat lookup of an identical (file, k, isolation) is a pure cache hit -- no re-timing.
    cache.get(a, 5, tmp_path, isolation="bwrap")
    assert len(calls) == anchors_after_three

    # k and isolation are both part of the key: changing either yields a fresh reference.
    cache.get(a, 6, tmp_path, isolation="bwrap")
    cache.get(a, 5, tmp_path, isolation="verifier-container")
    assert len(cache._store) == 5


def test_grade_one_dataset_timings_are_reported_on_a_common_reproducible_basis(
    monkeypatch, tmp_path,
):
    """End-to-end guard on the speed-score accounting: the per-probe runtime SHARE is ADDED to all
    three timings (submission share to the submission, reference share to BOTH reference anchors),
    the shared sandbox-entry overhead is SUBTRACTED inside systems_quality, and the fields the
    artifact publishes must feed straight back into systems_quality to reproduce the reported
    time_quality. This catches a sign error (probe share subtracted, or overhead added), a
    double-subtraction, or a mixed basis -- any of which would bias every speed score.
    """
    monkeypatch.setattr(grade, "_normalize_vcf", lambda path, _workdir: path)
    monkeypatch.setattr(grade, "subspace_accuracy", lambda S, ref, w: {"accuracy": 1.0})
    monkeypatch.setattr(grade, "per_pc_report", lambda S, ref, spectrum: {})

    def fake_run(submission_dir, vcf, k, out, *, isolation, timeout):
        Path(out).write_text("sample_id\tPC1\ns0\t1.0\ns1\t-1.0\n")
        return {"returncode": 0, "stderr": "", "seconds": 3.0,
                "peak_pss_bytes": 0, "peak_storage_bytes": 0}

    monkeypatch.setattr(grade, "run_submission", fake_run)

    class Cache:
        def get(self, *args, **kwargs):
            return {"sample_ids": ["s0", "s1"], "scores": np.array([[1.0], [-1.0]]),
                    "weights": np.ones(1), "spectrum": np.ones(1), "kept": 10,
                    "seconds": 4.0, "fast_seconds": 2.0}

    src = tmp_path / "case.vcf"
    src.write_text("x")
    res = grade.grade_one_dataset(
        tmp_path, src, {"spec": {"category": "subtle"}}, 1, Cache(), tmp_path,
        lib={"name": "library_scan", "factor": 1.0, "severity": 0.0},
        probe_runtime_share={"submission_seconds": 1.2, "reference_seconds": 1.4},
        isolation="bwrap", execution_overhead=0.5,
    )

    # Probe share is ADDED (not subtracted): the submission and both references get slower, and the
    # reference share lands on BOTH anchors so they stay on one basis.
    assert res["sub_seconds"] == pytest.approx(3.0 + 1.2)
    assert res["ref_seconds"] == pytest.approx(4.0 + 1.4)
    assert res["fast_ref_seconds"] == pytest.approx(2.0 + 1.4)
    assert res["execution_overhead_seconds"] == pytest.approx(0.5)

    # The published fields reproduce the published score exactly -- the auditability property.
    reproduced = systems_quality(
        res["sub_seconds"], res["ref_seconds"], res["fast_ref_seconds"],
        overhead=res["execution_overhead_seconds"],
    )
    assert reproduced == pytest.approx(res["time_quality"], abs=1e-12)


def test_dataset_score_is_bounded_and_quality_dominant():
    assert dataset_score(0.0, 1.0, 1.0) == 0.0
    assert dataset_score(1.0, 1.0, 1.0) == 1.0
    assert dataset_score(1.0, 0.5, 1.0) == pytest.approx(0.55)
    # Correctness alone -- whenever systems_quality reaches 0 -- earns the floor and no more. It
    # stays nonzero so a sound-but-slow fit remains
    # distinguishable from a fast wrong one; it stays small so it cannot pay for systems work that
    # was not done. A naive full scan measured 0.6841 while failing 16 of 18 categories under the
    # old 0.30.
    assert dataset_score(1.0, 0.0, 1.0) == pytest.approx(0.10)
    assert dataset_score(1.0, 0.0, 1.0) > 0.0
    # The climb from full-scan parity to the achievable ceiling is the gradient the task is about,
    # and it must dominate what correctness alone pays.
    assert (dataset_score(1.0, 1.0, 1.0) - dataset_score(1.0, 0.5, 1.0)) > dataset_score(1.0, 0.0, 1.0)
    # Unlimited speed cannot rescue a mediocre scientific fit: below the unlock band the systems
    # term is worth nothing at all, so a mediocre fit earns its share of the floor and no more.
    assert dataset_score(0.4, 1.0, 1.0) == pytest.approx(0.04)
    assert dataset_score(0.5, 0.0, 1.0) == dataset_score(0.5, 1.0, 1.0)
    assert dataset_score(1.0, 1.0, 0.5) == pytest.approx(0.5)


def test_speed_is_worthless_to_a_broken_pca_and_never_beats_a_correct_slow_fit():
    """The load-bearing property: a PCA that does not recover the structure has not computed the
    thing being timed, so its wall time measures nothing and must buy it nothing. A plain
    accuracy*speed product did not deliver this -- at accuracy 0.5 a maximally optimized program
    banked 0.50, five times a scientifically perfect but slow fit's 0.10."""
    correct_but_slow = dataset_score(1.0, 0.0, 1.0)

    for broken in (0.0, 0.2, 0.5, _SYSTEMS_UNLOCK_MIN):
        # Optimizing a wrong answer moves the score by exactly nothing.
        assert dataset_score(broken, 1.0, 1.0) == dataset_score(broken, 0.0, 1.0)
        # And however fast it is, it stays worse than being right and slow.
        assert dataset_score(broken, 1.0, 1.0) < correct_but_slow


def test_the_unlock_ramp_leaves_every_genuine_solve_whole():
    """`scripts/validate.py` pins 0.90 as the worst accuracy a genuine PCA may score (FN_MIN), and
    real implementations saturate at 1.000. The ramp must therefore be fully open by 0.90, or the
    gate aimed at impostors would quietly tax honest submissions instead."""
    assert _SYSTEMS_UNLOCK_FULL <= 0.90
    assert systems_unlock(0.90) == pytest.approx(1.0)
    # A genuine fast solve still earns the full ceiling.
    assert dataset_score(1.0, 1.0, 1.0) == pytest.approx(1.0)
    # The ramp is a ramp, not a cliff: accuracy in the band degrades smoothly and monotonically.
    band = np.linspace(_SYSTEMS_UNLOCK_MIN, _SYSTEMS_UNLOCK_FULL, 25)
    scores = [dataset_score(a, 1.0, 1.0) for a in band]
    assert all(b >= a for a, b in zip(scores, scores[1:]))


def test_systems_work_is_most_of_a_perfect_run():
    """The first line of the task is "Build a FAST population-genetics PCA". Correctness is the key
    that unlocks the systems term; the systems term is the prize."""
    perfect = dataset_score(1.0, 1.0, 1.0)
    correctness_alone = dataset_score(1.0, 0.0, 1.0)
    assert (perfect - correctness_alone) / perfect > 0.5


def test_scientific_method_failures_keep_small_partial_credit():
    perfect = {"factor": 1.0}
    zero = {"factor": 0.0}

    assert effective_gate_product({
        "library_scan": perfect,
        "validity": perfect,
        "hwe_norm": zero,
        "coverage": perfect,
        "representation_equivalence": perfect,
    }, "continental") == pytest.approx(0.15)
    assert effective_gate_product({
        "library_scan": perfect,
        "validity": perfect,
        "hwe_norm": perfect,
        "coverage": perfect,
        "representation_equivalence": zero,
    }, "variable_width") == pytest.approx(0.35)


def test_representation_fragility_is_local_to_parser_stress_categories():
    gates = {
        "library_scan": {"factor": 1.0},
        "validity": {"factor": 1.0},
        "hwe_norm": {"factor": 1.0},
        "coverage": {"factor": 1.0},
        "representation_equivalence": {"factor": 0.0},
    }

    assert effective_gate_product(gates, "continental") == 1.0
    assert effective_gate_product(gates, "messy") == pytest.approx(0.35)
    assert effective_gate_product(gates, "variable_width") == pytest.approx(0.35)


def test_integrity_failure_still_hard_zeros():
    assert effective_gate_product({
        "library_scan": {"factor": 0.0},
        "validity": {"factor": 1.0},
        "hwe_norm": {"factor": 1.0},
    }, "continental") == 0.0


def test_rollup_balances_categories_and_preserves_partial_credit():
    per_dataset = [
        {"category": "synthetic", "weight": 1.0, "reward": 0.9},
        {"category": "synthetic", "weight": 3.0, "reward": 0.5},
        {"category": "observed", "weight": 1.0, "reward": 0.2},
    ]

    out = rollup(per_dataset)

    assert out == {
        "benchmark": pytest.approx(7.0 / 15.0),
        "category_scores": {"synthetic": 0.6, "observed": 0.2},
    }


def test_rollup_scores_the_memory_ladder_as_independent_graduated_tiers():
    """The biobank memory ladder (commit "Turn the biobank memory cliff into a continuous scale
    ladder") is three SEPARATE single-dataset categories -- biobank_a (24k, weight 0.4),
    biobank_b (50k, weight 0.5), biobank (64k, weight 0.6). Each is its own capability axis, so
    rollup must:

      * weight each tier by its OWN weight (a heavier tier counts more), and
      * hand out graduated partial credit -- clearing the easy tier while OOMing the hard ones is
        strictly better than clearing nothing and strictly worse than clearing all.

    This pins the newly-added categories against a silent regression that would collapse the ladder
    back into an all-or-nothing cliff (e.g. by averaging tiers into one category or by weighting
    every category equally regardless of its declared weight).
    """
    def ladder(reward_a, reward_b, reward_c):
        return rollup([
            {"category": "biobank_a", "weight": 0.4, "reward": reward_a},
            {"category": "biobank_b", "weight": 0.5, "reward": reward_b},
            {"category": "biobank", "weight": 0.6, "reward": reward_c},
        ])

    # Each tier is its own category, scored at its own reward, weighted by its own weight.
    partial = ladder(1.0, 0.0, 0.0)
    assert partial["category_scores"] == {"biobank_a": 1.0, "biobank_b": 0.0, "biobank": 0.0}
    # Cross-category weighting is by each tier's own weight: 0.4/(0.4+0.5+0.6) of the ladder.
    assert partial["benchmark"] == pytest.approx(0.4 / 1.5, abs=1e-5)

    none = ladder(0.0, 0.0, 0.0)
    climbing = ladder(1.0, 0.5, 0.0)
    all_tiers = ladder(1.0, 1.0, 1.0)
    # Graduated, monotone credit: clearing more (and harder, heavier) tiers scores strictly higher.
    assert none["benchmark"] < partial["benchmark"] < climbing["benchmark"] < all_tiers["benchmark"]
    assert all_tiers["benchmark"] == pytest.approx(1.0)
    # The heaviest tier (biobank, 0.6) moves the benchmark more than the lightest (biobank_a, 0.4).
    assert ladder(0.0, 0.0, 1.0)["benchmark"] > ladder(1.0, 0.0, 0.0)["benchmark"]


def test_the_speed_scale_does_not_saturate_into_a_sign_bit():
    """A metric whose ordinary variation saturates its scale is a sign bit, not a measurement.

    This is the regression for a defect an Opus rollout (run_019f6688) exposed: ELEVEN datasets
    returned time_quality EXACTLY 0.000 while the submission ran only 2-3x slower than the plain
    full scan. Anchoring solely on parity gives a rate of 0.5/headroom, and a straight line through
    0.5 at parity reaches zero at 2*headroom -- which on the deployed folds put the zero at 1.65x
    slower than optimal (`spatial_ld`), 1.75x (`ill_conditioned`) and 2.4x (`admixed`). The env could
    not distinguish "unoptimized" from "catastrophic", and scored Opus below a naive full scan for
    work that was merely slow.

    The documented "8x slower earns nothing" was silently true only where 0.5/h == 1/3 (h = 1.5).
    """
    full = 10.0
    # Every fold with real-but-modest headroom must still resolve a 3x-off-optimal submission.
    for headroom in (0.36, 0.405, 0.62, 0.825, 1.214):
        best = full / 2 ** headroom
        three_x_off = systems_quality(best * 3, full, best)
        assert 0.05 < three_x_off < 0.45, (
            f"headroom={headroom}: a 3x-off-optimal submission scores {three_x_off:.3f}; the scale "
            f"has saturated and can no longer measure speed"
        )
        # ...and must still be strictly ordered against something genuinely worse.
        assert three_x_off > systems_quality(best * 6, full, best)


def test_parity_is_exact_and_the_former_zero_is_small_positive_on_every_fold():
    """Parity stays exact while catastrophic successful runtimes retain a tiny ordering signal."""
    full = 10.0
    for headroom in (0.3, 0.405, 0.62, 0.825, 1.214, 1.5):
        best = full / 2 ** headroom
        # "a full scan earns half" -- exactly, on every fold with real headroom
        assert systems_quality(full, full, best) == pytest.approx(0.5), f"parity broke at h={headroom}"
        former_zero = systems_quality(best * 8, full, best)
        assert 0.0 < former_zero < 0.05, f"8x operational zero broke at h={headroom}"
        assert systems_quality(best * 40, full, best) < former_zero


def test_folds_with_large_headroom_keep_their_useful_linear_region():
    """Softening the terminal clamp must not materially move ordinary above-zero scores.

    `biobank` (headroom 2.784 octaves) is the deployed instance; its useful linear region must remain
    effectively unchanged because parity is the binding calibration there -- a full scan is itself
    more than 8x behind optimal.
    """
    full = 10.0
    headroom = 2.784
    best = full / 2 ** headroom
    for behind in (0.0, 1.0, 2.0, 2.784, 4.0):
        expected = max(0.0, 1.0 - 0.5 * behind / headroom)
        assert systems_quality(best * 2 ** behind, full, best) == pytest.approx(expected, abs=3e-4)


def test_catastrophic_successful_solvers_remain_strictly_ordered():
    """A 5.5x eigensolver improvement must not disappear into the correctness floor.

    These are retained ``biobank_a`` measurements: both fits were exact and gate-clean, but the
    126-second iterative solve and 700-second dense solve formerly received identical reward 0.10.
    """
    full, fast = 17.492, 3.550
    iterative = systems_quality(126.460, full, fast)
    dense = systems_quality(699.841, full, fast)
    assert 0.0 < dense < iterative < 0.05
    iterative_reward = dataset_score(1.0, iterative, 1.0)
    dense_reward = dataset_score(1.0, dense, 1.0)
    assert iterative_reward > dense_reward > 0.10
    # The distinction must survive the legacy display precision too; merely differing at 1e-8
    # does not provide a useful RL signal.
    assert round(iterative_reward, 5) > round(dense_reward, 5) > 0.10


def test_rollup_preserves_full_precision_for_the_successful_runtime_tail():
    tiny_success = 0.10000004
    result = rollup([
        {"category": "slow", "weight": 1.0, "reward": tiny_success},
    ])

    assert result["category_scores"]["slow"] == pytest.approx(tiny_success, abs=1e-12)
    assert result["benchmark"] == pytest.approx(tiny_success, abs=1e-12)


def test_reported_timings_reproduce_time_quality():
    """The artifact must be able to check itself: recompute the score from the fields it publishes.

    `sub_seconds`, `ref_seconds` and `fast_ref_seconds` are the three inputs to `systems_quality`,
    so a reader must be able to feed them back in and land on the reported `time_quality`. If they
    cannot, the number is unfalsifiable no matter how many fields we print -- which is exactly how a
    saturated scale survived: the anchor was computed, reported, and still useless.

    This caught a real defect in the fix that added the anchor. `fast_ref_seconds` was published RAW
    while `sub_seconds`/`ref_seconds` were published SCORED (probe share included), so recomputation
    came out systematically low -- 0.305 against a reported 0.594 on `high_rank` -- and the reader
    could not tell whether the grader or their own arithmetic was at fault. Mixed bases are not a
    rounding issue; they are an unreproducible score wearing an auditable one's clothes.
    """
    # A fold with real headroom, exercised the way grade_one_dataset assembles the numbers.
    raw_sub, raw_ref, raw_fast = 6.0, 4.0, 2.0
    probe_sub, probe_ref, entry = 1.2, 1.4, 0.5

    scored_sub = raw_sub + probe_sub
    scored_ref = raw_ref + probe_ref
    scored_fast = raw_fast + probe_ref          # the basis the artifact must publish

    expected = systems_quality(scored_sub, scored_ref, scored_fast, overhead=entry)
    # Exactly what a reader does with the published fields, and nothing else.
    reproduced = systems_quality(scored_sub, scored_ref, scored_fast, overhead=entry)
    assert reproduced == pytest.approx(expected)

    # The bug this pins: publishing the RAW fast anchor alongside SCORED neighbours does NOT
    # reproduce the score, and is silently wrong rather than loudly wrong.
    wrong = systems_quality(scored_sub, scored_ref, raw_fast + entry, overhead=entry)
    assert wrong != pytest.approx(expected), (
        "the raw and scored fast anchors coincide in this fixture, so it cannot detect the defect"
    )
    assert wrong < expected, "the mixed-basis error is the low-side one that was observed"
