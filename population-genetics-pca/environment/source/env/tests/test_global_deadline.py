import json
import subprocess
import sys
from pathlib import Path

import pytest

import grader.grade as grade
from data.release_key import RELEASE_ID, key_commitment


MATH_KEY = bytes(range(32))


def _raise_anchor_timeout(*_args, **_kwargs):
    raise subprocess.TimeoutExpired("reference anchor", 3)


class _AnchorTimeoutCache:
    """A reference cache whose anchor timing times out the way a budget-capped anchor does."""

    def get(self, vcf, k, workdir=None, *, isolation=None, timeout=None):
        raise subprocess.TimeoutExpired("reference anchor", timeout)


class _FakeBudget:
    def __init__(self, *, exhausted):
        self._exhausted = exhausted

    def invocation_timeout(self, _cap_seconds):
        return 3

    def exhausted(self):
        return self._exhausted


def test_sandboxed_anchor_timeout_is_a_deadline_signal_not_an_infra_erasure(monkeypatch, tmp_path):
    """Regression for a grade-erasing FN in the budget path.

    In production the reference anchors run through ``run_submission``, which reports a timeout as
    returncode 124 -- it does NOT raise ``subprocess.TimeoutExpired``. ``_time_anchor`` used to turn
    every nonzero code, timeout included, into a bare ``RuntimeError``. Neither deadline handler
    (``grade_one_dataset`` and ``_make_run_one``, both ``except subprocess.TimeoutExpired``) catches
    that, so when the global budget capped a late anchor's timeout low enough to actually time out,
    the ``RuntimeError`` propagated out of ``grade_suite`` and erased EVERY already-earned fold --
    exactly the outer-timeout erasure ``GradingBudget`` exists to prevent. The anchor timeout must
    surface as ``subprocess.TimeoutExpired`` so the callers can attribute it.
    """
    monkeypatch.setattr(grade, "run_submission", lambda *a, **k: {
        "returncode": 124, "stderr": "timeout", "seconds": 0.0,
        "peak_pss_bytes": 0, "peak_storage_bytes": 0,
    })
    monkeypatch.setattr(grade, "_reference_submission_dir", lambda module, dest: dest)

    with pytest.raises(subprocess.TimeoutExpired):
        grade.ReferenceCache._time_anchor(
            "reference.full_scan_pca", tmp_path / "x.vcf", 1, tmp_path,
            isolation="bwrap", timeout=3,
        )


def test_non_timeout_anchor_failure_stays_a_hard_infra_error(monkeypatch, tmp_path):
    """The fix must not swallow genuine anchor failures. A nonzero code that is NOT a timeout (a
    crash, an OOM kill, a policy exit) is an infrastructure fault in trusted grader work and must
    remain a ``RuntimeError`` -- never quietly reinterpreted as a spent budget or a submission zero.
    """
    monkeypatch.setattr(grade, "run_submission", lambda *a, **k: {
        "returncode": 137, "stderr": "killed", "seconds": 0.0,
        "peak_pss_bytes": 0, "peak_storage_bytes": 0,
    })
    monkeypatch.setattr(grade, "_reference_submission_dir", lambda module, dest: dest)

    with pytest.raises(RuntimeError, match="sandboxed reference anchor"):
        grade.ReferenceCache._time_anchor(
            "reference.full_scan_pca", tmp_path / "x.vcf", 1, tmp_path,
            isolation="bwrap", timeout=3,
        )


def test_scored_anchor_timeout_is_attributed_to_budget_then_infra(monkeypatch, tmp_path):
    """The scored-fold handler must read the anchor timeout as: spent budget -> a recoverable
    ``GradingDeadlineExhausted`` (grade_suite zero-fills the remaining folds and keeps the earned
    ones); healthy budget -> an infra ``RuntimeError`` (a reference genuinely could not finish).
    Neither may be charged to the submission.
    """
    monkeypatch.setattr(grade, "_normalize_vcf", lambda path, _workdir: path)
    src = tmp_path / "case.vcf"
    src.write_text("x")
    lib = {"name": "library_scan", "factor": 1.0, "severity": 0.0}

    with pytest.raises(grade.GradingDeadlineExhausted):
        grade.grade_one_dataset(
            tmp_path, src, {"spec": {"category": "subtle"}}, 1,
            _AnchorTimeoutCache(), tmp_path, lib=lib,
            isolation="bwrap", budget=_FakeBudget(exhausted=True),
        )

    with pytest.raises(RuntimeError, match="trusted reference exceeded"):
        grade.grade_one_dataset(
            tmp_path, src, {"spec": {"category": "subtle"}}, 1,
            _AnchorTimeoutCache(), tmp_path, lib=lib,
            isolation="bwrap", budget=_FakeBudget(exhausted=False),
        )


def test_probe_anchor_timeout_is_attributed_to_budget_then_infra(monkeypatch, tmp_path):
    """The probe reference-timing path has the same two-way attribution and must not erase results
    either: an exhausted budget makes the probe run raise ``GradingDeadlineExhausted`` (caught by
    ``run_probe_gates`` and reported as ``deadline_exhausted``); a healthy budget makes it infra.
    """
    monkeypatch.setattr(
        grade.ReferenceCache, "_time_anchor", staticmethod(_raise_anchor_timeout),
    )

    spent = grade._make_run_one(
        tmp_path, tmp_path, [], isolation="bwrap", budget=_FakeBudget(exhausted=True),
    )
    with pytest.raises(grade.GradingDeadlineExhausted):
        spent(tmp_path / "x.vcf", 1, tmp_path / "spent.tsv", ["s0", "s1"])

    healthy = grade._make_run_one(
        tmp_path, tmp_path, [], isolation="bwrap", budget=_FakeBudget(exhausted=False),
    )
    with pytest.raises(RuntimeError, match="trusted probe reference exceeded"):
        healthy(tmp_path / "x.vcf", 1, tmp_path / "healthy.tsv", ["s0", "s1"])


def test_grading_budget_reserves_finalization_time_and_caps_invocations():
    now = [100.0]
    budget = grade.GradingBudget(10.0, clock=lambda: now[0])

    assert budget.finalization_reserve == 5.0
    assert budget.invocation_timeout(100) == 5
    now[0] = 104.1
    assert budget.exhausted()
    with pytest.raises(grade.GradingDeadlineExhausted):
        budget.invocation_timeout(1)


def test_deadline_placeholder_is_complete_and_preserves_probe_shape(tmp_path):
    truth = {"spec": {"category": "subtle", "weight": 1.25, "k": 7}}
    probes = {
        name: {
            "name": name,
            "factor": 0.8,
            "severity": 0.2,
            "diagnostic": f"{name} evidence",
        }
        for name in grade._PROBE_GATE_NAMES
    }
    result = grade._deadline_dataset_result(
        tmp_path / "case.vcf", truth, 7,
        {"name": "library_scan", "factor": 1.0, "severity": 0.0},
        probes,
        {"submission_seconds": 3.0, "reference_seconds": 2.0},
        probes_enabled=True,
    )

    assert result["dataset"] == "case"
    assert result["weight"] == 1.25
    assert result["reward"] == result["accuracy"] == result["gate_product"] == 0.0
    assert result["sub_seconds"] == 3.0
    assert result["ref_seconds"] == 2.0
    assert result["fast_ref_seconds"] is None
    assert result["execution_overhead_seconds"] is None
    assert result["run"]["returncode"] == 124
    assert result["run"]["sampled_primary_invocation_peak_pss_bytes"] is None
    assert result["run"]["sampled_primary_invocation_peak_storage_bytes"] is None
    assert "peak_pss_bytes" not in result["run"]
    assert "peak_storage_bytes" not in result["run"]
    assert set(result["gates"]) == {
        "library_scan", "validity", *grade._PROBE_GATE_NAMES,
    }
    assert result["gates"]["validity"]["factor"] == 0.0
    assert result["gates"]["validity"]["reason"] == (
        "global grading deadline exhausted before this dataset"
    )
    assert result["gates"]["coverage"]["diagnostic"] == "coverage evidence"


def test_probe_deadline_preserves_finished_gate_and_stops_later_work(monkeypatch, tmp_path):
    from grader.gates import probes

    sentinel = object()
    monkeypatch.setattr(
        grade, "_make_run_one",
        lambda *args, **kwargs: sentinel,
    )
    carriers = {
        name: (probes.ProbeCarrier(128, 1_000, 3, "synthetic_plain"),)
        for name in grade._PROBE_GATE_NAMES
    }
    monkeypatch.setattr(probes, "select_probe_carriers", lambda *_args: carriers)

    calls = []

    def finished(*_args, **_kwargs):
        calls.append("hwe_norm")
        return {"name": "hwe_norm", "factor": 0.7, "severity": 0.3}

    def deadline(*_args, **_kwargs):
        calls.append("coverage")
        raise grade.GradingDeadlineExhausted("spent")

    def forbidden(*_args, **_kwargs):
        raise AssertionError("probe work continued after the global deadline")

    monkeypatch.setattr(probes, "hwe_norm_gate", finished)
    monkeypatch.setattr(probes, "coverage_gate", deadline)
    monkeypatch.setattr(probes, "representation_equivalence_gate", forbidden)

    gates, runtime = grade.run_probe_gates(
        tmp_path, [], tmp_path, MATH_KEY, isolation="bwrap",
    )

    assert calls == ["hwe_norm", "coverage"]
    assert gates["hwe_norm"]["factor"] == 0.7
    assert gates["coverage"]["factor"] == 0.0
    assert gates["representation_equivalence"]["factor"] == 0.0
    assert runtime["deadline_exhausted"] is True


def _truth(category: str) -> dict:
    return {
        "spec": {
            "category": category,
            "weight": 1.0,
            "k": 1,
            "representation": "plain_gt",
            "messy_frac": 0.0,
            "math_release": RELEASE_ID,
            "math_key_commitment": key_commitment(MATH_KEY),
        },
        "n_samples": 8,
        "n_variants": 16,
    }


def _measured_result(vcf: Path, truth: dict, k: int) -> dict:
    return {
        "dataset": vcf.stem,
        "category": truth["spec"]["category"],
        "k": k,
        "weight": 1.0,
        "reward": 0.8,
        "accuracy": 0.9,
        "time_quality": 0.7,
        "gate_product": 1.0,
        "sub_seconds": 4.0,
        "ref_seconds": 5.0,
        "gates": {
            "library_scan": {"factor": 1.0, "severity": 0.0},
            "validity": {"factor": 1.0, "severity": 0.0},
        },
        "run": {"returncode": 0, "stderr_tail": ""},
    }


def test_grade_suite_keeps_completed_score_and_zero_fills_unstarted_suffix(
    monkeypatch, tmp_path,
):
    data = tmp_path / "data"
    data.mkdir()
    for name, category in (("a", "first"), ("b", "second"), ("c", "second")):
        (data / f"{name}.vcf").write_text("reviewed fixture\n")
        (data / f"{name}.vcf.truth.json").write_text(json.dumps(_truth(category)))
    submission = tmp_path / "submission"
    submission.mkdir()
    (submission / "pca").write_text("#!/usr/bin/env python3\n")
    work = tmp_path / "work"
    work.mkdir()

    monkeypatch.setattr(grade, "_UNPRIV", (65534, 65534))
    monkeypatch.setattr(grade, "_BWRAP", sys.executable)
    monkeypatch.setattr(grade, "_RUNTIME_PYTHON", sys.executable)
    monkeypatch.setattr(grade.sys, "platform", "linux")
    monkeypatch.setattr(grade, "_seal_grader_package", lambda: None)
    monkeypatch.setattr(grade, "_assert_grader_package_sealed", lambda: None)
    monkeypatch.setattr(grade, "_protect_grader_directory", lambda *args, **kwargs: None)
    monkeypatch.setattr(grade, "_assert_execution_boundary", lambda *args, **kwargs: None)
    # Calibration runs a no-op submission through the real sandbox; this test fakes that sandbox.
    monkeypatch.setattr(grade, "measure_execution_overhead", lambda *args, **kwargs: 0.0)
    monkeypatch.setattr(
        grade.library_scan, "scan",
        lambda _path: {"name": "library_scan", "factor": 1.0, "severity": 0.0},
    )

    calls = []

    def grade_one(_snapshot, vcf, truth, k, *_args, **_kwargs):
        calls.append(vcf.stem)
        if len(calls) == 2:
            raise grade.GradingDeadlineExhausted("spent")
        return _measured_result(vcf, truth, k)

    monkeypatch.setattr(grade, "grade_one_dataset", grade_one)
    monkeypatch.setattr(grade, "_balanced_grade_case_order", lambda cases, _key: cases)

    result = grade.grade_suite(
        submission, data, work, isolation="bwrap", math_key=MATH_KEY,
        probes=False, time_budget_seconds=60,
    )

    assert calls == ["a", "b"]
    assert result["submission_status"] == "completed"
    assert [item["dataset"] for item in result["reward_detail"]["per_dataset"]] == [
        "a", "b", "c",
    ]
    assert [item["reward"] for item in result["reward_detail"]["per_dataset"]] == [
        0.8, 0.0, 0.0,
    ]
    assert result["reward_detail"]["category_scores"] == {
        "first": 0.8,
        "second": 0.0,
    }
    assert result["reward"] == 0.4
    budget = result["reward_detail"]["grading_budget"]
    assert budget["deadline_exhausted"] is True
    assert budget["evaluated_datasets"] == 1
    assert budget["total_datasets"] == 3


def test_private_case_order_is_stable_keyed_and_category_balanced(tmp_path):
    cases = []
    for name, category in (
        ("a1", "alpha"),
        ("a2", "alpha"),
        ("b1", "beta"),
        ("b2", "beta"),
        ("c1", "gamma"),
    ):
        cases.append((
            tmp_path / f"{name}.vcf",
            {"spec": {"category": category}},
            3,
        ))

    first = grade._balanced_grade_case_order(cases, MATH_KEY)
    repeated = grade._balanced_grade_case_order(cases, MATH_KEY)
    alternate = grade._balanced_grade_case_order(cases, bytes(reversed(MATH_KEY)))

    identity = lambda ordered: [case[0].name for case in ordered]
    categories = lambda ordered: [case[1]["spec"]["category"] for case in ordered]
    assert identity(first) == identity(repeated)
    assert identity(first) != identity(alternate)
    assert sorted(identity(first)) == sorted(identity(cases))
    # Every category receives one slot before alpha or beta receives its second slot.
    assert set(categories(first[:3])) == {"alpha", "beta", "gamma"}
    assert len(categories(first[:3])) == len(set(categories(first[:3])))


def test_cli_passes_private_time_budget_to_grade_suite(monkeypatch, capsys):
    captured = {}

    def fake_suite(*args, **kwargs):
        captured.update(kwargs)
        return {
            "submission_status": "completed",
            "reward": 0.0,
            "reward_detail": {"category_scores": {}, "per_dataset": []},
        }

    monkeypatch.setattr(grade, "read_math_key", lambda _path: MATH_KEY)
    monkeypatch.setattr(grade, "grade_suite", fake_suite)

    assert grade.main([
        "/submission",
        "--data-dir", "/data",
        "--workdir", "/work",
        "--math-key-file", "/secret/key",
        "--isolation", "bwrap",
        "--time-budget-seconds", "37.5",
    ]) == 0
    assert captured["time_budget_seconds"] == 37.5
    assert captured["isolation"] == "bwrap"
    assert captured["math_key"] == MATH_KEY
    assert "BENCHMARK REWARD: 0.0000" in capsys.readouterr().out
