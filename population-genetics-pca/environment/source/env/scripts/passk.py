"""Report binary pass@k without discarding pcabench's continuous reward.

The platform already defines one reviewed binary status for each capability category.  This script
uses those statuses directly; it never invents a reward threshold after seeing model results.

Reported quantities have deliberately different names:

* ``pass@k``: probability that at least one of k sampled rollouts passes the complete reviewed
  category suite in a single rollout;
* ``category_mastery_coverage@k``: mean, over categories, of the probability that at least one of k
  rollouts masters that category.  This is useful decomposed signal, but it is not pass@k;
* ``expected_best_score@k``: expected best native continuous reward among k sampled rollouts.

The loader fails closed on incomplete category reports, duplicate rollout IDs, mixed evaluation
cohorts, and unknown terminal errors.  Model-controlled terminal failures count as all-category
failures.  Provider rejection and evaluator/infrastructure failures are disclosed and excluded
because they are not samples of model capability.

Usage:
    python scripts/passk.py run1.json run2.json
    python scripts/passk.py --allow-mixed-commits run1.json run2.json
"""
from __future__ import annotations

import argparse
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path


CATEGORY_PREFIX = "category:"
TERMINAL_PROBLEM_STATUSES = {"pass", "fail", "error"}
PROVIDER_ERROR_FRAGMENTS = (
    "rate limit",
    "provider reset",
    "provider rejected",
    "service unavailable",
    "model unavailable",
    "overloaded",
    "provider capacity",
    "unexpected capacity constraints",
    "stream idle timeout",
)
MODEL_FAILURE_FRAGMENTS = (
    "exceeded maximum runtime",
    "agent timed out",
    "agent timeout",
    "maximum turns",
    "token budget",
)
INFRA_ERROR_FRAGMENTS = (
    "invalid generated truth contract",
    "grader produced an incomplete",
    "grader produced an unconfined",
    "environment setup",
    "provisioning",
    "cloud-init",
)


@dataclass(frozen=True)
class Cohort:
    environment_id: str
    problem_id: str
    model: str
    instance_type: str
    commit_sha: str


@dataclass(frozen=True)
class Observation:
    rollout_id: str
    kind: str
    categories: Mapping[str, bool] | None
    score: float | None
    reason: str


@dataclass(frozen=True)
class Sample:
    rollout_id: str
    categories: Mapping[str, bool]
    score: float
    source_kind: str


@dataclass(frozen=True)
class Analysis:
    cohort: Cohort
    commits: tuple[str, ...]
    categories: tuple[str, ...]
    samples: tuple[Sample, ...]
    excluded_provider: tuple[Observation, ...]
    excluded_infrastructure: tuple[Observation, ...]


def _comb(n: int, k: int) -> int:
    return math.comb(n, k) if 0 <= k <= n else 0


def pass_at_k(n: int, successes: int, k: int) -> float:
    """Unbiased probability that k draws contain at least one of ``successes`` successes."""
    if n < 1:
        raise ValueError("n must be positive")
    if not 0 <= successes <= n:
        raise ValueError("successes must lie in [0, n]")
    if not 1 <= k <= n:
        raise ValueError("k must lie in [1, n]")
    if n - successes < k:
        return 1.0
    return 1.0 - _comb(n - successes, k) / _comb(n, k)


def expected_best_score_at_k(scores: Sequence[float], k: int) -> float:
    """Expected maximum score under k draws without replacement from observed rollouts."""
    n = len(scores)
    if not 1 <= k <= n:
        raise ValueError("k must lie in [1, len(scores)]")
    ordered = sorted(float(score) for score in scores)
    if any(not math.isfinite(score) or not 0.0 <= score <= 1.0 for score in ordered):
        raise ValueError("scores must be finite and in [0, 1]")
    denominator = _comb(n, k)
    return sum(
        score * _comb(index, k - 1) / denominator
        for index, score in enumerate(ordered)
        if index >= k - 1
    )


def _required_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"run is missing nonempty {field}")
    return value


def _finite_score(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    score = float(value)
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError(f"{field} must be finite and in [0, 1]")
    return score


def cohort_for_run(run: Mapping[str, object]) -> Cohort:
    compute = run.get("compute")
    compute_map = compute if isinstance(compute, Mapping) else {}
    return Cohort(
        environment_id=_required_text(run.get("environmentId"), "environmentId"),
        problem_id=_required_text(run.get("problemId"), "problemId"),
        model=_required_text(run.get("resolvedModel") or run.get("requestedModel"), "model"),
        instance_type=_required_text(
            compute_map.get("instanceType") or run.get("instanceType"), "instance type",
        ),
        commit_sha=_required_text(run.get("commitSha"), "commitSha"),
    )


def _category_statuses(rollout: Mapping[str, object]) -> dict[str, bool] | None:
    raw_results = rollout.get("testResults")
    if raw_results is None:
        return None
    if not isinstance(raw_results, list):
        raise ValueError("testResults must be a list")
    categories: dict[str, bool] = {}
    for result in raw_results:
        if not isinstance(result, Mapping):
            raise ValueError("testResults entries must be objects")
        test_id = result.get("testId")
        if not isinstance(test_id, str) or not test_id.startswith(CATEGORY_PREFIX):
            continue
        category = test_id[len(CATEGORY_PREFIX):]
        if not category or category in categories:
            raise ValueError(f"invalid or duplicate category result {test_id!r}")
        status = result.get("status")
        if status not in {"passed", "failed"}:
            raise ValueError(f"category {category!r} has non-binary status {status!r}")
        # Status decides mastery. Score is validated only to catch truncated/corrupt run records.
        _finite_score(result.get("score"), f"{test_id}.score")
        categories[category] = status == "passed"
    return categories or None


def _error_text(rollout: Mapping[str, object]) -> str:
    error = rollout.get("error")
    return error if isinstance(error, str) else ""


def _contains_any(text: str, fragments: Iterable[str]) -> bool:
    lowered = text.lower()
    return any(fragment in lowered for fragment in fragments)


def classify_rollout(rollout: Mapping[str, object]) -> Observation:
    rollout_id = _required_text(rollout.get("id"), "rollout id")
    categories = _category_statuses(rollout)
    if categories is not None:
        score = _finite_score(rollout.get("rolloutScore"), f"{rollout_id}.rolloutScore")
        return Observation(rollout_id, "evaluated", categories, score, "complete category report")

    problem_status = rollout.get("problemStatus")
    if problem_status not in TERMINAL_PROBLEM_STATUSES:
        raise ValueError(
            f"rollout {rollout_id} is not terminal (problemStatus={problem_status!r})",
        )
    error = _error_text(rollout)
    raw_results = rollout.get("testResults")
    results = raw_results if isinstance(raw_results, list) else []
    statuses = {
        result.get("status")
        for result in results
        if isinstance(result, Mapping)
    }

    if _contains_any(error, PROVIDER_ERROR_FRAGMENTS):
        return Observation(rollout_id, "provider", None, None, error or "provider failure")
    if _contains_any(error, MODEL_FAILURE_FRAGMENTS):
        return Observation(rollout_id, "model_failure", None, 0.0, error)
    if "errored" in statuses or _contains_any(error, INFRA_ERROR_FRAGMENTS):
        return Observation(
            rollout_id, "infrastructure", None, None, error or "evaluator returned an error",
        )

    grade_failures = [
        result for result in results
        if isinstance(result, Mapping)
        and result.get("testId") == "grade"
        and result.get("status") == "failed"
    ]
    if grade_failures or problem_status == "fail":
        return Observation(
            rollout_id,
            "model_failure",
            None,
            0.0,
            error or "submission failed before a complete category report",
        )

    raise ValueError(
        f"rollout {rollout_id} ended without category results and has an unclassified error: "
        f"{error!r}",
    )


def _read_run(path: str | Path) -> Mapping[str, object]:
    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path}: run JSON must be an object")
    return payload


def build_analysis(
    runs: Sequence[Mapping[str, object]],
    *,
    allow_mixed_commits: bool = False,
    expected_categories: Sequence[str] | None = None,
) -> Analysis:
    if not runs:
        raise ValueError("at least one run is required")
    cohorts = [cohort_for_run(run) for run in runs]
    base = cohorts[0]
    for cohort in cohorts[1:]:
        if (
            cohort.environment_id,
            cohort.problem_id,
            cohort.model,
            cohort.instance_type,
        ) != (
            base.environment_id,
            base.problem_id,
            base.model,
            base.instance_type,
        ):
            raise ValueError(f"mixed evaluation cohorts: {base!r} versus {cohort!r}")
        if not allow_mixed_commits and cohort.commit_sha != base.commit_sha:
            raise ValueError(
                "mixed commits are forbidden by default; audit compatibility and pass "
                "--allow-mixed-commits explicitly",
            )

    observations: list[Observation] = []
    seen_rollout_ids: set[str] = set()
    for run in runs:
        raw_rollouts = run.get("rollouts")
        if not isinstance(raw_rollouts, list):
            raise ValueError("run.rollouts must be a list")
        for raw_rollout in raw_rollouts:
            if not isinstance(raw_rollout, Mapping):
                raise ValueError("run.rollouts entries must be objects")
            observation = classify_rollout(raw_rollout)
            if observation.rollout_id in seen_rollout_ids:
                raise ValueError(f"duplicate rollout id {observation.rollout_id!r}")
            seen_rollout_ids.add(observation.rollout_id)
            observations.append(observation)

    evaluated = [observation for observation in observations if observation.kind == "evaluated"]
    declared = tuple(sorted(set(expected_categories or ())))
    if expected_categories is not None and len(declared) != len(expected_categories):
        raise ValueError("expected categories must be unique")
    if not declared:
        if not evaluated:
            raise ValueError("cannot infer the category contract without an evaluated rollout")
        declared = tuple(sorted(evaluated[0].categories or {}))
    if not declared:
        raise ValueError("the expected category set is empty")

    expected_set = set(declared)
    samples: list[Sample] = []
    for observation in observations:
        if observation.kind == "evaluated":
            actual = set(observation.categories or {})
            if actual != expected_set:
                missing = sorted(expected_set - actual)
                extra = sorted(actual - expected_set)
                raise ValueError(
                    f"rollout {observation.rollout_id} has an incomplete category contract; "
                    f"missing={missing}, extra={extra}",
                )
            samples.append(Sample(
                rollout_id=observation.rollout_id,
                categories=dict(observation.categories or {}),
                score=float(observation.score),
                source_kind="evaluated",
            ))
        elif observation.kind == "model_failure":
            samples.append(Sample(
                rollout_id=observation.rollout_id,
                categories={category: False for category in declared},
                score=0.0,
                source_kind="model_failure",
            ))

    if not samples:
        raise ValueError("no model-capability samples remain after exogenous failures are excluded")
    return Analysis(
        cohort=base,
        commits=tuple(sorted({cohort.commit_sha for cohort in cohorts})),
        categories=declared,
        samples=tuple(samples),
        excluded_provider=tuple(
            observation for observation in observations if observation.kind == "provider"
        ),
        excluded_infrastructure=tuple(
            observation for observation in observations if observation.kind == "infrastructure"
        ),
    )


def metrics(analysis: Analysis) -> dict[str, object]:
    samples = analysis.samples
    categories = analysis.categories
    n = len(samples)
    category_successes = {
        category: sum(bool(sample.categories[category]) for sample in samples)
        for category in categories
    }
    suite_successes = sum(
        all(sample.categories[category] for category in categories)
        for sample in samples
    )
    return {
        "cohort": asdict(analysis.cohort),
        "commits": list(analysis.commits),
        "n_model_samples": n,
        "n_evaluated": sum(sample.source_kind == "evaluated" for sample in samples),
        "n_model_failures": sum(sample.source_kind == "model_failure" for sample in samples),
        "excluded_provider": [asdict(item) for item in analysis.excluded_provider],
        "excluded_infrastructure": [asdict(item) for item in analysis.excluded_infrastructure],
        "categories": list(categories),
        "category_successes": category_successes,
        "suite_successes": suite_successes,
        "pass_at_k": {
            str(k): pass_at_k(n, suite_successes, k)
            for k in range(1, n + 1)
        },
        "category_mastery_coverage_at_k": {
            str(k): sum(
                pass_at_k(n, category_successes[category], k)
                for category in categories
            ) / len(categories)
            for k in range(1, n + 1)
        },
        "expected_best_score_at_k": {
            str(k): expected_best_score_at_k([sample.score for sample in samples], k)
            for k in range(1, n + 1)
        },
    }


def _print_report(result: Mapping[str, object]) -> None:
    n = int(result["n_model_samples"])
    categories = list(result["categories"])
    successes = dict(result["category_successes"])
    print(
        f"n = {n} model-capability samples; {len(categories)} reviewed categories; "
        f"suite passes = {result['suite_successes']}/{n}",
    )
    print(f"commits = {', '.join(result['commits'])}")
    print(
        f"excluded: provider={len(result['excluded_provider'])}, "
        f"infrastructure={len(result['excluded_infrastructure'])}\n",
    )
    print("category mastery (official platform status, c_i / n), hardest first:")
    for category in sorted(categories, key=lambda name: (successes[name], name)):
        print(f"  {category:22s} {successes[category]}/{n}")

    print("\nstrict suite pass@k (all reviewed categories mastered in one rollout):")
    for k, value in result["pass_at_k"].items():
        print(f"  pass@{k:<2s} = {value:.4f}")

    print("\ncategory_mastery_coverage@k (decomposed diagnostic; NOT pass@k):")
    for k, value in result["category_mastery_coverage_at_k"].items():
        print(f"  coverage@{k:<2s} = {value:.4f}")

    print("\nexpected_best_score@k (native continuous reward):")
    for k, value in result["expected_best_score_at_k"].items():
        print(f"  best_score@{k:<2s} = {value:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", help="hfdev run --json dumps")
    parser.add_argument(
        "--allow-mixed-commits",
        action="store_true",
        help="explicitly pool commits after a human compatibility audit",
    )
    parser.add_argument(
        "--expected-category",
        action="append",
        default=None,
        help="declare one expected category (repeatable); otherwise infer from a full report",
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args()

    analysis = build_analysis(
        [_read_run(path) for path in args.runs],
        allow_mixed_commits=args.allow_mixed_commits,
        expected_categories=args.expected_category,
    )
    result = metrics(analysis)
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        _print_report(result)


if __name__ == "__main__":
    main()
