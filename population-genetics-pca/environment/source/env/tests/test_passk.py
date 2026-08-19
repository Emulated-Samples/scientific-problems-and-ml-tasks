import pytest

from scripts.passk import build_analysis, expected_best_score_at_k, metrics, pass_at_k


def category(name, status, score):
    return {"testId": f"category:{name}", "status": status, "score": score}


def rollout(identifier, results=None, *, score=None, status="fail", error=""):
    value = {
        "id": identifier,
        "problemStatus": status,
        "error": error,
    }
    if results is not None:
        value["testResults"] = results
    if score is not None:
        value["rolloutScore"] = score
    return value


def run(*rollouts, commit="a" * 40):
    return {
        "environmentId": "sc-pcabench",
        "problemId": "from_scratch_pca",
        "requestedModel": "claude-opus-4-8",
        "resolvedModel": "claude-opus-4-8",
        "compute": {"instanceType": "r5.2xlarge"},
        "commitSha": commit,
        "rollouts": list(rollouts),
    }


def test_strict_passk_is_not_average_category_coverage():
    analysis = build_analysis([
        run(
            rollout("r1", [category("a", "passed", 0.10), category("b", "failed", 0.99)], score=0.20),
            rollout("r2", [category("a", "passed", 0.80), category("b", "passed", 0.80)], score=0.90),
            rollout("r3", [category("a", "failed", 0.99), category("b", "passed", 0.10)], score=0.50),
        ),
    ])
    result = metrics(analysis)

    # Status, not a post-hoc score threshold, is the reviewed binary event.
    assert result["category_successes"] == {"a": 2, "b": 2}
    assert result["suite_successes"] == 1
    assert result["pass_at_k"] == pytest.approx({"1": 1 / 3, "2": 2 / 3, "3": 1})
    assert result["category_mastery_coverage_at_k"] == pytest.approx(
        {"1": 2 / 3, "2": 1, "3": 1},
    )
    assert result["expected_best_score_at_k"] == pytest.approx(
        {"1": (0.2 + 0.5 + 0.9) / 3, "2": 23 / 30, "3": 0.9},
    )


def test_incomplete_category_contract_fails_closed():
    with pytest.raises(ValueError, match="incomplete category contract"):
        build_analysis([
            run(
                rollout("r1", [category("a", "passed", 1), category("b", "passed", 1)], score=1),
                rollout("r2", [category("a", "passed", 1)], score=1),
            ),
        ])


def test_mixed_commits_require_explicit_human_audit():
    first = run(rollout("r1", [category("a", "failed", 0)], score=0), commit="a" * 40)
    second = run(rollout("r2", [category("a", "failed", 0)], score=0), commit="b" * 40)
    with pytest.raises(ValueError, match="mixed commits"):
        build_analysis([first, second])
    analysis = build_analysis([first, second], allow_mixed_commits=True)
    assert analysis.commits == ("a" * 40, "b" * 40)


def test_provider_failure_is_excluded_but_model_timeout_is_a_zero_sample():
    analysis = build_analysis([
        run(
            rollout("evaluated", [category("a", "passed", 0.8)], score=0.8),
            rollout(
                "provider",
                status="error",
                error="Claude Code rate limit rejected; provider reset",
            ),
            rollout("timeout", status="error", error="Exceeded maximum runtime (8h)"),
        ),
    ])
    result = metrics(analysis)
    assert result["n_model_samples"] == 2
    assert result["n_model_failures"] == 1
    assert len(result["excluded_provider"]) == 1
    assert result["category_successes"] == {"a": 1}
    assert result["suite_successes"] == 1
    assert result["pass_at_k"] == pytest.approx({"1": 0.5, "2": 1.0})
    assert result["expected_best_score_at_k"] == pytest.approx({"1": 0.4, "2": 0.8})


def test_generic_capacity_text_is_not_misclassified_as_provider_failure():
    """Only provider capacity is exogenous; a solver's own capacity error is not."""
    with pytest.raises(ValueError, match="unclassified error"):
        build_analysis([
            run(rollout("solver", status="error", error="matrix capacity exceeded")),
        ])
    analysis = build_analysis([
        run(
            rollout("evaluated", [category("a", "failed", 0)], score=0),
            rollout("provider", status="error", error="provider capacity unavailable"),
        ),
    ])
    assert len(analysis.excluded_provider) == 1

    analysis = build_analysis([
        run(
            rollout("evaluated", [category("a", "failed", 0)], score=0),
            rollout("provider", status="error", error="unexpected capacity constraints"),
        ),
    ])
    assert len(analysis.excluded_provider) == 1


def test_duplicate_rollout_ids_cannot_be_double_counted():
    duplicate = rollout("same", [category("a", "failed", 0)], score=0)
    with pytest.raises(ValueError, match="duplicate rollout id"):
        build_analysis([run(duplicate, duplicate)])


def test_estimators_validate_their_domains():
    assert pass_at_k(3, 0, 3) == 0
    assert pass_at_k(3, 3, 1) == 1
    assert expected_best_score_at_k([0.2, 0.5, 0.9], 2) == pytest.approx(23 / 30)
    with pytest.raises(ValueError):
        pass_at_k(0, 0, 1)
    with pytest.raises(ValueError):
        expected_best_score_at_k([0.5], 2)
