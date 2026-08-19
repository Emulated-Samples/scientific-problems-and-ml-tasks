import json

import pytest

from scripts.validate import (
    assess_shortcut,
    require_validation_categories,
    select_validation_cases,
)


def _result(reward, **categories):
    return {
        "reward": reward,
        "reward_detail": {"category_scores": categories},
    }


def test_wrong_object_shortcut_must_stay_below_ceiling_in_every_category():
    accepted, _, _ = assess_shortcut(
        "raw_covariance", _result(0.08, clean=0.15, messy=0.02)
    )
    assert accepted

    accepted, _, _ = assess_shortcut(
        "raw_covariance", _result(0.08, clean=0.151, messy=0.02)
    )
    assert not accepted


def test_format_fragile_keeps_legitimate_clean_category_credit():
    accepted, detail, _ = assess_shortcut(
        "format_fragile",
        _result(0.356, clean=0.467, messy=0.0, variable_width=0.04),
    )
    assert accepted
    assert "parser_best=0.040/0.05" in detail


def test_format_fragile_must_remain_bounded_overall():
    accepted, _, _ = assess_shortcut(
        "format_fragile",
        _result(0.401, clean=0.467, messy=0.0, variable_width=0.04),
    )
    assert not accepted


def test_format_fragile_must_fail_parser_stress_categories():
    accepted, _, _ = assess_shortcut(
        "format_fragile",
        _result(0.35, clean=0.467, messy=0.051, variable_width=0.04),
    )
    assert not accepted


def test_format_fragile_policy_fails_closed_if_parser_category_is_missing():
    accepted, detail, _ = assess_shortcut(
        "format_fragile", _result(0.35, clean=0.467, messy=0.0)
    )
    assert not accepted
    assert detail == "missing_parser_categories=variable_width"


def test_validation_catalog_requires_both_parser_stress_categories(tmp_path):
    for name, category in (("a", "messy"), ("b", "continental")):
        (tmp_path / f"{name}.truth.json").write_text(
            json.dumps({"spec": {"category": category}})
        )

    with pytest.raises(RuntimeError, match="variable_width"):
        require_validation_categories(tmp_path)


def test_validation_catalog_accepts_complete_parser_stress_pair(tmp_path):
    for name, category in (("a", "messy"), ("b", "variable_width")):
        (tmp_path / f"{name}.truth.json").write_text(
            json.dumps({"spec": {"category": category}})
        )

    require_validation_categories(tmp_path)


def test_validation_case_selection_is_exact_and_rejects_typos(tmp_path):
    genuine = [tmp_path / "good" / "pca"]
    shortcuts = [tmp_path / "bad" / "pca"]

    selected_genuine, selected_shortcuts = select_validation_cases(
        genuine, shortcuts, ["bad"]
    )
    assert selected_genuine == []
    assert selected_shortcuts == shortcuts

    with pytest.raises(RuntimeError, match="unknown validation case"):
        select_validation_cases(genuine, shortcuts, ["missing"])
