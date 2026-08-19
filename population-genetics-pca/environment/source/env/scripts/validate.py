"""Validate the grader's two guarantees through the real grading path:

  * fn_resistance/  -> a genuine PCA must retain high accuracy and method factors;
  * cheats/         -> every known shortcut must remain within its reviewed low-credit ceiling.

Usage:  python scripts/validate.py [--data-dir data/dev]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from grader.grade import grade_suite, ReferenceCache
from data.release_key import read_math_key

FN_MIN = 0.90          # a genuine PCA must clear this on accuracy and method identity
CHEAT_BEST_CATEGORY_MAX = 0.15
FORMAT_FRAGILE_BENCHMARK_MAX = 0.40
FORMAT_FRAGILE_PARSER_CATEGORY_MAX = 0.05
FORMAT_FRAGILE_PARSER_CATEGORIES = ("messy", "variable_width")


def require_validation_categories(data_dir: Path) -> None:
    """Reject stale self-test bundles before running an expensive, inconclusive ladder."""
    categories: set[str] = set()
    for truth_path in data_dir.glob("*.truth.json"):
        payload = json.loads(truth_path.read_text())
        category = payload.get("spec", {}).get("category")
        if isinstance(category, str):
            categories.add(category)
    missing = sorted(set(FORMAT_FRAGILE_PARSER_CATEGORIES) - categories)
    if missing:
        raise RuntimeError(
            "validation data is stale; regenerate it with parser categories: "
            + ", ".join(missing)
        )


def select_validation_cases(
    genuine: list[Path], shortcuts: list[Path], requested: list[str] | None,
) -> tuple[list[Path], list[Path]]:
    """Select named ladder cases for focused maintainer reruns."""
    if not requested:
        return genuine, shortcuts
    wanted = set(requested)
    available = {path.parent.name for path in [*genuine, *shortcuts]}
    unknown = sorted(wanted - available)
    if unknown:
        raise RuntimeError("unknown validation case(s): " + ", ".join(unknown))
    return (
        [path for path in genuine if path.parent.name in wanted],
        [path for path in shortcuts if path.parent.name in wanted],
    )


def assess_shortcut(name: str, result: dict) -> tuple[bool, str, float]:
    """Apply the reviewed ceiling that matches a shortcut's actual failure mode.

    Wrong-object and undercoverage programs must remain weak in every category. The
    format-fragile fixture is different: it computes a genuine Patterson PCA on the clean
    representation it understands, so clean-category credit is legitimate. It must instead
    remain weak on both parser-stress categories and bounded overall.
    """
    categories = result["reward_detail"]["category_scores"]
    benchmark = float(result["reward"])

    if name == "format_fragile":
        missing = [category for category in FORMAT_FRAGILE_PARSER_CATEGORIES
                   if category not in categories]
        if missing:
            return False, f"missing_parser_categories={','.join(missing)}", float("inf")
        parser_best = max(float(categories[category])
                          for category in FORMAT_FRAGILE_PARSER_CATEGORIES)
        ok = (benchmark <= FORMAT_FRAGILE_BENCHMARK_MAX and
              parser_best <= FORMAT_FRAGILE_PARSER_CATEGORY_MAX)
        detail = (f"benchmark={benchmark:.3f}/{FORMAT_FRAGILE_BENCHMARK_MAX:.2f} "
                  f"parser_best={parser_best:.3f}/{FORMAT_FRAGILE_PARSER_CATEGORY_MAX:.2f}")
        return ok, detail, max(
            benchmark / FORMAT_FRAGILE_BENCHMARK_MAX,
            parser_best / FORMAT_FRAGILE_PARSER_CATEGORY_MAX,
        )

    best = max((float(score) for score in categories.values()), default=0.0)
    return (best <= CHEAT_BEST_CATEGORY_MAX,
            f"best_cat={best:.3f}/{CHEAT_BEST_CATEGORY_MAX:.2f}", best)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/dev")
    ap.add_argument("--workdir", default="/tmp/pcabench_validate")
    ap.add_argument("--math-key-file", required=True)
    ap.add_argument("--case", action="append", dest="cases")
    a = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    data_dir = Path(a.data_dir)
    require_validation_categories(data_dir)
    fn = sorted((root / "fn_resistance").glob("*/pca"))
    cheats = sorted((root / "cheats").glob("*/pca"))
    fn, cheats = select_validation_cases(fn, cheats, a.cases)
    shared_cache = ReferenceCache()   # full-scan reference is submission-independent
    math_key = read_math_key(a.math_key_file)

    print("=" * 70)
    print("FN-RESISTANCE (accuracy AND gates must be >= %.2f on every dataset):" % FN_MIN)
    # The "genuine PCA is never rejected" guarantee is about the ANTI-CHEAT layer -- accuracy
    # recovered and object-identity gates not firing -- NOT speed. A correct-but-simple full
    # scan legitimately earns a speed penalty; that must not read as a false rejection.
    fn_fail = []
    for pca in fn:
        r = grade_suite(pca.parent, data_dir,
                        Path(a.workdir) / pca.parent.name,
                        isolation="bwrap", math_key=math_key, cache=shared_cache)
        pds = r["reward_detail"]["per_dataset"]
        worst_acc = min((d["accuracy"] for d in pds), default=0.0)
        worst_gate = min((d["gate_product"] for d in pds), default=0.0)
        worst = min(worst_acc, worst_gate)
        flag = "OK " if worst >= FN_MIN else "FAIL"
        if worst < FN_MIN:
            fn_fail.append(pca.parent.name)
        print(f"  [{flag}] {pca.parent.name:24s} worst_accuracy={worst_acc:.3f} "
              f"worst_gate={worst_gate:.3f}")

    print("-" * 70)
    print("KNOWN SHORTCUTS (reviewed bounded-credit ceilings):")
    cheat_fail = []
    for pca in cheats:
        r = grade_suite(pca.parent, data_dir,
                        Path(a.workdir) / pca.parent.name,
                        isolation="bwrap", math_key=math_key, cache=shared_cache)
        accepted, policy_detail, leak_metric = assess_shortcut(pca.parent.name, r)
        flag = "OK " if accepted else "LEAK"
        if not accepted:
            cheat_fail.append((pca.parent.name, leak_metric))
        # show which gate/axis caught it (from continental row)
        pd = {d["dataset"]: d for d in r["reward_detail"]["per_dataset"]}
        any_ds = next(iter(pd.values()), {})
        gates = any_ds.get("gates", {})
        print(f"  [{flag}] {pca.parent.name:24s} {policy_detail} "
              f"gates={ {n: g['factor'] for n,g in gates.items()} }")

    print("=" * 70)
    ok = not fn_fail and not cheat_fail
    if fn_fail:
        print("FN-RESISTANCE FAILURES:", fn_fail)
    if cheat_fail:
        print("CHEAT LEAKS:", cheat_fail)
    print("RESULT:", "ALL GUARANTEES HOLD" if ok else "GUARANTEES VIOLATED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
