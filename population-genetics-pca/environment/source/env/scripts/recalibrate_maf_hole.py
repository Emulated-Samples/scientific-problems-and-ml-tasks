#!/usr/bin/env python3
"""Turnkey MAF-filter FN-hole check for pcabench (BOX-ONLY, PRINT-ONLY).

Grades a deliberate MAF<0.05-filter submission (forbidden by the prompt) AND the genuine
non-filtering reference through the REAL grader entrypoint (``grader.grade.grade_suite``) over the
generated 18-dataset synthetic grade suite, then prints, per dataset, accuracy vs the category
mastery floor and reward vs the non-filtering reference, and a PASS/FAIL verdict on the question:

    Does the current DGP already close the historical "illegal MAF filter costs nothing" hole?
    (i.e. does the mandatory rare_structure capability drop BELOW its mastery floors under a MAF
    filter, while the genuine non-filtering reference stays above them?)

It never edits any grader/threshold; it only prints the verdict for a human to act on.

Why box-only: ``grade_suite`` runs each submission through bwrap+setpriv+``pcasub`` and requires
Linux root, the pinned interpreter (``/opt/hyperfocal/pcabench/bin/python``), and a provisioned
sandbox. Run on a provisioned grading box, e.g.:

    sudo /opt/hyperfocal/pcabench/bin/python scripts/recalibrate_maf_hole.py \
        --math-key-file /opt/hyperfocal/pcabench-secrets/release-v3.math-key

No drift: SCORING is 100% grade_suite (same call validate.py uses). The MAF-filter submission
uses the format-complete reference decoder and exact full-scan PCA, changing only the addition of
a MAF>=0.05 filter. The genuine baseline is the shipped reference/fast_submission. Only the
mastery FLOORS are mirrored from
environment/src/reporting.ts (MASTERY_POLICIES) -- keep them in sync if that file changes.
"""

from __future__ import annotations

import argparse
import secrets
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from grader.grade import grade_suite, ReferenceCache          # noqa: E402
from data.release_key import read_math_key                    # noqa: E402

# --- MIRROR of environment/src/reporting.ts MASTERY_POLICIES (meanAccuracy, minimumAccuracy).
#     KEEP IN SYNC with that file. Only the accuracy floors are needed here. ------------------
MASTERY: dict[str, tuple[float, float]] = {
    "admixed": (0.96, 0.92), "continental": (0.97, 0.94), "crossover": (0.95, 0.90),
    "haploid": (0.95, 0.90), "high_rank": (0.95, 0.90),
    "ill_conditioned": (0.93, 0.87), "io_wide": (0.97, 0.92), "messy": (0.95, 0.90),
    "observed": (0.95, 0.90), "rare_structure": (0.94, 0.88),
    "sample_heavy": (0.97, 0.92), "scaling": (0.97, 0.92), "spatial_ld": (0.94, 0.88),
    "spectral_selection": (0.95, 0.90), "subtle": (0.94, 0.88), "variable_width": (0.95, 0.90),
    "very_high_rank": (0.93, 0.88),
}
# ``rare_structure`` is the dedicated low-frequency capability. ``subtle`` tests weak eigengaps
# and ``spectral_selection`` tests leading-rank selection; duplicating the MAF intervention across
# those categories would reduce task diversity. Since every category must reach mastery, failure
# here is sufficient to make the binary pass event reject an illegal MAF-filtered implementation.
FOCUS = ("rare_structure",)

# MAF-filter probe submission: format-complete exact Patterson PCA with one deliberate violation.
_MAF_FILTER_PCA = r'''#!/usr/bin/env python3
"""Exact, format-complete Patterson PCA that deliberately drops MAF<0.05 markers."""
import sys
from pathlib import Path
import numpy as np
from _pcacore import _decode_lines, _standardize_kept, gram_to_scores

path, k, out = Path(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3])
with path.open("rb") as fh:
    for raw in fh:
        if raw.startswith(b"#CHROM"):
            ids = [value.decode() for value in raw.rstrip(b"\r\n").split(b"\t")[9:]]
            break
    else:
        raise ValueError("no #CHROM header line found")
n = len(ids)
if not 1 <= k < n:
    raise ValueError("k must satisfy 1 <= k < number of samples")

gram = None
blocks = []
kept = 0
lines = []

def flush():
    global gram, kept
    matrix = _decode_lines(lines, n)
    lines.clear()
    if matrix is None:
        return
    missing = matrix < 0
    n_valid = matrix.shape[1] - missing.sum(axis=1)
    allele_count = matrix.sum(axis=1, dtype=np.int64) + missing.sum(axis=1)
    minor_count = np.minimum(allele_count, 2 * n_valid - allele_count)
    # THE ONE DELIBERATE VIOLATION: discard every marker below MAF 0.05.
    keep_maf = (n_valid > 1) & (minor_count >= 0.10 * n_valid)
    standardized = _standardize_kept(matrix[keep_maf])
    if standardized is None:
        return
    kept += standardized.shape[0]
    if gram is None:
        blocks.append(standardized)
        if kept > n:
            gram = np.zeros((n, n), dtype=np.float64)
            for block in blocks:
                gram += block.T @ block
            blocks.clear()
    else:
        gram += standardized.T @ standardized

with path.open("rb") as fh:
    for raw in fh:
        if raw[:1] == b"#":
            continue
        line = raw.rstrip(b"\r\n")
        if line:
            lines.append(line)
            if len(lines) >= 20000:
                flush()
if lines:
    flush()
if kept < 1:
    raise ValueError("MAF filter retained no informative markers")

if gram is None:
    design = np.vstack(blocks)
    _, singular_values, right = np.linalg.svd(design, full_matrices=False)
    rank_floor = float(singular_values[0]) * np.sqrt(32.0 * np.finfo(design.dtype).eps)
    if singular_values.size < k or float(singular_values[k - 1]) <= rank_floor:
        raise ValueError("filtered design has rank below requested k")
    scores = right[:k].T * (singular_values[:k] / np.sqrt(kept))[None, :]
    for component in range(k):
        column = scores[:, component]
        if column[np.argmax(np.abs(column))] < 0:
            scores[:, component] = -column
else:
    scores, _, _ = gram_to_scores(gram, kept, k)

with out.open("w") as fh:
    fh.write("sample_id\t" + "\t".join(f"PC{i + 1}" for i in range(k)) + "\n")
    for sample_id, row in zip(ids, scores, strict=True):
        fh.write(sample_id + "\t" + "\t".join(f"{value:.8g}" for value in row) + "\n")
'''


def _build_maf_submission(dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    pca = dest / "pca"
    pca.write_text(_MAF_FILTER_PCA)
    pca.chmod(0o755)
    shutil.copyfile(_REPO / "reference" / "pca_core.py", dest / "_pcacore.py")
    return dest


def _generate_grade_suite(out_dir: Path, math_key_file: Path) -> None:
    """Generate the synthetic grade suite exactly as environment/src/index.ts does."""
    rep_key = out_dir.parent / "representation.key"
    rep_key.write_bytes(secrets.token_bytes(32))
    subprocess.run(
        [sys.executable, "-m", "data.generate", "--suite", "grade",
         "--out-dir", str(out_dir), "--math-key-file", str(math_key_file),
         "--representation-key-file", str(rep_key)],
        cwd=str(_REPO), check=True,
    )
    rep_key.unlink(missing_ok=True)


def _per_category(per_dataset: list[dict]) -> dict[str, dict]:
    by_cat: dict[str, list[dict]] = {}
    for d in per_dataset:
        by_cat.setdefault(d["category"], []).append(d)
    out = {}
    for cat, rows in by_cat.items():
        accs = [float(r["accuracy"]) for r in rows]
        out[cat] = {"mean": sum(accs) / len(accs), "min": min(accs), "n": len(rows)}
    return out


def _passes_mastery(m: dict | None, floors: tuple[float, float]) -> bool:
    return m is not None and m["mean"] >= floors[0] and m["min"] >= floors[1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--math-key-file",
                    default=str(_REPO / "grader" / "private" / "release-v3.math-key"))
    ap.add_argument("--data-dir", default=None,
                    help="pre-generated grade suite; if omitted, one is generated in a tmp dir")
    ap.add_argument("--workdir", default="/tmp/pcabench_maf_hole")
    args = ap.parse_args()

    math_key_file = Path(args.math_key_file)
    math_key = read_math_key(str(math_key_file))
    genuine_dir = _REPO / "reference" / "fast_submission"
    if not (genuine_dir / "pca").is_file():
        print(f"ERROR: genuine baseline {genuine_dir}/pca missing", file=sys.stderr)
        return 2

    tmp = Path(tempfile.mkdtemp(prefix="maf_hole_"))
    try:
        data_dir = Path(args.data_dir) if args.data_dir else (tmp / "suite")
        if args.data_dir is None:
            data_dir.mkdir(parents=True, exist_ok=True)
            print(f"[generate] synthetic grade suite -> {data_dir}", file=sys.stderr)
            _generate_grade_suite(data_dir, math_key_file)

        maf_dir = _build_maf_submission(tmp / "maf_filter")
        cache = ReferenceCache()          # sealed mathematical reference is submission-independent
        workdir = Path(args.workdir)

        print("[grading] genuine non-filtering reference (reference/fast_submission) ...",
              file=sys.stderr)
        genuine = grade_suite(genuine_dir, data_dir, workdir / "genuine",
                              isolation="bwrap", math_key=math_key, cache=cache)
        print("[grading] MAF<0.05-filter probe ...", file=sys.stderr)
        maf = grade_suite(maf_dir, data_dir, workdir / "maf",
                          isolation="bwrap", math_key=math_key, cache=cache)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    g_pd = {d["dataset"]: d for d in genuine["reward_detail"]["per_dataset"]}
    m_pd = {d["dataset"]: d for d in maf["reward_detail"]["per_dataset"]}
    g_cat = _per_category(genuine["reward_detail"]["per_dataset"])
    m_cat = _per_category(maf["reward_detail"]["per_dataset"])

    # --- per-dataset table ---
    print("\n=== per-dataset: MAF-filter vs genuine non-filtering reference ===")
    print(f"{'dataset':>26} {'category':>18} {'floor(min)':>10} "
          f"{'maf_acc':>8} {'gen_acc':>8} {'maf_rew':>8} {'gen_rew':>8} {'acc<floor':>9}")
    for name in sorted(g_pd):
        gd, md = g_pd[name], m_pd.get(name, {})
        cat = gd["category"]
        floor_min = MASTERY.get(cat, (float("nan"), float("nan")))[1]
        maf_acc = float(md.get("accuracy", float("nan")))
        below = "YES" if maf_acc < floor_min else "no"
        print(f"{name:>26} {cat:>18} {floor_min:>10.2f} {maf_acc:>8.3f} "
              f"{float(gd['accuracy']):>8.3f} {float(md.get('reward', float('nan'))):>8.3f} "
              f"{float(gd['reward']):>8.3f} {below:>9}")

    # --- focus-category verdict ---
    print("\n=== RARE-STRUCTURE MASTERY (does the MAF filter lose the dedicated capability?) ===")
    print(f"{'category':>20} {'floors(mean/min)':>18} {'genuine(mean/min)':>20} "
          f"{'maf(mean/min)':>20} {'genuine':>8} {'maf':>8}")
    hole_open = []
    for cat in FOCUS:
        floors = MASTERY[cat]
        gm, mm = g_cat.get(cat), m_cat.get(cat)
        gen_pass = _passes_mastery(gm, floors)
        maf_pass = _passes_mastery(mm, floors)
        # The hole is CLOSED for this category iff the MAF filter fails mastery while genuine passes.
        if maf_pass or not gen_pass:
            hole_open.append(cat)

        def _fmt(m):
            return "n/a" if m is None else f"{m['mean']:.3f}/{m['min']:.3f}"
        print(f"{cat:>20} {f'{floors[0]:.2f}/{floors[1]:.2f}':>18} {_fmt(gm):>20} "
              f"{_fmt(mm):>20} {('PASS' if gen_pass else 'FAIL'):>8} "
              f"{('PASS' if maf_pass else 'FAIL'):>8}")

    print(f"\nsynthetic-suite reward: genuine={float(genuine['reward']):.4f}  "
          f"maf_filter={float(maf['reward']):.4f}")
    print("\n=== VERDICT ===")
    if not hole_open:
        print("HOLE CLOSED: the MAF filter fails rare-structure mastery while the genuine "
              "non-filtering reference passes -- structure now lives in rare variants, so MAF "
              "filtering costs "
              "real accuracy. No DGP change needed on this axis.")
        return 0
    print("HOLE OPEN on:", ", ".join(hole_open))
    print("The MAF filter still clears mastery (or genuine does not) on the above categories, so "
          "MAF filtering is not sufficiently penalized. Strengthen the DGP so those categories "
          "carry structure in rare (MAF<0.05) variants -- lower Fst / rarer causal markers -- "
          "rather than adding a MAF-specific gate.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
