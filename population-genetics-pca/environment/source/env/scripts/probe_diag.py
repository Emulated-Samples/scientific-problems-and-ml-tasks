"""Diagnostic for the object-identity probes: print the raw alignment/agreement numbers a
given submission produces on each probe, so gate thresholds can be tuned against real fits
(the genuine FN guard vs. the wrong-object cheats) rather than guessed.

Usage:  /opt/hyperfocal/pcabench/bin/python scripts/probe_diag.py <submission_dir> \
          --data-dir <active-private-suite> --math-key-file <release-key> --isolation bwrap
"""
from __future__ import annotations

import argparse
import json
import secrets
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data.release_key import derive_seed, read_math_key  # noqa: E402
from grader.grade import _make_run_one, _probe_carrier_from_truth  # noqa: E402
from grader.gates.probes import (  # noqa: E402
    coverage_gate,
    hwe_norm_gate,
    select_probe_carriers,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("submission_dir")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--math-key-file", required=True)
    ap.add_argument("--workdir", default="/tmp/pcabench_probe_diag")
    ap.add_argument("--isolation", choices=("bwrap", "verifier-container"), required=True)
    a = ap.parse_args()
    sub = Path(a.submission_dir)
    wd = Path(a.workdir) / sub.name
    wd.mkdir(parents=True, exist_ok=True)
    timings = []
    run_one = _make_run_one(sub, wd, timings, isolation=a.isolation)
    math_key = read_math_key(a.math_key_file)
    truths = [json.loads(path.read_text())
              for path in sorted(Path(a.data_dir).glob("*.truth.json"))]
    carriers = select_probe_carriers(
        [_probe_carrier_from_truth(truth) for truth in truths], math_key)
    representation_key = secrets.token_bytes(32)
    h = hwe_norm_gate(
        run_one, wd, carriers["hwe_norm"], derive_seed(math_key, "probe/hwe"),
        representation_key=representation_key,
    )
    c = coverage_gate(
        run_one, wd, carriers["coverage"], derive_seed(math_key, "probe/coverage"),
        representation_key=representation_key,
    )
    print(f"{sub.name}:")
    print(f"  hwe_norm  factor={h['factor']:.3f} severity={h.get('severity',0):.3f} "
          f"align_patterson={h.get('align_patterson')} align_rawcov={h.get('align_rawcov')}")
    print(f"  coverage  factor={c['factor']:.3f} severity={c.get('severity',0):.3f} "
          f"subtle_agreement={c.get('subtle_agreement')}")
    print(f"  runtime   submission={sum(x['submission_seconds'] for x in timings):.3f}s "
          f"reference={sum(x['reference_seconds'] for x in timings):.3f}s")


if __name__ == "__main__":
    main()
