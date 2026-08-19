"""Sweep the fast reference's block budget on a disjoint calibration VCF.

Find the fastest setting that still recovers the full-scan subspace to accuracy ~1 on a subtle
structure calibration arm, because that bounds how few variants the policy can retain. Never use a
scored or candidate grading corpus here: doing so tunes the speed/accuracy anchor to hidden answers.
The next evaluator release must wrap this tool in a committed calibration manifest; the current
script cannot establish provenance by itself.

Prints time, selected core bytes, variants kept, and subspace accuracy for each setting
so we can pick a systems-optimized reference default without sacrificing correctness.

Usage:  python scripts/tune_reference.py <vcf> [--k 10]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from reference import fast_pca
from reference.full_scan_pca import fit as full_scan_fit
from grader.metrics.subspace import subspace_accuracy, structure_weights


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("vcf")
    ap.add_argument("--k", type=int, default=10)
    a = ap.parse_args()
    vcf = Path(a.vcf)

    print(f"full scan of {vcf.name} ...", flush=True)
    t0 = time.perf_counter()
    ids, ref_scores, kept, spectrum = full_scan_fit(vcf, a.k)
    t_ref = time.perf_counter() - t0
    w = structure_weights(spectrum, ref_scores.shape[1])[:ref_scores.shape[1]]
    print(f"  full scan: {kept} variants, {t_ref:.1f}s (baseline)\n")

    settings = [
        (256, 131072), (512, 131072), (1024, 131072),
        (512, 262144), (1024, 262144), (2048, 262144),
        (2048, 131072), (4096, 131072),
    ]
    print(f"{'blocks':>7} {'blk_KB':>7} {'MB_core':>8} {'kept':>8} {'time_s':>7} "
          f"{'speedup':>8} {'accuracy':>9}")
    for n_blocks, bsz in settings:
        t0 = time.perf_counter()
        _, scores, kvar, meta = fast_pca.fit(vcf, a.k, n_blocks=n_blocks, block_size=bsz)
        dt = time.perf_counter() - t0
        acc = subspace_accuracy(scores, ref_scores, w)["accuracy"]
        mb = meta["selected_bytes"] / 1e6
        print(f"{n_blocks:>7} {bsz//1024:>7} {mb:>8.0f} {kvar:>8} {dt:>7.2f} "
              f"{t_ref/dt:>7.1f}x {acc:>9.4f}", flush=True)


if __name__ == "__main__":
    main()
