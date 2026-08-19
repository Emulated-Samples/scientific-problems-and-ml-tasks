"""Shared helpers for the cheat corpus (test scaffolding, not a submission).

Each cheat is one known hack packaged as a ``pca`` program; ``cheats/run_cheats.py`` grades
them all through the real grading path and asserts each is crushed to reward ~0. A cheat that
scored here would score in the real benchmark -- exactly the false positive the gates exist
to catch.
"""
from __future__ import annotations

import numpy as np


def read_dosage_matrix(vcf: str, max_variants: int | None = None):
    """Return (sample_ids, X) with X (n_variants x n_samples) float dosage, missing=nan.

    Deliberately simple full-scan reader so cheats can focus on the *modelling* hack.
    """
    sample_ids = None
    rows = []
    with open(vcf, "rb") as fh:
        for raw in fh:
            if raw[:2] == b"##":
                continue
            if raw[:6] == b"#CHROM":
                sample_ids = [s.decode() for s in raw.rstrip(b"\n").split(b"\t")[9:]]
                continue
            line = raw.rstrip(b"\n")
            f = line.split(b"\t", 9)
            if len(f) < 10:
                continue
            ref, alt = f[3], f[4]
            if len(ref) != 1 or len(alt) != 1 or ref not in b"ACGT" or alt not in b"ACGT":
                continue
            block = f[9]
            n = len(sample_ids)
            arr = np.frombuffer(block, dtype=np.uint8)
            if arr.size != 4 * n - 1:
                continue
            left = arr[0::4].astype(np.float64)
            right = arr[2::4].astype(np.float64)
            miss = (left == 46) | (right == 46)
            dose = (left - 48) + (right - 48)
            dose[miss] = np.nan
            rows.append(dose)
            if max_variants and len(rows) >= max_variants:
                break
    X = np.asarray(rows)
    return sample_ids, X


def write_scores(out: str, sample_ids, scores):
    k = scores.shape[1]
    with open(out, "w") as fh:
        fh.write("sample_id\t" + "\t".join(f"PC{i+1}" for i in range(k)) + "\n")
        for sid, row in zip(sample_ids, scores):
            fh.write(sid + "\t" + "\t".join(f"{v:.6f}" for v in row) + "\n")
