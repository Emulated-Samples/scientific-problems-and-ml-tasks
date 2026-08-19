"""Local helper module for the eigsh fn-resistance submission: an independent, FORMAT-aware VCF
reader that Patterson-standardises every clean biallelic SNV. It implements the complete visible
GT contract independently: case-insensitive bases, phased/unphased diploids, pseudo-diploid
haploids, rich FORMAT rows, and call-local missing/malformed handling."""
from __future__ import annotations

import numpy as np


def read_standardized(vcf: str):
    sample_ids = None
    cols = []
    with open(vcf, "rb") as fh:
        for raw in fh:
            if raw[:2] == b"##":
                continue
            if raw[:6] == b"#CHROM":
                sample_ids = [s.decode() for s in raw.rstrip(b"\r\n").split(b"\t")[9:]]
                continue
            f = raw.rstrip(b"\r\n").split(b"\t", 9)
            if len(f) < 10:
                continue
            ref, alt = f[3], f[4]
            if (len(ref) != 1 or len(alt) != 1
                    or ref.upper() not in b"ACGT" or alt.upper() not in b"ACGT"):
                continue
            if f[8] != b"GT" and not f[8].startswith(b"GT:"):
                continue
            n = len(sample_ids)
            fields = f[9].split(b"\t")
            if len(fields) != n:
                continue
            dose = np.empty(n, dtype=np.float64)
            for i, fld in enumerate(fields):
                gt = fld.split(b":", 1)[0]
                if gt == b"0":
                    dose[i] = 0.0
                elif gt == b"1":
                    dose[i] = 2.0
                elif (len(gt) == 3 and gt[1:2] in (b"/", b"|")
                      and gt[0:1] in (b"0", b"1") and gt[2:3] in (b"0", b"1")):
                    dose[i] = (gt[0:1] == b"1") + (gt[2:3] == b"1")
                else:
                    dose[i] = np.nan
            obs = ~np.isnan(dose)
            if obs.sum() < 2 or np.var(dose[obs]) <= 0.0:
                continue
            p = dose[obs].sum() / (2.0 * obs.sum())
            if not (1e-6 < p < 1 - 1e-6):
                continue
            s = np.sqrt(2 * p * (1 - p))
            dose = np.where(np.isnan(dose), 2 * p, dose)
            cols.append((dose - 2 * p) / s)
    Z = np.asarray(cols)
    return sample_ids, Z
