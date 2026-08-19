"""Audit every grade fold's STRUCTURE-DETECTION MARGIN on the real release key.

Fairness check the suite has no other guard for: the grader scores a submission against the
reference's *structured* axes (``structure_weights`` > 0). If a fold's weakest planted axis sits too
close to the Marchenko-Pastur noise edge, the reference may resolve fewer axes than the fold's stated
``k`` -- still self-consistent, but the fold then scores on fewer directions than intended and is
fragile to a seed/box change. This regenerates each fold from the ACTUAL release key and reports the
margin between the weakest scored eigenvalue and the noise edge.

Reading the output:
  * ``struct == k``  -> the fold resolves its full requested rank; ``margin_x`` is the safety factor.
  * ``struct <  k``  -> EXPECTED for folds whose k exceeds planted rank (n_pops-1): continental_b,
    subtle_a, subtle_b, messy, scaling. Those extra axes are noise, get exactly zero weight, and are
    never scored -- so this is fair by design, and their ``margin_x`` (computed at ev[k-1], a noise
    eigenvalue) is meaningless. Judge those folds by ``struct == n_pops - 1`` instead.

Measured 2026-07-18 on release-v3 (r5.2xlarge): every fold resolves its planted rank; margins for the
folds that resolve their full k range 1.14x (io_wide, the thinnest in the suite) to 43x (biobank_b).
io_wide's margin is set by its 96-sample count and fixed +-0.24 contrast, NOT by marker count -- the
1.5M-marker enlargement did not widen it. Re-run this whenever the release key, a fold's shape, or the
noise-edge estimator changes.

Usage (on an hfdev box, never locally):  uv run python scripts/audit_structure_margins.py
"""
import sys, os
sys.path.insert(0,"/hyperfocal/env")
import numpy as np
from pathlib import Path
from data.generate import grade_specs, write_vcf
from data.release_key import read_math_key, key_commitment
from reference.full_scan_pca import fit as full_fit
from grader.metrics.subspace import structure_weights

mk = read_math_key("/hyperfocal/env/grader/private/release-v3.math-key")
com = key_commitment(mk); rk = os.urandom(32)
out = Path("/tmp/fairdata"); out.mkdir(exist_ok=True)
specs = grade_specs(mk)
print("%-24s %4s %6s %8s %10s %10s %7s" % ("fold","k","struct","ev[k-1]","noise_edge","margin_x","VERDICT"))
for spec in specs:
    p = out/(spec.name+".vcf")
    if not p.exists():
        write_vcf(spec,p,representation_key=rk,math_commitment=com)
    try:
        ids,ref,kept,spectrum = full_fit(p,spec.k)
        w = structure_weights(spectrum, ref.shape[1])[:ref.shape[1]]
        nstruct = int((w>0).sum())
        # weakest SCORED axis eigenvalue vs the first unstructured (noise) eigenvalue
        full_w = structure_weights(spectrum, spec.k)
        pos = spectrum[spectrum>0]
        edge_idx = int((full_w>0).sum())
        ev_weakest = float(spectrum[spec.k-1])
        ev_noise = float(spectrum[edge_idx]) if edge_idx < spectrum.size else float('nan')
        margin = ev_weakest/ev_noise if ev_noise>0 else float('inf')
        verdict = "THIN" if (nstruct < spec.k or margin < 1.15) else "ok"
        print("%-24s %4d %6d %8.3f %10.3f %10.3f %7s" % (spec.name, spec.k, nstruct, ev_weakest, ev_noise, margin, verdict), flush=True)
    except Exception as e:
        print("%-24s %4d  ERROR %s" % (spec.name, spec.k, str(e)[:60]), flush=True)
    p.unlink(missing_ok=True); Path(str(p)+".truth.json").unlink(missing_ok=True)
print("FAIR_DONE")
