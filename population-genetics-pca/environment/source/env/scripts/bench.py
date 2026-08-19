"""Rapid A/B harness for iterating on the fast reference.

For each candidate change to reference/fast_pca.py, this reports -- in one shot, on real data --
the three things that matter: wall-clock (min of N runs, robust to a loaded box), subspace
accuracy vs the full-scan anchor, and population-structure recovery vs known labels.

The full-scan anchor (slow: minutes on a real chromosome) is computed once and **cached to
disk**, so every subsequent iteration only pays the fast-path time. That is what makes tight
edit -> measure loops possible.

Usage:
  python scripts/bench.py data/real/chr22.vcf --repeat 5
  python scripts/bench.py data/generated/big_subtle.vcf --repeat 3     # hardest synthetic
  python scripts/bench.py --all                                         # chr22 + big_subtle
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from reference.full_scan_pca import fit as full_scan_fit          # noqa: E402
from grader.metrics.subspace import subspace_accuracy, structure_weights  # noqa: E402
from grader.metrics.structure import population_separation        # noqa: E402

CACHE = Path("/tmp/pcabench_anchor_cache")


def _anchor(vcf: Path, k: int) -> dict:
    """Full-scan reference scores + weights, cached to disk keyed by (path, mtime, size, k)."""
    CACHE.mkdir(exist_ok=True)
    st = vcf.stat()
    key = hashlib.md5(
        f"full-spectrum-v1:{vcf.resolve()}:{st.st_mtime}:{st.st_size}:{k}".encode()
    ).hexdigest()
    f = CACHE / f"{key}.npz"
    if f.exists():
        d = np.load(f, allow_pickle=True)
        return {"scores": d["scores"], "weights": d["weights"],
                "sample_ids": list(d["sample_ids"]), "seconds": float(d["seconds"])}
    print(f"  [anchor] full scan of {vcf.name} (one-time, cached)...", flush=True)
    t0 = time.perf_counter()
    ids, scores, kept, spectrum = full_scan_fit(vcf, k)
    dt = time.perf_counter() - t0
    weights = structure_weights(spectrum, scores.shape[1])[:scores.shape[1]]
    np.savez(f, scores=scores, weights=weights, sample_ids=np.array(ids), seconds=dt)
    return {"scores": scores, "weights": weights, "sample_ids": ids, "seconds": dt}


def bench_one(vcf: Path, k: int, repeat: int) -> dict:
    import reference.fast_pca as fp
    importlib.reload(fp)                       # pick up edits without a fresh process
    anchor = _anchor(vcf, k)

    # Labels for structure recovery, if a truth sidecar exists.
    labels = None
    tp = Path(str(vcf) + ".truth.json")
    if tp.exists():
        labels = np.array(json.loads(tp.read_text()).get("sample_pop", []))

    times, last = [], None
    for _ in range(repeat):
        t0 = time.perf_counter()
        ids, scores, kept, meta = fp.fit(vcf, k)
        times.append(time.perf_counter() - t0)
        last = (scores, kept, meta)
    scores, kept, meta = last
    acc = subspace_accuracy(scores, anchor["scores"], anchor["weights"])["accuracy"]
    tmin = min(times)
    row = {"vcf": vcf.name, "t_min": tmin, "t_anchor": anchor["seconds"],
           "speedup": anchor["seconds"] / tmin, "accuracy": acc, "kept": kept,
           "mb_selected": meta["selected_bytes"] / 1e6}
    if labels is not None and len(set(x for x in labels.tolist() if x >= 0)) >= 2:
        row["ncc"] = population_separation(scores, labels)["ncc_accuracy"]
        row["ncc_anchor"] = population_separation(anchor["scores"], labels)["ncc_accuracy"]
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("vcf", nargs="?")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--repeat", type=int, default=5)
    a = ap.parse_args()
    targets = []
    if a.all:
        for p in ["data/real/chr22.vcf", "data/generated/big_subtle.vcf",
                  "data/generated/big_continental.vcf"]:
            if Path(p).exists():
                targets.append(Path(p))
    elif a.vcf:
        targets = [Path(a.vcf)]
    else:
        ap.error("give a VCF or --all")

    print(f"{'dataset':22s} {'t_min':>7} {'anchor':>8} {'speedup':>8} {'accuracy':>9} "
          f"{'ncc/anch':>10} {'MBcore':>6} {'kept':>8}")
    for vcf in targets:
        r = bench_one(vcf, a.k, a.repeat)
        ncc = f"{r.get('ncc', float('nan')):.3f}/{r.get('ncc_anchor', float('nan')):.3f}" \
            if "ncc" in r else "   -"
        print(f"{r['vcf']:22s} {r['t_min']:7.2f} {r['t_anchor']:8.1f} {r['speedup']:7.1f}x "
              f"{r['accuracy']:9.4f} {ncc:>10} {r['mb_selected']:6.0f} {r['kept']:8d}", flush=True)


if __name__ == "__main__":
    main()
