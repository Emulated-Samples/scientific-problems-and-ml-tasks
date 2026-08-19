#!/usr/bin/env python3
"""Assemble measured benchmark numbers into the plot_bench.py JSON schema.

Numbers are REAL measurements (see docs/benchmark_vs_tools.md for provenance):
  * disk bytes read (cold cache) are contention-independent -- measured on the shared box via
    /usr/bin/time -v "File system inputs"; these carry the headline.
  * wall-clock for the 1.29 GB synthetic is from a QUIET Mac (warm cache) so the ratio is clean;
    the shared benchmark box's wall-clock was load-distorted and is NOT used for absolute claims.
  * accuracy is subspace agreement vs the full-scan reference (grader metric); load-independent.
Filled by the driver once the box computes fullscan disk-read + per-tool accuracy.
"""
import json, sys

SECT = 512

# --- filled from measurements ---
data = {
  "meta": {"k": 10, "cold": True, "note": "disk-read cold/contention-proof; wall from quiet Mac (warm)"},
  "datasets": [
    {
      "vcf": "1000G chr22 (real)", "size_bytes": 11_212_370_718,
      "n_samples": 2504, "kept_variants": 1_050_000, "anchor_seconds": 128.0,
      "tools": [
        {"tool": "ours (fast_pca)", "kind": "ours", "ok": True,
         "warm_seconds": 2.4, "cold_seconds": 18.09, "cold_read_bytes": 547080*SECT,
         "accuracy": 0.99, "ncc_ratio_vs_anchor": 0.997},
        {"tool": "plink2 --pca", "kind": "external", "ok": True,
         "warm_seconds": None, "cold_seconds": 475.0, "cold_read_bytes": 21920008*SECT,
         "accuracy": 0.98, "ncc_ratio_vs_anchor": 1.0},
        {"tool": "full-scan (numpy anchor)", "kind": "anchor", "ok": True,
         "warm_seconds": None, "cold_seconds": 128.0, "cold_read_bytes": 21920008*SECT,
         "accuracy": 1.0, "ncc_ratio_vs_anchor": 1.0},
      ],
    },
    {
      "vcf": "big_continental (synthetic)", "size_bytes": 1_292_922_305,
      "n_samples": 800, "kept_variants": 390000, "anchor_seconds": None,
      "tools": [
        # cold_read_bytes = measured sectors * 512; warm_seconds = quiet-Mac clean wall.
        {"tool": "ours (fast_pca)", "kind": "ours", "ok": True,
         "warm_seconds": 0.48, "cold_seconds": None, "cold_read_bytes": 608000*SECT,
         "accuracy": None},
        {"tool": "plink2 --pca", "kind": "external", "ok": True,
         "warm_seconds": 1.85, "cold_seconds": None, "cold_read_bytes": 2546016*SECT,
         "accuracy": None},
        {"tool": "plink2 --pca approx", "kind": "external", "ok": True,
         "warm_seconds": None, "cold_seconds": None, "cold_read_bytes": 2547144*SECT,
         "accuracy": None},
        {"tool": "plink1.9 --pca", "kind": "external", "ok": True,
         "warm_seconds": None, "cold_seconds": None, "cold_read_bytes": 2542616*SECT,
         "accuracy": None},
        {"tool": "scikit-allel", "kind": "external", "ok": True,
         "warm_seconds": None, "cold_seconds": None, "cold_read_bytes": 2622136*SECT,
         "accuracy": None},
        {"tool": "full-scan (numpy anchor)", "kind": "anchor", "ok": True,
         "warm_seconds": None, "cold_seconds": None, "cold_read_bytes": None,  # FILLED
         "accuracy": 1.0},
      ],
    },
  ],
}

if __name__ == "__main__":
    # optional overrides: pass "fullscan_sectors=NNN acc_ours=.. acc_plink2=.." as argv
    ov = dict(a.split("=") for a in sys.argv[1:] if "=" in a)
    if "fullscan_sectors" in ov:
        data["datasets"][1]["tools"][-1]["cold_read_bytes"] = int(ov["fullscan_sectors"]) * SECT
    print(json.dumps(data, indent=2))
