#!/usr/bin/env python3
"""Render the tool benchmark (bench_vs_tools.py JSON) into publication-ready figures.

Four panels, one story each:

  1. **End-to-end wall-clock** (cold cache) per tool -- the headline "how long from VCF to PCs".
  2. **Speedup vs plink2** -- our tool as a multiple of the standard workhorse.
  3. **Speed vs accuracy** scatter -- proves we are in the fast *and* correct corner, not
     trading correctness for speed.
  4. **Bytes read from disk** (cold cache) -- the mechanism: we read a fraction of the file.

Usage:  python scripts/plot_bench.py bench_results.json --out-dir figs
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# colour by kind so "ours" pops
_COLORS = {"ours": "#d62728", "anchor": "#7f7f7f", "external": "#1f77b4"}


def _color(kind, tool):
    if kind == "ours":
        return _COLORS["ours"]
    if kind == "anchor":
        return _COLORS["anchor"]
    return _COLORS["external"]


def _t(tool, cold_pref=True):
    """Preferred time: cold if present, else warm."""
    if cold_pref and tool.get("cold_seconds"):
        return tool["cold_seconds"]
    return tool["warm_seconds"]


def plot_dataset(ds: dict, out_dir: Path, cold: bool):
    tools = [t for t in ds["tools"] if t["ok"]]
    if not tools:
        return
    name = ds["vcf"]
    size_gb = ds["size_bytes"] / 1e9
    label = f"{name}  ({size_gb:.1f} GB, {ds['n_samples']} samples, {ds['kept_variants']:,} SNVs)"

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(f"Population-genetics PCA: fast_pca vs standard tools\n{label}",
                 fontsize=13, fontweight="bold")

    # -- Panel 1: wall-clock bars (log) --
    ax = axes[0, 0]
    order = sorted(tools, key=lambda t: _t(t, cold))
    names = [t["tool"] for t in order]
    times = [_t(t, cold) for t in order]
    colors = [_color(t["kind"], t["tool"]) for t in order]
    bars = ax.barh(names, times, color=colors)
    ax.set_xscale("log")
    ax.set_xlabel("wall-clock seconds (log)  " + ("[cold cache]" if cold else "[warm cache]"))
    ax.set_title("End-to-end: VCF on disk → PC scores", fontsize=11)
    ax.invert_yaxis()
    for b, t in zip(bars, times):
        ax.text(t, b.get_y() + b.get_height() / 2, f" {t:.2f}s",
                va="center", fontsize=9)

    # -- Panel 2: speedup vs plink2 --
    ax = axes[0, 1]
    ref = next((t for t in tools if t["tool"] == "plink2 --pca"), None)
    if ref:
        base = _t(ref, cold)
        speed = [(t["tool"], base / _t(t, cold), t["kind"]) for t in order]
        sn = [s[0] for s in speed]
        sv = [s[1] for s in speed]
        sc = [_color(s[2], s[0]) for s in speed]
        bars = ax.barh(sn, sv, color=sc)
        ax.axvline(1.0, color="k", ls="--", lw=1, alpha=0.6)
        ax.set_xlabel("× faster than plink2 --pca")
        ax.set_title("Speedup relative to plink2", fontsize=11)
        ax.invert_yaxis()
        for b, v in zip(bars, sv):
            ax.text(v, b.get_y() + b.get_height() / 2, f" {v:.1f}×", va="center", fontsize=9)
    else:
        ax.set_visible(False)

    # -- Panel 3: speed vs accuracy scatter --
    ax = axes[1, 0]
    for t in tools:
        ax.scatter(_t(t, cold), t["accuracy"], s=120, color=_color(t["kind"], t["tool"]),
                   edgecolor="k", zorder=3)
        ax.annotate(t["tool"], (_t(t, cold), t["accuracy"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("wall-clock seconds (log)  " + ("[cold]" if cold else "[warm]"))
    ax.set_ylabel("subspace accuracy vs full-scan (1.0 = anchor)")
    ax.set_title("Fast AND correct? (top-left is best)", fontsize=11)
    ax.axhline(0.9, color="green", ls=":", lw=1, alpha=0.6)
    ax.grid(True, alpha=0.3)

    # -- Panel 4: bytes read from disk (cold) --
    ax = axes[1, 1]
    have_io = [t for t in order if t.get("cold_read_bytes")]
    if have_io and cold:
        bn = [t["tool"] for t in have_io]
        bv = [t["cold_read_bytes"] / 1e9 for t in have_io]
        bc = [_color(t["kind"], t["tool"]) for t in have_io]
        bars = ax.barh(bn, bv, color=bc)
        ax.axvline(size_gb, color="k", ls="--", lw=1, alpha=0.6)
        ax.text(size_gb, -0.5, f" file = {size_gb:.1f} GB", fontsize=8, color="k")
        ax.set_xlabel("GB read from disk (cold cache)")
        ax.set_title("Why we win: we read a fraction of the file", fontsize=11)
        ax.invert_yaxis()
        for b, v in zip(bars, bv):
            ax.text(v, b.get_y() + b.get_height() / 2, f" {v:.2f} GB", va="center", fontsize=9)
    else:
        ax.text(0.5, 0.5, "no cold-cache disk-read data\n(run with --cold as root on Linux)",
                ha="center", va="center", transform=ax.transAxes, fontsize=10, color="gray")
        ax.set_axis_off()

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = out_dir / f"bench_{Path(name).stem}.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--out-dir", default="figs")
    a = ap.parse_args(argv)
    data = json.loads(Path(a.results).read_text())
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cold = data.get("meta", {}).get("cold", False)
    for ds in data["datasets"]:
        plot_dataset(ds, out_dir, cold)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(__import__("sys").argv[1:]))
