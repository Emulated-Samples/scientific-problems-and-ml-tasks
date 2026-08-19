#!/usr/bin/env python3
"""Render the fair fast_pca-vs-tools comparison from measured numbers.

Two panels, both from real measurements (provenance in docs/benchmark_vs_tools.md):
  1. Bytes read from disk (cold cache) -- the mechanism, and CONTENTION-INDEPENDENT so it is the
     rigorous headline: our sampler reads a fixed ~0.3 GB budget; every streaming tool reads the
     whole file, so our advantage grows with file size (4x at 1.3 GB -> 40x at 11 GB).
  2. Wall-clock, VCF -> PCs, on the measurements taken on a QUIET machine (so the ratio is honest):
     ours vs plink2 at 1.3 GB (warm Mac), and ours vs the full-scan reference at 11 GB.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RED, GREY, BLUE = "#d62728", "#7f7f7f", "#1f77b4"


def _color(kind):
    return {"ours": RED, "anchor": GREY}.get(kind, BLUE)


def main(argv):
    data = json.loads(Path(argv[0]).read_text())
    out_dir = Path(argv[1]) if len(argv) > 1 else Path("docs/figs")
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.2))
    fig.suptitle("fast_pca vs. standard population-genetics PCA tools — VCF → top-10 PCs",
                 fontsize=14, fontweight="bold", y=0.99)

    # ---- Panel 1: disk read (cold), grouped by dataset ----
    ax = axes[0]
    ds_list = data["datasets"]
    # collect union of tool names in a stable order (ours first, anchor last)
    def order_key(t):
        return (0 if t["kind"] == "ours" else 2 if t["kind"] == "anchor" else 1, t["tool"])
    labels_order = []
    for ds in ds_list:
        for t in sorted(ds["tools"], key=order_key):
            if t["tool"] not in labels_order and t.get("cold_read_bytes"):
                labels_order.append(t["tool"])
    n_ds = len(ds_list)
    y = np.arange(len(labels_order))
    h = 0.8 / n_ds
    for di, ds in enumerate(ds_list):
        by_name = {t["tool"]: t for t in ds["tools"]}
        vals, colors, ypos = [], [], []
        for i, name in enumerate(labels_order):
            t = by_name.get(name)
            if t and t.get("cold_read_bytes"):
                vals.append(t["cold_read_bytes"] / 1e9)
                colors.append(_color(t["kind"]))
                ypos.append(y[i] + (di - (n_ds - 1) / 2) * h)
        bars = ax.barh(ypos, vals, height=h * 0.92, color=colors,
                       edgecolor="white", linewidth=0.5)
        size_gb = ds["size_bytes"] / 1e9
        for b, v in zip(bars, vals):
            frac = v / size_gb * 100
            ax.text(v * 1.03, b.get_y() + b.get_height() / 2,
                    f"{v:.2f} GB ({frac:.0f}%)", va="center", fontsize=7.5)
    ax.set_yticks(y)
    ax.set_yticklabels(labels_order)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("GB read from disk (cold cache, log)")
    ax.set_title("Why we win: we read a fraction of the file\n"
                 "(bar groups: 1.3 GB synthetic • 11 GB real 1000G chr22)", fontsize=10.5)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=RED),
                       plt.Rectangle((0, 0), 1, 1, color=BLUE),
                       plt.Rectangle((0, 0), 1, 1, color=GREY)],
              labels=["ours (fast_pca)", "standard tool", "full-scan reference"],
              fontsize=8, loc="lower right")

    # ---- Panel 2: clean wall-clock comparisons ----
    ax = axes[1]
    # Three SAME-CONDITION pairs (never compare across conditions):
    bars_data = [
        ("ours — 1.3 GB (Mac, warm)", 0.48, RED),
        ("plink2 --pca — 1.3 GB (Mac, warm)", 1.85, BLUE),
        ("ours — 11 GB real (shared box, cold)", 18.0, RED),
        ("plink2 --pca — 11 GB real (shared box, cold)", 475.0, BLUE),
        ("ours fast-path — 11 GB real (clean)", 2.4, RED),
        ("full-scan reference — 11 GB real (clean)", 128.0, GREY),
    ]
    names = [b[0] for b in bars_data]
    vals = [b[1] for b in bars_data]
    colors = [b[2] for b in bars_data]
    bars = ax.barh(names, vals, color=colors, edgecolor="white")
    ax.set_xscale("log")
    ax.invert_yaxis()
    ax.set_xlabel("wall-clock seconds, VCF → PC scores (log)")
    ax.set_title("Wall-clock, same-condition pairs: ~4× faster than plink2\n"
                 "(1.3 GB, quiet Mac); ~53× faster than a full scan (11 GB real)", fontsize=10.5)
    for b, v in zip(bars, vals):
        ax.text(v * 1.05, b.get_y() + b.get_height() / 2,
                f"{v:.2f}s" if v < 10 else f"{v:.0f}s", va="center", fontsize=8.5)
    # bracket the three comparison pairs
    for (y0, y1, mult) in [(0, 1, "3.9×"), (2, 3, "26× (both under shared load)"), (4, 5, "53×")]:
        ax.annotate("", xy=(0.012, y1), xytext=(0.012, y0), xycoords=("axes fraction", "data"),
                    arrowprops=dict(arrowstyle="-", color="black", lw=1))
        ax.text(0.02, (y0 + y1) / 2, mult, transform=ax.get_yaxis_transform(),
                rotation=90, va="center", ha="left", fontsize=7.5, color="black")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = out_dir / "fast_pca_vs_tools.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1:])
