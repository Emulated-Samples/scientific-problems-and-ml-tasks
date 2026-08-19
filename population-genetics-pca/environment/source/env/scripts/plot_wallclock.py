#!/usr/bin/env python3
"""One wall-clock chart (cold vs warm page cache, linear axis) for computing a population-genetics
PCA from the real 11 GB 1000 Genomes chromosome 22 VCF, with the actual recovered PC1/PC2 structure
overlaid as bare coloured points in the whitespace past the (tiny) "Ours" bars -- so the same
five-continent structure is visible right beside how little time it took. Numbers are measured on
one idle machine; plink2 / full-scan are static clean measurements."""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# --- measured on the real 11 GB 1000 Genomes chr22 VCF, clean same-machine (AWS 8-core, idle) ---
# (cold, warm) seconds.  __OURS__ filled from the 200k-marker measurement.
ROWS = [
    ("Ours",                  19.7,  9.0),
    ("plink2",                134,   51),
    ("Agent solution",        160,   122),
]

COLD_LO, COLD_HI = "#2563eb", "#1e5bff"   # blue  = cold cache (off disk); Ours slightly brighter
WARM_LO, WARM_HI = "#f59e0b", "#ff9e0a"   # amber = warm cache (from RAM)
INK = "#0f172a"

# super-population palette (1000 Genomes) -- points only, no legend
POP_COLORS = {"AFR": "#d62728", "AMR": "#9467bd", "EAS": "#2ca02c",
              "EUR": "#1f77b4", "SAS": "#ff7f0e"}

HERE = Path(__file__).resolve().parent.parent
PCA_DIR = HERE / "docs/figs/chr22_pca"


def load_scores(path):
    ids, rows = [], []
    with open(path) as f:
        f.readline()
        for line in f:
            p = line.rstrip("\n").split("\t")
            ids.append(p[0]); rows.append([float(x) for x in p[1:3]])  # PC1, PC2
    return ids, np.array(rows)


def load_panel(path):
    m = {}
    with open(path) as f:
        f.readline()
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 3:
                m[p[0]] = p[2]
    return m


fig, ax = plt.subplots(figsize=(13, 5.4))
fig.patch.set_facecolor("white"); ax.set_facecolor("white")

# ---------- the bars ----------
y = np.arange(len(ROWS))[::-1]
h = 0.36
xmax = max(c for _, c, _ in ROWS) * 1.16

for yi, (label, cold, warm) in zip(y, ROWS):
    is_ours = label == "Ours"
    ax.barh(yi + h/2, cold, height=h, color=COLD_HI if is_ours else COLD_LO, zorder=3)
    ax.barh(yi - h/2, warm, height=h, color=WARM_HI if is_ours else WARM_LO, zorder=3)
    fw = "bold" if is_ours else "normal"
    ax.text(cold + xmax*0.008, yi + h/2, f"{cold:g}s", va="center", ha="left",
            fontsize=11, color=COLD_LO, fontweight=fw)
    ax.text(warm + xmax*0.008, yi - h/2, f"{warm:g}s", va="center", ha="left",
            fontsize=11, color="#b45309" if is_ours else "#a1863f", fontweight=fw)

ax.set_yticks(y)
ax.set_yticklabels([r[0] for r in ROWS], fontsize=14, color=INK)
for lab in ax.get_yticklabels():
    if lab.get_text() == "Ours":
        lab.set_fontweight("bold")
xmax = 235   # extended past the longest bar (160s) to make room for the plink2 overlay
ax.set_xlim(0, xmax)
ax.set_ylim(-0.6, len(ROWS) - 0.4)
ax.set_xlabel("Seconds to fit 10 PCs", fontsize=11, color="#475569")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("bottom", "left"):
    ax.spines[s].set_color("#cbd5e1")
ax.tick_params(colors="#64748b", length=4)
ax.set_axisbelow(True)

ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=COLD_LO),
                   plt.Rectangle((0, 0), 1, 1, color=WARM_LO)],
          labels=["cold cache (off disk)", "warm cache (from RAM)"],
          fontsize=10, loc="lower right", frameon=False)

ax.set_title("Population genetics PCA from a VCF (11 GB 1000 Genomes chromosome 22)",
             fontsize=15, color=INK, loc="left", fontweight="bold", pad=12)

# ---------- the recovered structure: one bare-point PCA overlaid next to each tool's bars -------
panel = load_panel(PCA_DIR / "panel.txt")


def overlay_pca(scores_path, box):
    """Draw a frameless, transparent PC1/PC2 scatter (points only) in `box` (axes-fraction),
    with PC1/PC2 as interior labels so they sit inside the cloud and never land on a bar."""
    ids, xy = load_scores(scores_path)
    xy = xy.copy()
    for c in range(2):                       # deterministic PC signs (PCA is sign-free)
        if np.sum(xy[:, c]) < 0:
            xy[:, c] = -xy[:, c]
    sub = ax.inset_axes(box)
    sub.set_facecolor("none")
    for pop, col in POP_COLORS.items():
        sel = np.array([panel.get(i) == pop for i in ids])
        sub.scatter(xy[sel, 0], xy[sel, 1], s=4, c=col, alpha=0.8, linewidths=0)
    sub.margins(0.12)
    sub.set_xticks([]); sub.set_yticks([])
    for s in sub.spines.values():
        s.set_visible(False)
    # interior labels: PC1 lifted up inside the axes, PC2 just inside the left edge
    sub.text(0.5, 0.06, "PC1", transform=sub.transAxes, ha="center", va="bottom",
             fontsize=9, color="#64748b")
    sub.text(0.04, 0.5, "PC2", transform=sub.transAxes, ha="left", va="center",
             rotation=90, fontsize=9, color="#64748b")


# Ours PCA in the whitespace past the short "Ours" bars (top band); plink2 PCA past its bar (middle
# band).  Both are the same object -- the recovered five-continent structure -- one per tool.
overlay_pca(PCA_DIR / "ours.tsv",   [0.15, 0.60, 0.42, 0.36])
overlay_pca(PCA_DIR / "plink2.tsv", [0.615, 0.31, 0.36, 0.31])

fig.tight_layout()
fig.savefig(HERE / "docs/figs/fast_pca_wallclock.png", dpi=200, facecolor="white")
print("wrote docs/figs/fast_pca_wallclock.png")
