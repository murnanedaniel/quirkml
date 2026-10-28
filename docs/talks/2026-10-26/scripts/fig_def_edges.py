"""Definitions slide: the kinds of edge in the GNN graph, on a toy drawing (no text in the image; the slide holds
the legend as native line samples so the same PNG can be reused at any size).

Hits: quirk A (dark blue circles), quirk B (light orange squares), one ordinary particle (grey circles).
Edge kinds and line styles (same on the slide legend):
  true edge of an ordinary particle      grey solid
  true quirk edge (neighbouring hits of one quirk)   solid, in the quirk's colour
  edge joining quirk A to quirk B        purple dashed   (wrong for the per-quirk GNN, right for the pair GNN)
  edge joining a quirk hit to an ordinary hit   magenta dotted (always wrong)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qstyle import C, plt, save, setup  # noqa: E402

AB = "#5b3f9e"     # edge joining quirk A to quirk B
QO = "#c2185b"     # edge joining a quirk hit to an ordinary hit

N = 9
x = np.arange(1, N + 1, dtype=float)
yA = 0.62 * np.sin(np.pi * (x - 0.3) / 3.2)
yB = -0.62 * np.sin(np.pi * (x - 0.3) / 3.2) - 0.05
yO = 1.45 - 0.045 * x  # ordinary particle, a straight track passing above the pair

setup(15)
fig, ax = plt.subplots(figsize=(900 / 120, 640 / 120))
for xx in x:
    ax.axvline(xx, color=C["grid"], lw=1.2, zorder=0)

# true edges
ax.plot(x, yO, "-", color=C["other"], lw=3.5, zorder=1)
ax.plot(x, yA, "-", color=C["A"], lw=4, zorder=2)
ax.plot(x, yB, "-", color=C["B"], lw=4, zorder=2)

# edges joining quirk A to quirk B (neighbouring layers or same layer, where the two are close)
ab = [(1, 1), (2, 3), (4, 4), (5, 6), (7, 8)]
for i, j in ab:
    ax.plot([x[i], x[j]], [yA[i], yB[j]], "--", color=AB, lw=2.6, dashes=(5, 3), zorder=1.5)

# edges joining a quirk hit to an ordinary hit
qo = [((1, "A"), 2), ((7, "A"), 6), ((4, "B"), 5)]
for (i, q), j in qo:
    yq = yA[i] if q == "A" else yB[i]
    ax.plot([x[i], x[j]], [yq, yO[j]], ":", color=QO, lw=3.2, zorder=1.5)

ax.scatter(x, yO, s=170, color=C["other"], marker="o", edgecolor="white", lw=1.5, zorder=3)
ax.scatter(x, yA, s=190, color=C["A"], marker="o", edgecolor="white", lw=1.5, zorder=4)
ax.scatter(x, yB, s=190, color=C["B"], marker="s", edgecolor="white", lw=1.5, zorder=4)
ax.set_xlim(0.7, N + 0.3)
ax.set_ylim(-0.82, 1.5)
ax.axis("off")
fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
save(fig, "def_edges")


# small two-panel picture of the two GNN training targets, for the models slide (no text; labels on the slide)
def targets_mini():
    fig, axs = plt.subplots(1, 2, figsize=(820 / 120, 190 / 120))
    n = 7
    xs = np.arange(1, n + 1, dtype=float)
    ya = 0.62 * np.sin(np.pi * (xs - 0.3) / 3.2)
    yb = -ya - 0.05
    pairs = [(0, 1), (1, 1), (2, 3), (3, 2), (4, 4), (5, 6), (6, 5)]
    for k, ax in enumerate(axs):
        for i, j in pairs:
            if k == 0:
                ax.plot([xs[i], xs[j]], [ya[i], yb[j]], "--", color=C["axis"], lw=1.8, dashes=(4, 3), zorder=1)
            else:
                ax.plot([xs[i], xs[j]], [ya[i], yb[j]], "-", color=AB, lw=2.6, zorder=1)
        ax.plot(xs, ya, "-", color=C["A"], lw=3.5, zorder=2)
        ax.plot(xs, yb, "-", color=C["B"], lw=3.5, zorder=2)
        ax.scatter(xs, ya, s=110, color=C["A"], marker="o", edgecolor="white", lw=1.2, zorder=3)
        ax.scatter(xs, yb, s=110, color=C["B"], marker="s", edgecolor="white", lw=1.2, zorder=3)
        ax.set_xlim(0.6, n + 0.4)
        ax.set_ylim(-0.85, 0.8)
        ax.axis("off")
    fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0, wspace=0.12)
    save(fig, "def_targets_mini")


targets_mini()
