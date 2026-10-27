"""Toy schematic of the three GNN edge targets on one quirk pair (8 layers, one hit per quirk per layer).

1. per-quirk: consecutive hits of the same quirk in true-time order (gnn-v2 / e1 / g2 rows);
2. pair chain: one chain over the union of both quirks' hits in true-time order (gnn-e2-pair-*), drawn here with
   the two quirks alternating in time;
3. all same-pair edges true: every graph edge with both hits in the pair (gnn-p3-pairv3-s0, --pair-all-true),
   drawn as the edges between neighbouring layers.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qstyle import C, plt, save, setup  # noqa: E402

N = 8
zig = np.array([1, 1, -1, -1, 1, 1, -1, -1], float) * 0.5
P = {("A", k): (k + 1, zig[k]) for k in range(N)}
P.update({("B", k): (k + 1, -zig[k]) for k in range(N)})

setup(15)
fig, axs = plt.subplots(1, 3, figsize=(15.6, 4.3), sharey=True)
titles = ["per-quirk: same-quirk chain", "pair: one time-ordered chain", "pair: every same-pair edge"]
for i, ax in enumerate(axs):
    if i == 0:
        for q in "AB":
            for k in range(N - 1):
                ax.plot(*zip(P[(q, k)], P[(q, k + 1)]), "-", color=C[q], lw=3.5, zorder=1)
    elif i == 1:
        seq = [h for k in range(N) for h in (("A", k), ("B", k))]
        for a, b in zip(seq[:-1], seq[1:]):
            ax.plot(*zip(P[a], P[b]), "-", color=C["gnn"], lw=3.5, zorder=1)
    else:
        for k in range(N - 1):
            for a in "AB":
                for b in "AB":
                    ax.plot(*zip(P[(a, k)], P[(b, k + 1)]), "-", color=C["gnn"], lw=3.0, zorder=1)
        for k in range(N):
            ax.plot(*zip(P[("A", k)], P[("B", k)]), "-", color=C["gnn"], lw=3.0, zorder=1)
    for k in range(N):
        ax.scatter(*P[("A", k)], s=120, color=C["A"], marker="o", edgecolor="white", lw=1.2, zorder=3)
        ax.scatter(*P[("B", k)], s=120, color=C["B"], marker="s", edgecolor="white", lw=1.2, zorder=3)
    ax.set_title(titles[i], fontsize=17)
    ax.set_xlim(0.5, N + 0.5)
    ax.set_ylim(-0.8, 0.8)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlabel("detector layer (outwards)", color=C["ink2"])
fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.10, wspace=0.08)
save(fig, "targets_toy")
