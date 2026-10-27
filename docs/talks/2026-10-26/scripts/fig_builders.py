"""Toy schematic of the two track builders on a quirk pair (10 layers, one hit per quirk per layer).

Rows: edge scores of a pair-target GNN (every same-pair edge high) and of a per-quirk GNN (same-quirk edges high,
a few A-B edges near 0.6, one A-A edge missing from the graph). Columns: walk (greedy, outward, one successor per
hit, starting from the innermost unassigned hit) and connected components above a 0.5 cut.
The real builders (tracking/gnn/build_tracks.py) add stitching, min hits and orientation handling; this only
shows the mechanism. Hit counts per resulting track are printed on the panels.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qstyle import C, TRACK_COLS, plt, record, save, setup  # noqa: E402

N = 10
zig = np.array([1, 1, -1, -1, 1, 1, -1, -1, 1, 1], float) * 0.5
nodes = [("A", k) for k in range(N)] + [("B", k) for k in range(N)]
pos = {("A", k): (k + 1, zig[k]) for k in range(N)}
pos.update({("B", k): (k + 1, -zig[k]) for k in range(N)})


def edges(mode):
    e = {}
    for k in range(N - 1):
        for a in "AB":
            for b in "AB":
                if mode == "pair":
                    s = 0.9 - 0.01 * ((k + (a != b)) % 2)  # all high; small offsets pick one successor
                else:
                    if a == b:
                        if a == "A" and k == 4:
                            continue  # A-A edge missing from the graph
                        s = 0.92
                    else:
                        s = 0.6 if k in (2, 4, 6) else 0.1
                e[((a, k), (b, k + 1))] = s
    return e


def walk(e, cut=0.5):
    unassigned = set(nodes)
    tracks = []
    while unassigned:
        start = min(unassigned, key=lambda n: (n[1], n[0]))
        t = [start]
        unassigned.discard(start)
        cur = start
        while True:
            nxt = [(s, b) for (a, b), s in e.items() if a == cur and b in unassigned and s > cut]
            if not nxt:
                break
            cur = max(nxt)[1]
            t.append(cur)
            unassigned.discard(cur)
        tracks.append(t)
    return tracks


def components(e, cut=0.5):
    par = {n: n for n in nodes}

    def f(n):
        while par[n] != n:
            par[n] = par[par[n]]
            n = par[n]
        return n
    for (a, b), s in e.items():
        if s > cut:
            par[f(a)] = f(b)
    groups = {}
    for n in nodes:
        groups.setdefault(f(n), []).append(n)
    return list(groups.values())


setup(15)
fig, axs = plt.subplots(2, 2, figsize=(13.9, 6.6), sharex=True, sharey=True)
out = {}
for i, mode in enumerate(("pair", "quirk")):
    e = edges(mode)
    for j, (bname, fn) in enumerate((("walk", walk), ("components", components))):
        ax = axs[i, j]
        for (a, b), s in e.items():
            if s > 0.5:
                ax.plot(*zip(pos[a], pos[b]), "-", color=C["grid"], lw=1.6, zorder=0)
        tracks = sorted(fn(e), key=len, reverse=True)
        desc = []
        for t_i, t in enumerate(tracks):
            pts = sorted((pos[n] for n in t), key=lambda p: (p[0], p[1]))
            col = TRACK_COLS[t_i % len(TRACK_COLS)]
            if len(t) > 1:
                ax.plot([p[0] for p in pts], [p[1] for p in pts], "-", color=col, lw=7, alpha=0.32,
                        solid_capstyle="round", zorder=1)
            na, nb = sum(n[0] == "A" for n in t), sum(n[0] == "B" for n in t)
            desc.append(f"{na}A+{nb}B")
        for n in nodes:
            ax.scatter(*pos[n], s=95, color=C[n[0]], marker="o" if n[0] == "A" else "s", edgecolor="white",
                       lw=1.1, zorder=3)
        if mode == "quirk":
            ax.plot([5.5], [0.5], marker="x", color=C["fake"], ms=13, mew=3, zorder=4)
        ax.text(0.5, -0.88, "tracks: " + ", ".join(desc[:4]) + (" ..." if len(desc) > 4 else ""),
                fontsize=14, color=C["ink"])
        out[f"{mode}_{bname}"] = desc
        ax.set_ylim(-1.0, 0.75)
        ax.set_xlim(0.4, 10.6)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
axs[0, 0].set_title("walk: one successor per hit", fontsize=16)
axs[0, 1].set_title("connected components above 0.5", fontsize=16)
axs[0, 0].set_ylabel("pair-target scores", fontsize=16)
axs[1, 0].set_ylabel("per-quirk scores", fontsize=16)
fig.subplots_adjust(left=0.04, right=0.995, top=0.93, bottom=0.01, wspace=0.04, hspace=0.08)
save(fig, "builders_toy")
record("builders_toy", out)
print(out)
