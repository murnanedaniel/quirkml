"""Definitions slide: the two quirks, per-quirk vs quirk-event vs coupled efficiency, on a toy pair.

Quirk A (dark blue circles) and quirk B (light orange squares) leave 12 hits each on 12 layers and cross between
layers. Three panels: each quirk on its own track; only quirk A on its own track (B split in two halves); one track
holding both quirks. Outcomes use strict loose matching (purity and completeness > 50%, majority particle), the
deck's standard, and are computed here, not typed in.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qstyle import C, TRACK_COLS, plt, record, save, setup  # noqa: E402

N = 12
x = np.arange(1, N + 1, dtype=float)
yA = 0.55 * np.sin(np.pi * (x - 0.5) / 3.0)
yB = -yA
A = {("A", k) for k in range(N)}
B = {("B", k) for k in range(N)}
pos = {("A", k): (x[k], yA[k]) for k in range(N)}
pos.update({("B", k): (x[k], yB[k]) for k in range(N)})


def found(tracks, truth, ratio=0.5):
    out = {n: False for n in truth}
    for t in tracks:
        counts = {n: len(t & h) for n, h in truth.items()}
        maj = max(counts, key=lambda n: counts[n])
        if counts[maj] / len(t) > ratio and counts[maj] / len(truth[maj]) > ratio:
            out[maj] = True
    return out


scen = [
    ("each quirk on its own track", [A, B]),
    ("quirk B split in two tracks", [A, {("B", k) for k in range(6)}, {("B", k) for k in range(6, N)}]),
    ("one track holds both quirks", [A | B]),
]
res = []
for title, tracks in scen:
    pq = found(tracks, {"A": A, "B": B})
    cp = found(tracks, {"pair": A | B})
    res.append({"title": title, "A": pq["A"], "B": pq["B"], "event": pq["A"] and pq["B"], "pair": cp["pair"]})

setup(15)
W, H = 1664 / 120, 500 / 120
fig, axs = plt.subplots(1, 3, figsize=(W, H))
band_cols = {0: [TRACK_COLS[0], TRACK_COLS[0]], 1: [TRACK_COLS[0], TRACK_COLS[1], TRACK_COLS[3]], 2: [TRACK_COLS[0]]}
for i, (ax, (title, tracks), r) in enumerate(zip(axs, scen, res)):
    for j, t in enumerate(tracks):
        col = band_cols[i][j]
        if i == 2:
            # one track: a band around the whole braid
            ax.fill_between([0.55, N + 0.45], -0.78, 0.78, color=col, alpha=0.16, lw=0, zorder=0)
            ax.plot([0.55, N + 0.45, N + 0.45, 0.55, 0.55], [-0.78, -0.78, 0.78, 0.78, -0.78], color=col, lw=2,
                    alpha=0.7, zorder=0)
            continue
        pts = sorted((pos[h] for h in t), key=lambda p: p[0])
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "-", color=col, lw=16, alpha=0.22,
                solid_capstyle="round", zorder=1)
    ax.plot(x, yA, "-", color=C["A"], lw=1.2, alpha=0.5, zorder=2)
    ax.plot(x, yB, "-", color=C["B"], lw=1.2, alpha=0.7, zorder=2)
    ax.scatter(x, yA, s=120, color=C["A"], marker="o", edgecolor="white", lw=1.2, zorder=3)
    ax.scatter(x, yB, s=120, color=C["B"], marker="s", edgecolor="white", lw=1.2, zorder=3)
    ax.set_title(title, color=C["ink"], fontsize=16, pad=8)
    ax.set_xlim(0.2, N + 0.8)
    ax.set_ylim(-0.95, 0.95)
    ax.axis("off")
    yes = lambda b: "found" if b else "not found"  # noqa: E731
    nq = int(r["A"]) + int(r["B"])
    lines = [("per-quirk:", f"{nq} of 2 quirks found"),
             ("quirk event:", yes(r["event"])),
             ("coupled:", yes(r["pair"]))]
    for k, (lab, val) in enumerate(lines):
        yy = -0.06 - 0.16 * k
        ax.text(0.02, yy, lab, transform=ax.transAxes, ha="left", va="top", fontsize=15, color=C["ink2"])
        good = (val == "found") or (k == 0 and nq > 0)
        ax.text(0.40, yy, val, transform=ax.transAxes, ha="left", va="top", fontsize=15,
                color=C["gnn"] if good else C["ink"], fontweight="bold" if good else "normal")
axs[0].scatter([], [], s=120, color=C["A"], marker="o", label="quirk A hit")
axs[0].scatter([], [], s=120, color=C["B"], marker="s", label="quirk B hit")
axs[0].plot([], [], "-", color=TRACK_COLS[0], lw=12, alpha=0.3, label="track")
fig.legend(loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=15, handletextpad=0.6, handlelength=2.2,
           columnspacing=1.6)
fig.subplots_adjust(left=0.01, right=0.99, top=0.79, bottom=0.33, wspace=0.10)
save(fig, "def_quirks")
record("def_quirks", res)
print(res)
