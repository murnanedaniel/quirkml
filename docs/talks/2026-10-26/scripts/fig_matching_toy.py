"""Toy two-quirk example: per-quirk vs coupled matching (ACTS double matching with the majority step).

Quirk A and quirk B leave 10 hits each on 10 layers and cross every two layers (interleaved regime).
Three panels: truth; per-quirk style tracks that switch quirk; one track holding the whole pair.
Outcomes are computed here with STRICT matching (ATLAS convention, the deck's standard from 2026-10-26): majority
particle, purity > r and completeness > r, r = 0.5 (loose) or 0.9 (tight); ties go to A. The ACTS matcher uses >=;
the ">=" outcomes are recorded too (numbers.json) for the strict-vs-ge slide. The exact-50% cases are annotated.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qstyle import C, TRACK_COLS, plt, record, save, setup  # noqa: E402

N = 10
r = np.arange(1, N + 1, dtype=float)
zig = np.array([1, 1, -1, -1, 1, 1, -1, -1, 1, 1], float) * 0.55
hits = {("A", k): (r[k], zig[k]) for k in range(N)}
hits.update({("B", k): (r[k], -zig[k]) for k in range(N)})


def match(tracks, truth, ratio, strict=True):
    """truth: name -> set of hit keys; returns {name: matched?} under double matching with the majority step."""
    out = {name: False for name in truth}
    for t in tracks:
        counts = {name: len(t & h) for name, h in truth.items()}
        maj = max(counts, key=lambda n: counts[n])  # dict order breaks ties (A first)
        pur = counts[maj] / len(t)
        comp = counts[maj] / len(truth[maj])
        ok = (pur > ratio and comp > ratio) if strict else (pur >= ratio and comp >= ratio)
        if ok:
            out[maj] = True
    return out


A = {("A", k) for k in range(N)}
B = {("B", k) for k in range(N)}
scen = {
    "truth": [A, B],
    "switch": [{("A", k) for k in range(6)} | {("B", 6), ("B", 7)},
               {("B", k) for k in range(6)} | {("A", 6), ("A", 7)},
               {("A", 8), ("A", 9), ("B", 8), ("B", 9)}],
    "pair": [A | B],
}
res = {}
for strict in (True, False):
    for s, tr in scen.items():
        key = s if strict else s + "_ge"
        res[key] = {}
        for wp, ratio in (("50%", 0.5), ("90%", 0.9)):
            pq = match(tr, {"A": A, "B": B}, ratio, strict)
            cp = match(tr, {"pair": A | B}, ratio, strict)
            res[key][wp] = {"A": pq["A"], "B": pq["B"], "pair": cp["pair"]}
NOTE = {"truth": "each track holds exactly 10/20 of the pair:\ncompleteness 0.5 is not > 0.5, pair not found",
        "switch": "A: purity 6/8, completeness 6/10\nboth > 0.5: A found (B likewise)",
        "pair": "for A: purity exactly 10/20 = 0.5,\nnot > 0.5: no quirk found; pair found"}

setup(15)
fig, axs = plt.subplots(1, 3, figsize=(13.9, 5.4), sharey=True)
titles = {"truth": "truth: one track per quirk", "switch": "tracks that switch quirk",
          "pair": "one track holds the pair"}
for ax, s in zip(axs, scen):
    for j, t in enumerate(scen[s]):
        pts = sorted((hits[h] for h in t), key=lambda p: (p[0], p[1]))
        if s == "truth":
            col = C["A"] if ("A", 0) in t else C["B"]
            ax.plot([p[0] for p in pts], [p[1] for p in pts], "-", color=col, lw=2.5, alpha=0.6, zorder=1)
        else:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], "-", color=TRACK_COLS[j], lw=7, alpha=0.30,
                    solid_capstyle="round", zorder=1)
            first = pts[0] if j != 2 else pts[1]
            ax.text(first[0] - (0.95 if j != 2 else -0.25), first[1] + (0.12 if first[1] > 0 else -0.2), f"T{j + 1}", color=TRACK_COLS[j],
                    fontsize=15, fontweight="bold")
    for k in range(N):
        ax.scatter(*hits[("A", k)], s=110, color=C["A"], marker="o", edgecolor="white", lw=1.2, zorder=3)
        ax.scatter(*hits[("B", k)], s=110, color=C["B"], marker="s", edgecolor="white", lw=1.2, zorder=3)
    for x in r:
        ax.axvline(x, color=C["grid"], lw=1, zorder=0)
    ax.set_title(titles[s], color=C["ink"], fontsize=16, pad=10)
    ax.set_xlim(-0.2, 10.9)
    ax.set_ylim(-1.0, 1.0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("detector layer (outwards)", color=C["ink2"])
    ax.text(0.5, -0.13, NOTE[s], transform=ax.transAxes, ha="center", va="top", fontsize=14, color=C["fake"],
            linespacing=1.3)
    ok = lambda b: "found" if b else "not found"  # noqa: E731
    lines = []
    for wp in ("50%", "90%"):
        v = res[s][wp]
        lines.append(f"{wp}:  A {ok(v['A'])},  B {ok(v['B'])};  pair {ok(v['pair'])}")
axs[0].scatter([], [], s=110, color=C["A"], marker="o", label="quirk A hit")
axs[0].scatter([], [], s=110, color=C["B"], marker="s", label="quirk B hit")
axs[0].legend(loc="upper left", bbox_to_anchor=(0.0, 1.0), ncol=2, fontsize=13, handletextpad=0.2,
              columnspacing=0.8)
fig.subplots_adjust(left=0.01, right=0.99, top=0.90, bottom=0.24, wspace=0.08)
save(fig, "matching_toy")
record("matching_toy", res)
print(res)
