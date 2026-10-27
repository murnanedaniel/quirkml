"""Figures for the splitter (A1) and fake-rate (B) slides. Numbers from tracking/EXPERIMENTS.md sections
"A1 learned splitter (2026-10-27)" and "Fake-rate reduction (work package B)"; B2 curves from
/pscratch/sd/d/danieltm/quirk_tracking/fakes/b2/<model>/curve_*_test.csv (test 17000-19999, strict).

split_scaling.png  val strict loose / tight per quirk vs training pairs (oracle pairs), plus test points
split_errors.png   schematic of the A1 error type: segments swapped between crossings (illustration, not data)
b1_fakes.png       B1 fake decomposition per model (fakes per event, stacked by category)
b2_curves.png      B2 efficiency vs fake rate for the quality-score variants and the hit-count cut
"""
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qstyle import C, plt, record, save, setup  # noqa: E402

# ---------------------------------------------------------------- A1 scaling (ledger: A1 data scaling, 1M, longer/larger)
VAL = [  # training pairs, loose, tight (val strict, no direction, augmentation, d128 4 layers)
    (16_000, 0.902, 0.160), (50_000, 0.918, 0.243), (200_000, 0.933, 0.375), (971_000, 0.973, 0.705)]
VAL16EP = (971_000, 0.985, 0.824)   # 16 epochs, d128 4 layers
VALBIG = (971_000, 0.987, 0.859)    # 8 epochs, d256 6 layers
TEST = [(200_000, 0.939, 0.403), (971_000, 0.986, 0.865)]  # a1s-200k-nodir-aug, a1s-1m-d256l6 (oracle pairs)
TEST16_DIR = (16_000, 0.878, 0.123)  # a1r-oracle-none: with direction, no augmentation
REAL = [("200k", 0.739, 0.165), ("971k\n6 epochs", 0.804, 0.394), ("971k\n16 epochs", 0.814, 0.476),
        ("971k\nlarger", 0.828, 0.505)]  # gated real tracks: a1g-200k, a1s-1m, a1s-1m-e16, a1s-1m-d256l6
setup(15)
fig, axs = plt.subplots(1, 2, figsize=(15.6, 6.2), gridspec_kw={"width_ratios": [1.5, 1]})
ax = axs[0]
n = [v[0] for v in VAL]
ax.plot(n, [v[1] for v in VAL], "-o", color=C["gnn"], ms=9, label="validation, loose (6-40 epochs)")
ax.plot(n, [v[2] for v in VAL], "-o", color=C["mf"], ms=9, label="validation, tight")
for v, mk, lab in ((VAL16EP, "^", "validation, 971k, 16 epochs"), (VALBIG, "*", "validation, 971k, larger model")):
    ax.plot([v[0]] * 2, v[1:], mk, color=C["ink2"], ms=14, label=lab)
ax.plot([t[0] for t in TEST], [t[1] for t in TEST], "D", color=C["gnn"], ms=12, mfc="white", mew=2.5,
        label="test, loose")
ax.plot([t[0] for t in TEST], [t[2] for t in TEST], "D", color=C["mf"], ms=12, mfc="white", mew=2.5,
        label="test, tight")
ax.plot([TEST16_DIR[0]] * 2, TEST16_DIR[1:], "x", color=C["muted"], ms=11, mew=2.5,
        label="test, 16k, with direction")
ax.set_xscale("log")
ax.set_xlabel("training pairs (true pairs from simulation)")
ax.set_ylabel("per-quirk efficiency, strict")
ax.set_ylim(0, 1.02)
ax.grid(True, axis="y")
ax.legend(loc="lower right", fontsize=12)
ax.set_title("splitter on true pairs", fontsize=16, loc="left")
ax = axs[1]
x = np.arange(len(REAL))
ax.plot(x, [r[1] for r in REAL], "-o", color=C["gnn"], ms=10, label="loose (> 50%)")
ax.plot(x, [r[2] for r in REAL], "-o", color=C["mf"], ms=10, label="tight (> 90%)")
ax.axhline(0.748, color=C["gnn"], ls=":", lw=1.6)
ax.axhline(0.103, color=C["mf"], ls=":", lw=1.6)
ax.text(3.0, 0.70, "per-quirk GNN, graph v3", ha="right", fontsize=12, color=C["ink2"])
ax.set_xticks(x, [r[0] for r in REAL])
ax.set_ylim(0, 1.02)
ax.grid(True, axis="y")
ax.legend(loc="lower right", fontsize=12)
ax.set_title("reconstructed pair tracks after the quirk tagger", fontsize=16, loc="left")
fig.subplots_adjust(left=0.07, right=0.99, top=0.92, bottom=0.16, wspace=0.18)
save(fig, "split_scaling")

# ---------------------------------------------------------------- error schematic
setup(15)
fig, ax = plt.subplots(figsize=(11.5, 3.6))
n = 14
wrong = {4, 5, 6, 10}  # a run of three and a single: about 2 runs, ~4 wrong of ~14 (ledger medians)
for i in range(n):
    col = C["B"] if i in wrong else C["A"]
    ax.scatter(i, 1, s=260, color=C["A"], marker="o", zorder=2)
    ax.scatter(i, 0, s=260, color=col, marker="s" if i in wrong else "o", zorder=2)
ax.text(-0.8, 1, "truth: quirk A's hits\nin time order", ha="right", va="center", fontsize=14)
ax.text(-0.8, 0, "splitter labels", ha="right", va="center", fontsize=14)
ax.annotate("", xy=(3.6, -0.45), xytext=(6.4, -0.45), arrowprops=dict(arrowstyle="<->", color=C["fake"], lw=2))
ax.text(5, -0.75, "segment swapped between two crossings", ha="center", fontsize=13, color=C["fake"])
ax.set_xlim(-6.5, n)
ax.set_ylim(-1.0, 1.5)
ax.axis("off")
fig.subplots_adjust(left=0.01, right=0.99, top=0.98, bottom=0.02)
save(fig, "split_errors")

# ---------------------------------------------------------------- B1 decomposition (fakes per event)
B1 = [  # model, fakes/ev, counts (a),(b),(c),(d),(e),(f)
    ("per-quirk GNN v3", 37.8, [930, 1276, 51, 36406, 74789, 8]),
    ("pair GNN", 24.8, [280, 1150, 19, 12382, 60536, 9]),
    ("MaskFormer, obj. threshold 0.1", 5.3, [546, 1894, 24, 1711, 11820, 15]),
    ("MaskFormer, obj. threshold 0.2", 2.3, [364, 1794, 15, 715, 4057, 6]),
]
CATS = [("quirk piece", C["A"]), ("two quirks", C["B"]), ("quirk secondaries", "#6d4c41"),
        ("merged ordinary", C["ink2"]), ("piece of an ordinary track", C["other"]), ("noise", C["fake"])]
setup(15)
fig, ax = plt.subplots(figsize=(13.9, 5.6))
y = np.arange(len(B1))[::-1]
for j, (m, fpe, cnt) in enumerate(B1):
    tot = sum(cnt)
    left = 0.0
    for k, (lab, col) in enumerate(CATS):
        w = fpe * cnt[k] / tot
        ax.barh(y[j], w, left=left, color=col, height=0.6, label=lab if j == 0 else None)
        left += w
    ax.text(left + 0.4, y[j], f"{fpe:.1f}", va="center", fontsize=14)
ax.set_yticks(y, [b[0] for b in B1])
ax.set_xlabel("fake tracks per event (strict, normal truth)")
ax.legend(loc="lower right", fontsize=13)
ax.set_xlim(0, 44)
fig.subplots_adjust(left=0.2, right=0.99, top=0.97, bottom=0.14)
save(fig, "b1_fakes")

# ---------------------------------------------------------------- B2 curves
B2 = "/pscratch/sd/d/danieltm/quirk_tracking/fakes/b2"


def curve(model, var):
    rows = list(csv.DictReader(open(f"{B2}/{model}/curve_{var}_test.csv")))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}


VAR = [("n_hits", "hit-count cut", C["ink2"], ":"), ("quality", "quality score", C["B"], "-"),
       ("quality_qw", "quality, quirk tracks x30", C["gnn"], "-"),
       ("quality_or_tag", "quality OR quirk tag", C["mf"], "--")]
PANELS = [("gnn_v3_s0_walk", "per-quirk GNN, graph v3", "fake_rate", "quirk_eff", "per-quirk efficiency", (0, 0.125), (0, 0.8)),
          ("pair_gnn_s0_cc", "pair GNN, connected components", "fake_rate_pair", "coupled_eff", "coupled efficiency", (0, 0.085), (0.8, 0.96)),
          ("mf_long_ot0.1", "MaskFormer, object threshold 0.1", "fake_rate_pair", "coupled_eff", "coupled efficiency", (0, 0.07), (0.8, 0.96))]
setup(14)
fig, axs = plt.subplots(1, 3, figsize=(15.6, 5.6))
for ax, (m, title, fx, fy, yl, xlim, ylim) in zip(axs, PANELS):
    for var, lab, col, ls in VAR:
        c = curve(m, var)
        ax.plot(c[fx], c[fy], ls, color=col, lw=2.2, label=lab)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_title(title, fontsize=16, loc="left")
    ax.set_xlabel("fake rate (strict)" + (", pair as one particle" if "pair" in fx else ""))
    ax.set_ylabel(yl)
    ax.grid(True)
axs[0].legend(loc="lower right", fontsize=12.5)
fig.subplots_adjust(left=0.06, right=0.99, top=0.91, bottom=0.14, wspace=0.28)
save(fig, "b2_curves")
record("split_fakes", {"val": VAL, "val16ep": VAL16EP, "valbig": VALBIG, "test": TEST, "test16_dir": TEST16_DIR, "real": REAL, "b1": B1})
