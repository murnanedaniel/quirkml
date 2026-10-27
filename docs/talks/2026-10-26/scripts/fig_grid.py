"""STRICT matching (> 50% / > 90%; tracking/grid/gnn_grid_strict.csv, ledger section "Strict matching").
GNN grid generalisation (Stage F, no retraining) from tracking/grid/gnn_grid.csv: 110 test sets of 2000 events.

Per-quirk: graph v3 + GNN g2_q50_dir_s0, walk + stitch 0.7, >= 4 hits. Coupled: pair-arm graph + pair GNN
e2_pair_dir_s0, components at 0.5, >= 4 hits, on the pair-truth copy. ACTS, very_loose (50%) / tight (90%).
Writes grid_perquirk.png, grid_coupled.png (maps) and grid_slices.png (vs Lambda at four masses).
"""
import csv
import os
import sys

import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qstyle import C, plt, record, save, setup  # noqa: E402

CSV = "/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/grid/gnn_grid_strict.csv"
rows = [r for r in csv.DictReader(open(CSV)) if r["m"] != "400"]  # m400 L1000 = development point, off the grid
for r in rows:  # strict columns replace the >= ones; fraction of quirks / pairs matched only at exactly 50%
    for c in ("quirk_eff", "quirk_eff_tight", "quirk_event_eff", "coupled_eff", "coupled_eff_tight"):
        r[c] = r[c + "_strict"]
    r["frac_at50"] = float(r["quirks_at_50"]) / float(r["quirks"])
    r["pair_frac_at50"] = float(r["pairs_at_50"]) / float(r["pairs"])
# the training point is filled from its own test set (17000-19999), as tracking/grid/plot_gnn_grid.py does:
# strict table rows gnn-g2-v3-dir-s0 (per-quirk: 0.746 / 0.098, event 0.620, 383 of 5962 at exactly 50%) and
# gnn-e2-pair-dir-s0-cc (coupled 0.941 / 0.692, 15 of 2991 pairs at 50%); strict ACTS fake 0.117 / coupled 0.079
rows.append(dict(m="200", Lambda="774", quirk_eff="0.746", quirk_eff_tight="0.098", quirk_event_eff="0.620",
                 coupled_eff="0.941", coupled_eff_tight="0.692", fake="0.117", coupled_fake="0.079",
                 technical_sm="0.991", frac_at50=383 / 5962, pair_frac_at50=15 / 2991))
M = sorted({int(r["m"]) for r in rows})
L = sorted({int(r["Lambda"]) for r in rows})
CMAP = LinearSegmentedColormap.from_list("q", ["#f7f6f2", "#9ec2e6", "#1f5fbf", "#0d2f66"])


def grid(key):
    g = np.full((len(L), len(M)), np.nan)
    for r in rows:
        g[L.index(int(r["Lambda"])), M.index(int(r["m"]))] = float(r[key])
    return g


def heat(ax, g, title, vmax=1.0):
    ax.imshow(g, origin="lower", cmap=CMAP, vmin=0, vmax=vmax, aspect="auto")
    for i in range(len(L)):
        for j in range(len(M)):
            v = g[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=12.5,
                    color="white" if v > 0.55 * vmax else C["ink"])
    ax.add_patch(Rectangle((M.index(200) - 0.5, L.index(774) - 0.5), 1, 1, fill=False, ec=C["B"], lw=3))
    ax.set_xticks(range(len(M)), M)
    ax.set_yticks(range(len(L)), L)
    ax.set_xlabel("quirk mass m [GeV]")
    ax.set_title(title, fontsize=16)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)


def maps(keys, titles, name, vmaxes=(1.0, 1.0)):
    setup(14)
    fig, axs = plt.subplots(1, 2, figsize=(15.6, 6.6))
    for ax, k, t, vm in zip(axs, keys, titles, vmaxes):
        heat(ax, grid(k), t, vm)
    axs[0].set_ylabel("Λ [eV]")
    fig.subplots_adjust(left=0.06, right=0.995, top=0.93, bottom=0.10, wspace=0.10)
    save(fig, name)


maps(("quirk_eff", "quirk_eff_tight"), ("per-quirk efficiency, loose (> 50%)", "per-quirk efficiency, tight (> 90%)"),
     "grid_perquirk_strict")
maps(("coupled_eff", "coupled_eff_tight"), ("coupled efficiency, loose (> 50%)", "coupled efficiency, tight (> 90%)"),
     "grid_coupled_strict")
maps(("quirk_event_eff", "frac_at50"), ("both quirks found, loose matching (> 50%)",
                                        "quirks found with the old ≥ 50% rule, not at loose"),
     "grid_event_at50_strict", vmaxes=(1.0, 0.2))

# slices vs Lambda
setup(15)
fig, axs = plt.subplots(1, 3, figsize=(15.6, 5.6), sharey=True)
mcol = {50: "#9ec2e6", 125: "#5a8fd6", 200: "#1f5fbf", 300: "#0d2f66"}
for ax, (k, t) in zip(axs, (("quirk_eff", "per-quirk, loose (> 50%)"), ("quirk_eff_tight", "per-quirk, tight (> 90%)"),
                            ("coupled_eff", "coupled, loose (> 50%)"))):
    g = grid(k)
    for m, col in mcol.items():
        ax.plot(L, g[:, M.index(m)], "-o", color=col, ms=6, label=f"m = {m} GeV")
    if k == "coupled_eff":
        gt = grid("coupled_eff_tight")
        ax.plot(L, gt[:, M.index(200)], "--o", color=C["B"], ms=6, label="m = 200 GeV, tight")
    ax.set_xscale("log")
    ax.set_title(t, fontsize=16)
    ax.set_xlabel("Λ [eV]")
    ax.grid(True, axis="y")
    ax.axvline(774, color=C["B"], lw=1.2, ls=":")
axs[0].set_ylabel("efficiency")
axs[0].set_ylim(0, 1.03)
axs[2].legend(loc="lower right", fontsize=13)
fig.subplots_adjust(left=0.06, right=0.995, top=0.91, bottom=0.14, wspace=0.08)
save(fig, "grid_slices_strict")
record("grid_strict", {k: {f"m{m}_L{lam}": round(float(grid(k)[i, j]), 4) for i, lam in enumerate(L) for j, m in enumerate(M)}
                for k in ("quirk_eff", "quirk_eff_tight", "coupled_eff", "coupled_eff_tight", "quirk_event_eff", "frac_at50", "pair_frac_at50")})
f = np.array([float(r["fake"]) for r in rows])
cf = np.array([float(r["coupled_fake"]) for r in rows])
ts = np.array([float(r["technical_sm"]) for r in rows])
print("fake", f.min(), f.max(), "coupled fake", cf.min(), cf.max(), "tech_sm", ts.min(), ts.max(), len(rows))
