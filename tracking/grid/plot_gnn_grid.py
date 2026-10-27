"""Maps of the GNN grid results (gnn_grid.csv) over (m, Lambda): per-quirk loose/tight and quirk event efficiency of the
v3 GNN (seed 0, walk + stitch, >= 4 hits) and coupled loose/tight of the pair GNN (seed 0, components at 0.5, >= 4
hits). The training point m200 Lambda 774 eV is filled from its test numbers (17000-19999) and outlined.
m400 Lambda 1000 eV (development point, off the grid) is left out of the maps.
  python plot_gnn_grid.py            # >= 50% / >= 90% (gnn_grid.csv), gnn_grid_<metric>.png
  python plot_gnn_grid.py --strict   # > 50% / > 90%, the standard from 2026-10-26 (gnn_grid_strict.csv),
                                     # gnn_grid_<metric>_strict.png
"""
import argparse
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--strict", action="store_true", help="> 50% / > 90% matching (gnn_grid_strict.csv)")
STRICT = ap.parse_args().strict
SUF = "_strict" if STRICT else ""
d = pd.read_csv(os.path.join(HERE, "gnn_grid_strict.csv" if STRICT else "gnn_grid.csv"))
d = d[d.m != 400]
if STRICT:  # the plotted columns hold the strict numbers
    for c in ("quirk_eff", "quirk_eff_tight", "quirk_event_eff", "coupled_eff", "coupled_eff_tight"):
        d[c] = d[c + "_strict"]
    # training point, test 17000-19999 (strict50/strict_main.csv: g2_q50_dir_s0 stitch 0.7 mh4, e2_pair_dir_s0 cc 0.5 mh4)
    train = dict(m=200, Lambda=774, quirk_eff=0.746, quirk_eff_tight=0.098, quirk_event_eff=0.620, coupled_eff=0.942,
                 coupled_eff_tight=0.692)
else:
    train = dict(m=200, Lambda=774, quirk_eff=0.810, quirk_eff_tight=0.103, quirk_event_eff=0.688, coupled_eff=0.947,
                 coupled_eff_tight=0.698)
d = pd.concat([d, pd.DataFrame([train])], ignore_index=True)
L, T = ("> 50%", "> 90%") if STRICT else ("50%", "90%")
panels = [("quirk_eff", f"Per-quirk efficiency, very_loose ({L})"), ("quirk_eff_tight", f"Per-quirk efficiency, tight ({T})"),
          ("quirk_event_eff", f"Quirk event efficiency (both quirks, {L})"),
          ("coupled_eff", f"Coupled efficiency, very_loose ({L})"), ("coupled_eff_tight", f"Coupled efficiency, tight ({T})")]
ms, ls = sorted(d.m.unique()), sorted(d.Lambda.unique())
BG, INK, MUTED = "#fcfcfb", "#0b0b0b", "#52514e"


def draw(ax, col, title):
    z = d.pivot(index="Lambda", columns="m", values=col).reindex(index=ls, columns=ms).values
    im = ax.imshow(z, origin="lower", cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for i in range(len(ls)):
        for j in range(len(ms)):
            if np.isfinite(z[i, j]):
                ax.text(j, i, f"{z[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="#fcfcfb" if z[i, j] > 0.6 else INK)
    j, i = ms.index(200), ls.index(774)
    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec="#eb6834", lw=2))
    ax.set_xticks(range(len(ms)), [str(m) for m in ms], fontsize=8, color=MUTED)
    ax.set_yticks(range(len(ls)), [str(l) for l in ls], fontsize=8, color=MUTED)
    ax.set_xlabel("quirk mass m [GeV]", color=MUTED)
    ax.set_ylabel("Lambda [eV]", color=MUTED)
    ax.set_title(title, color=INK, fontsize=10)
    for s in ax.spines.values():
        s.set_visible(False)
    return im


for col, title in panels:
    fig, ax = plt.subplots(figsize=(7.5, 5.2), facecolor=BG)
    im = draw(ax, col, title)
    fig.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, f"gnn_grid_{col}{SUF}.png"), dpi=140, facecolor=BG)
    plt.close(fig)
fig, axs = plt.subplots(2, 3, figsize=(20, 10.5), facecolor=BG)
for ax, (col, title) in zip(axs.flat, panels):
    im = draw(ax, col, title)
axs.flat[-1].axis("off")
axs.flat[-1].text(0.02, 0.6, "GNN, no retraining (trained on m200, Lambda 774 eV: orange box,\nfilled from its own test set).\n"
                  "Per-quirk: graph v3 + GNN seed 0, walk + stitch, >= 4 hits.\nCoupled: pair GNN seed 0, components at 0.5, >= 4 hits.\n"
                  "2000 test events per point; "
                  + ("strict matching (> 50% / > 90%), Python = ACTS." if STRICT else "ACTS."), fontsize=11, color=MUTED, va="top")
fig.colorbar(im, ax=axs, fraction=0.015)
fig.savefig(os.path.join(HERE, f"gnn_grid_maps{SUF}.png"), dpi=120, facecolor=BG)
print("written")
