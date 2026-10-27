"""Two figures for the 2026-10-27 additions to the deck.

params_valley.png: per-event parameter fits with perfect hit labels (EXPERIMENTS.md "Quirk parameters with truth hit
assignment", fits/fits_table.csv): fitted minus true ln m (x) against fitted minus true ln(Lambda^2/m) (y), one point per
event, fits from a 3% start, no hit times; left the toy's own hits (closure), right the Geant4 hits; m = 100/200/300 GeV,
Lambda = 167 and 774 eV. Failed fits (chi2 more than 9 above the better start) are left out.

scaling_both.png: data-scaling arms (EXPERIMENTS.md "MaskFormer data scaling" and "GNN data scaling"), strict coupled
efficiency at loose / tight matching on test 17000-19999, against training events, with the fully trained 16k controls.
Numbers copied from the ledger tables (MaskFormer at object threshold 0.1).
"""
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qstyle import C, plt, save, setup  # noqa: E402

T = "/pscratch/sd/d/danieltm/quirk_tracking/params/fits/fits_table.csv"
rows = [r for r in csv.DictReader(open(T)) if r["times"] == "none" and abs(float(r["f"]) - 0.03) < 1e-6
        and r["fail"] == "False" and int(r["lam"]) in (167, 774)]
setup(14)
fig, axs = plt.subplots(1, 2, figsize=(15.6, 5.6), sharey=True)
for ax, hs, title in ((axs[0], "closure", "fit model's own hits (closure)"), (axs[1], "g4", "Geant4 hits")):
    for lam, col, mk in ((167, C["A"], "o"), (774, C["B"], "s")):
        rr = [r for r in rows if r["hits"] == hs and int(r["lam"]) == lam]
        x = np.array([float(r["dlnm"]) for r in rr])
        y = np.array([float(r["dlsm"]) for r in rr])
        out = int(np.sum((x < -1.5) | (x > 3.5) | (np.abs(y) > 0.3)))
        ax.scatter(np.clip(x, -1.5, 3.5), np.clip(y, -0.3, 0.3), s=40, color=col, marker=mk, edgecolor="white", lw=0.5,
                   label=f"Λ = {lam} eV ({len(rr)} events, {out} at the plot edge)")
    ax.axhline(0, color=C["muted"], lw=1)
    ax.axvline(0, color=C["muted"], lw=1)
    ax.set_title(title, fontsize=16, loc="left")
    ax.set_xlabel("fitted minus true ln m")
    ax.grid(True, alpha=0.4)
axs[0].set_ylabel("fitted minus true ln(Λ²/m)")
axs[0].set_ylim(-0.32, 0.32)
for ax in axs:
    ax.set_xlim(-1.6, 3.6)
for ax in axs:
    ax.legend(loc="upper right", fontsize=12)
fig.subplots_adjust(left=0.07, right=0.99, top=0.92, bottom=0.13, wspace=0.08)
save(fig, "params_valley")

# ---- scaling
N = [16, 50, 150]
MF = {"loose": [0.817, 0.807, 0.865], "tight": [0.245, 0.238, 0.309]}
MF16 = {"loose": (0.842, 0.026), "tight": (0.288, 0.043)}
GN = {"loose": [0.925, 0.915, 0.918], "tight": [0.624, 0.586, 0.619]}
GN16 = {"loose": (0.944, 0.003), "tight": (0.687, 0.006)}
fig, axs = plt.subplots(1, 2, figsize=(15.6, 5.6))
for ax, arms, ctl, title in ((axs[0], MF, MF16, "MaskFormer, pair as one target"), (axs[1], GN, GN16, "pair GNN, connected components")):
    for k, col in (("loose", C["gnn"]), ("tight", C["mf"])):
        ax.plot(N, arms[k], "-o", color=col, ms=9, lw=2, label=f"{k}: stopped early, learning rate not decayed")
        ax.errorbar([16 * 1.07], [ctl[k][0]], yerr=[ctl[k][1]], fmt="D", color=col, mfc="white", mew=2.2, ms=11, capsize=4,
                    label=f"{k}: 16k, fully trained (3 trainings)")
    ax.set_xscale("log")
    ax.set_xticks(N, ["16k", "50k", "150k"])
    ax.minorticks_off()
    ax.set_ylim(0, 1)
    ax.set_xlabel("training events")
    ax.set_title(title, fontsize=16, loc="left")
    ax.grid(True, axis="y", alpha=0.4)
axs[0].set_ylabel("coupled efficiency, test")
axs[1].legend(loc="lower right", fontsize=12)
fig.subplots_adjust(left=0.07, right=0.99, top=0.92, bottom=0.13, wspace=0.12)
save(fig, "scaling_both")
