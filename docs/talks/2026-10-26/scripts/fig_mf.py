"""MaskFormer figures (STRICT matching for every efficiency).

mf_val_curves.png: val (16000-16999) efficiency vs fake rate over the object threshold (ot 0.1, 0.2, 0.3, 0.5;
mask threshold 0.5), ACTS. Left: per-quirk efficiency of the per-quirk-target variants (fixed loss). Right: coupled
efficiency (Python coupled_eval) of the pair-target LR scan. Values from tracking/EXPERIMENTS.md rows (ids below).
mf_long.png: the 100k-step pair + direction run (mf-E-pairdir-long): train loss, final-layer val loss and LR vs
step from its two metrics.csv files (resumed once at 70k), with the val coupled efficiency of three checkpoints
(ledger row).
"""
import csv
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qstyle import C, plt, record, save, setup  # noqa: E402

OT = [0.1, 0.2, 0.3, 0.5]
# STRICT matching (> 50%), val 16000-16999, by object threshold; Python strict scoring of the kept predictions
# (/pscratch/sd/d/danieltm/quirk_tracking/strict50/strict_main.csv: gt_vl for per-quirk, ce_gt for coupled).
# No strict fake rates exist for these val runs, so efficiency is plotted against the object threshold.
PERQ = [  # (label, colour, marker, strict per-quirk eff per ot)
    ("per-quirk targets, 30k (control)", C["mf"], "o", [.1091, .1101, .0995, .0603]),  # t2fix_lr5e-5
    ("100k steps", "#8e7cc3", "s", [.1101, .1065, .0754, .0397]),  # t2_long_lr5e-5
    ("quirk weight 10, LR 5e-5", "#9e9d24", "v", [.0894, .0889, .0899, .0930]),  # t2qw10_lr5e-5
    ("quirk weight 10, LR 1e-4", "#c0ca33", "^", [.1131, .1131, .1121, .1116]),  # t2qw10_lr1e-4
    ("no origin features", C["acts"], "D", [.1251, .1231, .1186, .0869]),  # t2d_noorigin
    ("no origin + direction", C["gnn"], "P", [.1367, .1533, .1473, .1035]),  # t2d_dir
    ("batch 1", C["muted"], "x", [.0583, .0532, .0437, .0]),  # t2d_bs1
]
PAIR = [  # strict coupled efficiency (Python) on val, pair-target LR scan
    ("pair target, LR 2.5e-5", "#b6a2dd", "v", [.8308, .8197, .7341, .5085]),  # t2d_pair_lr2.5e-5
    ("pair target, LR 5e-5", C["mf"], "o", [.8499, .8147, .7270, .4259]),  # t2d_pair_lr5e-5
    ("pair target, LR 1e-4", "#2e1d5c", "s", [.7129, .7180, .6797, .4169]),  # t2d_pair_lr1e-4
]

setup(14)
fig, axs = plt.subplots(1, 2, figsize=(15.6, 6.2))
for ax, rows, yl in ((axs[0], PERQ, "per-quirk efficiency, loose"), (axs[1], PAIR, "coupled efficiency, loose")):
    for lab, col, mk, eff in rows:
        ax.plot(OT, eff, "-" + mk, color=col, ms=10, mec="white" if mk != "x" else col, mew=2 if mk == "x" else 1,
                label=lab)
    ax.set_xlabel("object threshold")
    ax.set_xticks(OT)
    ax.set_ylabel(yl)
    ax.grid(True, axis="y")
    ax.legend(loc="lower left", fontsize=12.5) if ax is axs[1] else ax.legend(loc="upper center", ncol=2, fontsize=12.5)
axs[0].set_ylim(0, 0.27)
axs[1].set_ylim(0.35, 0.9)
axs[0].set_title("per-quirk targets: variants", fontsize=16)
axs[1].set_title("pair as one target: LR scan", fontsize=16)
fig.subplots_adjust(left=0.06, right=0.99, top=0.92, bottom=0.12, wspace=0.18)
save(fig, "mf_val_curves_strict")

# long run
RUN = "/pscratch/sd/d/danieltm/quirk_tracking/maskformer/runs/t2e_pairdir_long"
tr, va, lr = {}, {}, {}
for f in sorted(glob.glob(f"{RUN}/*/t2e_pairdir_long/version_0/metrics.csv")):
    for x in csv.DictReader(open(f)):
        st = int(x["step"])
        if x.get("train/final_loss"):
            tr[st] = float(x["train/loss"]) if x.get("train/loss") else np.nan
        if x.get("val/final_loss"):
            va[st] = float(x["val/final_loss"])
        if x.get("lr-Lion"):
            lr[st] = float(x["lr-Lion"])
ts = np.array(sorted(tr))
tl = np.array([tr[s] for s in ts])
k = 200
tsm = np.convolve(tl, np.ones(k) / k, mode="valid")
vs = np.array(sorted(va))
vl = np.array([va[s] for s in vs])
setup(15)
fig, axs = plt.subplots(1, 3, figsize=(15.6, 5.4))
axs[0].plot(ts[k - 1:] / 1e3, tsm, color=C["ink2"], lw=1.6)
axs[0].set_title("train loss (all layers, 200-step mean)", fontsize=15)
axs[0].set_ylim(np.nanpercentile(tsm, 1) * 0.95, np.nanpercentile(tsm, 99.5))
axs[1].plot(vs / 1e3, vl, "-o", color=C["mf"], ms=4)
axs[1].set_title("val final-layer loss", fontsize=15)
axs[1].set_ylim(0.5, 1.0)
ls = np.array(sorted(lr))
axb = axs[1].twinx()
axb.plot(ls / 1e3, [lr[s] * 1e5 for s in ls], color=C["B"], lw=1.6)
axb.set_ylabel("LR [1e-5]", color=C["B"])
axb.tick_params(axis="y", colors=C["B"])
axb.spines["right"].set_visible(True)
# strict val coupled (Python, ot 0.1 / 0.2): 92k = best-val-loss ckpt (t2e_pairdir_long_val), 100k = final (e049);
# strict_main.csv ce_gt. The 60k checkpoint (>= 0.877 / 0.874 in the ledger) was not rescored strictly.
ck = {92: (0.8982, 0.8892), 100: (0.9073, 0.9033)}
axs[2].plot(list(ck), [v[0] for v in ck.values()], "-o", color=C["mf"], ms=9, label="ot 0.1")
axs[2].plot(list(ck), [v[1] for v in ck.values()], "-s", color=C["mf2"], ms=9, label="ot 0.2")
axs[2].set_title("val coupled eff., loose, by checkpoint", fontsize=15)
axs[2].set_ylim(0.87, 0.92)
axs[2].set_xlim(84, 102)
axs[2].legend(loc="lower right")
for ax in axs:
    ax.set_xlabel("step [k]")
    ax.grid(True, axis="y")
    ax.axvspan(85, 100, color=C["grid"], alpha=0.6, lw=0)
fig.subplots_adjust(left=0.05, right=0.99, top=0.90, bottom=0.14, wspace=0.32)
save(fig, "mf_long_strict")
i92 = int(np.argmin(np.abs(vs - 92000)))
record("mf_long", {"val_final_loss_min": float(vl.min()), "step_of_min": int(vs[np.argmin(vl)]),
                   "val_final_loss_last": float(vl[-1]), "val_final_loss_92k": float(vl[i92]),
                   "val_steps": len(vs)})
print(vl.min(), vs[np.argmin(vl)], vl[-1], vl[i92])
