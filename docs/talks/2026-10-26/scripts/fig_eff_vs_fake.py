"""STRICT matching. Quirk efficiency vs fake rate for every model on m200_L774_mu0 test events 17000-19999 (ACTS, proto-track level).

Two figures: per-quirk (normal truth) and coupled (pair-truth copy). Left panel very_loose (50%), right tight (90%);
x is always the very_loose fake rate of the row. Points with several seeds: mean +- sample std (error bars on both
axes). Ceilings (perfect edge scores, no model) as horizontal lines.
Numbers are copied from tracking/EXPERIMENTS.md (row ids in the comments); the Stage C baselines (ACTS CKF,
truth-seeded fits) and the LRT-style T1 arms are read from their summary.json files (paths below), which the
ledger table does not hold.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qstyle import C, plt, record, save, setup  # noqa: E402

SC = "/pscratch/sd/d/danieltm/quirk_tracking/stageC"
T1 = "/pscratch/sd/d/danieltm/quirk_tracking/gnn/main/{arm}/test_gnn/eval_optuned_mh5/summary.json"


def acts(path, wp="very_loose"):
    s = json.load(open(path))
    return s["quirk_track_efficiency"][wp]["eff"], s["fake_rate"]["very_loose"]


ckf_vl, ckf_f = acts(f"{SC}/eval_ckf/summary.json")
ckf_t, _ = acts(f"{SC}/eval_ckf/summary.json", "tight")
tt_vl, tt_f = acts(f"{SC}/eval_truth_tracks/summary.json")
tt_t, _ = acts(f"{SC}/eval_truth_tracks/summary.json", "tight")
ckfp_vl, ckfp_f = acts(f"{SC}/eval_ckf_pair/summary.json")
ckfp_t, _ = acts(f"{SC}/eval_ckf_pair/summary.json", "tight")
ttp_vl, ttp_f = acts(f"{SC}/eval_truth_tracks_pair/summary.json")
ttp_t, _ = acts(f"{SC}/eval_truth_tracks_pair/summary.json", "tight")
t1 = {a: (acts(T1.format(arm=a)), acts(T1.format(arm=a), "tight")) for a in ("t1_qq10", "t1_qq0.1")}

# STRICT matching (> 50%, > 90%), tracking/EXPERIMENTS.md "Strict matching" table and "Strict ACTS reruns";
# per-seed values from /pscratch/sd/d/danieltm/quirk_tracking/strict50/strict_main.csv (gt_vl, gt_tight,
# acts_fake_vl_strict). Rows: (label, colour, marker, [eff loose], [eff tight], [fake], fake_is_strict).
# Where no strict ACTS rerun exists, the >= fake rate is used and the marker is hollow.
PER_QUIRK = [
    ("GNN v2, all tracks (2 seeds; fake: seed 0)", C["gnn2"], "o", [0.650, 0.641], [0.065, 0.064], [0.1335], True),
    ("GNN v2, no cluster shape", C["gnn2"], "v", [0.627], [0.057], [0.113], False),
    ("GNN v2 + direction (2 seeds)", C["gnn"], "D", [0.656, 0.662], [0.065, 0.068], [0.1305, 0.1289], True),
    ("GNN graph v3 + direction (2 seeds)", C["gnn"], "o", [0.746, 0.751], [0.098, 0.109], [0.1175, 0.1164], True),
    ("GNN, LRT-style, 2 arms, ≥ 5 hits", "#9e9d24", "P", [0.453, 0.415], [0.013, 0.007],
     [t1["t1_qq10"][0][1], t1["t1_qq0.1"][0][1]], False),
    ("MaskFormer, per-quirk targets", C["mf2"], "s", [0.110], [0.001], [0.092], False),
    ("MaskFormer, pair + dir, 100k", C["mf"], "s", [0.214], [0.000], [0.0715], True),
    ("ACTS CKF", C["acts"], "^", [0.001], [0.000], [ckf_f], False),
    ("ACTS truth-seeded fits", C["acts"], "<", [0.271], [0.131], [tt_f], False),
]
PER_QUIRK_CEIL = [("perfect scores, graph v2", 0.887, 0.601), ("perfect scores, graph v3", 0.966, 0.828),
                  ("graph v3, components", 0.976, 0.854)]

COUPLED = [
    ("GNN v2, per-quirk training", C["gnn2"], "o", [0.440], [0.006], [0.109], False),
    ("pair GNN, walk + stitch (3 seeds)", C["gnn"], "D", [0.909, 0.914, 0.913], [0.120, 0.122, 0.114],
     [0.1299, 0.1279, 0.1298], True),
    ("pair GNN, components (3 seeds)", C["gnn"], "o", [0.941, 0.944, 0.947], [0.692, 0.689, 0.681],
     [0.0787, 0.0774, 0.0795], True),
    ("pair GNN on graph v3, components", C["gnn"], "h", [0.980], [0.813], [0.1892], True),
    ("MaskFormer, per-quirk targets", C["mf2"], "s", [0.165], [0.007], [0.092], False),
    ("MaskFormer, pair target (3 seeds)", C["mf2"], "D", [0.846, 0.815, 0.866], [0.299, 0.241, 0.325],
     [0.0785, 0.0862, 0.0769], True),
    ("MaskFormer, pair + dir (3 seeds)", C["mf"], "D", [0.841, 0.815, 0.849], [0.289, 0.322, 0.404],
     [0.0759, 0.0812, 0.0657], True),
    ("MaskFormer, pair + dir, 100k, ot 0.1", C["mf"], "s", [0.907], [0.553], [0.0651], True),
    ("MaskFormer, pair + dir, 100k, ot 0.2", C["mf"], "X", [0.900], [0.556], [0.0276], True),
    ("ACTS CKF", C["acts"], "^", [0.001], [0.000], [ckfp_f], False),
    ("ACTS truth-seeded fits", C["acts"], "<", [0.065], [0.000], [ttp_f], False),
]
COUPLED_CEIL = [("same-pair edges, v3, components", 0.998, 0.974),
                ("same, pair GNN graph", 0.989, 0.877)]


def ms(v):
    v = np.asarray(v, float)
    return v.mean(), (v.std(ddof=1) if len(v) > 1 else 0.0)


def draw(rows, ceils, name, title):
    setup(14)
    fig, axs = plt.subplots(1, 2, figsize=(15.6, 6.4), sharex=True)
    rec = {}
    for k, ax in enumerate(axs):
        for lab, col, mk, vl, tg, fk, fstrict in rows:
            eff = vl if k == 0 else tg
            if eff is None:
                continue
            (y, ye), (x, xe) = ms(eff), ms(fk)
            ax.errorbar(x, y, yerr=ye or None, xerr=xe or None, fmt=mk, color=col, ms=11,
                        mec=col if not fstrict else "white", mfc=col if fstrict else "white", mew=2.0 if not fstrict else 1.0,
                        elinewidth=2, capsize=4, label=lab + ("" if fstrict else " (old-rule fake)"), zorder=3)
            rec.setdefault(lab, {})["vl" if k == 0 else "tight"] = [round(y, 4), round(ye, 4)]
            rec[lab]["fake"] = [round(x, 4), round(xe, 4)]
        for j, (lab, y50, y90) in enumerate(ceils):
            y = y50 if k == 0 else y90
            ax.axhline(y, color=C["ceil"], lw=1.6, ls=[(0, (6, 3)), (0, (1, 2)), (0, (3, 1, 1, 1))][j], zorder=1,
                       label=f"ceiling: {lab}")
        ax.set_xlim(-0.005, 0.265)
        ax.set_ylim(-0.02, 1.06)
        ax.grid(True, axis="y")
        ax.set_xlabel("fake rate (loose matching)")
        ax.set_title("loose matching (> 50%)" if k == 0 else "tight matching (> 90%)", fontsize=16)
    axs[0].set_ylabel(title)
    h, lab = axs[0].get_legend_handles_labels()
    for ax in axs:  # ceiling values at the right edge, per panel
        for j, (clab, y50, y90) in enumerate(ceils):
            y = y50 if ax is axs[0] else y90
            pass  # values are on the slide
    fig.legend(h, lab, loc="center left", bbox_to_anchor=(0.705, 0.5), fontsize=12, handletextpad=0.3,
               labelspacing=0.75)
    fig.subplots_adjust(left=0.055, right=0.70, top=0.92, bottom=0.11, wspace=0.12)
    save(fig, name)
    record(name, rec)


draw(PER_QUIRK, PER_QUIRK_CEIL, "effake_perquirk", "per-quirk efficiency")
draw(COUPLED, COUPLED_CEIL, "effake_coupled", "coupled efficiency")
