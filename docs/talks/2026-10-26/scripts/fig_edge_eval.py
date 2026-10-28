"""Edge-level efficiency vs purity of the quirk GNNs on test 17000-19999 (from tracking/gnn/edge_eval.py outputs).

Figures: edge_eff_purity.png (three panels: quirk edges with the per-quirk target, quirk edges with the pair target,
ordinary edges; curves over the score cut, markers at the track-building cuts) and edge_cross_kept.png (per-quirk
models: fraction of same-quirk consecutive edges and of cross-quirk graph edges above the cut). The table at the
operating cuts goes to figs/numbers.json under "edge_eval".

Definitions (edge_eval.py): efficiency = edges of the class above the cut / all truth edges of the class (including
those the graph misses); consecutive = neighbouring hits of one particle in true time order (pair: of the merged,
time-ordered sequence of both quirks' hits); purity "same particle" = both hits of one particle (for the pair models
the pair is one particle), so same-particle edges that are not consecutive (masked in training) count as true.

Run with ~/.conda/envs/collider-env/bin/python (or any env with numpy + matplotlib).
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qstyle import C, plt, record, save, setup  # noqa: E402

RES = "/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirk_tracking_data/results/edge_eval"
R = json.load(open(os.path.join(RES, "edge_eval.json")))
CV = np.load(os.path.join(RES, "curves.npz"))
CUTS = R["cuts"]

GROUPS = {  # label, seeds, colour, linestyle, operating cuts
    "pq_v2": ("per-quirk GNN, graph v2", ["v2"], "#eda100", "-", [0.5]),
    "pq_v3": ("per-quirk GNN, graph v3", ["v3_s0", "v3_s1"], "#1baf7a", "-", [0.5]),
    "pair_own": ("pair GNN, its own graph", ["e2_s0", "e2_s1", "e2_s2"], "#6250d6", "--", [0.5]),
    "pair_v3": ("pair GNN, graph v3", ["p3"], "#e87ba4", "--", [0.5]),
}
MK = {0.1: "o", 0.5: "s", 0.6: "s", 0.7: "D"}


def curve(m, k, split="test", v="primary"):
    return CV[f"{m}|{split}|{v}|{k}"]


def at(m, k, cut, split="test", v="primary"):
    return R["models"][m][split]["at_cuts"][v][k][CUTS.index(cut)]


def frac_cross(m, cut, split="test", v="primary"):
    a = R["models"][m][split]["at_cuts"][v]
    return a["n_cross_kept"][CUTS.index(cut)] / a["cross_in_graph"]


def frac_pair_any(m, cut, split="test", v="primary"):
    a = R["models"][m][split]["at_cuts"][v]
    j = CUTS.index(cut)
    return (a["n_cross_kept"][j] + a["n_samequirk_any_kept"][j]) / (a["cross_in_graph"] + a["samequirk_any_in_graph"])


def frac_samequirk_any(m, cut, split="test", v="primary"):
    a = R["models"][m][split]["at_cuts"][v]
    return a["n_samequirk_any_kept"][CUTS.index(cut)] / a["samequirk_any_in_graph"]


def panel(ax, groups, xk, yk, title, xlabel, ylabel, xlim, ylim):
    lo, hi = 10, 990  # cuts 0.01 .. 0.99
    for g in groups:
        lab, ms, col, ls, ops = GROUPS[g]
        for i, m in enumerate(ms):
            x, y = curve(m, xk)[lo:hi], curve(m, yk)[lo:hi]
            ax.plot(x, y, color=col, ls=ls, lw=2.2, label=lab if i == 0 else None)
            for c in ops:
                ax.plot(at(m, xk, c), at(m, yk, c), MK[c], ms=10, color=col, mec="white", mew=1.5, zorder=5)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_title(title, loc="left", color=C["ink"])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True)


def fig_main():
    setup(15)
    fig, axs = plt.subplots(1, 3, figsize=(13.87, 3.75), layout="constrained")
    panel(axs[0], ["pq_v2", "pq_v3"], "eff_samequirk", "purity_quirk_same", "GNN trained per quirk",
          "quirk edge efficiency", "purity, edges with a quirk hit", (0.3, 1.0), (0.4, 1.0))
    panel(axs[1], ["pair_own", "pair_v3"], "eff_pairchain", "purity_quirk_same", "GNN trained on the pair",
          "quirk pair edge efficiency", "purity, edges with a quirk hit", (0.3, 1.0), (0.4, 1.0))
    panel(axs[2], list(GROUPS), "eff_ordinary", "purity_all_same", "Ordinary particles",
          "ordinary-particle edge efficiency", "purity, all edges", (0.75, 1.0), (0.75, 1.0))
    # model legend in the empty lower-left of the pair panel, marker legend in the per-quirk panel
    h, l = [], []
    for ax in axs:
        for a_, b_ in zip(*ax.get_legend_handles_labels()):
            if b_ not in l:
                h.append(a_)
                l.append(b_)
    axs[1].legend(h, l, loc="lower left", fontsize=12.5, handlelength=1.8)
    mh = [axs[0].plot([], [], MK[c], color=C["ink2"], ms=9)[0] for c in (0.5,)]
    axs[0].legend(mh, ["score cut 0.5"],
                  loc="lower left", fontsize=12.5)
    save(fig, "edge_eff_purity")


def fig_cross():
    setup(15)
    fig, ax = plt.subplots(figsize=(7.4, 5.4), layout="constrained")
    cuts = np.arange(1000) / 1000
    for g in ["pq_v2", "pq_v3"]:
        lab, ms, col, _, ops = GROUPS[g]
        for i, m in enumerate(ms):
            nk = curve(m, "n_cross_kept")
            ax.plot(cuts[10:990], (nk / nk[0])[10:990], color=col, ls="--", lw=2.2)
            ax.plot(cuts[10:990], curve(m, "eff_samequirk")[10:990], color=col, ls="-", lw=2.2,
                    label=lab if i == 0 else None)
    ax.plot([], [], "-", color=C["ink2"], label="same-quirk consecutive edges (efficiency)")
    ax.plot([], [], "--", color=C["ink2"], label="cross-quirk edges kept (of those in the graph)")
    for c in (0.1, 0.6):
        ax.axvline(c, color=C["axis"], lw=1.2, zorder=0)
    ax.set_xlabel("edge score cut")
    ax.set_ylabel("fraction above the cut")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True)
    fig.legend(loc="outside lower center", ncol=2, fontsize=12)
    save(fig, "edge_cross_kept")


def mean_sd(vals):
    v = np.asarray(vals, float)
    return [float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else None]


def table():
    out = {"definitions": __doc__, "test_events": "17000-19999", "val_events": "16000-16999", "rows": []}
    for g, (lab, ms, _, _, ops) in GROUPS.items():
        for c in sorted(set(ops + [0.5])):
            row = {"model": lab, "seeds": ms, "cut": c, "operating": c in ops}
            for split in ["test", "val"]:
                d = {}
                for k in ["eff_ordinary", "eff_ordinary_tech", "eff_samequirk", "eff_pairchain", "eff_pairchain_cross",
                          "purity_all_same", "purity_quirk_same", "purity_all_target", "purity_quirk_target"]:
                    d[k] = mean_sd([at(m, k, c, split) for m in ms])
                d["frac_cross_kept_in_graph"] = mean_sd([frac_cross(m, c, split) for m in ms])
                d["cross_kept_per_event"] = mean_sd([R["models"][m][split]["at_cuts"]["primary"]["n_cross_kept"][CUTS.index(c)]
                                                     / R["models"][m][split]["n_events"] for m in ms])
                d["cross_in_graph_per_event"] = mean_sd([R["models"][m][split]["at_cuts"]["primary"]["cross_in_graph"]
                                                         / R["models"][m][split]["n_events"] for m in ms])
                d["frac_pair_any_kept_in_graph"] = mean_sd([frac_pair_any(m, c, split) for m in ms])
                d["frac_samequirk_any_kept_in_graph"] = mean_sd([frac_samequirk_any(m, c, split) for m in ms])
                d["quirk_rule"] = {k: mean_sd([at(m, k, c, split, "rule") for m in ms])
                                   for k in ["eff_samequirk", "eff_pairchain_cross", "purity_quirk_same", "purity_all_same"]}
                d["quirk_rule"]["frac_cross_kept_in_graph"] = mean_sd([frac_cross(m, c, split, "rule") for m in ms])
                row[split] = d
            out["rows"].append(row)
    out["val_vs_training_log_at_0.5"] = {
        m: {"training_log": {k: R["models"][m]["train_cut_scan_at_0.5"][k] for k in
                             ["eff", "eff_quirk", "eff_tech", "purity", "purity_quirk"]},
            "recomputed_val_16000_16199": {k: R["models"][m]["val200"]["logdef_at_0.5"][k] for k in
                                           ["eff", "eff_quirk", "eff_tech", "purity", "purity_quirk"]},
            "val_16000_16999": {k: R["models"][m]["val"]["logdef_at_0.5"][k] for k in
                                ["eff", "eff_quirk", "eff_tech", "purity", "purity_quirk"]},
            "test": {k: R["models"][m]["test"]["logdef_at_0.5"][k] for k in
                     ["eff", "eff_quirk", "eff_tech", "purity", "purity_quirk"]}}
        for m in R["models"]}
    record("edge_eval", out)
    json.dump(out, open(os.path.join(RES, "edge_table.json"), "w"), indent=1)
    for r in out["rows"]:
        t = r["test"]
        f = lambda k: "%.3f" % t[k][0] + ("" if t[k][1] is None else "+-%.3f" % t[k][1])  # noqa: E731
        print(f"{r['model']:28s} cut {r['cut']:.1f} op={r['operating']!s:5s} ord {f('eff_ordinary')} sameq {f('eff_samequirk')}"
              f" pairchain {f('eff_pairchain')} pccross {f('eff_pairchain_cross')} crosskept {f('frac_cross_kept_in_graph')}"
              f" crossPerEv {f('cross_kept_per_event')} pairany {f('frac_pair_any_kept_in_graph')} pur_all {f('purity_all_same')} pur_q {f('purity_quirk_same')}"
              f" | consec pur_all {f('purity_all_target')} pur_q {f('purity_quirk_target')}")


if __name__ == "__main__":
    fig_main()
    fig_cross()
    table()
