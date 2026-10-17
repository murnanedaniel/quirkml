"""Fig 10: GNN chain quirk track efficiency vs fake rate (ACTS evaluator, proto-track level), m = 200 GeV,
Lambda = 774 eV. v1 on test events 17000-17199 (200 events); v2 (cluster = node_edge) on 17000-19999 (3000 events),
the v2 points on the same 200 events as v1 go to numbers.json only (not drawn). Track builders: ccwalk and ccwalk + stitch (th 0.7),
each with a minimum-hits cut. Filled = very_loose (matching ratio 0.5), open = tight (0.9)."""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qcommon import C, GNN, record, save, setup, plt, np  # noqa: E402

M = f"{GNN}/main"
SERIES = {
    ("v1", "stitch"): [f"{M}/v1/test200/eval_stitch0.7_mh{k}" for k in (3, 4, 5, 6)],
    ("v1", "ccwalk"): [f"{M}/v1/eval_ccwalk", f"{M}/v1/test200/eval_ccwalk_mh4", f"{M}/v1/test200/eval_ccwalk_mh5"],
    ("v2", "stitch"): [f"{M}/v2/test_node_edge/eval_stitch0.7_mh{k}" for k in (3, 4, 5, 6)],
    ("v2", "ccwalk"): [f"{M}/v2/test_node_edge/eval_ccwalk", f"{M}/v2/test_node_edge/eval_ccwalk_mh5"],
    ("v2_200", "stitch"): [f"{M}/v2/test_node_edge/eval200_stitch0.7_mh5"],
    ("v2_200", "ccwalk"): [f"{M}/v2/test_node_edge/eval200_ccwalk"],
}


def load(d):
    s = json.load(open(f"{d}/summary.json"))
    q = s["quirk_track_efficiency"]
    cfg = s["predictions_meta"]["config"]
    return {"eff_vl": q["very_loose"]["eff"], "err_vl": q["very_loose"]["err"], "eff_t": q["tight"]["eff"],
            "err_t": q["tight"]["err"], "fake": s["fake_rate"]["very_loose"], "events": s["events"],
            "min_hits": cfg.get("min_hits"), "method": cfg.get("method"),
            "event_eff_vl": s["quirk_event_efficiency"]["very_loose"]["eff"],
            "tech_eff_vl": s["technical_efficiency"]["very_loose"]["eff"]}


def main():
    setup()
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    vals, srcs = {}, {}
    mk = {"stitch": "o", "ccwalk": "s"}
    for (ver, b), dirs in SERIES.items():
        pts = [load(d) for d in dirs]
        for d, p in zip(dirs, pts):
            key = os.path.relpath(d, M)
            vals[key] = p
            srcs[key] = f"{d}/summary.json"
        if ver == "v2_200":
            continue
        col = C["v1"] if ver == "v1" else C["v2"]
        small = False
        f = np.array([p["fake"] for p in pts])
        e = np.array([p["eff_vl"] for p in pts])
        ee = np.array([p["err_vl"] for p in pts])
        t = np.array([p["eff_t"] for p in pts])
        te = np.array([p["err_t"] for p in pts])
        ms = 6 if small else 9
        lab = {"v1": "v1", "v2": "v2", "v2_200": None}[ver]
        lab = f"{lab}, {'ccwalk + stitch' if b == 'stitch' else 'ccwalk'}" if lab else None
        ax.errorbar(f, e, yerr=ee, marker=mk[b], ms=ms, color=col, mec="white", mew=1, lw=1.5,
                    ls="-" if (b == "stitch" and not small) else "none", elinewidth=1.2, capsize=0, label=lab, zorder=3)
        ax.errorbar(f, t, yerr=te, marker=mk[b], ms=ms, mfc="white", mec=col, mew=1.5, color=col, lw=1.0,
                    ls="-" if (b == "stitch" and not small) else "none", elinewidth=1.0, capsize=0, zorder=3)
        if b == "stitch" and ver == "v2":
            for p in (pts[0], pts[-1]):
                ax.annotate(f"≥{p['min_hits']} hits", (p["fake"], p["eff_vl"]), xytext=(0, -18),
                            textcoords="offset points", ha="center", fontsize=9, color=C["ink2"])
    ax.text(0.065, 0.50, "filled: very loose matching (\u2265 50% of hits)", fontsize=10, color=C["ink2"])
    ax.text(0.065, 0.12, "open: tight matching (\u2265 90% of hits)", fontsize=10, color=C["ink2"])
    ax.set_xlabel("Fake rate (very loose matching)")
    ax.set_ylabel("Quirk track efficiency")
    ax.set_xlim(0.06, 0.33)
    ax.set_ylim(0, 0.85)
    ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.43))
    ax.text(0.99, 0.16, "m = 200 GeV, \u039b = 774 eV\nv1: 200 test events, v2: 3000", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=9.5, color=C["ink2"])
    save(fig, "gnn_eff_vs_fake")
    record("gnn_eff_vs_fake", vals, srcs)


if __name__ == "__main__":
    main()
