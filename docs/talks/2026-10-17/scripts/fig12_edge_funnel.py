"""Fig 12: edges per event and quirk-edge efficiency (fraction of same-quirk truth-chain edges kept) through the GNN
chain: metric-learning radius graph -> edge filter (cut 0.005) -> GNN (score > 0.1 and > 0.5).
m = 200 GeV, Lambda = 774 eV, validation events 16000-16199 (v2 filter scan: 100 of them).
v1: graph and filter from logs/v1_graph.log and logs/v1_filt_apply.log (events 16000-16199), GNN from
main/v1/gnn/cut_scan.json. v2: filter from main/v2/filter/cut_scan.json, GNN from main/v2/gnn_node_edge/cut_scan.json."""

import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qcommon import C, GNN, record, save, setup, plt, np  # noqa: E402

STAGES = ["metric-learning\ngraph", "filter\n(score > 0.005)", "GNN\n(score > 0.1)", "GNN\n(score > 0.5)"]


def from_log(path, tag):
    for line in open(path):
        if line.startswith(tag):
            return json.loads(re.search(r"(\{.*\})", line).group(1))
    raise KeyError(tag)


def cut(path, c):
    return next(r for r in json.load(open(path)) if abs(r["cut"] - c) < 1e-9)


def main():
    setup()
    L, M = f"{GNN}/logs", f"{GNN}/main"
    v1 = [from_log(f"{L}/v1_graph.log", "graph events 16000-16199"),
          from_log(f"{L}/v1_filt_apply.log", "filtered events 16000-16199"),
          cut(f"{M}/v1/gnn/cut_scan.json", 0.1), cut(f"{M}/v1/gnn/cut_scan.json", 0.5)]
    v2 = [None, cut(f"{M}/v2/filter/cut_scan.json", 0.005),
          cut(f"{M}/v2/gnn_node_edge/cut_scan.json", 0.1), cut(f"{M}/v2/gnn_node_edge/cut_scan.json", 0.5)]
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(12.5, 4.8))
    x = np.arange(len(STAGES))
    for ver, rows, col, mk in (("v1", v1, C["v1"], "s"), ("v2", v2, C["v2"], "o")):
        xs = np.array([i for i, r in enumerate(rows) if r is not None])
        ne = np.array([r["edges_per_event"] for r in rows if r is not None])
        eq = np.array([r["eff_quirk"] for r in rows if r is not None])
        dx = -0.1 if ver == "v1" else 0.1
        axa.plot(xs + dx, ne, marker=mk, ms=9, color=col, mec="white", mew=1, lw=2, label=ver)
        axb.plot(xs + dx, eq, marker=mk, ms=9, color=col, mec="white", mew=1, lw=2, label=ver)
        for xi, n, e in zip(xs, ne, eq):
            off, ha = ((-9, "right") if ver == "v1" else (9, "left"))
            axa.annotate(f"{n / 1000:.1f}k" if n < 1e4 else f"{n / 1000:.0f}k", (xi + dx, n), xytext=(off, 0),
                         textcoords="offset points", ha=ha, va="center", fontsize=9.5, color=C["ink2"], bbox=dict(fc="white", ec="none", pad=0.5), zorder=5)
            axb.annotate(f"{e:.3f}", (xi + dx, e), xytext=(off, 0), textcoords="offset points", ha=ha,
                         va="center", fontsize=9.5, color=C["ink2"], bbox=dict(fc="white", ec="none", pad=0.5), zorder=5)
    for ax in (axa, axb):
        ax.set_xticks(x)
        ax.set_xticklabels(STAGES, fontsize=10.5)
        ax.set_xlim(-0.4, len(STAGES) - 0.6)
        ax.tick_params(axis="x", length=0)
    axa.set_yscale("log")
    axa.set_ylim(1e3, 1e6)
    axa.set_ylabel("Edges per event")
    axb.set_ylim(0, 1.05)
    axb.set_ylabel("Quirk-edge efficiency")
    axa.legend(loc="upper right")
    axb.text(0.02, 0.04, "quirk edges: consecutive hits of the same quirk\nvalidation events, m = 200 GeV, Λ = 774 eV",
             transform=axb.transAxes, fontsize=9.5, color=C["ink2"], va="bottom")
    fig.tight_layout(w_pad=2.5)
    save(fig, "edge_funnel")
    keys = ["edges_per_event", "eff_quirk", "eff", "eff_tech", "purity"]
    record("edge_funnel", {"stages": [s.replace("\n", " ") for s in STAGES],
                           "v1": [{k: r[k] for k in keys} for r in v1],
                           "v2": [None if r is None else {k: r[k] for k in keys} for r in v2]},
           {"v1_graph": f"{L}/v1_graph.log (events 16000-16199)", "v1_filter": f"{L}/v1_filt_apply.log (events 16000-16199)",
            "v1_gnn": f"{M}/v1/gnn/cut_scan.json", "v2_filter": f"{M}/v2/filter/cut_scan.json (n_val 100)",
            "v2_gnn": f"{M}/v2/gnn_node_edge/cut_scan.json"})


if __name__ == "__main__":
    main()
