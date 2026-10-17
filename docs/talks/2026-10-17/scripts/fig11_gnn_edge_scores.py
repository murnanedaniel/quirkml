"""Fig 11: GNN edge-score distributions on validation events (m = 200 GeV, Lambda = 774 eV), v1 (events
16000-16199) and v2 cluster = node_edge (events 16000-16999). Edge categories follow
quirkml/tracking/gnn/score_dists.py:
  same-quirk chain   both ends on the same quirk and the edge is a truth-chain edge (consecutive hits)
  quirk-other quirk  one end on each quirk of the event
  other true edges   truth-chain edges of non-quirk particles
  false edges        everything else (different particles, or same particle but not consecutive)
Cache (pid, hit_quirk, chain per event) and scored graphs (src, dst, score) from the GNN chain outputs."""

import glob
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qcommon import C, GNN, record, save, setup, plt, np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

CACHE = f"{GNN}/cache/m200_L774_mu0"
RUNS = {"v1": (f"{GNN}/main/v1/gnn/scored", 16000, 16199), "v2": (f"{GNN}/main/v2/gnn_node_edge/scored", 16000, 16999)}
CATS = [("same", "same-quirk chain edges", C["qplus"]), ("other_q", "quirk to other-quirk edges", C["qminus"]),
        ("true", "other true edges", C["ink2"]), ("false", "false edges", C["other"])]
BINS = np.linspace(0, 1, 41)


def cache_events(first, last):
    out = {}
    for f in glob.glob(f"{CACHE}/events*.pkl"):
        a, b = os.path.basename(f)[6:-4].split("-")
        if int(b) < first or int(a) > last:
            continue
        with open(f, "rb") as fh:
            for ev in pickle.load(fh):
                if first <= ev["event_id"] <= last:
                    out[ev["event_id"]] = ev
    return out


def scores(scored_dir, first, last):
    evs = cache_events(first, last)
    hist = {k: np.zeros(len(BINS) - 1) for k, _, _ in CATS}
    count = {k: 0 for k, _, _ in CATS}
    frac = {k: [0, 0] for k, _, _ in CATS}  # (> 0.5, total)
    n_ev = 0
    for f in sorted(glob.glob(f"{scored_dir}/*.parquet")):
        t = pq.read_table(f, columns=["event_id", "src", "dst", "score"])
        eids = t.column("event_id").to_numpy()
        cols = {}
        for c in ("src", "dst", "score"):
            arr = t.column(c).combine_chunks()
            cols[c] = (arr.offsets.to_numpy(), arr.values.to_numpy())
        for i, e in enumerate(eids):
            if not (first <= e <= last) or int(e) not in evs:
                continue
            ev = evs[int(e)]
            s, d, sc = (cols[c][1][cols[c][0][i]:cols[c][0][i + 1]] for c in ("src", "dst", "score"))
            s, d = s.astype(np.int64), d.astype(np.int64)
            n = len(ev["pid"])
            ch = ev["chain"].astype(np.int64)
            key = np.minimum(ch[0], ch[1]) * n + np.maximum(ch[0], ch[1])
            y = np.isin(np.minimum(s, d) * n + np.maximum(s, d), key)
            pid, hq = ev["pid"], ev["hit_quirk"]
            both = hq[s] & hq[d]
            same = pid[s] == pid[d]
            m = {"same": both & same & y, "other_q": both & ~same, "true": y & ~(hq[s] | hq[d])}
            m["false"] = ~(m["same"] | m["other_q"] | m["true"] | (both & same))
            for k, mm in m.items():
                hist[k] += np.histogram(sc[mm], bins=BINS)[0]
                count[k] += int(mm.sum())
                frac[k][0] += int((sc[mm] > 0.5).sum())
                frac[k][1] += int(mm.sum())
            n_ev += 1
    return hist, count, frac, n_ev


def main():
    setup()
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharey=True)
    vals, srcs = {}, {}
    for ax, (ver, (sd, a, b)) in zip(axes, RUNS.items()):
        hist, count, frac, n_ev = scores(sd, a, b)
        for k, lab, col in CATS:
            dens = hist[k] / max(hist[k].sum(), 1) / np.diff(BINS)
            ax.stairs(dens, BINS, color=col, lw=2, label=f"{lab} ({count[k] / n_ev:,.0f} / event)")
        ax.set_yscale("log")
        ax.set_xlim(0, 1)
        ax.set_xlabel("GNN edge score")
        ax.text(0.5, 0.97, f"{ver}, {n_ev} validation events", transform=ax.transAxes, ha="center", va="top",
                fontsize=11, color=C["ink"])
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0.9), fontsize=9.5)
        vals[ver] = {"events": n_ev, "edges_per_event": {k: count[k] / n_ev for k in count},
                     "frac_score_gt_0.5": {k: frac[k][0] / max(frac[k][1], 1) for k in frac}}
        srcs[ver] = {"scored": sd, "cache": CACHE, "events": [a, b]}
        print(ver, vals[ver])
    axes[0].set_ylabel("Edges (normalised density)")
    lo, hi = axes[0].get_ylim()
    axes[0].set_ylim(lo, hi * 80)
    fig.tight_layout(w_pad=1.5)
    save(fig, "gnn_edge_scores")
    record("gnn_edge_scores", vals, srcs)


if __name__ == "__main__":
    main()
