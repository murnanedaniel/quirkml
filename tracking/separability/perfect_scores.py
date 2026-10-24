"""Separability ceiling, part 1: the GNN v2 candidate graph (after the filter) with perfect edge scores.

Writes copies of the filter graphs with score = truth (1 for a truth chain edge: consecutive hits of one target
particle in true-time order, as in gnn/make_cache.py and common.edge_truth; else 0), to be fed to
gnn/build_tracks.py unchanged. Also the graph ceiling: per quirk (R1 first-particle labels, >= 3 hits, the
evaluator's quirk selection), the fraction of its truth chain edges present in the graph.

  python perfect_scores.py --graphs 'G/events*.parquet' --cache C --events 17000 19999 --out DIR
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "gnn"))
from common import edge_truth, load_events, read_graphs, write_graphs  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--events", type=int, nargs=2, required=True, help="first last (inclusive)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "scored"), exist_ok=True)
    evs = {e["event_id"]: e for e in load_events(a.cache, *a.events)}
    frac_q, n_q_edges, n_q_found = [], 0, 0
    frac_sm, pair_all = [], []
    n_y_mismatch = 0
    for f in sorted(glob.glob(a.graphs)):
        g = read_graphs(f, *a.events)
        if not g:
            continue
        rows = []
        for eid in sorted(g):
            ev, gr = evs[eid], g[eid]
            src, dst = gr["src"].astype(np.int64), gr["dst"].astype(np.int64)
            y, _ = edge_truth(ev, src, dst)
            n_y_mismatch += int((y != gr["y"]).sum())
            rows.append(dict(event_id=eid, src=src, dst=dst, y=y, score=y.astype(np.float32)))
            # graph ceiling
            n = len(ev["pid"])
            ch = ev["chain"].astype(np.int64)
            ck = np.minimum(ch[0], ch[1]) * n + np.maximum(ch[0], ch[1])
            ek = np.minimum(src, dst) * n + np.maximum(src, dst)
            found = np.isin(ck, ek)
            cpid = ev["pid"][ch[0]]
            pid = ev["pid"]
            u, cnt = np.unique(pid, return_counts=True)
            nh = dict(zip(u.tolist(), cnt.tolist()))
            quirks = [int(q) for q in np.unique(pid[ev["hit_quirk"]]) if nh[int(q)] >= 3]
            fq = []
            for q in quirks:
                m = cpid == q
                n_q_edges += int(m.sum())
                n_q_found += int(found[m].sum())
                fq.append(found[m].mean() if m.any() else 1.0)
            frac_q += fq
            if len(quirks) == 2:
                pair_all.append(min(fq))
            tech = np.unique(pid[(~ev["hit_quirk"]) & ev["hit_target"] & (ev["hit_pt"] > 1) & (ev["hit_nhits"] >= 7)])
            for p in tech:
                m = cpid == p
                frac_sm.append(found[m].mean())
        write_graphs(os.path.join(a.out, "scored", os.path.basename(f)), rows)
        print("wrote", os.path.basename(f), len(rows), "events", flush=True)
    fq, fs, fp = np.array(frac_q), np.array(frac_sm), np.array(pair_all)
    S = dict(events=a.events, quirks=len(fq), y_mismatch_vs_stored=n_y_mismatch,
             quirk_chain_edges=n_q_edges, quirk_chain_edges_found=n_q_found, quirk_edge_eff=n_q_found / max(n_q_edges, 1),
             quirk_all_edges_frac=float((fq == 1).mean()), quirk_ge90_frac=float((fq >= 0.9).mean()),
             quirk_ge50_frac=float((fq >= 0.5).mean()), quirk_edge_frac_q10_50=np.percentile(fq, [10, 50]).tolist(),
             pairs=len(fp), pair_both_all_edges_frac=float((fp == 1).mean()), pair_both_ge90_frac=float((fp >= 0.9).mean()),
             tech_sm_particles=len(fs), tech_sm_all_edges_frac=float((fs == 1).mean()), tech_sm_ge90_frac=float((fs >= 0.9).mean()))
    json.dump(S, open(os.path.join(a.out, "graph_ceiling.json"), "w"), indent=1)
    print(json.dumps(S, indent=1))


if __name__ == "__main__":
    main()
