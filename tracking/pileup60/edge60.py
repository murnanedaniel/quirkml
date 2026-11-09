"""Edge-level breakdown of the mu 60 pair GNN on test events (CPU): what limits the coupled efficiency.

Universe: the edges of the GNN input graph (graph v3 + mu 0 filter at 0.1, graph_apply60.py); GNN score from the
scored files (edges with score <= 0.01 were not written: they get score 0, so the AUC treats them as tied at the
bottom). Truth from the pair cache (both quirks = one particle).
  pair edge       both hits quirk hits (the p3 training target: every same-pair edge true)
  pair-chain edge consecutive hits of the time-ordered pair chain (cache chain with both ends quirk hits)
  wrong quirk edge one quirk hit and one non-quirk hit
Per cut: pair-chain efficiency (found with score > cut / all pair-chain edges, missing-from-graph counted as lost),
purity of quirk-touching edges above the cut (pair edges / edges touching a quirk hit), AUC pair edges vs wrong quirk
edges. Ceilings per pair (pairs with >= 3 hits): largest connected component of pair hits using (i) all pair edges
in the graph, (ii) pair edges with score > cut; a pair "can be found" at loose (tight) if that component holds > 50%
(> 90%) of its hits.
  python edge60.py RUN_DIR OUT_JSON [cut ...]
"""
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "gnn"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_events  # noqa: E402
from graph_apply60 import read_compact  # noqa: E402

CP = "/pscratch/sd/d/danieltm/quirk_pu60/cache/m200_L774_mu60_dir_pair"
G = "/pscratch/sd/d/danieltm/quirk_pu60/graphs/v3_mu0f_cut0.1"
RUN, CUTS = None, None
NB = 2000  # score histogram bins for the AUC
BLOCKS = [int(b) for b in os.environ.get("EDGE_BLOCKS", "0 1").split()]
DEG_BINS = np.array([0, 50, 100, 200, 400, 800, 1600, 10**9])


def largest_frac(n, idx, s, t):
    """largest connected component among the nodes idx using edges (s, t) restricted to idx; fraction of len(idx)."""
    if len(idx) == 0:
        return 0.0
    loc = -np.ones(n, np.int64); loc[idx] = np.arange(len(idx))
    m = (loc[s] >= 0) & (loc[t] >= 0)
    _, lab = connected_components(coo_matrix((np.ones(m.sum()), (loc[s[m]], loc[t[m]])), shape=(len(idx), len(idx))), directed=False)
    return np.bincount(lab).max() / len(idx)


def block(b):
    e0, e1 = 1000 * b, 1000 * b + 999
    evs = load_events(CP, e0, e1)
    g = read_compact(os.path.join(G, f"events{e0}-{e1}.d"))
    t = pq.read_table(os.path.join(RUN, "scored", f"events{e0}-{e1}.parquet"), columns=["event_id", "src", "dst", "score"])
    eids = t["event_id"].to_numpy()
    off = np.concatenate([[0], np.cumsum(np.asarray(pc.list_value_length(t["src"])))])
    S = {k: np.asarray(pc.list_flatten(t[k]).to_numpy(zero_copy_only=False)) for k in ["src", "dst", "score"]}
    row = {int(e): i for i, e in enumerate(eids)}
    hp, hw = np.zeros(NB, np.int64), np.zeros(NB, np.int64)
    acc = {c: dict(chain_found=0, touch=0, touch_pair=0) for c in CUTS}
    acc_chain = 0
    pairs = []
    deg_all = np.zeros(len(DEG_BINS) - 1, np.int64); deg_kept = np.zeros(len(DEG_BINS) - 1, np.int64)
    for ev in evs:
        n = len(ev["x"]); hq = ev["hit_quirk"]
        gg = g[ev["event_id"]]
        s, d = gg["src"].astype(np.int64), gg["dst"].astype(np.int64)
        key = np.minimum(s, d) * n + np.maximum(s, d)
        i = row[ev["event_id"]]
        ss, dd, sc = S["src"][off[i]:off[i + 1]].astype(np.int64), S["dst"][off[i]:off[i + 1]].astype(np.int64), S["score"][off[i]:off[i + 1]]
        skey = np.minimum(ss, dd) * n + np.maximum(ss, dd)
        o = np.argsort(skey); skey, sc = skey[o], sc[o]
        pos = np.searchsorted(skey, key); pos = np.minimum(pos, len(skey) - 1)
        score = np.where((len(skey) > 0) & (skey[pos] == key), sc[pos], 0.0) if len(skey) else np.zeros(len(key))
        pair_e = hq[s] & hq[d]
        wrong_q = hq[s] ^ hq[d]
        deg = np.bincount(s, minlength=n) + np.bincount(d, minlength=n)
        mind = np.minimum(deg[s[pair_e]], deg[d[pair_e]])
        bi = np.digitize(mind, DEG_BINS) - 1
        deg_all += np.bincount(bi, minlength=len(DEG_BINS) - 1)[:len(DEG_BINS) - 1]
        deg_kept += np.bincount(bi[score[pair_e] > 0.5], minlength=len(DEG_BINS) - 1)[:len(DEG_BINS) - 1]
        bins = np.minimum((score * NB).astype(np.int64), NB - 1)
        hp += np.bincount(bins[pair_e], minlength=NB); hw += np.bincount(bins[wrong_q], minlength=NB)
        ch = ev["chain"].astype(np.int64)
        cq = hq[ch[0]] & hq[ch[1]]
        ckey = np.minimum(ch[0][cq], ch[1][cq]) * n + np.maximum(ch[0][cq], ch[1][cq])
        acc_chain += len(ckey)
        gk = np.argsort(key); ks = key[gk]
        p2 = np.minimum(np.searchsorted(ks, ckey), max(len(ks) - 1, 0))
        in_g = (len(ks) > 0) & (ks[p2] == ckey) if len(ks) else np.zeros(len(ckey), bool)
        cscore = np.where(in_g, score[gk][p2], -1.0) if len(ks) else np.full(len(ckey), -1.0)
        touch = hq[s] | hq[d]
        for c in CUTS:
            a = acc[c]
            a["chain_found"] += int((cscore > c).sum())
            k = score > c
            a["touch"] += int((k & touch).sum()); a["touch_pair"] += int((k & pair_e).sum())
        qi = np.flatnonzero(hq)
        if len(qi) >= 3:
            r = {"n": int(len(qi)), "graph": float(largest_frac(n, qi, s[pair_e], d[pair_e]))}
            for c in CUTS:
                k = pair_e & (score > c)
                r[f"gnn_{c:g}"] = float(largest_frac(n, qi, s[k], d[k]))
            pairs.append(r)
    return hp, hw, acc, acc_chain, pairs, deg_all, deg_kept, len(evs)


def auc(hp, hw):
    # P(score_pos > score_neg) + 0.5 P(tie), from the histograms
    cw = np.concatenate([[0], np.cumsum(hw)])[:-1]
    return float(((hp * cw).sum() + 0.5 * (hp * hw).sum()) / max(hp.sum() * hw.sum(), 1))


def main():
    global RUN, CUTS
    RUN, out = sys.argv[1], sys.argv[2]
    CUTS = [float(x) for x in sys.argv[3:]] or [0.5, 0.6]
    with Pool(len(BLOCKS)) as p:
        res = p.map(block, BLOCKS)
    nev = sum(r[7] for r in res)
    hp = sum(r[0] for r in res); hw = sum(r[1] for r in res)
    nchain = sum(r[3] for r in res)
    pairs = [x for r in res for x in r[4]]
    o = {"run": RUN, "blocks": BLOCKS, "n_events": nev, "pairs": len(pairs), "auc_pair_vs_wrong_quirk": auc(hp, hw),
         "pair_edges": int(hp.sum()), "wrong_quirk_edges": int(hw.sum()), "pair_chain_edges": int(nchain)}
    for c in CUTS:
        f = sum(r[2][c]["chain_found"] for r in res); tt = sum(r[2][c]["touch"] for r in res); tp = sum(r[2][c]["touch_pair"] for r in res)
        o[f"cut_{c:g}"] = {"pair_chain_eff": f / max(nchain, 1), "purity_quirk_touching": tp / max(tt, 1),
                           "pair_edges_kept_frac": float(hp[int(c * NB):].sum() / max(hp.sum(), 1)),
                           "wrong_quirk_kept_per_event": float(hw[int(c * NB):].sum() / nev)}
    # pair-edge recall at a fixed number of wrong quirk-touching edges per event (threshold from the histograms)
    cw = np.cumsum(hw[::-1])[::-1] / nev; cp_ = np.cumsum(hp[::-1])[::-1] / max(hp.sum(), 1)
    o["pair_edge_recall_at_wrong_per_event"] = {}
    for W in (2, 5, 10, 20, 50, 100):
        i = int(np.argmax(cw <= W)) if (cw <= W).any() else NB - 1
        o["pair_edge_recall_at_wrong_per_event"][str(W)] = {"threshold": i / NB, "pair_edge_recall": float(cp_[i]), "wrong_per_event": float(cw[i])}
    da = sum(r[5] for r in res); dk = sum(r[6] for r in res)
    o["pair_edge_recall_by_min_degree_cut0.5"] = [{"deg_lo": int(DEG_BINS[i]), "deg_hi": int(DEG_BINS[i + 1]), "pair_edges": int(da[i]),
                                                   "recall": float(dk[i] / max(da[i], 1))} for i in range(len(da))]
    for key in ["graph"] + [f"gnn_{c:g}" for c in CUTS]:
        v = np.array([x[key] for x in pairs])
        o[f"ceiling_{key}"] = {"loose_gt50": float((v > 0.5).mean()), "tight_gt90": float((v > 0.9).mean())}
    json.dump(o, open(out, "w"), indent=1)
    print(json.dumps(o, indent=1))


if __name__ == "__main__":
    main()
