"""Track-building study for the mu 60 pair GNN on val (CPU). For each builder setting: coupled strict efficiency
(> 50% / > 90%, pair truth = pair cache, pairs with >= 3 hits) and strict fake rate (pair cache), plus a failure
breakdown of every pair not matched: the pair's best track (most pair hits) has completeness <= f (fragmentation:
pair hits spread over tracks / unassigned) or completeness > f but purity <= f (absorption: ordinary hits merged in,
typically through a wrong bridge edge); "no track" if no pair hit is on any track.

Builders (min 4 hits):
  cc:c                  connected components of edges with score > c (the current builder)
  tt:hi:lo              two-threshold: components at hi; then every hit not in a component of >= 2 hits joins the
                        component of its best edge with score > lo (single hits only; components never merge)
  br:c:t                components at c, then edges with score < t that are bridges with no triangle support
                        (no common neighbour above c) are removed and components recomputed
Paired event bootstrap of the difference to the reference builder (cc:0.5) on the loose and tight coupled efficiency.

  python builders60.py RUN_DIR OUT_JSON SPEC...       (val events 2000-2999)
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
from common import load_events  # noqa: E402

CP = "/pscratch/sd/d/danieltm/quirk_pu60/cache/m200_L774_mu60_dir_pair"
E0, E1 = 2000, 2999
SPECS = None
D = {}


def comps(n, s, t):
    _, lab = connected_components(coo_matrix((np.ones(len(s)), (s, t)), shape=(n, n)), directed=False)
    return lab


def finalize(lab, min_hits=4):
    u, inv, cnt = np.unique(lab, return_inverse=True, return_counts=True)
    good = (cnt[inv] >= min_hits) & (lab >= 0)
    out = np.full(len(lab), -1, np.int64)
    _, out[good] = np.unique(lab[good], return_inverse=True)
    return out


def build(n, s, t, sc, spec):
    p = spec.split(":")
    if p[0] == "cc":
        k = sc > float(p[1])
        return finalize(comps(n, s[k], t[k]))
    if p[0] == "tt":
        hi, lo = float(p[1]), float(p[2])
        k = sc > hi
        lab = comps(n, s[k], t[k])
        size = np.bincount(lab)
        multi = size[lab] >= 2
        # single hits: attach through their best edge > lo to a multi-hit component
        m = (sc > lo) & (sc <= hi)
        a = np.concatenate([s[m], t[m]]); b = np.concatenate([t[m], s[m]]); w = np.concatenate([sc[m], sc[m]])
        ok = (~multi[a]) & multi[b]
        a, b, w = a[ok], b[ok], w[ok]
        o = np.lexsort((-w, a)); a, b = a[o], b[o]
        first = np.concatenate([[True], a[1:] != a[:-1]]) if len(a) else np.zeros(0, bool)
        lab = lab.copy(); lab[a[first]] = lab[b[first]]
        return finalize(lab)
    if p[0] == "br":
        c, tw = float(p[1]), float(p[2])
        k = sc > c
        s2, t2, w2 = s[k], t[k], sc[k]
        # triangle support: does the edge (u, v) have a common neighbour among edges above c?
        adj = coo_matrix((np.ones(len(s2)), (s2, t2)), shape=(n, n)).tocsr()
        adj = ((adj + adj.T) > 0).astype(np.float32).tocsr()
        common = np.asarray(adj[s2].multiply(adj[t2]).sum(1)).ravel()
        keep = (w2 >= tw) | (common > 0)
        return finalize(comps(n, s2[keep], t2[keep]))
    raise ValueError(spec)


def match(lab, pid, hq, opid=None):
    """per pair (>= 3 hits): (loose ok, tight ok, failure class at 0.5); plus (tracks, fakes) at 0.5 (pair cache truth)."""
    qi = np.flatnonzero(hq)
    res = []
    if len(qi) >= 3:
        on = lab[qi] >= 0
        if not on.any():
            res = [(0, 0, "no_track", "tight_no_track")]
        else:
            tr, c = np.unique(lab[qi][on], return_counts=True)
            best = tr[np.argmax(c)]; nb = c.max()
            tsize = (lab == best).sum()
            comp, pur = nb / len(qi), nb / tsize
            lo = comp > 0.5 and pur > 0.5
            ti = comp > 0.9 and pur > 0.9
            cls = "ok" if lo else ("fragmentation" if comp <= 0.5 else "absorption")
            if cls == "fragmentation" and opid is not None:  # is the best track one quirk alone (pair split into quirks)?
                bq = qi[lab[qi] == best]
                u, cu = np.unique(opid[bq], return_counts=True)
                q1 = u[np.argmax(cu)]
                if cu.max() > 0.9 * len(bq) and cu.max() > 0.5 * (opid[qi] == q1).sum():
                    cls = "fragmentation_one_quirk"
            tcls = "tight_ok" if ti else ("tight_incomplete" if comp <= 0.9 and pur > 0.9 else ("tight_impure" if pur <= 0.9 and comp > 0.9 else "tight_both"))
            res = [(int(lo), int(ti), cls, tcls)]
    m = lab >= 0
    if not m.any():
        return res, 0, 0
    P = int(pid.max()) + 1
    k, c = np.unique(lab[m] * P + pid[m], return_counts=True)
    trk, par = k // P, k % P
    tsize = np.bincount(lab[m])
    u, pc_ = np.unique(pid, return_counts=True)
    psz = dict(zip(u.tolist(), pc_.tolist()))
    ps = np.array([psz[int(x)] for x in par])
    ok = (c > 0.5 * tsize[trk]) & (c > 0.5 * ps)
    good = np.zeros(len(tsize), bool); good[trk[ok]] = True
    nt = int((tsize > 0).sum())
    return res, nt, int((~good & (tsize > 0)).sum())


def one(i):
    ev, (s, t, sc) = D["evs"][i], D["g"][i]
    n = len(ev["x"])
    out = {}
    for spec in SPECS:
        lab = build(n, s, t, sc, spec)
        out[spec] = match(lab, ev["pid"], ev["hit_quirk"], D["opid"][i])
    return out


def main():
    global SPECS
    run, dest = sys.argv[1], sys.argv[2]
    SPECS = sys.argv[3:]
    evs = load_events(CP, E0, E1)
    D["opid"] = [e["pid"] for e in load_events(CP.replace("_dir_pair", "_dir"), E0, E1)]  # original particle ids
    tab = pq.read_table(os.path.join(run, "scored", f"events{E0}-{E1}.parquet"), columns=["event_id", "src", "dst", "score"])
    off = np.concatenate([[0], np.cumsum(np.asarray(pc.list_value_length(tab["src"])))])
    F = {k: np.asarray(pc.list_flatten(tab[k]).to_numpy(zero_copy_only=False)) for k in ["src", "dst", "score"]}
    row = {int(e): i for i, e in enumerate(tab["event_id"].to_numpy())}
    D["evs"] = evs
    D["g"] = [(F["src"][off[row[e["event_id"]]]:off[row[e["event_id"]] + 1]].astype(np.int64),
               F["dst"][off[row[e["event_id"]]]:off[row[e["event_id"]] + 1]].astype(np.int64),
               F["score"][off[row[e["event_id"]]]:off[row[e["event_id"]] + 1]]) for e in evs]
    with Pool(64) as p:
        per = p.map(one, range(len(evs)), chunksize=8)
    res = {}
    rng = np.random.default_rng(0)
    B = rng.integers(0, len(per), size=(1000, len(per)))
    ref = SPECS[0]

    def arr(spec, j):
        return np.array([sum(x[j] for x in pe[spec][0]) for pe in per], float), np.array([len(pe[spec][0]) for pe in per], float)
    rl, nq = arr(ref, 0); rt, _ = arr(ref, 1)
    for spec in SPECS:
        lo, _ = arr(spec, 0); ti, _ = arr(spec, 1)
        cls = {}
        for pe in per:
            for x in pe[spec][0]:
                cls[x[2]] = cls.get(x[2], 0) + 1
                cls[x[3]] = cls.get(x[3], 0) + 1
        nt = sum(pe[spec][1] for pe in per); nf = sum(pe[spec][2] for pe in per)
        dl = (lo[B].sum(1) - rl[B].sum(1)) / nq[B].sum(1); dt = (ti[B].sum(1) - rt[B].sum(1)) / nq[B].sum(1)
        res[spec] = {"loose": lo.sum() / nq.sum(), "tight": ti.sum() / nq.sum(), "fake": nf / max(nt, 1), "tracks_per_event": nt / len(per),
                     "failures": cls, "d_loose_vs_ref": [float(np.mean(dl)), float(np.percentile(dl, 2.5)), float(np.percentile(dl, 97.5))],
                     "d_tight_vs_ref": [float(np.mean(dt)), float(np.percentile(dt, 2.5)), float(np.percentile(dt, 97.5))],
                     "per_event": {"loose": lo.astype(int).tolist(), "tight": ti.astype(int).tolist(), "pairs": nq.astype(int).tolist(),
                                   "event_id": [int(e["event_id"]) for e in D["evs"]]}}
        r = res[spec]
        print(f"{spec:14s} loose {r['loose']:.3f} tight {r['tight']:.3f} fake {r['fake']:.3f} tracks/ev {r['tracks_per_event']:.0f} "
              f"d_loose {r['d_loose_vs_ref'][0]:+.3f} [{r['d_loose_vs_ref'][1]:+.3f},{r['d_loose_vs_ref'][2]:+.3f}] "
              f"d_tight {r['d_tight_vs_ref'][0]:+.3f} [{r['d_tight_vs_ref'][1]:+.3f},{r['d_tight_vs_ref'][2]:+.3f}] {cls}", flush=True)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    json.dump({"run": run, "events": [E0, E1], "pairs": int(nq.sum()), "results": res}, open(dest, "w"), indent=1)


if __name__ == "__main__":
    main()
