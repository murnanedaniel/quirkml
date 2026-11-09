"""Kept-hit caches for mu 60 training/inference after the hit filter: every per-hit array is subset to the kept hits
(keep_prob >= cut from the hit-mask parquet), `orig_hit` holds the hit's index in the full event (R1 hit order), and
each truth chain (one path per particle in the full cache) is contracted over the dropped hits, so the kept hits of a
particle stay linked in the same (time) order. hit_nhits / hit_target keep their full-event values (the truth of the
particle, not of the kept subset).

  python make_kept_cache.py --cache FULL_CACHE --masks MASK_DIR --cut CUT --out OUT_CACHE --blocks 0 1 2 [--workers N]
"""
import argparse
import glob
import os
import pickle
from multiprocessing import Pool

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

A = None
PER_HIT = ["x", "y", "z", "vol_idx", "layer", "cluster", "pid", "hit_quirk", "hit_pt", "hit_nhits", "hit_target",
           "axa", "axb", "dphi", "deta", "diag"]


def contract(chain, keep):
    """chain (2, E): union of per-particle paths in the full event; returns the contracted paths on kept hits,
    re-indexed to the kept numbering."""
    n = len(keep)
    if chain.shape[1] == 0:
        return np.zeros((2, 0), np.int32)
    nxt = np.full(n, -1, np.int64)
    nxt[chain[0]] = chain[1]
    has_prev = np.zeros(n, bool); has_prev[chain[1]] = True
    starts = np.setdiff1d(np.unique(chain[0]), chain[1])
    new = np.full(n, -1, np.int64); new[keep] = np.arange(int(keep.sum()))
    src, dst = [], []
    for s in starts:
        prev, cur = -1, s
        while cur >= 0:
            if keep[cur]:
                if prev >= 0:
                    src.append(new[prev]); dst.append(new[cur])
                prev = cur
            cur = nxt[cur]
    return np.array([src, dst], np.int32).reshape(2, -1)


def one(path):
    tag = os.path.basename(path)[:-4]
    out = os.path.join(A.out, tag + ".pkl")
    if os.path.exists(out):
        return tag, "exists"
    mf = glob.glob(os.path.join(A.masks, f"*.hitmask.{tag}.parquet"))
    assert len(mf) == 1, (tag, mf)
    t = pq.read_table(mf[0], columns=["event_id", "keep_prob"])
    eids = t["event_id"].to_numpy().astype(np.int64)
    n = np.asarray(pc.list_value_length(t["keep_prob"]))
    off = np.concatenate([[0], np.cumsum(n)])
    prob = np.asarray(pc.list_flatten(t["keep_prob"]).to_numpy(zero_copy_only=False))
    row = {int(e): i for i, e in enumerate(eids)}
    with open(path, "rb") as fh:
        evs = pickle.load(fh)
    res, kept, nq, nqk = [], 0, 0, 0
    for ev in evs:
        i = row[ev["event_id"]]
        p = prob[off[i]:off[i + 1]]
        assert len(p) == len(ev["x"])
        keep = p >= A.cut
        e2 = {k: (v[keep] if k in PER_HIT else v) for k, v in ev.items()}
        e2["orig_hit"] = np.flatnonzero(keep).astype(np.int32)
        e2["chain"] = contract(ev["chain"], keep)
        e2["n_hits_full"] = len(p)
        res.append(e2)
        kept += int(keep.sum()); nq += int(ev["hit_quirk"].sum()); nqk += int(ev["hit_quirk"][keep].sum())
    with open(out + ".tmp", "wb") as fh:
        pickle.dump(res, fh, protocol=5)
    os.replace(out + ".tmp", out)
    return tag, len(res), kept / len(res), nqk / max(nq, 1)


def main():
    global A
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True); ap.add_argument("--masks", required=True); ap.add_argument("--cut", type=float, required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--blocks", type=int, nargs="+", required=True)
    ap.add_argument("--workers", type=int, default=4)
    A = ap.parse_args()
    os.makedirs(A.out, exist_ok=True)
    paths = [os.path.join(A.cache, f"events{1000 * b}-{1000 * b + 999}.pkl") for b in A.blocks]
    with Pool(A.workers) as p:
        for r in p.imap_unordered(one, paths):
            print(*r, flush=True)


if __name__ == "__main__":
    main()
