"""Stage 4: track building from GNN-scored graphs -> predictions.parquet (+ meta.json).

Methods:
  ccwalk : acorn CC & walkthrough (score_cut_cc 0.01, th_min 0.1, th_add 0.6). Edges are oriented outward in
           R^2 = r^2 + z^2 (acorn remove_cycles). For quirks that turn inwards this orientation splits the track at
           the turning point unless the segment is an isolated simple path (the CC step handles those whatever the
           orientation). Road growth capped at 256 roads per start (safety; acorn = unlimited).
  greedy : orientation-free greedy path cover. Edges with score > --th-min are taken in decreasing score order and
           accepted if both hits have fewer than 2 accepted edges and the edge closes no cycle (union-find). The
           result is a set of disjoint paths; each path with >= --min-hits hits is a track. No direction is
           assumed, so a track that turns back inwards stays one path.
  ccwalk_stitch : ccwalk, then fragments are joined end to end: each track's two end hits (lowest and highest
           R^2) are its end points; edges between end points of different tracks with score > --th-stitch are taken in
           decreasing score order, each end point used once, no cycles. A track that turns back is split by ccwalk at
           its turning points; the pieces end at the same place and are joined again here.
  hybrid : connected components of the edges above --cc-cut with >= --min-hits hits are tracks; the rest of the hits
           and edges go to ccwalk_stitch.
  cc     : plain connected components of edges with score > --cc-cut (orientation-free).
Tracks with fewer than --min-hits hits are dropped (hits -> -1).

  python build_tracks.py --cache C --scored 'S/*.parquet' --events A B --method ccwalk --out DIR
"""

import argparse
import datetime
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ccwalk_core as cw  # noqa: E402
from common import load_events, read_graphs  # noqa: E402

cw.MAX_ROADS = 256
ARGS = None


def stitch(tracks, src, dst, sc, R, n):
    tid = np.full(n, -1, np.int64)
    end = np.zeros(n, bool)
    for i, t in enumerate(tracks):
        t = np.asarray(t, np.int64)
        tid[t] = i
        o = t[np.argsort(R[t])]
        end[o[0]] = end[o[-1]] = True
    k = (sc > ARGS.th_stitch) & end[src] & end[dst] & (tid[src] >= 0) & (tid[dst] >= 0) & (tid[src] != tid[dst])
    idx = np.flatnonzero(k)
    idx = idx[np.argsort(-sc[idx], kind="stable")]
    parent = list(range(len(tracks)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    used = np.zeros(n, bool)
    for e in idx:
        u, v = src[e], dst[e]
        if used[u] or used[v]:
            continue
        a, b = find(tid[u]), find(tid[v])
        if a == b:
            continue
        parent[a] = b
        used[u] = used[v] = True
    groups = {}
    for i, t in enumerate(tracks):
        groups.setdefault(find(i), []).extend(t)
    return list(groups.values())


def build_event(item):
    ev, g = item
    n = len(ev["x"])
    src, dst, sc = g["src"].astype(np.int64), g["dst"].astype(np.int64), g["score"].astype(np.float32)
    labels = np.full(n, -1, np.int64)
    if ARGS.method == "greedy":
        k = np.flatnonzero(sc > ARGS.th_min)
        k = k[np.argsort(-sc[k], kind="stable")]
        parent = np.arange(n)
        deg = np.zeros(n, np.int8)

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for e in k:
            u, v = src[e], dst[e]
            if deg[u] >= 2 or deg[v] >= 2:
                continue
            ru, rv = find(u), find(v)
            if ru == rv:
                continue
            parent[ru] = rv
            deg[u] += 1
            deg[v] += 1
        roots = np.array([find(i) for i in range(n)])
        _, inv, cnt = np.unique(roots, return_inverse=True, return_counts=True)
        good = cnt[inv] >= ARGS.min_hits
        _, lab = np.unique(roots[good], return_inverse=True)
        labels[good] = lab
        return ev["event_id"], labels
    taken = None
    if ARGS.method == "hybrid":
        # components of the edges above --cc-cut with >= min_hits hits become tracks; the remaining hits and edges
        # go to ccwalk + stitch
        k = sc > ARGS.cc_cut
        m = coo_matrix((np.ones(k.sum()), (src[k], dst[k])), shape=(n, n))
        _, lab = connected_components(m, directed=False)
        _, inv, cnt = np.unique(lab, return_inverse=True, return_counts=True)
        good = cnt[inv] >= ARGS.min_hits
        _, labels_good = np.unique(lab[good], return_inverse=True)
        labels[good] = labels_good
        taken = good
        keep_e = ~(good[src] | good[dst])
        src, dst, sc = src[keep_e], dst[keep_e], sc[keep_e]
    if ARGS.method == "cc":
        k = sc > ARGS.cc_cut
        m = coo_matrix((np.ones(k.sum()), (src[k], dst[k])), shape=(n, n))
        _, lab = connected_components(m, directed=False)
        _, inv, cnt = np.unique(lab, return_inverse=True, return_counts=True)
        good = cnt[inv] >= ARGS.min_hits
        _, labels_good = np.unique(lab[good], return_inverse=True)
        labels[good] = labels_good
        return ev["event_id"], labels
    r = np.hypot(ev["x"], ev["y"]).astype(np.float32)
    R = r ** 2 + ev["z"].astype(np.float32) ** 2
    flip = (R[src] > R[dst]) | ((R[src] == R[dst]) & (src > dst))
    u, v = np.where(flip, dst, src), np.where(flip, src, dst)
    keep = sc > ARGS.score_cut_cc
    succ, pred = cw.build_adjacency(u, v, sc, keep)
    order = cw.topo_order(succ, pred)
    cc_tracks, removed = cw.simple_path_components(succ, pred, order)
    claimed, _, _ = cw.walk(succ, pred, order, removed, ARGS.th_min, ARGS.th_add)
    tracks = cw.flatten_partition(cc_tracks + claimed)
    if ARGS.method in ("ccwalk_stitch", "hybrid"):
        tracks = stitch([t for t in tracks if len(t) >= 2], src, dst, sc, R, n)
    tracks = [t for t in tracks if len(t) >= ARGS.min_hits]
    off = 0 if taken is None else int(labels.max()) + 1
    for i, t in enumerate(tracks):
        labels[np.asarray(t, np.int64)] = off + i
    return ev["event_id"], labels


def main():
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--scored", required=True)
    ap.add_argument("--events", type=int, nargs=2, required=True)
    ap.add_argument("--method", choices=["ccwalk", "ccwalk_stitch", "cc", "greedy", "hybrid"], default="ccwalk")
    ap.add_argument("--score-cut-cc", type=float, default=0.01)
    ap.add_argument("--th-min", type=float, default=0.1)
    ap.add_argument("--th-add", type=float, default=0.6)
    ap.add_argument("--th-stitch", type=float, default=0.5)
    ap.add_argument("--cc-cut", type=float, default=0.5)
    ap.add_argument("--min-hits", type=int, default=3)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", required=True)
    ap.add_argument("--run-id", default=None)
    ARGS = ap.parse_args()
    t0 = time.time()
    evs = load_events(ARGS.cache, *ARGS.events)
    g = read_graphs(ARGS.scored, *ARGS.events)
    items = [(e, g[e["event_id"]]) for e in evs]
    with Pool(ARGS.workers) as pool:
        res = pool.map(build_event, items, chunksize=4)
    eids = np.concatenate([np.full(len(l), e, np.uint64) for e, l in res])
    hids = np.concatenate([np.arange(len(l), dtype=np.uint64) for _, l in res])
    tids = np.concatenate([l for _, l in res])
    os.makedirs(ARGS.out, exist_ok=True)
    pq.write_table(pa.table({"event_id": pa.array(eids, pa.uint64()), "hit_id": pa.array(hids, pa.uint64()),
                             "predicted_track_id": pa.array(tids, pa.int64()),
                             "score": pa.array(np.full(len(tids), np.nan, np.float32), pa.float32())}),
                   os.path.join(ARGS.out, "predictions.parquet"))
    meta = {"run_id": ARGS.run_id or datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"),
            "implementation": f"quirkml:gnn_{ARGS.method}", "config": vars(ARGS),
            "wrote_at": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    json.dump(meta, open(os.path.join(ARGS.out, "meta.json"), "w"), indent=1)
    ntr = np.mean([l.max() + 1 for _, l in res])
    print(f"{len(res)} events, {ntr:.1f} tracks/event, {(tids >= 0).mean():.3f} hits assigned, "
          f"{time.time() - t0:.1f}s, capped starts {cw.CAPPED['starts']}", flush=True)


if __name__ == "__main__":
    main()
