"""Where quirk truth-chain edges are lost: metric-learning graph vs filter graph, by step length, detector region,
layer skip and position along the chain. Val events, CPU.

  python lost_edges.py --cache C --ml 'G1/*.parquet' --filt 'G2/*.parquet' --events 16000 16199
"""
import argparse
import json
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import VOLUMES, load_events, read_graphs  # noqa: E402

REG = {16: "pix-EC", 17: "pix-B", 18: "pix-EC", 23: "sstrip-EC", 24: "sstrip-B", 25: "sstrip-EC", 28: "lstrip-EC",
       29: "lstrip-B", 30: "lstrip-EC"}


def keys(s, d, n):
    s, d = s.astype(np.int64), d.astype(np.int64)
    return np.minimum(s, d) * n + np.maximum(s, d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--ml", required=True)
    ap.add_argument("--filt", required=True)
    ap.add_argument("--events", type=int, nargs=2, default=[16000, 16199])
    a = ap.parse_args()
    evs = load_events(a.cache, *a.events)
    gm = read_graphs(a.ml, *a.events)
    gf = read_graphs(a.filt, *a.events)
    rows = []
    for ev in evs:
        n = len(ev["x"])
        ch = ev["chain"].astype(np.int64)
        q = ev["hit_quirk"][ch[0]]
        ch = ch[:, q]
        km = keys(gm[ev["event_id"]]["src"], gm[ev["event_id"]]["dst"], n)
        kf = keys(gf[ev["event_id"]]["src"], gf[ev["event_id"]]["dst"], n)
        k = keys(ch[0], ch[1], n)
        pos = np.stack([ev["x"], ev["y"], ev["z"]], 1)
        vol = np.array(VOLUMES)[ev["vol_idx"].astype(int)]
        pid = ev["pid"]
        for j in range(ch.shape[1]):
            u, v = ch[0, j], ch[1, j]
            # position along the particle's chain
            mine = np.flatnonzero(pid[ch[0]] == pid[u])
            rows.append(dict(in_ml=bool(np.isin(k[j], km)), in_f=bool(np.isin(k[j], kf)),
                             step=float(np.linalg.norm(pos[u] - pos[v])), reg=f"{REG[vol[u]]}->{REG[vol[v]]}",
                             same_layer=bool(vol[u] == vol[v] and ev["layer"][u] == ev["layer"][v]),
                             idx=int(np.searchsorted(mine, j)), nlen=len(mine),
                             r=float(np.hypot(*pos[u, :2]))))
    st = np.array([r["step"] for r in rows])
    ml = np.array([r["in_ml"] for r in rows])
    f = np.array([r["in_f"] for r in rows])
    out = dict(n=len(rows), recall_ml=ml.mean(), recall_filter=f.mean(),
               recall_ml_step_gt150=ml[st > 150].mean(), recall_filter_step_gt150=f[st > 150].mean(),
               frac_step_gt150=(st > 150).mean(),
               step_median_kept_lost_filter=[float(np.median(st[f])), float(np.median(st[~f]))])
    for nm, lo, hi in (("<50", 0, 50), ("50-150", 50, 150), ("150-300", 150, 300), (">300", 300, 1e9)):
        m = (st >= lo) & (st < hi)
        out[f"step {nm}"] = dict(n=int(m.sum()), ml=float(ml[m].mean()), filt=float(f[m].mean()))
    reg = Counter(r["reg"] for r in rows)
    out["by_region"] = {g: dict(n=c, ml=float(np.mean([r["in_ml"] for r in rows if r["reg"] == g])),
                                filt=float(np.mean([r["in_f"] for r in rows if r["reg"] == g])))
                        for g, c in reg.most_common(10)}
    idx = np.array([min(r["idx"], 6) for r in rows])
    out["by_chain_position(6=6+)"] = {int(i): dict(n=int((idx == i).sum()), ml=float(ml[idx == i].mean()), filt=float(f[idx == i].mean()))
                                     for i in range(7)}
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
