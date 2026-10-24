"""Per-quirk matching of the perfect-score tracks split by how complete the quirk's truth chain is in the graph.
Python double matching on R1 first-particle labels (equals ACTS per quirk on this sample: see README).

  python ceiling_breakdown.py --pred P/predictions.parquet --graphs 'G/events*.parquet' --events 17000 19999
"""

import argparse
import json
import os
import sys

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "gnn"))
from common import load_events, read_graphs  # noqa: E402

CACHE = "/pscratch/sd/d/danieltm/quirk_tracking/gnn/cache/m200_L774_mu0"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--graphs", required=True)
    ap.add_argument("--events", type=int, nargs=2, default=[17000, 19999])
    a = ap.parse_args()
    evs = load_events(CACHE, *a.events)
    g = read_graphs(a.graphs, *a.events)
    pred = pq.read_table(a.pred).to_pandas()
    pred = {int(e): x.sort_values("hit_id").predicted_track_id.to_numpy() for e, x in pred.groupby("event_id")}
    rows = []
    for ev in evs:
        eid = ev["event_id"]
        n = len(ev["pid"])
        gr = g[eid]
        ek = np.minimum(gr["src"], gr["dst"]).astype(np.int64) * n + np.maximum(gr["src"], gr["dst"])
        ch = ev["chain"].astype(np.int64)
        found = np.isin(np.minimum(ch[0], ch[1]) * n + np.maximum(ch[0], ch[1]), ek)
        cpid = ev["pid"][ch[0]]
        lab = pred[eid]
        pid = ev["pid"]
        u, cnt = np.unique(pid[ev["hit_quirk"]], return_counts=True)
        for q in u[cnt >= 3]:
            hq = np.flatnonzero(pid == q)
            fr = found[cpid == q].mean()
            nmiss = int((~found[cpid == q]).sum())
            best = {0.5: False, 0.9: False}
            for t in np.unique(lab[hq]):
                if t < 0:
                    continue
                th = np.flatnonzero(lab == t)
                s = np.isin(th, hq).sum()
                for f in best:
                    best[f] |= (s >= f * len(hq)) and (s >= f * len(th))
            ntr = len([t for t in np.unique(lab[hq]) if t >= 0])
            rows.append(dict(frac=fr, nmiss=nmiss, vl=best[0.5], tight=best[0.9], ntracks=ntr))
    fr = np.array([r["frac"] for r in rows])
    nm = np.array([r["nmiss"] for r in rows])
    vl = np.array([r["vl"] for r in rows])
    ti = np.array([r["tight"] for r in rows])
    nt = np.array([r["ntracks"] for r in rows])
    S = {}
    for nmz, m in (("all", np.ones(len(fr), bool)), ("graph_complete", nm == 0), ("missing_1", nm == 1),
                   ("missing_ge2", nm >= 2)):
        S[nmz] = dict(n=int(m.sum()), vl=float(vl[m].mean()), tight=float(ti[m].mean()),
                      tracks_per_quirk_mean=float(nt[m].mean()))
    print(json.dumps(S, indent=1))


if __name__ == "__main__":
    main()
