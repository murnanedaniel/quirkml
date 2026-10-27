"""Fairer coupled ceiling: every graph edge whose two hits both belong to the event's quirk pair gets score 1,
everything else 0 (pair cache labels). Writes scored graphs for build_tracks.py.
  python pair_allscore.py --graphs 'G/events1[7-9]*.parquet' --cache PAIRCACHE --events 17000 19999 --out DIR
"""
import argparse, glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_events, read_graphs, write_graphs  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--graphs", required=True); ap.add_argument("--cache", required=True)
ap.add_argument("--events", type=int, nargs=2, required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()
os.makedirs(os.path.join(a.out, "scored"), exist_ok=True)
evs = {e["event_id"]: e for e in load_events(a.cache, *a.events)}
npos = []
for f in sorted(glob.glob(a.graphs)):
    g = read_graphs(f, *a.events)
    rows = []
    for eid in sorted(g):
        ev, gr = evs[eid], g[eid]
        s, d = gr["src"].astype(np.int64), gr["dst"].astype(np.int64)
        hq = ev["hit_quirk"]
        y = hq[s] & hq[d] & (ev["pid"][s] == ev["pid"][d])
        npos.append(int(y.sum()))
        rows.append(dict(event_id=eid, src=s, dst=d, y=y, score=y.astype(np.float32)))
    write_graphs(os.path.join(a.out, "scored", os.path.basename(f)), rows)
json.dump({"same_pair_edges_per_event": float(np.mean(npos))}, open(os.path.join(a.out, "stats.json"), "w"))
print("same-pair edges per event", np.mean(npos))
