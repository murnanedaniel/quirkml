"""Attach per-quirk GNN v3 edge scores (g2_q50_dir_s0) to splitter tracks: t["gs"] = int32 [m, 2] local hit pairs
plus float32 [m] scores, for every graph-v3 edge with both ends on the track (max over the two directions).
usage: add_gnn_scores.py IN.pkl OUT.pkl "SCORED_GLOB" """
import pickle, sys
import numpy as np
sys.path.insert(0, "/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/gnn")
from common import read_graphs  # noqa: E402

inp, outp, glob_ = sys.argv[1:4]
tracks = pickle.load(open(inp, "rb"))
ev = sorted({t["event_id"] for t in tracks})
g = read_graphs(glob_, ev[0], ev[-1])
miss = 0; ne = []
for t in tracks:
    gg = g.get(int(t["event_id"]))
    if gg is None:
        miss += 1; t["gs_ij"] = np.zeros((0, 2), np.int32); t["gs_s"] = np.zeros(0, np.float32); continue
    h = t["hits"]; loc = {int(x): i for i, x in enumerate(h)}
    lut = np.full(max(int(gg["src"].max()), int(gg["dst"].max()), int(h.max())) + 1, -1, np.int64); lut[h] = np.arange(len(h))
    a, b = lut[gg["src"].astype(np.int64)], lut[gg["dst"].astype(np.int64)]
    k = (a >= 0) & (b >= 0)
    ij = np.stack([a[k], b[k]], 1).astype(np.int32); s = gg["score"][k].astype(np.float32)
    t["gs_ij"], t["gs_s"] = ij, s; ne.append(len(s))
print("tracks", len(tracks), "missing events", miss, "edges/track mean", np.mean(ne), flush=True)
pickle.dump(tracks, open(outp, "wb"), protocol=5)
