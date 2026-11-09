"""Audit of the training crops of train_gnn60.py (phi window sized for ~400k edges; quirk-centred windows centred on a
random quirk hit): per quirk-centred crop, fraction of the pair's hits inside, fraction of the pair's hits in the largest
component of pair edges inside the crop (pair connectivity), and the phi distance of the pair's hits to the nearest
window edge. 5 crops per event on train events 3000-3199 (pair cache, graph v3 + mu 0 filter at 0.1).
  python crop_audit.py OUT_JSON"""
import json, math, os, random, sys
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "gnn"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_events  # noqa: E402
from graph_apply60 import read_compact  # noqa: E402

evs = load_events("/pscratch/sd/d/danieltm/quirk_pu60/cache/m200_L774_mu60_dir_pair", 3000, 3199)
g = read_compact("/pscratch/sd/d/danieltm/quirk_pu60/graphs/v3_mu0f_cut0.1/events3000-3999.d", 3000, 3199)
rng = random.Random(0)
rows = []
for ev in evs:
    gg = g[ev["event_id"]]; s = gg["src"].astype(np.int64); d = gg["dst"].astype(np.int64)
    hq = ev["hit_quirk"]; qi = np.flatnonzero(hq)
    if len(qi) < 3:
        continue
    n = len(hq); phi = np.arctan2(ev["y"], ev["x"])
    width = 2 * math.pi * min(1.0, 400000 / len(s))
    pe = hq[s] & hq[d]
    qphi = phi[qi]
    spread = float(np.ptp(np.unwrap(np.sort(qphi))))
    for _ in range(5):
        c = float(phi[rng.choice(list(qi))])
        dd = lambda x: (x - c + math.pi) % (2 * math.pi) - math.pi  # noqa: E731
        ins = np.abs(dd(phi)) < width / 2
        frac_in = float(ins[qi].mean())
        k = pe & ins[s] & ins[d]
        loc = -np.ones(n, np.int64); loc[qi] = np.arange(len(qi))
        _, lab = connected_components(coo_matrix((np.ones(k.sum()), (loc[s[k]], loc[d[k]])), shape=(len(qi), len(qi))), directed=False)
        conn = float(np.bincount(lab).max() / len(qi))
        _, lab2 = connected_components(coo_matrix((np.ones(pe.sum()), (loc[s[pe]], loc[d[pe]])), shape=(len(qi), len(qi))), directed=False)
        conn_full = float(np.bincount(lab2).max() / len(qi))
        dist = width / 2 - np.abs(dd(qphi))
        rows.append(dict(width=width, frac_in=frac_in, conn=conn, conn_full=conn_full, min_dist=float(dist.min()),
                         frac_near_edge=float((np.abs(dist) < 0.05).mean()), pair_phi_spread=spread))
R = {k: np.array([r[k] for r in rows]) for k in rows[0]}
out = {"crops": len(rows), "window_width_rad_mean": float(R["width"].mean()), "pair_phi_spread_median": float(np.median(R["pair_phi_spread"])),
       "frac_pair_hits_inside": {"mean": float(R["frac_in"].mean()), "frac_crops_all_inside": float((R["frac_in"] == 1).mean()),
                                 "frac_crops_lt_0.9": float((R["frac_in"] < 0.9).mean()), "frac_crops_le_0.5": float((R["frac_in"] <= 0.5).mean())},
       "pair_connectivity_in_crop": {"mean": float(R["conn"].mean()), "full_graph_mean": float(R["conn_full"].mean()),
                                     "frac_crops_gt_0.5": float((R["conn"] > 0.5).mean()), "frac_full_gt_0.5": float((R["conn_full"] > 0.5).mean())},
       "frac_pair_hits_within_0.05rad_of_edge_mean": float(R["frac_near_edge"].mean()),
       "frac_crops_some_pair_hit_outside": float((R["min_dist"] < 0).mean())}
json.dump(out, open(sys.argv[1], "w"), indent=1); print(json.dumps(out, indent=1))
