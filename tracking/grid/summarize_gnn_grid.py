"""Collect the GNN grid (Stage F) ACTS summaries into tracking/grid/gnn_grid.csv: per test set (m, Lambda),
per-quirk numbers of the v3 GNN (seed 0, walk + stitch, >= 4 hits; normal truth) and coupled numbers of the pair GNN
(seed 0, components at 0.5, >= 4 hits; pair-truth copy).
  python summarize_gnn_grid.py [--root /pscratch/sd/d/danieltm/quirk_tracking/gnn/grid]
"""
import argparse, csv, json, os, re

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="/pscratch/sd/d/danieltm/quirk_tracking/gnn/grid")
ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "gnn_grid.csv"))
a = ap.parse_args()
rows = []
for s in sorted(os.listdir(a.root)):
    m = re.match(r"m(\d+)_L(\d+)_mu(\d+)$", s)
    if not m:
        continue
    pq, pp = os.path.join(a.root, s, "eval_v3_walk_mh4/summary.json"), os.path.join(a.root, s, "evalpair_pair_cc05_mh4/summary.json")
    if not (os.path.exists(pq) and os.path.exists(pp)):
        continue
    q, p = json.load(open(pq)), json.load(open(pp))
    rows.append(dict(set=s, m=int(m[1]), Lambda=int(m[2]), mu=int(m[3]),
                     quirk_eff=q["quirk_track_efficiency"]["very_loose"]["eff"], quirk_eff_tight=q["quirk_track_efficiency"]["tight"]["eff"],
                     quirks=q["quirk_track_efficiency"]["very_loose"]["total"],
                     quirk_event_eff=q["quirk_event_efficiency"]["very_loose"]["eff"],
                     technical_sm=q["technical_sm_efficiency"]["very_loose"]["eff"], fake=q["fake_rate"]["very_loose"],
                     coupled_eff=p["quirk_track_efficiency"]["very_loose"]["eff"], coupled_eff_tight=p["quirk_track_efficiency"]["tight"]["eff"],
                     pairs=p["quirk_track_efficiency"]["very_loose"]["total"], coupled_fake=p["fake_rate"]["very_loose"]))
with open(a.out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]) if rows else ["set"])
    w.writeheader()
    w.writerows(rows)
print(len(rows), "sets ->", a.out)
