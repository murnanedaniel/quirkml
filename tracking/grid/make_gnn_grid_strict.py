"""gnn_grid_strict.csv: gnn_grid.csv plus the strict-matching (> 50% / > 90%) numbers of every set, from the Python
scoring of the same predictions (/pscratch/sd/d/danieltm/quirk_tracking/strict50/strict_all.py -> strict_grid.csv;
its >= numbers equal the ACTS ones of gnn_grid.csv on all 220 evaluations, checked there).
New columns: quirk_eff_strict, quirk_eff_tight_strict, quirk_event_eff_tight, quirk_event_eff_strict,
quirk_event_eff_tight_strict, coupled_eff_strict, coupled_eff_tight_strict, quirks_at_50 / pairs_at_50 (matched at
>= 50% but not > 50%: purity or completeness exactly 50%).
  python make_gnn_grid_strict.py [--strict-csv .../strict50/strict_grid.csv]
"""
import argparse
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--strict-csv", default="/pscratch/sd/d/danieltm/quirk_tracking/strict50/strict_grid.csv")
a = ap.parse_args()
g = pd.read_csv(os.path.join(HERE, "gnn_grid.csv"))
s = pd.read_csv(a.strict_csv)
s["set"] = s.label.str.split("/").str[0]
q = s[s.label.str.endswith("/eval_v3_walk_mh4")].set_index("set")
p = s[s.label.str.endswith("/evalpair_pair_cc05_mh4")].set_index("set")
assert q.check_vl.all() and q.check_tight.all() and p.check_vl.all() and p.check_tight.all()
g = g.set_index("set")
assert (abs(g.quirk_eff - q.ge_vl.reindex(g.index)) < 1e-9).all() and (abs(g.coupled_eff - p.ge_vl.reindex(g.index)) < 1e-9).all()
g["quirk_eff_strict"] = q.gt_vl
g["quirk_eff_tight_strict"] = q.gt_tight
g["quirk_event_eff_tight"] = q.ev_ge_tight
g["quirk_event_eff_strict"] = q.ev_gt_vl
g["quirk_event_eff_tight_strict"] = q.ev_gt_tight
g["coupled_eff_strict"] = p.gt_vl
g["coupled_eff_tight_strict"] = p.gt_tight
g["quirks_at_50"] = q.at_vl
g["pairs_at_50"] = p.at_vl
g.reset_index().to_csv(os.path.join(HERE, "gnn_grid_strict.csv"), index=False)
print(len(g), "sets ->", os.path.join(HERE, "gnn_grid_strict.csv"))
