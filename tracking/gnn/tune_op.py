"""Operating-point choice on val (ROADMAP T1): grid over ccwalk_stitch thresholds (th_min, th_add, th_stitch) and
min-hits, scored with quick_eval truth matching; reports quirk efficiency at the matched fake rate for each threshold
set and writes the best to <out>.json. Dev tool for choosing the operating point; test numbers come from ACTS.

  python tune_op.py --cache C --scored 'S/*.parquet' --events 16200 16999 --out O.json [--fake-target 0.10]
"""

import argparse
import itertools
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_tracks as bt  # noqa: E402
import valtracks as vt  # noqa: E402
from common import load_events, read_graphs  # noqa: E402

EVS, G = None, None


def run_cfg(cfg):
    th_min, th_add, th_stitch = cfg
    bt.ARGS = argparse.Namespace(method="ccwalk_stitch", score_cut_cc=min(0.01, th_min), th_min=th_min, th_add=th_add,
                                 th_stitch=th_stitch, min_hits=3, cc_cut=0.5)
    labels = {}
    for ev in EVS:
        g = G[ev["event_id"]]
        _, labels[ev["event_id"]] = bt.build_event((ev, {"src": g["src"], "dst": g["dst"], "score": g["score"]}))
    rows = vt.scan(EVS, labels)
    return dict(th_min=th_min, th_add=th_add, th_stitch=th_stitch, rows=rows)


def main():
    global EVS, G
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--scored", required=True)
    ap.add_argument("--events", type=int, nargs=2, default=[16200, 16999])
    ap.add_argument("--fake-target", type=float, default=0.10)
    ap.add_argument("--th-min", type=float, nargs="+", default=[0.1, 0.3, 0.5])
    ap.add_argument("--th-add", type=float, nargs="+", default=[0.6, 0.8, 0.9, 0.97])
    ap.add_argument("--th-stitch", type=float, nargs="+", default=[0.7, 0.9])
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    EVS = load_events(a.cache, *a.events)
    G = read_graphs(a.scored, *a.events)
    cfgs = [c for c in itertools.product(a.th_min, a.th_add, a.th_stitch) if c[1] >= c[0]]
    with Pool(min(a.workers, len(cfgs))) as pool:
        res = pool.map(run_cfg, cfgs)
    for r in res:
        r["quirk_eff_at_fake"] = vt.at_fake(r["rows"], a.fake_target)
        f = [x["fake"] for x in r["rows"]]
        r["fake_reachable"] = bool(min(f) <= a.fake_target)
        print(json.dumps({k: r[k] for k in ["th_min", "th_add", "th_stitch", "quirk_eff_at_fake", "fake_reachable"]}), flush=True)
    ok = [r for r in res if r["fake_reachable"]] or res
    best = max(ok, key=lambda r: r["quirk_eff_at_fake"])
    # min-hits cut on val nearest the fake target from above (fake <= target)
    rows = sorted(best["rows"], key=lambda x: x["min_hits"])
    mh = next((x["min_hits"] for x in rows if x["fake"] <= a.fake_target), rows[-1]["min_hits"])
    best["min_hits_at_target"] = mh
    json.dump({"fake_target": a.fake_target, "best": best, "all": res}, open(a.out, "w"), indent=1)
    print("best", json.dumps({k: best[k] for k in ["th_min", "th_add", "th_stitch", "quirk_eff_at_fake", "min_hits_at_target"]}))


if __name__ == "__main__":
    main()
