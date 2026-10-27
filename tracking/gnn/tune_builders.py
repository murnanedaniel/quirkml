"""Val sweep of track builders on GNN scores (quick_eval truth matching): ccwalk_stitch reference, plain connected
components and the hybrid at several score cuts, >= --min-hits hits. Prints one JSON line per setting.
  python tune_builders.py --cache C --scored 'S/events16000-16999.parquet' --events 16000 16999
"""
import argparse, json, os, sys
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_tracks as bt  # noqa: E402
import quick_eval as qe  # noqa: E402
from common import load_events, read_graphs  # noqa: E402

EVS = G = None


def run(cfg):
    meth, cut = cfg
    bt.ARGS = argparse.Namespace(method=meth, score_cut_cc=0.01, th_min=0.1, th_add=0.6, th_stitch=0.7,
                                 min_hits=A.min_hits, cc_cut=cut)
    lab = {ev["event_id"]: bt.build_event((ev, G[ev["event_id"]]))[1] for ev in EVS}
    r = qe.metrics(EVS, lab)
    return dict(method=meth, cut=cut, quirk_eff=r["very_loose"]["quirk_eff"], tight=r["tight"]["quirk_eff"],
                tech=r["very_loose"]["tech_eff"], fake=r["fake_rate"], tracks=r["tracks_per_event"])


ap = argparse.ArgumentParser()
ap.add_argument("--cache", required=True); ap.add_argument("--scored", required=True)
ap.add_argument("--events", type=int, nargs=2, default=[16000, 16999]); ap.add_argument("--min-hits", type=int, default=4)
A = ap.parse_args()
EVS = load_events(A.cache, *A.events)
G = read_graphs(A.scored, *A.events)
cfgs = [("ccwalk_stitch", 0.5)] + [("cc", c) for c in (0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98)] + \
       [("hybrid", c) for c in (0.5, 0.7, 0.8, 0.9, 0.95, 0.98)]
with Pool(len(cfgs)) as p:
    for r in p.map(run, cfgs):
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
