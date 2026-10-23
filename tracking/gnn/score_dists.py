"""Score distributions on val of quirk edges in a scored graph: same-quirk truth-chain edges, same-quirk non-chain
edges, and quirk-to-other-quirk edges (one end on each quirk of the event). Prints quantiles, fraction above
0.1 / 0.5, and edge counts per event.

  python score_dists.py --cache C --scored 'S/*.parquet' --events A B
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import edge_truth, load_events, read_graphs  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--scored", required=True)
    ap.add_argument("--events", type=int, nargs=2, required=True)
    a = ap.parse_args()
    evs = load_events(a.cache, *a.events)
    g = read_graphs(a.scored, *a.events)
    cat = {"same_quirk_chain": [], "same_quirk_nonchain": [], "quirk_to_other_quirk": []}
    for ev in evs:
        gg = g[ev["event_id"]]
        s, d, sc = gg["src"].astype(np.int64), gg["dst"].astype(np.int64), gg["score"]
        pid, hq = ev["pid"], ev["hit_quirk"]
        y, _ = edge_truth(ev, s, d)
        both = hq[s] & hq[d]
        same = pid[s] == pid[d]
        cat["same_quirk_chain"].append(sc[both & same & y])
        cat["same_quirk_nonchain"].append(sc[both & same & ~y])
        cat["quirk_to_other_quirk"].append(sc[both & ~same])
    out = {}
    for k, v in cat.items():
        v = np.concatenate(v)
        out[k] = {"n_per_event": len(v) / len(evs), "q10_50_90": np.round(np.percentile(v, [10, 50, 90]), 3).tolist() if len(v) else None,
                  "frac_gt_0.1": float((v > 0.1).mean()) if len(v) else None, "frac_gt_0.5": float((v > 0.5).mean()) if len(v) else None,
                  "hist_0_1_by_0.1": np.histogram(v, bins=10, range=(0, 1))[0].tolist()}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
