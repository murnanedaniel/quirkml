"""Fast truth-matching check of predictions.parquet against the cache (for scanning track-building settings on val;
the reported numbers come from tracking/evaluate.py). Double matching at 0.5 (very_loose) and 0.9 (tight):
a particle is matched if a track has >= f of its hits from it and holds >= f of the particle's hits.
Fake: track not double-matched at 0.5 (as ACTS with double matching).
Quirk-pair diagnostic (not an ACTS number): the two quirks of an event as one object (events with both quirks
having >= 3 hits). pair_single: one track has >= 50% of its hits from the pair and holds >= 50% of the pair's
hits. pair_union: the tracks with >= 50% of their hits from the pair together hold >= 50% of the pair's hits.

  python quick_eval.py --cache C --pred P/predictions.parquet --events A B
"""

import argparse
import json
import os
import sys

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_events  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--events", type=int, nargs=2, required=True)
    a = ap.parse_args()
    evs = load_events(a.cache, *a.events)
    t = pq.read_table(a.pred).to_pandas()
    t = t[(t.event_id >= a.events[0]) & (t.event_id <= a.events[1])].sort_values(["event_id", "hit_id"])
    groups = {e: g.predicted_track_id.to_numpy() for e, g in t.groupby("event_id")}
    print(json.dumps(metrics(evs, groups)))


def metrics(evs, groups):
    """groups: {event_id: predicted_track_id per hit}. Returns the dict printed by main."""
    res = {f: dict(q=0, qm=0, tech=0, techm=0) for f in (0.5, 0.9)}
    ntr, nfake = 0, 0
    npair, psingle, punion = 0, 0, 0
    for ev in evs:
        lab = groups[ev["event_id"]]
        pid = ev["pid"]
        u, inv, pcnt = np.unique(pid, return_inverse=True, return_counts=True)
        nh = pcnt[inv]
        m = lab >= 0
        # (track, particle) shared counts
        key = lab[m].astype(np.int64) * (pid.max() + 1) + pid[m]
        k, c = np.unique(key, return_counts=True)
        trk, par = k // (pid.max() + 1), k % (pid.max() + 1)
        tsize = np.bincount(lab[m], minlength=lab.max() + 1 if m.any() else 0)
        psize = dict(zip(u, pcnt))
        ntr += int((tsize > 0).sum())
        psz = np.array([psize[p] for p in par])
        dm = (c >= 0.5 * tsize[trk]) & (c >= 0.5 * psz)  # double match at 0.5
        good = np.zeros(len(tsize), bool)
        good[trk[dm]] = True
        nfake += int((~good & (tsize > 0)).sum())
        quirks = np.unique(pid[ev["hit_quirk"] & (nh >= 3)])
        if len(quirks) == 2:
            inpair = np.isin(pid, quirks)
            npair += 1
            npairhits = inpair.sum()
            if m.any():
                tp = np.bincount(lab[m], weights=inpair[m].astype(float), minlength=len(tsize))
                major = (tp >= 0.5 * np.maximum(tsize, 1)) & (tsize > 0)
                psingle += int(np.any(major & (tp >= 0.5 * npairhits)))
                punion += int(tp[major].sum() >= 0.5 * npairhits)
        tech = np.unique(pid[(~ev["hit_quirk"]) & ev["hit_target"] & (ev["hit_pt"] > 1) & (nh >= 7)])
        for f in (0.5, 0.9):
            ok = (c >= f * tsize[trk]) & (c >= f * np.array([psize[p] for p in par]))
            matched = set(par[ok].tolist())
            res[f]["q"] += len(quirks)
            res[f]["qm"] += sum(int(q) in matched for q in quirks)
            res[f]["tech"] += len(tech)
            res[f]["techm"] += sum(int(p) in matched for p in tech)
    out = {}
    for f, name in ((0.5, "very_loose"), (0.9, "tight")):
        r = res[f]
        out[name] = dict(quirk_eff=r["qm"] / max(r["q"], 1), tech_eff=r["techm"] / max(r["tech"], 1))
    out["fake_rate"] = nfake / max(ntr, 1)
    out["quirk_pair_single"] = psingle / max(npair, 1)
    out["quirk_pair_union"] = punion / max(npair, 1)
    out["tracks_per_event"] = ntr / max(len(evs), 1)
    return out


if __name__ == "__main__":
    main()
