"""Track-level validation used for checkpoint selection (ROADMAP T1): ccwalk_stitch (th_stitch 0.7) tracks on val,
the quick_eval truth matching, and quirk efficiency at a matched fake rate, interpolated over min-hits cuts.
Dev tool only: reported numbers come from tracking/evaluate.py (ACTS)."""

import argparse

import numpy as np

import build_tracks as bt
import quick_eval as qe

ARGS = argparse.Namespace(method="ccwalk_stitch", score_cut_cc=0.01, th_min=0.1, th_add=0.6, th_stitch=0.7,
                          min_hits=3, cc_cut=0.5)


def labels_for(ev, src, dst, score):
    bt.ARGS = ARGS
    _, lab = bt.build_event((ev, {"src": src, "dst": dst, "score": score}))
    return lab


def cut_min_hits(lab, mh):
    out = lab.copy()
    m = out >= 0
    if m.any():
        cnt = np.bincount(out[m])
        out[m & (cnt[np.maximum(out, 0)] < mh)] = -1
    return out


def scan(evs, labels, min_hits=(3, 4, 5, 6, 8, 10, 12, 15)):
    rows = []
    for mh in min_hits:
        r = qe.metrics(evs, {e: cut_min_hits(l, mh) for e, l in labels.items()})
        rows.append(dict(min_hits=mh, quirk_eff=r["very_loose"]["quirk_eff"], quirk_eff_tight=r["tight"]["quirk_eff"],
                         tech_eff=r["very_loose"]["tech_eff"], fake=r["fake_rate"], pair_union=r["quirk_pair_union"]))
    return rows


def at_fake(rows, target):
    """Quirk efficiency at fake rate = target, linear interpolation between neighbouring min-hits points
    (fake falls as min-hits rises); clamped to the end points."""
    f = np.array([r["fake"] for r in rows])
    e = np.array([r["quirk_eff"] for r in rows])
    o = np.argsort(f)
    return float(np.interp(target, f[o], e[o]))
