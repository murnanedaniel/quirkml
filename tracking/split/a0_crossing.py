"""A0: crossing ambiguity of the two quirks on test truth (m200_L774_mu0, events 17000-19999, R1 first-particle labels).

Per pair: A-B hit pairs on the same surface and on the same layer within detector resolution and within 1, 3, 10 mm;
hits shared by both quirks (one cluster from both, from the per-run particle_ids); minimum A-B separation along the
trajectory and number of crossings (local minima of the separation below 3 / 10 mm, positions interpolated in time).

Upper bound for a splitter that sees positions only: an A hit and a B hit closer than d on the same layer cannot be told
apart, so each such pair (a maximum matching between A and B hits within d) is labelled correctly with probability
1/2 at best, and a wrong guess costs one hit on each quirk. A quirk with n hits passes tight (> 90% purity and
completeness) only if the wrong guesses e satisfy n - e > 0.9 n; the bound per quirk is P(Binomial(m, 1/2) <= emax),
averaged over quirks (m = its ambiguous pairs). Same for > 50%.

  python a0_crossing.py --out DIR
"""
import argparse, glob, json, os, sys
import numpy as np
import pyarrow.parquet as pq
from scipy.stats import binom

sys.path.insert(0, "/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/gnn")
from common import load_events  # noqa: E402

R1 = "/global/cfs/cdirs/m4958/data/ColliderML/simulation/quirks/m200_L774_mu0/v1/parquet_r1/reco/tracker_hits"
RUNS = "/global/cfs/cdirs/m4958/data/ColliderML/simulation/quirks/m200_L774_mu0/v1/runs"
CACHE = "/pscratch/sd/d/danieltm/quirk_tracking/gnn/cache/m200_L774_mu0"
# position resolution scale per volume (mm): pixels 50 um pitch, short strips 80 um, long strips 0.1-0.125 mm in loc0
RES = {16: 0.05, 17: 0.05, 18: 0.05, 23: 0.08, 24: 0.08, 25: 0.08, 28: 0.125, 29: 0.1, 30: 0.125}
DS = ("res", 1.0, 3.0, 10.0)


def greedy_match(dist, d):
    """Maximum (greedy, closest first) one-to-one matching of A-B pairs with distance < d."""
    ia, ib = np.nonzero(dist < d)
    o = np.argsort(dist[ia, ib])
    ua, ub, m = set(), set(), 0
    for k in o:
        if ia[k] in ua or ib[k] in ub:
            continue
        ua.add(ia[k]); ub.add(ib[k]); m += 1
    return m


def bound(n, m, frac):
    emax = int(np.ceil(n - frac * n) - 1)  # n - e > frac n  <=>  e < n (1 - frac)
    if emax < 0:
        return 0.0
    return float(binom.cdf(emax, m, 0.5)) if m > 0 else 1.0


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    evs = load_events(CACHE, 17000, 19999)
    H = {}
    for f in sorted(glob.glob(R1 + "/*events1[7-9]*.parquet")):
        t = pq.read_table(f, columns=["event_id", "time", "volume_id", "layer_id", "surface_id"]).to_pydict()
        for i, e in enumerate(t["event_id"]):
            H[e] = {k: np.asarray(t[k][i]) for k in t if k != "event_id"}
    shared = {}
    for run in range(34, 40):
        T = pq.read_table(f"{RUNS}/{run}/tracker_hits/tracker_hits_000000-100000.parquet", columns=["event_id", "particle_ids"]).to_pydict()
        for e, pids in zip(T["event_id"], T["particle_ids"]):
            shared[run * 500 + e] = pids
    rows = []
    for ev in evs:
        pid, hq = ev["pid"], ev["hit_quirk"]
        qs = [q for q in np.unique(pid[hq]) if (pid == q).sum() >= 3]
        if len(qs) != 2:
            continue
        h = H[ev["event_id"]]
        A, B = qs
        ia, ib = np.flatnonzero(pid == A), np.flatnonzero(pid == B)
        pos = np.stack([ev["x"], ev["y"], ev["z"]], 1).astype(np.float64)
        D = np.linalg.norm(pos[ia][:, None] - pos[ib][None], axis=2)
        vol, lay, sur = h["volume_id"].astype(int), h["layer_id"].astype(int), h["surface_id"].astype(int)
        same_layer = (vol[ia][:, None] == vol[ib][None]) & (lay[ia][:, None] == lay[ib][None])
        same_surf = same_layer & (sur[ia][:, None] == sur[ib][None])
        res = np.array([RES[v] for v in vol[ia]])[:, None] * 3  # 3 x pitch
        r = dict(eid=ev["event_id"], nA=len(ia), nB=len(ib), same_surface_pairs=int(same_surf.sum()),
                 same_layer_pairs=int(same_layer.sum()))
        # shared clusters: hit with both quirks in particle_ids
        pl = shared[ev["event_id"]]
        r["shared_hits"] = int(sum(1 for x in pl if A in x and B in x))
        for d in DS:
            M = np.where(same_layer, D, np.inf)
            if d == "res":
                M = np.where(same_surf & (D < res), D, np.inf); dd = np.inf
            else:
                dd = d
            r[f"pairs_lt_{d}"] = int(np.isfinite(M).sum() if d == "res" else (M < dd).sum())
            r[f"match_lt_{d}"] = greedy_match(M, dd if d != "res" else np.inf)
        # separation along the trajectory (time interpolation over the overlap of the two time ranges)
        t = h["time"].astype(np.float64)
        tA, tB = t[ia], t[ib]
        lo, hi = max(tA.min(), tB.min()), min(tA.max(), tB.max())
        if hi > lo:
            ts = np.linspace(lo, hi, 400)
            oa, ob = np.argsort(tA), np.argsort(tB)
            pa = np.stack([np.interp(ts, tA[oa], pos[ia][oa, k]) for k in range(3)], 1)
            pb = np.stack([np.interp(ts, tB[ob], pos[ib][ob, k]) for k in range(3)], 1)
            sep = np.linalg.norm(pa - pb, axis=1)
            mins = np.flatnonzero((sep[1:-1] < sep[:-2]) & (sep[1:-1] <= sep[2:])) + 1
            r["min_sep"] = float(sep.min())
            r["crossings_lt3"] = int((sep[mins] < 3).sum())
            r["crossings_lt10"] = int((sep[mins] < 10).sum())
        else:
            r["min_sep"], r["crossings_lt3"], r["crossings_lt10"] = np.nan, 0, 0
        rows.append(r)
    out = {"pairs": len(rows)}
    g = lambda k: np.array([x[k] for x in rows], float)  # noqa: E731
    for k in ["same_surface_pairs", "same_layer_pairs", "shared_hits"] + [f"pairs_lt_{d}" for d in DS] + [f"match_lt_{d}" for d in DS] + ["crossings_lt3", "crossings_lt10"]:
        v = g(k); out[k] = dict(mean=float(v.mean()), median=float(np.median(v)), frac_pairs_gt0=float((v > 0).mean()))
    ms = g("min_sep"); out["min_sep_mm_q10_50_90"] = np.nanpercentile(ms, [10, 50, 90]).round(2).tolist()
    for frac, nm in ((0.5, "loose_strict"), (0.9, "tight_strict")):
        for d in DS:
            b = [bound(x["nA"], x[f"match_lt_{d}"], frac) for x in rows] + [bound(x["nB"], x[f"match_lt_{d}"], frac) for x in rows]
            out[f"bound_{nm}_d{d}"] = float(np.mean(b))
    json.dump(out, open(os.path.join(a.out, "a0_summary.json"), "w"), indent=1)
    json.dump(rows, open(os.path.join(a.out, "a0_pairs.json"), "w"))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
