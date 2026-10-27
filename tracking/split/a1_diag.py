"""Diagnose the A1 oracle-split errors on test (truth used for diagnosis only): per pair, which split set belongs to
which quirk (best of the two assignments), the wrongly assigned hits, their distance to the nearest hit of the other
quirk, layer, time distance to the nearest crossing (local minimum of the time-interpolated A-B separation < 3 mm), and
whether errors form runs (consecutive in true time along their own quirk) or isolated hits.
  python a1_diag.py --pred P/predictions.parquet --out DIR
"""
import argparse, glob, json, os, sys
import numpy as np
import pyarrow.parquet as pq
sys.path.insert(0, "/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/gnn")
from common import VOLUMES, load_events  # noqa: E402

R1 = "/global/cfs/cdirs/m4958/data/ColliderML/simulation/quirks/m200_L774_mu0/v1/parquet_r1/reco/tracker_hits"
CACHE = "/pscratch/sd/d/danieltm/quirk_tracking/gnn/cache/m200_L774_mu0"
ap = argparse.ArgumentParser(); ap.add_argument("--pred", required=True); ap.add_argument("--out", required=True); a = ap.parse_args()
evs = load_events(CACHE, 17000, 19999)
T = {}
for f in sorted(glob.glob(R1 + "/*events1[7-9]*.parquet")):
    t = pq.read_table(f, columns=["event_id", "time"]).to_pydict()
    for e, v in zip(t["event_id"], t["time"]):
        T[e] = np.asarray(v, np.float64) / 299.792458
P = pq.read_table(a.pred).to_pandas()
P = {int(e): g.sort_values("hit_id").predicted_track_id.to_numpy() for e, g in P.groupby("event_id")}
errs, runs, per_quirk = [], [], []
for ev in evs:
    pid, hq = ev["pid"], ev["hit_quirk"]
    qs = [q for q in np.unique(pid[hq]) if (pid == q).sum() >= 3]
    if len(qs) != 2:
        continue
    lab = P[ev["event_id"]]; t = T[ev["event_id"]]
    pos = np.stack([ev["x"], ev["y"], ev["z"]], 1).astype(np.float64)
    ia, ib = np.flatnonzero(pid == qs[0]), np.flatnonzero(pid == qs[1])
    sets = [s for s in np.unique(lab[np.concatenate([ia, ib])]) if s >= 0]
    if len(sets) < 1:
        continue
    # assignment of predicted sets to quirks: the one with more correct hits
    def correct(sa, sb):
        return (lab[ia] == sa).sum() + (lab[ib] == sb).sum()
    best = max([(sa, sb) for sa in sets + [-2] for sb in sets + [-2] if sa != sb], key=lambda p: correct(*p))
    wrong_a = ia[lab[ia] != best[0]]; wrong_b = ib[lab[ib] != best[1]]
    # crossings
    tA, tB = t[ia], t[ib]; lo, hi = max(tA.min(), tB.min()), min(tA.max(), tB.max())
    cross = np.array([])
    if hi > lo:
        ts = np.linspace(lo, hi, 600); oa, ob = np.argsort(tA), np.argsort(tB)
        pa = np.stack([np.interp(ts, tA[oa], pos[ia][oa, k]) for k in range(3)], 1)
        pb = np.stack([np.interp(ts, tB[ob], pos[ib][ob, k]) for k in range(3)], 1)
        sep = np.linalg.norm(pa - pb, axis=1)
        mins = np.flatnonzero((sep[1:-1] < sep[:-2]) & (sep[1:-1] <= sep[2:])) + 1
        cross = ts[mins[sep[mins] < 3]]
    for own, other, wrong in ((ia, ib, wrong_a), (ib, ia, wrong_b)):
        o = own[np.argsort(t[own])]
        w = np.isin(o, wrong)
        # runs of consecutive wrong hits along the quirk's own time order
        r, cur = [], 0
        for x in w:
            if x:
                cur += 1
            elif cur:
                r.append(cur); cur = 0
        if cur:
            r.append(cur)
        runs += r
        per_quirk.append(dict(n=len(own), wrong=int(w.sum()), runs=len(r), longest=max(r) if r else 0))
        for h in wrong:
            d = np.linalg.norm(pos[other] - pos[h], axis=1).min()
            same_layer = (ev["vol_idx"][other] == ev["vol_idx"][h]) & (ev["layer"][other] == ev["layer"][h])
            dl = np.linalg.norm(pos[other][same_layer] - pos[h], axis=1).min() if same_layer.any() else np.inf
            dc = np.abs(cross - t[h]).min() if len(cross) else np.inf
            errs.append(dict(d_other=float(d), d_other_same_layer=float(dl), vol=int(VOLUMES[ev["vol_idx"][h]]), r=float(np.hypot(*pos[h, :2])),
                             dt_cross_ns=float(dc)))
pq_ = per_quirk
nw = np.array([q["wrong"] for q in pq_]); nn = np.array([q["n"] for q in pq_])
out = dict(quirks=len(pq_), wrong_hits_per_quirk_mean=float(nw.mean()), frac_quirks_no_error=float((nw == 0).mean()),
           frac_quirks_err_le_10pct=float((nw <= 0.1 * nn).mean()),
           runs_per_quirk_mean=float(np.mean([q["runs"] for q in pq_])), run_length_hist=np.bincount(np.minimum(runs, 10)).tolist(),
           frac_wrong_in_runs_ge3=float(sum(r for r in runs if r >= 3) / max(sum(runs), 1)),
           longest_run_q50_90=np.percentile([q["longest"] for q in pq_], [50, 90]).tolist())
d = np.array([e["d_other"] for e in errs]); dl = np.array([e["d_other_same_layer"] for e in errs]); dc = np.array([e["dt_cross_ns"] for e in errs])
out.update(n_wrong=len(errs), d_other_mm_q10_50_90=np.percentile(d, [10, 50, 90]).round(2).tolist(),
           d_other_same_layer_mm_q10_50_90=np.percentile(dl[np.isfinite(dl)], [10, 50, 90]).round(2).tolist(),
           frac_no_other_on_layer=float(np.mean(~np.isfinite(dl))),
           frac_d_same_layer_lt1=float(np.mean(dl < 1)), frac_d_same_layer_lt3=float(np.mean(dl < 3)), frac_d_same_layer_lt10=float(np.mean(dl < 10)),
           dt_cross_ns_q10_50_90=np.percentile(dc[np.isfinite(dc)], [10, 50, 90]).round(3).tolist(),
           wrong_by_volume={int(v): int(c) for v, c in zip(*np.unique([e["vol"] for e in errs], return_counts=True))})
os.makedirs(a.out, exist_ok=True); json.dump(out, open(a.out + "/a1_diag.json", "w"), indent=1); print(json.dumps(out, indent=1))
