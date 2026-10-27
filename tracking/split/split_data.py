"""A1 data: per pair track, hit features and A/B/neither labels, saved as one pickle per event range.

Modes
  oracle : the pair track = all hits of both quirks (R1 labels); labels A (lower particle_id) / B.
  cc     : tracks of a predictions.parquet (e.g. pair GNN + components); every track with >= --min-hits hits that holds
           >= 1 quirk hit, plus a random --other-frac of the remaining tracks with >= --min-hits hits (they teach the
           model to leave ordinary tracks whole). Labels: the track's most frequent particle = A, the second most
           frequent (if >= 3 hits on the track) = B, the rest = neither (2).
Hit features (no truth): x, y, z, r, cos/sin phi, eta, positions relative to the track centroid, volume one-hot,
layer, cluster shape (4), direction (phi/eta_angle_abs, |diag_sign|, axis flag, the two axis candidates); time
(relative to the track's earliest hit, ns; true sim-hit time, so only for the "with time" arm) is stored separately.

  python split_data.py --mode oracle --events 0 15999 --out F.pkl
  python split_data.py --mode cc --pred P/predictions.parquet --events 0 15999 --out F.pkl
"""
import argparse, glob, os, pickle, sys
import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, "/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/gnn")
from common import VOLUMES, load_events  # noqa: E402

CACHE = "/pscratch/sd/d/danieltm/quirk_tracking/gnn/cache/m200_L774_mu0_dir"
R1 = "/global/cfs/cdirs/m4958/data/ColliderML/simulation/quirks/m200_L774_mu0/v1/parquet_r1/reco/tracker_hits"


def feats(ev, idx):
    x, y, z = ev["x"][idx].astype(np.float32), ev["y"][idx].astype(np.float32), ev["z"][idx].astype(np.float32)
    r = np.hypot(x, y); phi = np.arctan2(y, x); eta = np.arcsinh(z / np.maximum(r, 1e-3))
    c = np.stack([x, y, z], 1); rel = (c - c.mean(0)) / 100.0
    # offset from the centroid of the track's hits on the same layer (mm / 10): if both particles cross the layer this is
    # about +- half their separation, the fine-scale cue the global coordinates cannot carry
    key = ev["vol_idx"][idx].astype(np.int64) * 100 + ev["layer"][idx].astype(np.int64)
    off = np.zeros_like(c)
    nsame = np.zeros(len(idx), np.float32)
    for k in np.unique(key):
        m = key == k
        off[m] = c[m] - c[m].mean(0)
        nsame[m] = m.sum()
    off = np.clip(off / 10.0, -10, 10)
    oh = np.eye(len(VOLUMES), dtype=np.float32)[ev["vol_idx"][idx].astype(int)]
    cl = np.log1p(ev["cluster"][idx].astype(np.float32))
    ax = np.nan_to_num(np.concatenate([ev["axa"][idx], ev["axb"][idx]], 1).astype(np.float32))
    ok = np.isfinite(ev["axa"][idx, 0]).astype(np.float32)
    d = np.stack([np.nan_to_num(ev["dphi"][idx]), np.nan_to_num(ev["deta"][idx]), np.abs(ev["diag"][idx]).astype(np.float32), ok], 1)
    return np.concatenate([np.stack([x / 1000, y / 1000, z / 3000, r / 1000, np.cos(phi), np.sin(phi), eta / 3], 1), rel,
                           off, np.log1p(nsame)[:, None],
                           oh, (ev["layer"][idx].astype(np.float32) / 20)[:, None], cl, d, ax], 1).astype(np.float32)


def labels(pid, idx):
    u, c = np.unique(pid[idx], return_counts=True)
    o = np.argsort(-c)
    lab = np.full(len(idx), 2, np.int64)
    lab[pid[idx] == u[o[0]]] = 0
    if len(o) > 1 and c[o[1]] >= 3:
        lab[pid[idx] == u[o[1]]] = 1
    return lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["oracle", "cc"], required=True)
    ap.add_argument("--pred", default=None)
    ap.add_argument("--events", type=int, nargs=2, required=True)
    ap.add_argument("--min-hits", type=int, default=4)
    ap.add_argument("--other-frac", type=float, default=0.02)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    evs = load_events(CACHE, *a.events)
    T = {}
    for f in sorted(glob.glob(R1 + "/*.parquet")):
        t = pq.read_table(f, columns=["event_id", "time"])
        eid = t["event_id"].to_numpy()
        sel = np.flatnonzero((eid >= a.events[0]) & (eid <= a.events[1]))
        if len(sel):
            tt = t.take(sel).to_pydict()
            for e, v in zip(tt["event_id"], tt["time"]):
                T[e] = np.asarray(v, np.float64)
    P = None
    if a.mode == "cc":
        p = pq.read_table(a.pred, filters=[("event_id", ">=", a.events[0]), ("event_id", "<=", a.events[1])]).to_pandas()
        P = {int(e): g.sort_values("hit_id").predicted_track_id.to_numpy() for e, g in p.groupby("event_id")}
    out = []
    for ev in evs:
        e = ev["event_id"]; pid = ev["pid"]; hq = ev["hit_quirk"]; t = T[e]
        tracks = []
        if a.mode == "oracle":
            qs = [q for q in np.unique(pid[hq]) if (pid == q).sum() >= 3]
            if len(qs) == 2:
                tracks.append(np.flatnonzero(np.isin(pid, qs)))
        else:
            lab = P[e]
            for tid in np.unique(lab[lab >= 0]):
                idx = np.flatnonzero(lab == tid)
                if len(idx) < a.min_hits:
                    continue
                if hq[idx].any() or rng.random() < a.other_frac:
                    tracks.append(idx)
        for idx in tracks:
            if a.mode == "oracle":
                qs = sorted(np.unique(pid[idx]))
                lb = np.where(pid[idx] == qs[0], 0, 1)
            else:
                lb = labels(pid, idx)
            tt = t[idx] / 299.792458  # R1 m200 time is in mm/c -> ns
            u, c = np.unique(pid, return_counts=True)
            ntot = dict(zip(u.tolist(), c.tolist()))
            out.append(dict(event_id=e, hits=idx.astype(np.int32), ntot=np.array([ntot[p] for p in pid[idx]], np.int32), x=feats(ev, idx), time=(tt - tt.min()).astype(np.float32),
                            y=lb.astype(np.int8), quirk=hq[idx], pid=pid[idx]))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    pickle.dump(out, open(a.out, "wb"), protocol=5)
    print(a.mode, a.events, len(out), "tracks", "mean hits", np.mean([len(o["hits"]) for o in out]))


if __name__ == "__main__":
    main()
