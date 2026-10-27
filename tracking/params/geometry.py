"""Planar surface table for ODD v6 sensitive modules, from the packaged data (no ACTS geometry needed).

Every reco measurement lies exactly on its module plane: global = c + loc0 e0 + loc1 e1. For 2D modules
(pixels, short strips) a per-surface linear regression of (x, y, z) on (loc0, loc1) gives c, e0, e1 and the
normal n = e0 x e1. Long strips are 1D (loc1 = 0 for every hit), so their normal comes from an SVD of the
sim-hit positions on that surface, e0 from the regression on loc0 alone, and e1 = n x e0.

Output: npz with keys (int64 surface keys), c (n,3), e0, e1, n, kind (2 = from 2D regression, 1 = 1D strip),
nhit, rms (max regression residual, mm).

    python geometry.py --out /pscratch/sd/d/danieltm/quirk_tracking/params/odd_surfaces.npz
"""
import argparse
import sys
from collections import defaultdict

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, "/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/params")
from data import dataset_dir, _file, surface_key  # noqa: E402


def collect(datasets, n_events):
    R = defaultdict(list)
    S = defaultdict(list)
    for m, lam in datasets:
        ds = dataset_dir(m, lam)
        tr = pq.read_table(_file(ds, "reco/tracker_hits", 0), columns=["loc0", "loc1", "x", "y", "z", "volume_id", "layer_id", "surface_id"]).slice(0, n_events)
        ts = pq.read_table(_file(ds, "truth/tracker_simhits", 0), columns=["true_x", "true_y", "true_z", "volume_id", "layer_id", "surface_id"]).slice(0, n_events)
        c = {k: np.concatenate([np.asarray(v) for v in tr.column(k).to_pylist()]) for k in tr.column_names}
        key = (c["volume_id"].astype(np.int64) << 40) | (c["layer_id"].astype(np.int64) << 24) | c["surface_id"].astype(np.int64)
        arr = np.c_[c["loc0"], c["loc1"], c["x"], c["y"], c["z"]].astype(float)
        o = np.argsort(key, kind="stable"); key = key[o]; arr = arr[o]
        u, i0 = np.unique(key, return_index=True)
        for k, a in zip(u, np.split(arr, i0[1:])):
            R[int(k)].append(a)
        c = {k: np.concatenate([np.asarray(v) for v in ts.column(k).to_pylist()]) for k in ts.column_names}
        key = (c["volume_id"].astype(np.int64) << 40) | (c["layer_id"].astype(np.int64) << 24) | c["surface_id"].astype(np.int64)
        arr = np.c_[c["true_x"], c["true_y"], c["true_z"]].astype(float)
        o = np.argsort(key, kind="stable"); key = key[o]; arr = arr[o]
        u, i0 = np.unique(key, return_index=True)
        for k, a in zip(u, np.split(arr, i0[1:])):
            S[int(k)].append(a[:200])
        print(m, lam, "surfaces so far", len(R), flush=True)
    return R, S


def build(R, S):
    keys, C, E0, E1, N, KIND, NH, RMS = [], [], [], [], [], [], [], []
    for k, lst in R.items():
        a = np.vstack(lst)
        if len(a) > 400:
            a = a[np.linspace(0, len(a) - 1, 400).astype(int)]
        loc, X = a[:, :2], a[:, 2:]
        two_d = np.ptp(loc[:, 1]) > 1e-3
        if two_d:
            if len(a) < 3:
                continue
            A = np.c_[np.ones(len(a)), loc]
            sol, *_ = np.linalg.lstsq(A, X, rcond=None)
            if np.linalg.matrix_rank(A) < 3:
                continue
            c, e0, e1 = sol
            res = np.abs(A @ sol - X).max()
            n = np.cross(e0, e1); n /= np.linalg.norm(n)
            kind = 2
        else:
            if len(a) < 2 or k not in S:
                continue
            sp = np.vstack(S[k])
            if len(sp) < 3:
                continue
            A = np.c_[np.ones(len(a)), loc[:, 0]]
            if np.linalg.matrix_rank(A) < 2:
                continue
            sol, *_ = np.linalg.lstsq(A, X, rcond=None)
            c, e0 = sol
            res = np.abs(A @ sol - X).max()
            P = np.vstack([sp, X])
            _, _, vt = np.linalg.svd(P - P.mean(0))
            n = vt[2]
            e0 = e0 - (e0 @ n) * n
            e1 = np.cross(n, e0)
            kind = 1
        keys.append(k); C.append(c); E0.append(e0); E1.append(e1); N.append(n); KIND.append(kind); NH.append(len(a)); RMS.append(res)
    o = np.argsort(keys)
    f = lambda L: np.asarray(L)[o]
    return dict(keys=f(keys).astype(np.int64), c=f(C), e0=f(E0), e1=f(E1), n=f(N), kind=f(KIND), nhit=f(NH), rms=f(RMS))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--events", type=int, default=300)
    a = ap.parse_args()
    ds = [(200, 774), (100, 167), (300, 3594), (50, 100), (150, 1292), (250, 10000), (100, 774), (200, 167), (300, 167), (100, 3594), (200, 3594), (300, 774)]
    R, S = collect(ds, a.events)
    G = build(R, S)
    print("surfaces", len(G["keys"]), "2D", (G["kind"] == 2).sum(), "1D", (G["kind"] == 1).sum(), "max fit residual (mm)", G["rms"].max())
    np.savez(a.out, **G)
