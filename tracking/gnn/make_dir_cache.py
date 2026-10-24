"""Add the Stage-A cluster direction features to the m200_L774_mu0 cache (new cache directory, old one untouched).

Per hit (production / parquet_r1 hit order = cache order): axa, axb (N, 3) candidate cluster axes (unit, global frame;
NaN on long strips), dphi = phi_angle_abs, deta = eta_angle_abs (NaN on long strips), diag = diag_sign (int8).
Source: quirk_tracking_data/direction/m200_L774_mu0/*.parquet (build_direction_features.py).

  python make_dir_cache.py --cache C --direction D --out C2
"""
import argparse
import glob
import os
import pickle
import sys
from multiprocessing import Pool

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import cache_files  # noqa: E402

A = None


def one(path):
    with open(path, "rb") as fh:
        evs = pickle.load(fh)
    a, b = os.path.basename(path)[6:-4].split("-")
    t = pq.read_table(os.path.join(A.direction, f"quirks.m200_L774_mu0.v1.direction.events{a}-{b}.parquet")).sort_by("event_id")
    eids = t["event_id"].to_numpy()
    n = np.asarray(pc.list_value_length(t["axis_a_x"]))
    off = np.concatenate([[0], np.cumsum(n)])
    flat = {c: np.asarray(pc.list_flatten(t[c]).to_numpy(zero_copy_only=False)) for c in t.column_names if c != "event_id"}
    row = {int(e): i for i, e in enumerate(eids)}
    for ev in evs:
        i = row[ev["event_id"]]
        sl = slice(off[i], off[i + 1])
        assert off[i + 1] - off[i] == len(ev["x"]), ev["event_id"]
        ev["axa"] = np.stack([flat[f"axis_a_{c}"][sl] for c in "xyz"], 1).astype(np.float32)
        ev["axb"] = np.stack([flat[f"axis_b_{c}"][sl] for c in "xyz"], 1).astype(np.float32)
        ev["dphi"] = flat["phi_angle_abs"][sl].astype(np.float32)
        ev["deta"] = flat["eta_angle_abs"][sl].astype(np.float32)
        ev["diag"] = flat["diag_sign"][sl].astype(np.int8)
    out = os.path.join(A.out, os.path.basename(path))
    with open(out, "wb") as fh:
        pickle.dump(evs, fh, protocol=5)
    return out, len(evs), float(np.mean([np.isfinite(e["axa"][:, 0]).mean() for e in evs]))


def main():
    global A
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--direction", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=10)
    A = ap.parse_args()
    os.makedirs(A.out, exist_ok=True)
    with Pool(A.workers) as p:
        for r in p.imap_unordered(one, cache_files(A.cache)):
            print(*r, flush=True)


if __name__ == "__main__":
    main()
