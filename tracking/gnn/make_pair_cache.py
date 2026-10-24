"""Coupled-quirk (pair) truth cache: the two quirks of an event are one particle (ROADMAP: compare with MaskFormer's
pair target). From an existing cache (e.g. the direction cache), per event: hits of every quirk get the first quirk's
particle_id (pid and hit-level truth helpers follow), and the quirks' truth chains are replaced by ONE chain over the
union of their hits ordered by true sim-hit time (same-surface groups as in make_cache.py), so consecutive hits of the
merged sequence are linked whichever quirk they belong to. Everything else is unchanged.

  python make_pair_cache.py --cache C --r1 R1_ROOT --out C2
"""
import argparse
import glob
import os
import pickle
import sys
from multiprocessing import Pool

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import cache_files  # noqa: E402
from make_cache import chain_for_particle  # noqa: E402

A = None


def one(path):
    with open(path, "rb") as fh:
        evs = pickle.load(fh)
    a, b = os.path.basename(path)[6:-4].split("-")
    f = glob.glob(os.path.join(A.r1, "reco/tracker_hits", f"*.events{a}-{b}.parquet"))
    assert len(f) == 1, f
    H = pq.read_table(f[0], columns=["event_id", "time", "volume_id", "layer_id", "surface_id"]).to_pydict()
    row = {e: i for i, e in enumerate(H["event_id"])}
    nmerged = 0
    for ev in evs:
        i = row[ev["event_id"]]
        t = np.asarray(H["time"][i], np.float64)
        surf = (np.asarray(H["volume_id"][i], np.int64) * 10**8 + np.asarray(H["layer_id"][i], np.int64) * 10**5
                + np.asarray(H["surface_id"][i], np.int64))
        assert len(t) == len(ev["x"])
        hq = ev["hit_quirk"]
        qp = np.unique(ev["pid"][hq])
        if len(qp) < 2:
            continue
        nmerged += 1
        keep = qp[0]
        idx = np.flatnonzero(hq)
        ev["pid"] = ev["pid"].copy()
        ev["pid"][idx] = keep
        ev["hit_nhits"] = ev["hit_nhits"].copy()
        ev["hit_nhits"][idx] = len(idx)
        ev["hit_target"] = ev["hit_target"].copy()
        ev["hit_target"][idx] = len(idx) >= 3
        ch = ev["chain"]
        other = ch[:, ~hq[ch[0]]]
        pos = np.stack([ev["x"], ev["y"], ev["z"]], 1)
        pc = chain_for_particle(idx, t, surf, pos).astype(np.int32) if len(idx) >= 3 else np.zeros((2, 0), np.int32)
        ev["chain"] = np.concatenate([other, pc], 1)
    out = os.path.join(A.out, os.path.basename(path))
    with open(out, "wb") as fh:
        pickle.dump(evs, fh, protocol=5)
    return out, len(evs), nmerged


def main():
    global A
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--r1", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=10)
    A = ap.parse_args()
    os.makedirs(A.out, exist_ok=True)
    with Pool(A.workers) as p:
        for r in p.imap_unordered(one, cache_files(A.cache)):
            print(*r, flush=True)


if __name__ == "__main__":
    main()
