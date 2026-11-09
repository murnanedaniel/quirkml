"""Per-event caches for the mu 60 sets, one pickle per 1000-event block, in the formats of gnn/make_cache.py +
gnn/make_dir_cache.py (direction cache) and gnn/make_pair_cache.py (pair cache), in one pass and with vectorised
parquet reading (to_pydict of ~67M hits per block is too slow and too large).

Same definitions as the mu 0 caches: truth chain of every target particle (charge != 0, >= 3 hits) in true sim-hit
time order with same-surface groups (< 0.1 ns) ordered in space (make_cache.chain_for_particle, used as is for every
particle that has such a group; particles without one get the identical plain time-ordered links, vectorised);
direction columns from quirk_tracking_data/direction/<set>; pair cache: both quirks one particle, one chain over the
union of their hits (make_pair_cache.py logic).

  python make_cache60.py --r1 R1_ROOT --direction DIR --out-dir C_DIR --out-pair C_PAIR --blocks 0 1 2 [--workers N]
"""
import argparse
import glob
import os
import pickle
import sys
import time
from multiprocessing import Pool

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "gnn"))
from common import QUIRK_PDG  # noqa: E402
from make_cache import VOL_LUT, chain_for_particle  # noqa: E402

A = None


def flat_table(path, cols):
    t = pq.read_table(path, columns=["event_id"] + cols).sort_by("event_id")
    eids = t["event_id"].to_numpy().astype(np.int64)
    n = np.asarray(pc.list_value_length(t[cols[0]]))
    off = np.concatenate([[0], np.cumsum(n)])
    flat = {c: np.asarray(pc.list_flatten(t[c]).to_numpy(zero_copy_only=False)) for c in cols}
    return eids, off, flat


def chains(pid, t, surf, pos, target):
    """Truth chain (2, E) for all target particles; identical to make_cache's per-particle loop."""
    idx = np.flatnonzero(target)
    if len(idx) == 0:
        return np.zeros((2, 0), np.int32)
    o = idx[np.lexsort((t[idx], pid[idx]))]  # by particle, then time (stable as argsort on time per particle)
    p = pid[o]
    same_p = p[1:] == p[:-1]
    # group start condition of make_cache: same surface as the group's FIRST hit and dt < 0.1 from it; the cheap
    # pairwise test (same surface and dt < 0.1 to the previous hit) finds every particle that can have a group.
    cand = same_p & (surf[o][1:] == surf[o][:-1]) & (t[o][1:] - t[o][:-1] < 0.1)
    bad = np.unique(p[1:][cand])
    simple = same_p & ~np.isin(p[1:], bad)
    out = [np.stack([o[:-1][simple], o[1:][simple]])]
    for q in bad:
        g = idx[pid[idx] == q]
        out.append(chain_for_particle(g, t, surf, pos))
    return np.concatenate(out, 1).astype(np.int32)


def one(b):
    t0 = time.time()
    tag = f"events{1000 * b}-{1000 * b + 999}"
    fd, fp = os.path.join(A.out_dir, tag + ".pkl"), os.path.join(A.out_pair, tag + ".pkl")
    if os.path.exists(fd) and os.path.exists(fp):
        return b, "exists", 0.0
    hf = glob.glob(os.path.join(A.r1, "reco/tracker_hits", f"*.{tag}.parquet"))
    pf = glob.glob(os.path.join(A.r1, "truth/particles", f"*.{tag}.parquet"))
    df = glob.glob(os.path.join(A.direction, f"*.direction.{tag}.parquet"))
    assert len(hf) == 1 and len(pf) == 1 and len(df) == 1, (b, hf, pf, df)
    hcols = ["x", "y", "z", "time", "particle_id", "volume_id", "layer_id", "surface_id",
             "size_loc0", "size_loc1", "n_channels", "sum_activation"]
    he, ho, H = flat_table(hf[0], hcols)
    pe, po, P = flat_table(pf[0], ["particle_id", "pdg_id", "charge", "px", "py"])
    de, do, Dd = flat_table(df[0], ["axis_a_x", "axis_a_y", "axis_a_z", "axis_b_x", "axis_b_y", "axis_b_z",
                                    "phi_angle_abs", "eta_angle_abs", "diag_sign"])
    assert np.array_equal(he, pe) and np.array_equal(he, de) and np.array_equal(ho, do), b
    evs, pevs = [], []
    for k, eid in enumerate(he):
        hs, ps = slice(ho[k], ho[k + 1]), slice(po[k], po[k + 1])
        ppid = P["particle_id"][ps].astype(np.int64)
        pdg = P["pdg_id"][ps]
        charge = P["charge"][ps].astype(np.float32)
        ppt = np.hypot(P["px"][ps].astype(np.float32), P["py"][ps].astype(np.float32))
        row = np.full(int(ppid.max()) + 1, -1, np.int64)
        row[ppid] = np.arange(len(ppid))
        x, y, z = (H[c][hs].astype(np.float32) for c in "xyz")
        t = H["time"][hs].astype(np.float32)
        pid = H["particle_id"][hs].astype(np.int64)
        vol = H["volume_id"][hs].astype(np.int64)
        lay = H["layer_id"][hs].astype(np.int64)
        sid = H["surface_id"][hs].astype(np.int64)
        surf = vol * 10 ** 8 + lay * 10 ** 5 + sid
        n = len(x)
        has = pid >= 0
        prow = np.full(n, -1, np.int64)
        inr = has & (pid < len(row))
        prow[inr] = row[pid[inr]]
        ok = prow >= 0
        hit_quirk = np.zeros(n, bool); hit_quirk[ok] = np.abs(pdg[prow[ok]]) == QUIRK_PDG
        hit_pt = np.zeros(n, np.float32); hit_pt[ok] = ppt[prow[ok]]
        hit_charge = np.zeros(n, np.float32); hit_charge[ok] = charge[prow[ok]]
        u, inv, cnt = np.unique(pid, return_inverse=True, return_counts=True)
        hit_nhits = cnt[inv].astype(np.int32); hit_nhits[~has] = 0
        hit_target = ok & (hit_charge != 0) & (hit_nhits >= 3)
        pos = np.stack([x, y, z], 1)
        chain = chains(pid, t.astype(np.float64), surf, pos, hit_target)
        cl = np.stack([H[c][hs].astype(np.float32) for c in ["size_loc0", "size_loc1", "n_channels", "sum_activation"]], 1)
        ds = slice(do[k], do[k + 1])
        ev = dict(event_id=int(eid), x=x, y=y, z=z, vol_idx=VOL_LUT[vol].astype(np.int8), layer=lay.astype(np.int16),
                  cluster=cl, pid=pid, hit_quirk=hit_quirk, hit_pt=hit_pt, hit_nhits=hit_nhits, hit_target=hit_target,
                  chain=chain,
                  axa=np.stack([Dd[f"axis_a_{c}"][ds] for c in "xyz"], 1).astype(np.float32),
                  axb=np.stack([Dd[f"axis_b_{c}"][ds] for c in "xyz"], 1).astype(np.float32),
                  dphi=Dd["phi_angle_abs"][ds].astype(np.float32), deta=Dd["eta_angle_abs"][ds].astype(np.float32),
                  diag=Dd["diag_sign"][ds].astype(np.int8))
        evs.append(ev)
        # pair version (make_pair_cache.py)
        pe_ = dict(ev)
        qp = np.unique(pid[hit_quirk])
        if len(qp) >= 2:
            qi = np.flatnonzero(hit_quirk)
            pe_["pid"] = pid.copy(); pe_["pid"][qi] = qp[0]
            pe_["hit_nhits"] = hit_nhits.copy(); pe_["hit_nhits"][qi] = len(qi)
            pe_["hit_target"] = hit_target.copy(); pe_["hit_target"][qi] = len(qi) >= 3
            other = chain[:, ~hit_quirk[chain[0]]]
            pcn = chain_for_particle(qi, t.astype(np.float64), surf, pos).astype(np.int32) if len(qi) >= 3 else np.zeros((2, 0), np.int32)
            pe_["chain"] = np.concatenate([other, pcn], 1)
        pevs.append(pe_)
    for path, data in ((fd, evs), (fp, pevs)):
        with open(path + ".tmp", "wb") as fh:
            pickle.dump(data, fh, protocol=5)
        os.replace(path + ".tmp", path)
    return b, len(evs), time.time() - t0


def main():
    global A
    ap = argparse.ArgumentParser()
    ap.add_argument("--r1", required=True); ap.add_argument("--direction", required=True)
    ap.add_argument("--out-dir", required=True); ap.add_argument("--out-pair", required=True)
    ap.add_argument("--blocks", type=int, nargs="+", required=True); ap.add_argument("--workers", type=int, default=4)
    A = ap.parse_args()
    os.makedirs(A.out_dir, exist_ok=True); os.makedirs(A.out_pair, exist_ok=True)
    with Pool(A.workers) as p:
        for r in p.imap_unordered(one, A.blocks):
            print(*r, flush=True)


if __name__ == "__main__":
    main()
