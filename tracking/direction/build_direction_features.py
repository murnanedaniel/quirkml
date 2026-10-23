"""Cluster-shape direction features per hit for quirks/m200_L774_mu0/v1, from the cell lists of the re-digitised
runs (redigi/<run>/measurements.root), keyed to the production hit order (event_id + position in the event's lists,
as in parquet/reco/tracker_hits and parquet_r1).

Only the cluster shape is used: no origin line, no entry/exit orientation, no truth.
Local frame of the module: u = loc0, v = loc1, w = normal; cell-centre spans D_u, D_v (mm), thickness t (mm).
A single cluster fixes the line up to a mirror image in the module plane: (D_u, s D_v, t) and (-D_u, -s D_v, t),
where s = sign of the charge-weighted u-v covariance of the cells (which diagonal the cluster lies on).

Columns (one row per event, list per hit, float32 unless noted):
  axis_a_x/y/z, axis_b_x/y/z  the two candidate unit axes in the global frame, R (D_u, s D_v, t)/|.| and
                              R (-D_u, -s D_v, t)/|.|, both with positive component along the module normal.
                              Identical when D_u = D_v = 0. NaN on long strips (volumes 28-30: one bin in loc1).
  phi_angle_abs               atan(D_u / t): incidence angle in the u-w plane, magnitude only
  eta_angle_abs               atan(D_v / t), NaN on long strips
  diag_sign (int8)            s: +1 / -1; 0 when undetermined (D_u = 0 or D_v = 0, zero covariance, long strips)
Usage: python build_direction_features.py <block> [<block> ...]     block b = events 1000b .. 1000b+999 (runs 2b, 2b+1)
"""
import sys
from pathlib import Path

import awkward as ak
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import uproot

D = Path("/pscratch/sd/d/danieltm/quirk_stageA/direction")
PKG = Path("/global/cfs/cdirs/m4958/data/ColliderML/simulation/quirks/m200_L774_mu0/v1/parquet/reco/tracker_hits")
OUT = Path("/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirk_tracking_data/direction/m200_L774_mu0")
RUN_SIZE = 500
PITCH = {16: (0.05, 0.05), 17: (0.05, 0.05), 18: (0.05, 0.05), 23: (0.08, 0.5), 24: (0.08, 0.5),
         25: (0.08, 0.5), 28: (0.125, np.nan), 29: (0.1, np.nan), 30: (0.125, np.nan)}
THICK = {16: 0.125, 17: 0.125, 18: 0.125, 23: 0.2, 24: 0.2, 25: 0.2, 28: 0.25, 29: 0.25, 30: 0.25}
FRAMES = np.load(D / "frames.npy")  # volume, layer, sensitive, centre, e_u, e_v, normal (dump_frames.py)
KEY = {(int(r[0]), int(r[1]), int(r[2])): i for i, r in enumerate(FRAMES)}
F32 = ["axis_a_x", "axis_a_y", "axis_a_z", "axis_b_x", "axis_b_y", "axis_b_z", "phi_angle_abs", "eta_angle_abs"]


def read_run(run):
    """measurements.root of one run, rows grouped per global event id (ascending) in measurement order."""
    R = uproot.open(D / "redigi" / str(run) / "measurements.root")["measurements"].arrays(
        ["event_nr", "volume_id", "layer_id", "surface_id", "rec_loc0", "channel_loc0", "channel_loc1",
         "channel_value"], library="ak")
    ev = ak.to_numpy(R["event_nr"]).astype(np.int64)
    order = np.argsort(ev, kind="stable")
    return R[order], ev[order] + run * RUN_SIZE


def features(R):
    vol = ak.to_numpy(R["volume_id"]).astype(int)
    p0 = np.array([PITCH[v][0] for v in vol]); p1 = np.array([PITCH[v][1] for v in vol])
    has1 = np.isfinite(p1)
    t = np.array([THICK[v] for v in vol])
    b0, b1, a = R["channel_loc0"], R["channel_loc1"], R["channel_value"]
    k0 = ak.to_numpy(ak.max(b0, axis=1) - ak.min(b0, axis=1)) + 1
    k1 = ak.to_numpy(ak.max(b1, axis=1) - ak.min(b1, axis=1)) + 1
    du, dv = (k0 - 1) * p0, np.where(has1, (k1 - 1) * np.nan_to_num(p1), np.nan)
    w = ak.sum(a, axis=1)
    m0, m1 = ak.sum(a * b0, axis=1) / w, ak.sum(a * b1, axis=1) / w
    cov = ak.to_numpy(ak.sum(a * (b0 - m0) * (b1 - m1), axis=1))
    s = np.where(has1 & (du > 0) & (dv > 0), np.sign(cov), 0).astype(np.int8)
    idx = np.array([KEY[k] for k in zip(vol, ak.to_numpy(R["layer_id"]).astype(int),
                                        ak.to_numpy(R["surface_id"]).astype(int))])
    eu, ev_, en = FRAMES[idx, 6:9], FRAMES[idx, 9:12], FRAMES[idx, 12:15]
    sv = np.where(s == 0, 1, s) * dv  # undetermined diagonal: s = +1 by convention (diag_sign = 0 flags it)
    out = {}
    for name, sign in (("a", 1.0), ("b", -1.0)):
        loc = np.stack([sign * du, sign * sv, t], 1)
        loc /= np.linalg.norm(loc, axis=1, keepdims=True)
        g = loc[:, :1] * eu + loc[:, 1:2] * ev_ + loc[:, 2:3] * en
        for j, c in enumerate("xyz"):
            out[f"axis_{name}_{c}"] = g[:, j]
    out["phi_angle_abs"] = np.arctan2(du, t)
    out["eta_angle_abs"] = np.arctan2(dv, t)
    out["diag_sign"] = s
    return out, k0, k1


def build_block(b):
    runs = (2 * b, 2 * b + 1)
    parts = [read_run(r) for r in runs]
    R = ak.concatenate([p[0] for p in parts]); gev = np.concatenate([p[1] for p in parts])
    feats, k0, k1 = features(R)
    pkg = pq.read_table(PKG / f"quirks.m200_L774_mu0.v1.reco.tracker_hits.events{1000 * b}-{1000 * b + 999}.parquet",
                        columns=["event_id", "loc0", "volume_id", "size_loc0", "size_loc1"]).sort_by("event_id")
    ids = pkg["event_id"].to_numpy().astype(np.int64)
    assert ids.tolist() == list(range(1000 * b, 1000 * b + 1000)), "packaged block not complete"
    n = np.asarray(pc.list_value_length(pkg["loc0"]))
    # ROOT rows are grouped by global event in the same ascending order -> positions match after the sort
    assert np.array_equal(np.repeat(ids, n), gev), f"block {b}: event/hit counts differ from production"
    flat = lambda c: np.asarray(pc.list_flatten(pkg[c]).to_numpy(zero_copy_only=False))
    same = (np.isclose(ak.to_numpy(R["rec_loc0"]), flat("loc0"), atol=1e-4)
            & (ak.to_numpy(R["volume_id"]) == flat("volume_id"))
            & (k0 == flat("size_loc0")) & (np.where(np.isin(flat("volume_id"), [28, 29, 30]), 1, k1) == flat("size_loc1")))
    offs = pa.array(np.concatenate([[0], np.cumsum(n)]).astype(np.int32))
    cols = {"event_id": pa.array(ids.astype(np.uint32))}
    for c in F32:
        cols[c] = pa.ListArray.from_arrays(offs, pa.array(feats[c].astype(np.float32)))
    cols["diag_sign"] = pa.ListArray.from_arrays(offs, pa.array(feats["diag_sign"]))
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / f"quirks.m200_L774_mu0.v1.direction.events{1000 * b}-{1000 * b + 999}.parquet"
    tmp = target.with_suffix(".tmp")
    pq.write_table(pa.table(cols), tmp, compression="snappy")
    tmp.rename(target)
    return b, len(gev), int(same.sum())


if __name__ == "__main__":
    for b in map(int, sys.argv[1:]):
        b, nh, ns = build_block(b)
        print(f"block {b}: {nh} hits, identical to production (loc0, volume, sizes): {ns} ({ns / nh:.6f})", flush=True)
