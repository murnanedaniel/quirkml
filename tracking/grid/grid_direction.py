"""Cluster-direction features for one grid test set (blocks 0 and 1 = runs 0-3), reusing
tracking/direction/build_direction_features.py (same geometry frames and feature definitions); every hit is checked
against the production tracker_hits (loc0, volume, cluster sizes). Usage: grid_direction.py DS SCR PKG_HITS_DIR OUT"""
import sys
from pathlib import Path

import awkward as ak
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "direction"))
import build_direction_features as bdf  # noqa: E402

ds, scr, pkg_dir, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
bdf.D = scr  # read_run reads scr/redigi/<run>/measurements.root
out.mkdir(parents=True, exist_ok=True)
for b in (0, 1):
    parts = [bdf.read_run(r) for r in (2 * b, 2 * b + 1)]
    R = ak.concatenate([p[0] for p in parts]); gev = np.concatenate([p[1] for p in parts])
    feats, k0, k1 = bdf.features(R)
    pkg = pq.read_table(pkg_dir / f"quirks.{ds}.v1.reco.tracker_hits.events{1000 * b}-{1000 * b + 999}.parquet",
                        columns=["event_id", "loc0", "volume_id", "size_loc0", "size_loc1"]).sort_by("event_id")
    ids = pkg["event_id"].to_numpy().astype(np.int64)
    assert ids.tolist() == list(range(1000 * b, 1000 * b + 1000)), "packaged block not complete"
    n = np.asarray(pc.list_value_length(pkg["loc0"]))
    assert np.array_equal(np.repeat(ids, n), gev), f"block {b}: event/hit counts differ from production"
    flat = lambda c: np.asarray(pc.list_flatten(pkg[c]).to_numpy(zero_copy_only=False))  # noqa: E731
    same = (np.isclose(ak.to_numpy(R["rec_loc0"]), flat("loc0"), atol=1e-4)
            & (ak.to_numpy(R["volume_id"]) == flat("volume_id")) & (k0 == flat("size_loc0"))
            & (np.where(np.isin(flat("volume_id"), [28, 29, 30]), 1, k1) == flat("size_loc1")))
    if same.mean() < 0.9999:
        raise SystemExit(f"{ds} block {b}: only {same.mean():.6f} of hits identical to production")
    offs = pa.array(np.concatenate([[0], np.cumsum(n)]).astype(np.int32))
    cols = {"event_id": pa.array(ids.astype(np.uint32))}
    for c in bdf.F32:
        cols[c] = pa.ListArray.from_arrays(offs, pa.array(feats[c].astype(np.float32)))
    cols["diag_sign"] = pa.ListArray.from_arrays(offs, pa.array(feats["diag_sign"]))
    target = out / f"quirks.{ds}.v1.direction.events{1000 * b}-{1000 * b + 999}.parquet"
    pq.write_table(pa.table(cols), target.with_suffix(".tmp"), compression="snappy")
    target.with_suffix(".tmp").rename(target)
    print(f"{ds} block {b}: {len(gev)} hits, identical to production: {same.mean():.6f}", flush=True)
