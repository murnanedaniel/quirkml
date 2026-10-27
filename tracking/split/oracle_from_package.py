"""Oracle pair tracks (both quirks' hits, A/B labels) straight from the packaged m200_L774_mu0 v1 parquet, for the
data-scaling study of the A1 splitter. Events >= 20000 only (val 16000-16999 and test 17000-19999 untouched).
Same feature layout as split_data.py; the cluster-direction columns (28-37) are zero (direction features exist only
for events 0-19999), so the scaling arms are trained and validated without them.
  python oracle_from_package.py --files F1 F2 ... --out-dir DIR
"""
import argparse, os, pickle, sys
import numpy as np
import pyarrow.parquet as pq
sys.path.insert(0, "/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/gnn")
from common import VOLUMES  # noqa: E402

QPDG = 10000100
VOL_LUT = {v: i for i, v in enumerate(VOLUMES)}


def feats(x, y, z, vol, lay, cl):
    r = np.hypot(x, y); phi = np.arctan2(y, x); eta = np.arcsinh(z / np.maximum(r, 1e-3))
    c = np.stack([x, y, z], 1); rel = (c - c.mean(0)) / 100.0
    key = vol * 100 + lay
    off = np.zeros_like(c); nsame = np.zeros(len(x), np.float32)
    for k in np.unique(key):
        m = key == k; off[m] = c[m] - c[m].mean(0); nsame[m] = m.sum()
    off = np.clip(off / 10.0, -10, 10)
    oh = np.eye(len(VOLUMES), dtype=np.float32)[[VOL_LUT.get(v, len(VOLUMES) - 1) for v in vol]]
    return np.concatenate([np.stack([x / 1000, y / 1000, z / 3000, r / 1000, np.cos(phi), np.sin(phi), eta / 3], 1), rel, off,
                           np.log1p(nsame)[:, None], oh, (lay.astype(np.float32) / 20)[:, None], np.log1p(cl),
                           np.zeros((len(x), 10), np.float32)], 1).astype(np.float32)


def pa_list_m1(arr):
    import pyarrow as pa
    return pa.array([[2**63 - 1]] * len(arr), type=arr.type)


def one(path, out_dir):
    out_f = os.path.join(out_dir, os.path.basename(path).replace(".parquet", ".pkl"))
    if os.path.exists(out_f):
        return
    import pyarrow.compute as pc
    tb = pq.read_table(path, columns=["event_id", "x", "y", "z", "time", "volume_id", "layer_id", "size_loc0", "size_loc1",
                                      "n_channels", "sum_activation", "particle_ids"])
    off = tb["x"].combine_chunks().offsets.to_numpy()
    flat = {k: tb[k].combine_chunks().values.to_numpy(zero_copy_only=False) for k in
            ["x", "y", "z", "time", "volume_id", "layer_id", "size_loc0", "size_loc1", "n_channels", "sum_activation"]}
    hits_pids = tb["particle_ids"].combine_chunks().values  # list<uint64> per hit
    first = pc.list_element(pc.if_else(pc.greater(pc.list_value_length(hits_pids), 0), hits_pids, pa_list_m1(hits_pids)), 0)
    flat["pid"] = first.to_numpy(zero_copy_only=False).astype(np.int64)
    H = {"event_id": tb["event_id"].to_pylist()}
    P = pq.read_table(path.replace("/reco/tracker_hits/", "/truth/particles/").replace(".reco.tracker_hits.", ".truth.particles."),
                      columns=["event_id", "particle_id", "pdg_id"]).to_pydict()
    prow = {e: i for i, e in enumerate(P["event_id"])}
    out = []
    for i, e in enumerate(H["event_id"]):
        if e < 20000:
            continue
        j = prow[e]
        qs = set(np.asarray(P["particle_id"][j])[np.abs(np.asarray(P["pdg_id"][j])) == QPDG].tolist())
        sl = slice(off[i], off[i + 1])
        pid = flat["pid"][sl]
        qq = [q for q in qs if (pid == q).sum() >= 3]
        if len(qq) != 2:
            continue
        idx = np.flatnonzero(np.isin(pid, qq))
        g = lambda k: flat[k][sl][idx]  # noqa: E731
        x, y, z = g("x").astype(np.float32), g("y").astype(np.float32), g("z").astype(np.float32)
        cl = np.stack([g("size_loc0"), g("size_loc1"), g("n_channels"), g("sum_activation")], 1).astype(np.float32)
        t = g("time").astype(np.float64)
        lb = np.where(pid[idx] == min(qq), 0, 1)
        out.append(dict(event_id=int(e), hits=idx.astype(np.int32), x=feats(x, y, z, g("volume_id").astype(int), g("layer_id").astype(int), cl),
                        time=(t - t.min()).astype(np.float32), y=lb.astype(np.int8), quirk=np.ones(len(idx), bool), pid=pid[idx],
                        ntot=np.array([(pid == p).sum() for p in pid[idx]], np.int32)))
    pickle.dump(out, open(out_f + ".tmp", "wb"), protocol=5); os.rename(out_f + ".tmp", out_f)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--files", nargs="+"); ap.add_argument("--out-dir", required=True); a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    for f in a.files:
        one(f, a.out_dir)
