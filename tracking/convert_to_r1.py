"""Write a packaged quirk dataset in the ColliderML Release 1 layout, which the ACTS evaluator (pyacts 48,
ColliderMLRelease1InputConverter) and the hepattn data module read.

    python tracking/convert_to_r1.py <packaged parquet dir> <output dir> [processes]

Input: <in>/truth/{particles,tracker_simhits}, <in>/reco/tracker_hits (package_native_parquet.py layout).
Output: <out>/truth/particles (copied as is: same schema as Release 1) and <out>/reco/tracker_hits with the
Release 1 columns, per event:
  x, y, z, detector, volume_id, layer_id, surface_id   as in the input
  particle_id                    the hit's first truth particle (particle_ids[0]; 0 if none)
  true_x, true_y, true_z, time   position and time (ns) of the hit's first sim hit (simhit_ids[0])
plus the cluster columns kept as extras (loc0, loc1, size_loc0, size_loc1, n_channels, sum_activation,
local/global eta and phi, eta_angle, phi_angle). The Release 1 reader ignores them.
Hit order is unchanged, so hit_id = row position in the event, as in the packaged dataset.
"""
import shutil
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

EXTRA = ["loc0", "loc1", "size_loc0", "size_loc1", "n_channels", "sum_activation", "local_eta", "local_phi",
         "global_eta", "global_phi", "eta_angle", "phi_angle"]
R1_TYPES = {"x": pa.float32(), "y": pa.float32(), "z": pa.float32(), "true_x": pa.float32(),
            "true_y": pa.float32(), "true_z": pa.float32(), "time": pa.float32(), "particle_id": pa.uint64(),
            "detector": pa.uint8(), "volume_id": pa.uint8(), "layer_id": pa.uint16(), "surface_id": pa.uint32()}


def first(lists, fill):
    """First element of each inner list of a list<list<x>> column for one event, `fill` where empty."""
    return np.array([v[0] if len(v) else fill for v in lists])


def convert_file(hits_path, simhits_path, out_path):
    hits = pq.read_table(hits_path)
    sim = pq.read_table(simhits_path, columns=["event_id", "true_x", "true_y", "true_z", "true_time"])
    # rows are not in event_id order, and the order can differ between tables: match them by event_id
    sim_row = {e: j for j, e in enumerate(sim["event_id"].to_pylist())}
    assert set(sim_row) == set(hits["event_id"].to_pylist()), f"different events in {hits_path} and {simhits_path}"
    cols = {k: [] for k in ["event_id", *R1_TYPES, *EXTRA]}
    for i in range(hits.num_rows):
        h = {c: hits[c][i].as_py() for c in ["x", "y", "z", "detector", "volume_id", "layer_id", "surface_id",
                                             "particle_ids", "simhit_ids", *EXTRA]}
        j = sim_row[hits["event_id"][i].as_py()]
        s = {c: np.asarray(sim[c][j].as_py()) for c in ["true_x", "true_y", "true_z", "true_time"]}
        sid = first(h["simhit_ids"], -1).astype(np.int64)
        ok = sid >= 0
        cols["event_id"].append(hits["event_id"][i].as_py())
        for c in ["x", "y", "z", "detector", "volume_id", "layer_id", "surface_id", *EXTRA]:
            cols[c].append(h[c])
        cols["particle_id"].append(first(h["particle_ids"], 0).astype(np.uint64).tolist())
        for c, src in (("true_x", "true_x"), ("true_y", "true_y"), ("true_z", "true_z"), ("time", "true_time")):
            v = np.where(ok, s[src][np.clip(sid, 0, max(len(s[src]) - 1, 0))] if len(s[src]) else 0.0, np.nan)
            cols[c].append(v.astype(np.float32).tolist())
    fields = [pa.field("event_id", pa.uint32())]
    fields += [pa.field(c, pa.list_(t)) for c, t in R1_TYPES.items()]
    fields += [pa.field(c, hits.schema.field(c).type) for c in EXTRA]
    table = pa.table({f.name: pa.array(cols[f.name], type=f.type) for f in fields}, schema=pa.schema(fields))
    pq.write_table(table, out_path, compression="snappy")
    return hits.num_rows


def main():
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    (dst / "reco" / "tracker_hits").mkdir(parents=True, exist_ok=True)
    (dst / "truth" / "particles").mkdir(parents=True, exist_ok=True)
    for f in sorted((src / "truth" / "particles").glob("*.parquet")):
        shutil.copy2(f, dst / "truth" / "particles" / f.name)
    jobs = []
    for f in sorted((src / "reco" / "tracker_hits").glob("*.parquet")):
        s = src / "truth" / "tracker_simhits" / f.name.replace(".reco.tracker_hits.", ".truth.tracker_simhits.")
        jobs.append((f, s, dst / "reco" / "tracker_hits" / f.name))
    with Pool(int(sys.argv[3]) if len(sys.argv) > 3 else 8) as pool:
        n = sum(pool.starmap(convert_file, jobs))
    print(f"{len(jobs)} files, {n} events -> {dst}")

if __name__ == "__main__":
    main()
