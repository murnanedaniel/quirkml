"""Vectorised version of convert_to_r1.py (same output, checked bit-identical on m200_L774_mu0 events 0-999), for the
mu 60 packages (~67k hits per event, where the per-event Python loop of convert_to_r1.py is far too slow).

    python tracking/convert_to_r1_fast.py <packaged parquet dir> <output dir> [processes] [--events A B]

Output per 1000-event file: truth/particles copied as is; reco/tracker_hits with the Release 1 columns
(x, y, z, detector, volume_id, layer_id, surface_id as input; particle_id = particle_ids[0] or 0; true_x/y/z and time
of the hit's first sim hit, NaN without one) plus the cluster columns as extras. Hit order unchanged. Rows are matched
to the sim-hit table by event_id. Files are written to <name>.tmp and renamed; existing outputs are skipped.
--events A B: only files whose range lies inside [A, B].
"""
import re
import shutil
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

EXTRA = ["loc0", "loc1", "size_loc0", "size_loc1", "n_channels", "sum_activation", "local_eta", "local_phi",
         "global_eta", "global_phi", "eta_angle", "phi_angle"]
R1_TYPES = {"x": pa.float32(), "y": pa.float32(), "z": pa.float32(), "true_x": pa.float32(),
            "true_y": pa.float32(), "true_z": pa.float32(), "time": pa.float32(), "particle_id": pa.uint64(),
            "detector": pa.uint8(), "volume_id": pa.uint8(), "layer_id": pa.uint16(), "surface_id": pa.uint32()}


def first_of_inner(col: pa.ChunkedArray, fill):
    """For a list<list<x>> column: flat per-hit array of the first inner element (fill where the inner list is empty)."""
    hits = pc.list_flatten(col)  # list<x> per hit
    n = np.asarray(pc.list_value_length(hits).fill_null(0))
    vals = np.asarray(pc.list_flatten(hits))
    starts = np.zeros(len(n), dtype=np.int64)
    starts[1:] = np.cumsum(n)[:-1]
    out = np.full(len(n), fill, dtype=vals.dtype if len(vals) else np.int64)
    ok = n > 0
    out[ok] = vals[starts[ok]]
    return out, ok


def convert_file(hits_path, simhits_path, out_path):
    if out_path.exists():
        return 0
    hits = pq.read_table(hits_path)
    sim = pq.read_table(simhits_path, columns=["event_id", "true_x", "true_y", "true_z", "true_time"])
    order = {e: j for j, e in enumerate(sim["event_id"].to_pylist())}
    ev = hits["event_id"].to_pylist()
    assert set(order) == set(ev), f"different events in {hits_path} and {simhits_path}"
    sim = sim.take(pa.array([order[e] for e in ev]))  # sim rows in hit-table order
    nh = np.asarray(pc.list_value_length(hits["x"]))
    offs = np.zeros(len(nh) + 1, dtype=np.int32)
    offs[1:] = np.cumsum(nh)
    pid, _ = first_of_inner(hits["particle_ids"], 0)
    sid, ok = first_of_inner(hits["simhit_ids"], 0)
    ns = np.asarray(pc.list_value_length(sim["true_x"]))
    soffs = np.zeros(len(ns) + 1, dtype=np.int64)
    soffs[1:] = np.cumsum(ns)
    ev_of_hit = np.repeat(np.arange(len(nh)), nh)
    gidx = soffs[ev_of_hit] + sid.astype(np.int64)
    ok &= sid.astype(np.int64) < ns[ev_of_hit]
    cols = {"event_id": hits["event_id"].combine_chunks().cast(pa.uint32())}
    for c in ["x", "y", "z", "detector", "volume_id", "layer_id", "surface_id"]:
        cols[c] = pa.ListArray.from_arrays(pa.array(offs), pc.list_flatten(hits[c]).combine_chunks().cast(R1_TYPES[c]))
    cols["particle_id"] = pa.ListArray.from_arrays(pa.array(offs), pa.array(pid.astype(np.uint64)))
    for c, src in (("true_x", "true_x"), ("true_y", "true_y"), ("true_z", "true_z"), ("time", "true_time")):
        flat = np.asarray(pc.list_flatten(sim[src]).fill_null(np.nan), dtype=np.float32)
        v = np.full(len(pid), np.nan, dtype=np.float32)
        v[ok] = flat[gidx[ok]]
        cols[c] = pa.ListArray.from_arrays(pa.array(offs), pa.array(v))
    for c in EXTRA:
        cols[c] = hits[c].combine_chunks()
    fields = [pa.field("event_id", pa.uint32())] + [pa.field(c, pa.list_(t)) for c, t in R1_TYPES.items()]
    fields += [pa.field(c, hits.schema.field(c).type) for c in EXTRA]
    table = pa.table({f.name: cols[f.name] for f in fields}, schema=pa.schema(fields))
    tmp = out_path.with_suffix(".tmp")
    pq.write_table(table, tmp, compression="snappy")
    tmp.rename(out_path)
    return hits.num_rows


def main():
    args = [a for a in sys.argv[1:]]
    rng = None
    if "--events" in args:
        i = args.index("--events"); rng = (int(args[i + 1]), int(args[i + 2])); del args[i:i + 3]
    src, dst = Path(args[0]), Path(args[1])
    procs = int(args[2]) if len(args) > 2 else 8
    (dst / "reco" / "tracker_hits").mkdir(parents=True, exist_ok=True)
    (dst / "truth" / "particles").mkdir(parents=True, exist_ok=True)

    def inside(name):
        if rng is None:
            return True
        a, b = map(int, re.search(r"events(\d+)-(\d+)", name).groups())
        return rng[0] <= a and b <= rng[1]

    for f in sorted((src / "truth" / "particles").glob("*.parquet")):
        if inside(f.name) and not (dst / "truth" / "particles" / f.name).exists():
            shutil.copy2(f, dst / "truth" / "particles" / f.name)
    jobs = []
    for f in sorted((src / "reco" / "tracker_hits").glob("*.parquet")):
        if not inside(f.name):
            continue
        s = src / "truth" / "tracker_simhits" / f.name.replace(".reco.tracker_hits.", ".truth.tracker_simhits.")
        jobs.append((f, s, dst / "reco" / "tracker_hits" / f.name))
    with Pool(procs) as pool:
        n = sum(pool.starmap(convert_file, jobs))
    print(f"{len(jobs)} files, {n} events converted -> {dst}")


if __name__ == "__main__":
    main()
