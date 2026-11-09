"""Vectorised make_pair_truth.py (same rule and output; checked identical on m100_L100_mu60 events 0-1999), for mu 60.

Per event: the first quirk row (|pdg| = 10000100, in particle order) is kept; the other quirks' hits get its
particle_id, and the other quirk rows are removed from every truth/particles list column.

    python tracking/make_pair_truth_fast.py --dataset R1_ROOT --events A B --out OUT_ROOT [--name NAME]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r1_data import QUIRK_PDG, read_events  # noqa: E402


def offsets(col):
    n = np.asarray(pc.list_value_length(col).fill_null(0))
    o = np.zeros(len(n) + 1, dtype=np.int64)
    o[1:] = np.cumsum(n)
    return n, o


def relabel_block(root: Path, first: int, stop: int):
    parts = read_events(root, "particles", first, stop)
    hits = read_events(root, "tracker_hits", first, stop)
    assert parts["event_id"].equals(hits["event_id"])
    npart, po = offsets(parts["pdg_id"])
    pdg = np.asarray(pc.list_flatten(parts["pdg_id"]))
    pid = np.asarray(pc.list_flatten(parts["particle_id"]))
    ev_p = np.repeat(np.arange(len(npart)), npart)
    q = np.abs(pdg) == QUIRK_PDG
    qrows = np.nonzero(q)[0]
    qev = ev_p[qrows]
    first_in_ev = np.ones(len(qrows), dtype=bool)
    first_in_ev[1:] = qev[1:] != qev[:-1]
    keep_row = np.ones(len(pdg), dtype=bool)
    keep_row[qrows[~first_in_ev]] = False
    keep_id = np.full(len(npart), -1, dtype=np.int64)
    keep_id[qev[first_in_ev]] = pid[qrows[first_in_ev]].astype(np.int64)
    merged = np.zeros(len(npart), dtype=bool)
    merged[qev[~first_in_ev]] = True
    # hits: relabel the dropped quirks' ids (per event) to the kept quirk id
    nh, ho = offsets(hits["particle_id"])
    hp = np.asarray(pc.list_flatten(hits["particle_id"])).astype(np.uint64).copy()
    ev_h = np.repeat(np.arange(len(nh)), nh)
    drop_key = set(zip(qev[~first_in_ev].tolist(), pid[qrows[~first_in_ev]].tolist()))
    relabelled = 0
    if drop_key:
        cand = np.isin(hp, np.array(sorted({k[1] for k in drop_key}), dtype=np.uint64)) & merged[ev_h]
        for i in np.nonzero(cand)[0]:
            if (int(ev_h[i]), int(hp[i])) in drop_key:
                hp[i] = keep_id[ev_h[i]]
                relabelled += 1
    new_hits = hits.set_column(hits.column_names.index("particle_id"), "particle_id",
                               pa.ListArray.from_arrays(pa.array(ho.astype(np.int32)), pa.array(hp, type=pa.uint64())))
    # particles: drop rows, rebuild offsets
    kept_per_ev = np.bincount(ev_p[keep_row], minlength=len(npart))
    no = np.zeros(len(npart) + 1, dtype=np.int32)
    no[1:] = np.cumsum(kept_per_ev)
    mask = pa.array(keep_row)
    cols = {"event_id": parts["event_id"]}
    for c in parts.column_names:
        if c == "event_id":
            continue
        flat = pc.list_flatten(parts[c]).combine_chunks().filter(mask)
        cols[c] = pa.ListArray.from_arrays(pa.array(no), flat).cast(parts.schema.field(c).type)
    new_parts = pa.table(cols, schema=parts.schema)
    return new_parts, new_hits, {"events": len(npart), "events_merged": int(merged.sum()), "hits_relabelled": relabelled}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("--events", required=True, type=int, nargs=2)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--name", default="quirks.m200_L774_mu0.v1.pairtruth")
    args = ap.parse_args()
    first, stop = args.events
    total = {"events": 0, "events_merged": 0, "hits_relabelled": 0}
    for lo in range(first - first % 1000, stop, 1000):
        a, b = max(lo, first), min(lo + 1000, stop)
        parts, hits, stats = relabel_block(args.dataset, a, b)
        for obj, table in (("particles", parts), ("tracker_hits", hits)):
            d = args.out / ("truth" if obj == "particles" else "reco") / obj
            d.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, d / f"{args.name}.{obj}.events{a}-{b - 1}.parquet")
        for k in total:
            total[k] += stats[k]
        print(f"events [{a}, {b}): {stats}", flush=True)
    (args.out / f"pair_truth_manifest_events{first}-{stop - 1}.json").write_text(json.dumps(
        {"source": str(args.dataset.resolve()), "events": [first, stop], **total,
         "rule": "first quirk row kept; other quirks' hits relabelled to it; other quirk rows removed"}, indent=1))
    print(total)


if __name__ == "__main__":
    main()
