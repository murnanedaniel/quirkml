"""Relabelled R1 copy in which each event's quirk pair is ONE truth particle, to score coupled quirk efficiency with
the ACTS evaluator (evaluate.py's 'quirk' selection then selects the pair object).

Per event: the first quirk row (|pdg| = 10000100, in particle order) is kept; every other quirk's hits get the kept
quirk's particle_id, and the other quirk rows are removed from all truth/particles list columns. Everything else is
copied unchanged. The kept row keeps charge != 0, primary and mass, so the quirk selection (charged, primary,
mass > 10 GeV, >= 3 measurements) picks the pair object up with the union of both quirks' hits.

    python tracking/make_pair_truth.py --dataset R1_ROOT --events A B --out OUT_ROOT

Writes OUT_ROOT/{truth/particles,reco/tracker_hits}/<name>.pairtruth.<obj>.events<a>-<b>.parquet per 1000-event
block of [A, B) (B exclusive), rows sorted by event_id.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r1_data import QUIRK_PDG, read_events  # noqa: E402


def relabel_block(root: Path, first: int, stop: int) -> tuple[pa.Table, pa.Table, dict]:
    parts = read_events(root, "particles", first, stop)
    hits = read_events(root, "tracker_hits", first, stop)
    pcols = [c for c in parts.column_names if c != "event_id"]
    pd_parts = {c: parts[c].to_pylist() for c in pcols}
    hit_pid = hits["particle_id"].to_pylist()
    stats = {"events": 0, "events_merged": 0, "hits_relabelled": 0}
    for i in range(parts.num_rows):
        stats["events"] += 1
        pdg = np.asarray(pd_parts["pdg_id"][i])
        qi = np.nonzero(np.abs(pdg) == QUIRK_PDG)[0]
        if len(qi) < 2:
            continue
        stats["events_merged"] += 1
        ids = np.asarray(pd_parts["particle_id"][i])
        keep_id = int(ids[qi[0]])
        drop_ids = {int(ids[j]) for j in qi[1:]}
        h = np.asarray(hit_pid[i], dtype=np.uint64)
        sel = np.isin(h, np.array(sorted(drop_ids), dtype=np.uint64))
        stats["hits_relabelled"] += int(sel.sum())
        h[sel] = keep_id
        hit_pid[i] = h.tolist()
        keep = np.ones(len(ids), dtype=bool)
        keep[qi[1:]] = False
        for c in pcols:
            pd_parts[c][i] = [v for v, k in zip(pd_parts[c][i], keep, strict=True) if k]
    new_parts = pa.table({"event_id": parts["event_id"],
                          **{c: pa.array(pd_parts[c], type=parts.schema.field(c).type) for c in pcols}})
    new_hits = hits.set_column(hits.column_names.index("particle_id"), "particle_id",
                               pa.array(hit_pid, type=hits.schema.field("particle_id").type))
    return new_parts, new_hits, stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("--events", required=True, type=int, nargs=2, metavar=("A", "B"))
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--name", default="quirks.m200_L774_mu0.v1.pairtruth")
    args = ap.parse_args()
    first, stop = args.events
    total = {"events": 0, "events_merged": 0, "hits_relabelled": 0}
    for lo in range(first - first % 1000, stop, 1000):
        a, b = max(lo, first), min(lo + 1000, stop)
        parts, hits, stats = relabel_block(args.dataset, a, b)
        for obj, table in (("particles", parts), ("tracker_hits", hits)):
            group = "truth" if obj == "particles" else "reco"
            d = args.out / group / obj
            d.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, d / f"{args.name}.{obj}.events{a}-{b - 1}.parquet")
        for k in total:
            total[k] += stats[k]
        print(f"events [{a}, {b}): {stats}", flush=True)
    (args.out / "pair_truth_manifest.json").write_text(json.dumps(
        {"source": str(args.dataset.resolve()), "events": [first, stop], **total,
         "rule": "first quirk row kept; other quirks' hits relabelled to it; other quirk rows removed"}, indent=1))
    print(total)


if __name__ == "__main__":
    main()
