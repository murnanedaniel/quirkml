"""Turn the ACTS tracks stored in a packaged quirk dataset into predictions (tracking_core/schema.py) for evaluate.py.

    python tracking/ckf_predictions.py --packaged PACKAGED_ROOT --object tracks --events A B --out DIR

PACKAGED_ROOT is the package_native_parquet.py layout (<root>/reco/{tracks,truth_tracks,tracker_hits}). Objects:
  tracks        ACTS combinatorial Kalman filter (CKF) track finding of the production chain
  truth_tracks  ACTS fits of truth-seeded track candidates
Each track's hit_ids are positions in its event's tracker_hits list (the same hit_id convention as Release 1 and
predictions.parquet; checked here: the event's hit count bounds every hit_id).

One row per (track, hit): predicted_track_id = the track's index in its event. A hit on several tracks gets one row
per track. Outlier hits (hit_outlier) are kept on their track. Both choices follow the evaluator's "reference"
sample (tracking_core.eval, TracksToTrajectories + TrajectoriesToProtoTracks on the Release 1 tracks table), whose
schema has no outlier flag and whose prototracks keep every hit_id of every track. A hit on no track gets one row
with predicted_track_id -1, so every event of the range has rows.
"""
from __future__ import annotations

import argparse
import datetime
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r1_data import read_events  # noqa: E402
from tracking_core.schema import RunMeta, write_predictions  # noqa: E402

OBJECTS = ("tracks", "truth_tracks")


def ckf_predictions(packaged: Path, obj: str, first: int, stop: int) -> tuple[pl.DataFrame, dict]:
    tracks = read_events(packaged, obj, first, stop, columns=["event_id", "hit_ids", "hit_outlier"])
    hits = read_events(packaged, "tracker_hits", first, stop, columns=["event_id", "x"])
    n_hits = [len(v) for v in hits["x"].to_pylist()]
    events, hit_ids, track_ids = [], [], []
    stats = {"tracks": 0, "track_hits": 0, "outlier_hits": 0, "hits_on_several_tracks": 0, "unassigned_hits": 0}
    for i in range(tracks.num_rows):
        event = int(tracks["event_id"][i].as_py())
        lists = tracks["hit_ids"][i].as_py()
        outliers = tracks["hit_outlier"][i].as_py()
        n = n_hits[i]
        on_track = np.zeros(n, dtype=np.int64)
        for t, h in enumerate(lists):
            h = np.asarray(h, dtype=np.int64)
            if len(h) and (h.min() < 0 or h.max() >= n):
                raise ValueError(f"event {event} track {t}: hit_id outside the event's {n} hits")
            if len(np.unique(h)) != len(h):
                raise ValueError(f"event {event} track {t}: a hit_id appears twice on the track")
            on_track[h] += 1
            events.append(np.full(len(h), event, dtype=np.uint64))
            hit_ids.append(h.astype(np.uint64))
            track_ids.append(np.full(len(h), t, dtype=np.int64))
        stats["tracks"] += len(lists)
        stats["track_hits"] += sum(len(h) for h in lists)
        stats["outlier_hits"] += sum(sum(o) for o in outliers)
        stats["hits_on_several_tracks"] += int((on_track > 1).sum())
        free = np.flatnonzero(on_track == 0)
        stats["unassigned_hits"] += len(free)
        events.append(np.full(len(free), event, dtype=np.uint64))
        hit_ids.append(free.astype(np.uint64))
        track_ids.append(np.full(len(free), -1, dtype=np.int64))
    df = pl.DataFrame({"event_id": np.concatenate(events), "hit_id": np.concatenate(hit_ids),
                       "predicted_track_id": np.concatenate(track_ids)}).sort("event_id", maintain_order=True)
    df = df.with_columns(score=pl.lit(float("nan"), dtype=pl.Float32))
    return df, stats


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--packaged", required=True, type=Path, help="packaged dataset root (reco/, truth/)")
    parser.add_argument("--object", choices=OBJECTS, default="tracks")
    parser.add_argument("--events", required=True, type=int, nargs=2, metavar=("A", "B"),
                        help="event_id range A <= event_id < B")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    df, stats = ckf_predictions(args.packaged, args.object, *args.events)
    meta = RunMeta(run_id=f"acts_{args.object}_{args.events[0]}_{args.events[1]}",
                   implementation=f"acts:{args.object}",
                   config={"packaged": str(args.packaged.resolve()), "object": args.object,
                           "events": list(args.events), **stats},
                   wrote_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
    path = write_predictions(df, args.out, meta)
    print(f"wrote {path}: {stats}")


if __name__ == "__main__":
    main()
