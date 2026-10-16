"""Write truth-based predictions (predictions.parquet + meta.json, tracking_core.schema) for an event range.

    python tracking/oracle_predictions.py --dataset R1_ROOT --events A B --out DIR [--variant oracle]

Events A <= event_id < B. One row per hit: event_id, hit_id (position of the hit in its event), predicted_track_id,
score (NaN).

Variants:
  oracle             predicted_track_id = the hit's particle_id. Every hit in the R1 tables has a truth particle
                     (convert_to_r1.py writes 0 only for a hit without one, and the packaged quirk data have none),
                     and particle_id 0 is a real particle, often a quirk, so no hit is left unassigned.
  unassigned         every hit -1.
The others are evaluator checks; the oracle on all non-quirk hits, and for the quirk hits:
  quirk_halves_time  each quirk's hits split in two tracks by time order (first floor(n/2) hits, the rest).
  quirk_drop60       a random 60% of each quirk's hits unassigned (-1), the other 40% one track.
  quirk_coinflip     each quirk kept whole or all its hits unassigned, with probability 1/2 (seeded).
                     meta.json records the kept quirks per event, so the expected efficiencies are known exactly.
"""
from __future__ import annotations

import argparse
import datetime
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r1_data import QUIRK_PDG, read_events  # noqa: E402
from tracking_core.schema import RunMeta, write_predictions  # noqa: E402

VARIANTS = ("oracle", "unassigned", "quirk_halves_time", "quirk_drop60", "quirk_coinflip")
#: Track ids of the second half of a split quirk: particle_id + this offset (no particle id comes close).
SPLIT_OFFSET = 1 << 40


def make_predictions(dataset: Path, first: int, stop: int, variant: str = "oracle", seed: int = 0):
    """(predictions DataFrame, extra meta config) for events [first, stop)."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; known: {VARIANTS}")
    hits = read_events(dataset, "tracker_hits", first, stop, columns=["event_id", "particle_id", "time"])
    particles = read_events(dataset, "particles", first, stop, columns=["event_id", "particle_id", "pdg_id"])
    rng = np.random.default_rng(seed)
    event_ids, hit_ids, track_ids = [], [], []
    kept = {}
    for i in range(hits.num_rows):
        event = int(hits["event_id"][i].as_py())
        pid = np.asarray(hits["particle_id"][i].values, dtype=np.int64)
        n = len(pid)
        track = pid.copy()
        if variant == "unassigned":
            track[:] = -1
        elif variant != "oracle":
            ppid = np.asarray(particles["particle_id"][i].values, dtype=np.int64)
            pdg = np.asarray(particles["pdg_id"][i].values, dtype=np.int64)
            quirks = ppid[np.abs(pdg) == QUIRK_PDG]
            time = np.asarray(hits["time"][i].values, dtype=np.float64)
            kept_here = []
            for q in quirks:
                idx = np.flatnonzero(pid == q)
                if variant == "quirk_halves_time":
                    order = idx[np.argsort(time[idx], kind="stable")]
                    track[order[len(order) // 2:]] = q + SPLIT_OFFSET
                elif variant == "quirk_drop60":
                    drop = rng.permutation(idx)[: len(idx) - int(np.floor(0.4 * len(idx)))]
                    track[drop] = -1
                elif variant == "quirk_coinflip":
                    if rng.random() < 0.5:
                        track[idx] = -1
                    else:
                        kept_here.append(int(q))
            if variant == "quirk_coinflip":
                kept[event] = kept_here
        event_ids.append(np.full(n, event, dtype=np.uint64))
        hit_ids.append(np.arange(n, dtype=np.uint64))
        track_ids.append(track)
    df = pl.DataFrame({
        "event_id": np.concatenate(event_ids), "hit_id": np.concatenate(hit_ids),
        "predicted_track_id": np.concatenate(track_ids).astype(np.int64),
        "score": np.full(sum(len(t) for t in track_ids), np.nan, dtype=np.float32),
    })
    config = {"dataset": str(Path(dataset).resolve()), "events": [first, stop], "variant": variant, "seed": seed}
    if variant == "quirk_coinflip":
        config["kept_quirks"] = {str(k): v for k, v in kept.items()}
    return df, config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, type=Path, help="R1 dataset root (truth/, reco/)")
    parser.add_argument("--events", required=True, type=int, nargs=2, metavar=("A", "B"),
                        help="event_id range A <= event_id < B")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--variant", choices=VARIANTS, default="oracle")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    df, config = make_predictions(args.dataset, *args.events, variant=args.variant, seed=args.seed)
    meta = RunMeta(run_id=f"{args.variant}_{args.events[0]}_{args.events[1]}", implementation=f"truth:{args.variant}",
                   config=config, wrote_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
    path = write_predictions(df, args.out, meta)
    print(f"wrote {path}: {df.height} hits, {df['event_id'].n_unique()} events, "
          f"{(df['predicted_track_id'] >= 0).sum()} assigned")


if __name__ == "__main__":
    main()
