"""Convert ColliderML's stored ACTS tracks into canonical per-hit predictions."""

from __future__ import annotations

import argparse
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from tracking_core.schema import RunMeta, write_predictions

CHUNK_RE = re.compile(r"events(\d+)-(\d+)\.parquet$")


def _indexed_chunks(directory: Path) -> dict[tuple[int, int], Path]:
    result: dict[tuple[int, int], Path] = {}
    for path in directory.glob("*.parquet"):
        match = CHUNK_RE.search(path.name)
        if match is None:
            continue
        token = (int(match.group(1)), int(match.group(2)))
        if token in result:
            raise RuntimeError(f"duplicate event range {token} in {directory}")
        result[token] = path
    return result


def _select_chunks(
    chunks: list[tuple[int, int]],
    n_files: int | None,
    chunk_range: tuple[int, int] | list[int] | None,
) -> list[tuple[int, int]]:
    if chunk_range is not None:
        if len(chunk_range) != 2:
            raise ValueError("chunk_range must contain exactly two indices")
        first, last = chunk_range
        if first > last:
            raise ValueError("chunk_range first index must not exceed last index")
        chunks = [token for token in chunks if first <= token[0] // 1000 <= last]
    elif n_files is None:
        raise ValueError("n_files may be None only when chunk_range is set")
    if n_files is not None:
        if n_files < 1:
            raise ValueError("n_files must be at least 1")
        if len(chunks) < n_files:
            raise RuntimeError(f"requested {n_files} chunks; only {len(chunks)} are paired")
        chunks = chunks[:n_files]
    return chunks


def _event_rows(table: object, name: str) -> dict[int, int]:
    ids = table["event_id"].to_numpy(zero_copy_only=False)
    result = {int(event_id): row for row, event_id in enumerate(ids)}
    if len(result) != len(ids):
        raise RuntimeError(f"duplicate event_id in {name}")
    return result


def convert_acts_colliderml(
    dataset_root: Path | str,
    n_files: int | None,
    out_dir: Path | str,
    chunk_range: tuple[int, int] | list[int] | None = None,
) -> Path:
    """Convert selected paired ACTS/hit chunks and write predictions."""
    import polars as pl

    root = Path(dataset_root)
    tracks = _indexed_chunks(root / "reco/tracks")
    hits = _indexed_chunks(root / "reco/tracker_hits")
    selected = _select_chunks(sorted(set(tracks) & set(hits)), n_files, chunk_range)

    frames: list[pl.DataFrame] = []
    shared_hit_reassignments = 0
    for index, token in enumerate(selected, 1):
        print(f"[{index:02d}/{len(selected):02d}] events{token[0]}-{token[1]}", flush=True)
        track_table = pq.read_table(
            tracks[token], columns=["event_id", "track_id", "hit_ids"]
        )
        hit_table = pq.read_table(hits[token], columns=["event_id", "particle_id"])
        track_rows = _event_rows(track_table, "reco/tracks")
        hit_rows = _event_rows(hit_table, "reco/tracker_hits")
        if set(track_rows) != set(hit_rows):
            raise RuntimeError(f"event_id mismatch in events{token[0]}-{token[1]}")

        track_ids = track_table["track_id"].combine_chunks()
        track_hit_ids = track_table["hit_ids"].combine_chunks()
        hit_particles = hit_table["particle_id"].combine_chunks()
        event_parts: list[np.ndarray] = []
        hit_parts: list[np.ndarray] = []
        prediction_parts: list[np.ndarray] = []
        for event_id in sorted(hit_rows):
            hrow = hit_rows[event_id]
            trow = track_rows[event_id]
            n_hits = len(hit_particles[hrow].as_py())
            assigned = np.full(n_hits, -1, dtype=np.int64)
            ids = track_ids[trow].as_py()
            memberships = track_hit_ids[trow].as_py()
            if len(ids) != len(memberships):
                raise RuntimeError(f"track list length mismatch in event {event_id}")
            for track_id, member_ids in zip(ids, memberships):
                members = np.asarray(member_ids, dtype=np.int64)
                if np.any((members < 0) | (members >= n_hits)):
                    raise RuntimeError(f"out-of-range hit_id in event {event_id}")
                shared_hit_reassignments += int(np.count_nonzero(assigned[members] >= 0))
                # Canonical predictions allow one track per hit, so the last track wins.
                assigned[members] = int(track_id)
            event_parts.append(np.full(n_hits, event_id, dtype=np.uint64))
            hit_parts.append(np.arange(n_hits, dtype=np.uint64))
            prediction_parts.append(assigned)
        frames.append(pl.DataFrame({
            "event_id": np.concatenate(event_parts).astype(np.uint64),
            "hit_id": np.concatenate(hit_parts).astype(np.uint64),
            "predicted_track_id": np.concatenate(prediction_parts).astype(np.int64),
            "score": np.full(sum(map(len, event_parts)), np.nan, dtype=np.float32),
        }))

    predictions = pl.concat(frames, rechunk=False)
    meta = RunMeta(
        run_id=f"acts-colliderml-v1-{len(selected)}-chunks",
        implementation="baselines:acts_colliderml_v1",
        config={
            "dataset_root": str(root),
            "n_files": n_files,
            "chunk_event_ranges": [list(token) for token in selected],
            "shared_hit_reassignments": shared_hit_reassignments,
        },
        wrote_at=datetime.now(timezone.utc).isoformat(),
    )
    if shared_hit_reassignments:
        print(f"Warning: {shared_hit_reassignments} shared hit reassignments; last track wins")
    return write_predictions(predictions, out_dir, meta)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--n-files", type=int)
    parser.add_argument("--chunk-range", type=int, nargs=2, metavar=("FIRST", "LAST"))
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    n_files = args.n_files if args.n_files is not None or args.chunk_range else 50
    path = convert_acts_colliderml(
        args.dataset_root, n_files, args.out_dir, chunk_range=args.chunk_range
    )
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
