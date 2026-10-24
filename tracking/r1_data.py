"""Read an event range of a quirk dataset in the ColliderML Release 1 layout (written by convert_to_r1.py).

    <root>/truth/particles/*.parquet, <root>/reco/tracker_hits/*.parquet
    file names ...events<A>-<B>.parquet (A, B inclusive), one row per event, list columns per particle / hit.

Rows inside a file are not sorted by event_id, and the pyacts 48 ParquetReader reads events by row position, not by
event_id. Everything here therefore works with explicit event ids: `read_events` returns the rows of events
[first, stop) sorted by event_id, and `stage_events` writes them to one file per object, so that Sequencer event k
is event first + k. hit_id = position of the hit in its event's lists.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

OBJECTS = {"particles": ("truth", "particles"), "tracker_hits": ("reco", "tracker_hits")}
#: Objects of the packaged layout only (not staged): ACTS tracks, truth-seeded fits.
PACKAGED_OBJECTS = {"tracks": ("reco", "tracks"), "truth_tracks": ("reco", "truth_tracks")}
QUIRK_PDG = 10000100
_RANGE = re.compile(r"events(\d+)-(\d+)\.parquet$")


def object_files(root: Path | str, obj: str) -> list[tuple[int, int, Path]]:
    """[(first, last, path)] of one object directory, sorted by first event (last inclusive)."""
    directory = Path(root).joinpath(*{**OBJECTS, **PACKAGED_OBJECTS}[obj])
    files = []
    for path in directory.glob("*.parquet"):
        match = _RANGE.search(path.name)
        if not match:
            raise ValueError(f"{path}: no events<A>-<B> range in the file name")
        files.append((int(match.group(1)), int(match.group(2)), path))
    if not files:
        raise FileNotFoundError(f"no parquet files in {directory}")
    return sorted(files)


def read_events(root: Path | str, obj: str, first: int, stop: int, columns: list[str] | None = None) -> pa.Table:
    """Rows of events first <= event_id < stop of one object, sorted by event_id. Raises if any event is missing or
    duplicated."""
    tables = []
    for lo, hi, path in object_files(root, obj):
        if hi < first or lo >= stop:
            continue
        table = pq.read_table(path, columns=columns)
        ids = table["event_id"]
        keep = pc.and_(pc.greater_equal(ids, first), pc.less(ids, stop))
        tables.append(table.filter(keep))
    if not tables:
        raise ValueError(f"no {obj} rows for events [{first}, {stop}) in {root}")
    table = pa.concat_tables(tables).sort_by("event_id")
    ids = table["event_id"].to_numpy()
    expected = list(range(first, stop))
    if ids.tolist() != expected:
        missing = sorted(set(expected) - set(ids.tolist()))
        raise ValueError(f"{obj}: events [{first}, {stop}) not each present exactly once in {root} "
                         f"({len(ids)} rows, {len(missing)} missing, first missing {missing[:5]})")
    return table


def _source_stamp(root: Path, first: int, stop: int) -> dict:
    return {obj: [[str(path), os.stat(path).st_size, os.stat(path).st_mtime]
                  for lo, hi, path in object_files(root, obj) if not (hi < first or lo >= stop)]
            for obj in OBJECTS}


def stage_events(root: Path | str, first: int, stop: int, staging_root: Path | str) -> Path:
    """Write events [first, stop) of `root`, sorted by event_id, to
    <staging_root>/<dataset name>_<hash of its path>/events<first>-<stop-1>/{truth/particles,reco/tracker_hits}/ (one file each) and
    return that directory. Reuses an existing staging whose manifest matches the source files (path, size, mtime)."""
    root = Path(root).resolve()
    # The directory name carries a hash of the source path: two datasets with the same name (e.g. a relabelled copy
    # such as the pair truth) must not share, and overwrite, one staging copy.
    tag = hashlib.sha1(str(root).encode()).hexdigest()[:8]
    out = Path(staging_root) / f"{root.name}_{tag}" / f"events{first}-{stop - 1}"
    manifest = out / "manifest.json"
    stamp = {"source": str(root), "first": first, "stop": stop, "files": _source_stamp(root, first, stop)}
    if manifest.exists() and json.loads(manifest.read_text()) == stamp:
        return out
    for obj, parts in OBJECTS.items():
        directory = out.joinpath(*parts)
        directory.mkdir(parents=True, exist_ok=True)
        for old in directory.glob("*.parquet"):
            old.unlink()
        table = read_events(root, obj, first, stop)
        target = directory / f"{root.name}.{obj}.events{first}-{stop - 1}.parquet"
        tmp = target.with_suffix(".tmp")
        pq.write_table(table, tmp)
        tmp.rename(target)
    manifest.write_text(json.dumps(stamp, indent=1))
    return out
