"""ColliderML I/O.

Loads ColliderML configs through the official `colliderml` package and
returns split polars frames. Tensorisation is left to the consumer
(Lightning datamodule in `ml_tracking`, or upstream-specific adapter in
each baselines/<name>/).

Splits are deterministic by hash(event_id) so the same event lands in
the same split across reruns, machines, and object levels.
"""

from __future__ import annotations

import hashlib
from typing import Literal

import colliderml
import polars as pl

ObjectLevel = Literal["particles", "tracker_hits", "calo_hits", "tracks"]
Split = Literal["train", "val", "test"]

#: Default feature columns per object level. Names match the
#: CERN/ColliderML-Release-1 parquet schema.
FEATURE_COLUMNS: dict[ObjectLevel, list[str]] = {
    "particles": ["mass", "energy", "px", "py", "pz", "vx", "vy", "vz", "charge"],
    "tracker_hits": ["x", "y", "z", "time"],
    "calo_hits": ["x", "y", "z", "total_energy"],
    "tracks": ["d0", "z0", "phi", "theta", "qop"],
}


def _bucket(event_id: object) -> int:
    return int(hashlib.md5(str(event_id).encode()).hexdigest(), 16) % 100


def load_split(
    config_id: str = "ttbar_pu0",
    object_level: ObjectLevel = "particles",
    split: Split = "train",
    max_events: int | None = None,
) -> pl.DataFrame:
    """Load one (split, object_level) view of a ColliderML config.

    The first call downloads into `~/.cache/colliderml` (or
    `$COLLIDERML_DATA_DIR`); subsequent calls hit the cache.
    """
    frames = colliderml.load(config_id, max_events=max_events)
    if object_level not in frames:
        raise KeyError(
            f"object_level {object_level!r} not in colliderml.load output; "
            f"got keys {sorted(frames)}"
        )
    df = frames[object_level]
    bucket = [_bucket(e) for e in df["event_id"].to_list()]
    if split == "train":
        keep = [i for i, b in enumerate(bucket) if b < 90]
    elif split == "val":
        keep = [i for i, b in enumerate(bucket) if 90 <= b < 95]
    elif split == "test":
        keep = [i for i, b in enumerate(bucket) if b >= 95]
    else:
        raise ValueError(f"unknown split {split!r}")
    return df[keep]
