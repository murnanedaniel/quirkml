"""Canonical on-disk contract between baselines and the lego framework.

Every track-finding implementation — whether a `baselines/` Docker image
or a `packages/ml_tracking/` pipeline — emits predictions in the shape
defined here, and the eval routine in :mod:`tracking_core.eval` reads
exactly this format.

Per-run layout::

    <outdir>/
      predictions.parquet   # one row per hit
      meta.json             # run-level metadata (sidecar)

`predictions.parquet` columns:

================== ========= ================================================
column              type      meaning
================== ========= ================================================
event_id            uint64    groups hits into one event
hit_id              uint64    unique within event; matches ColliderML hit_id
predicted_track_id  int64     -1 ⇒ not assigned to any track
score               float32   optional per-hit confidence; NaN if unused
================== ========= ================================================
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import polars as pl

PREDICTIONS_COLUMNS: tuple[str, ...] = (
    "event_id",
    "hit_id",
    "predicted_track_id",
    "score",
)


@dataclass(frozen=True)
class RunMeta:
    """Sidecar metadata. Lives next to predictions.parquet as meta.json."""

    run_id: str
    implementation: str  # e.g. "ml_tracking:A4" or "baselines:trackformers"
    config: dict[str, Any]
    wrote_at: str  # ISO-8601 UTC


def write_predictions(df: pl.DataFrame, outdir: Path | str, meta: RunMeta) -> Path:
    """Write predictions.parquet + meta.json. Returns the parquet path."""
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    missing = [c for c in PREDICTIONS_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"predictions df missing required columns: {missing}. "
            f"Got: {df.columns}"
        )

    pred_path = out / "predictions.parquet"
    df.select(PREDICTIONS_COLUMNS).write_parquet(pred_path)
    (out / "meta.json").write_text(json.dumps(asdict(meta), indent=2))
    return pred_path


def read_predictions(outdir: Path | str) -> tuple[pl.DataFrame, RunMeta]:
    """Inverse of :func:`write_predictions`."""
    import polars as pl

    p = Path(outdir)
    df = pl.read_parquet(p / "predictions.parquet")
    meta = json.loads((p / "meta.json").read_text())
    return df, RunMeta(**meta)
