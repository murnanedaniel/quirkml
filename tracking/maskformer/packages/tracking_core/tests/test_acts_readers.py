"""Integration tests for the ColliderML → ACTS measurement pipeline.

Test 1 — test_truth_predictions (pure Python, no ACTS):
  Builds ground-truth predictions.parquet from ColliderML tracker_hits by
  grouping hits by particle_id (pT>1 GeV, |eta|<3). Validates round-trip
  via tracking_core.schema. Stores output path in a module-level variable
  so Test 2 can load it.

Test 2 — test_kf_efficiency:
  Calls tracking_core.eval.score() with the truth predictions from Test 1
  and asserts that KF and proto-track efficiency are ≥ 0.95.
  This validates that ColliderMLMeasurementReader produces correctly placed
  local coordinates.

Skipped if ColliderML data or surface map are absent.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path

import polars as pl
import pytest

PROJECT_ROOT = Path(__file__).parents[3]
PARTICLE_SHARD = (
    PROJECT_ROOT
    / "local_data/colliderml/CERN__ColliderML-Release-1"
    / "ttbar_pu0_particles/data/ttbar_pu0_particles/train-00000-of-01000.parquet"
)
HIT_SHARD = (
    PROJECT_ROOT
    / "local_data/colliderml/CERN__ColliderML-Release-1"
    / "ttbar_pu0_tracker_hits/data/ttbar_pu0_tracker_hits/train-00000-of-01000.parquet"
)
SURFACE_MAP = PROJECT_ROOT / "config/surface_map.csv.gz"
ODD_JSON = PROJECT_ROOT / "config/odd.json.gz"

requires_data = pytest.mark.skipif(
    not SURFACE_MAP.exists(),
    reason="ColliderML surface map not found — run scripts/build_colliderml_surface_map.py first",
)

# Written by test_truth_predictions, read by test_kf_efficiency.
_truth_predictions_dir: Path | None = None


# ---------------------------------------------------------------------------
# Helper — particle kinematic selection
# ---------------------------------------------------------------------------

def _selected_particle_ids(
    particles_df: pl.DataFrame,
    pt_min: float = 1.0,
    eta_max: float = 3.0,
) -> dict[int, set[int]]:
    """Return {event_id: set(particle_id)} for particles passing pT > pt_min GeV, |eta| < eta_max."""
    selected: dict[int, set[int]] = {}
    for row in particles_df.iter_rows(named=True):
        event_id = int(row["event_id"])
        passing: set[int] = set()
        for pid, px, py, pz in zip(row["particle_id"], row["px"], row["py"], row["pz"]):
            pt = math.sqrt(px**2 + py**2)
            if pt < pt_min:
                continue
            p = math.sqrt(px**2 + py**2 + pz**2)
            if p <= abs(pz):
                continue  # pathological — avoid log(0)
            cos_theta = pz / p
            if abs(cos_theta) >= 1.0:
                continue
            eta = -0.5 * math.log((1 + cos_theta) / (1 - cos_theta))
            if abs(eta) < eta_max:
                passing.add(int(pid))
        selected[event_id] = passing
    return selected


# ---------------------------------------------------------------------------
# Test 1 — pure Python, no ACTS
# ---------------------------------------------------------------------------

@requires_data
def test_truth_predictions(tmp_path):
    """Build truth predictions.parquet from ColliderML truth (pT>1 GeV, |eta|<3) and validate."""
    global _truth_predictions_dir

    from tracking_core.schema import RunMeta, read_predictions, write_predictions

    particles_df = pl.read_parquet(PARTICLE_SHARD)
    hits_df = pl.read_parquet(HIT_SHARD)
    selected = _selected_particle_ids(particles_df)

    rows: list[dict] = []
    for event_row in hits_df.iter_rows(named=True):
        event_id = int(event_row["event_id"])
        passing_pids = selected.get(event_id, set())
        particle_ids = event_row["particle_id"]
        track_id_map: dict[int, int] = {}
        next_track_id = 0
        for hit_idx, pid in enumerate(particle_ids):
            pid = int(pid)
            if pid not in passing_pids:
                rows.append({
                    "event_id": event_id,
                    "hit_id": hit_idx,
                    "predicted_track_id": -1,
                    "score": float("nan"),
                })
                continue
            if pid not in track_id_map:
                track_id_map[pid] = next_track_id
                next_track_id += 1
            rows.append({
                "event_id": event_id,
                "hit_id": hit_idx,
                "predicted_track_id": track_id_map[pid],
                "score": float("nan"),
            })

    df = pl.DataFrame(rows).with_columns([
        pl.col("event_id").cast(pl.UInt64),
        pl.col("hit_id").cast(pl.UInt64),
        pl.col("predicted_track_id").cast(pl.Int64),
        pl.col("score").cast(pl.Float32),
    ])

    write_predictions(
        df,
        tmp_path,
        RunMeta(
            run_id="test_truth",
            implementation="test:truth_tracks",
            config={"source": "colliderml_truth"},
            wrote_at=datetime.now(timezone.utc).isoformat(),
        ),
    )

    df_read, _meta = read_predictions(tmp_path)
    assert set(df_read.columns) >= {"event_id", "hit_id", "predicted_track_id"}
    assert len(df_read) == len(df)

    assigned = df_read.filter(pl.col("predicted_track_id") >= 0)
    assert assigned["predicted_track_id"].n_unique() > 0, "No tracks in truth predictions"
    hits_per_track = assigned.group_by("predicted_track_id").agg(pl.len().alias("n"))
    assert hits_per_track["n"].min() >= 1

    _truth_predictions_dir = tmp_path


# ---------------------------------------------------------------------------
# Test 2 — ACTS KF chain via score
# ---------------------------------------------------------------------------

pytest.importorskip("acts", reason="pyacts not installed")


@requires_data
def test_kf_efficiency():
    """KF with truth proto-tracks: validates that measurement local coords are correct."""
    if _truth_predictions_dir is None:
        pytest.skip("test_truth_predictions must run first to produce predictions")

    from tracking_core.eval import score

    metrics = score(
        predictions_dir=_truth_predictions_dir,
        hit_shard=HIT_SHARD,
        particle_shard=PARTICLE_SHARD,
        surface_map=SURFACE_MAP,
        odd_json=ODD_JSON,
    )
    print(f"\nKF efficiency:          {metrics['kf_efficiency']:.3f}")
    print(f"Proto-track efficiency: {metrics['proto_efficiency']:.3f}")
    assert metrics["kf_efficiency"] >= 0.95, (
        f"KF efficiency too low: {metrics['kf_efficiency']:.3f}"
    )
    assert metrics["proto_efficiency"] >= 0.95, (
        f"Proto-track efficiency too low: {metrics['proto_efficiency']:.3f}"
    )
