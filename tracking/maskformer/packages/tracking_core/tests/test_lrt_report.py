from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from tracking_core.adapters.acts import _select_chunks as _select_acts_chunks
from tracking_core.lrt_report import _select_chunks as _select_report_chunks
from tracking_core.lrt_report import generate_lrt_report
from tracking_core.schema import RunMeta, write_predictions


def _write_list_table(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def test_chunk_range_filters_before_n_files() -> None:
    chunks = [(1000, 1999), (796000, 796999), (797000, 797999), (816000, 816999)]
    for select in (_select_acts_chunks, _select_report_chunks):
        assert select(chunks, None, [796, 797]) == chunks[1:3]
        assert select(chunks, 1, (796, 797)) == [chunks[1]]


def test_lrt_efficiency_fakes_and_duplicates(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    suffix = "sample.events0-1.parquet"
    truth_rows = [
        {"event_id": 0, "particle_id": [1, 2, 3, 4, 5],
         "charge": [1., 1., 0., 0., 0.], "vx": [0., 3., 0., 0., 0.],
         "vy": [0., 0., 0., 0., 0.], "px": [2., 2., 1., 1., 1.],
         "py": [0., 0., 0., 0., 0.], "pz": [0., 0., 0., 0., 0.],
         "primary": [True, True, False, False, False]},
        {"event_id": 1, "particle_id": [6], "charge": [1.], "vx": [0.],
         "vy": [0.], "px": [2.], "py": [0.], "pz": [0.], "primary": [True]},
    ]
    hit_pids = [1] * 4 + [2] * 6 + [3, 4, 3, 4, 5]
    hit_rows = [
        {"event_id": 0, "particle_id": hit_pids,
         "volume_id": [1] * len(hit_pids), "layer_id": list(range(len(hit_pids)))},
        {"event_id": 1, "particle_id": [6] * 4,
         "volume_id": [1] * 4, "layer_id": [0, 1, 2, 3]},
    ]
    _write_list_table(root / "truth/particles" / suffix, truth_rows)
    _write_list_table(root / "reco/tracker_hits" / suffix, hit_rows)

    # Track 10 -> pid 1; tracks 20/21 duplicate pid 2; track 30 has an exact
    # 2/2 truth split and is a fake. Event 1's reconstructable pid 6 is missed.
    assignments = [10] * 4 + [20] * 3 + [21] * 3 + [30] * 4 + [-1] + [-1] * 4
    event_ids = [0] * 15 + [1] * 4
    hit_ids = list(range(15)) + list(range(4))
    pred = pl.DataFrame({"event_id": event_ids, "hit_id": hit_ids,
                         "predicted_track_id": assignments,
                         "score": [float("nan")] * 19}).cast({
        "event_id": pl.UInt64, "hit_id": pl.UInt64,
        "predicted_track_id": pl.Int64, "score": pl.Float32})
    pred_dir = tmp_path / "predictions"
    write_predictions(pred, pred_dir, RunMeta("test", "test:synthetic", {},
        datetime.now(timezone.utc).isoformat()))

    summary = generate_lrt_report(pred_dir, root, 1, tmp_path / "report", plots=False)
    fid = summary["selections"]["fiducial"]
    assert fid["overall"]["numerator"] == 2
    assert fid["overall"]["denominator"] == 3
    assert fid["prompt_r_lt_2_mm"]["efficiency"] == 0.5
    assert fid["displaced_r_ge_2_mm"]["efficiency"] == 1.0
    metrics = summary["track_metrics"]
    assert metrics["eligible_predicted_tracks"] == 4
    assert metrics["fake_tracks"] == 1
    assert metrics["fake_rate"] == 0.25
    assert metrics["duplicate_tracks"] == 1
    assert metrics["duplicate_rate"] == 0.25
    assert (tmp_path / "report/summary.json").exists()
