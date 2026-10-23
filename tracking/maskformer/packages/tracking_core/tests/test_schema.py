"""Round-trip test for the predictions contract — pure Python, no network."""

from __future__ import annotations

import polars as pl
from tracking_core.schema import (
    PREDICTIONS_COLUMNS,
    RunMeta,
    read_predictions,
    write_predictions,
)


def test_predictions_round_trip(tmp_path):
    df = pl.DataFrame(
        {
            "event_id": [0, 0, 0, 1, 1],
            "hit_id": [10, 11, 12, 20, 21],
            "predicted_track_id": [1, 1, 2, -1, 7],
            "score": [0.9, 0.8, 0.95, float("nan"), 0.5],
        }
    )
    meta = RunMeta(
        run_id="test-run",
        implementation="tracking_core:test",
        config={"foo": 1, "bar": [1, 2, 3]},
        wrote_at="2026-05-08T00:00:00Z",
    )

    pred_path = write_predictions(df, tmp_path, meta)
    assert pred_path.exists()
    assert (tmp_path / "meta.json").exists()

    df_back, meta_back = read_predictions(tmp_path)
    assert tuple(df_back.columns) == PREDICTIONS_COLUMNS
    assert df_back.height == df.height
    assert meta_back == meta


def test_missing_columns_raises(tmp_path):
    bad = pl.DataFrame({"event_id": [0], "hit_id": [1]})
    meta = RunMeta(
        run_id="r",
        implementation="t",
        config={},
        wrote_at="2026-05-08T00:00:00Z",
    )
    try:
        write_predictions(bad, tmp_path, meta)
    except ValueError as exc:
        assert "predicted_track_id" in str(exc)
    else:
        raise AssertionError("expected ValueError for missing columns")
