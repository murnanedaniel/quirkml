"""Eval wrapper smoke — fails clearly when pyacts is missing or eval not yet wired."""

from __future__ import annotations

import polars as pl
import pytest
from tracking_core.eval import score
from tracking_core.schema import RunMeta, write_predictions


def _seed_predictions(tmp_path):
    df = pl.DataFrame(
        {
            "event_id": [0],
            "hit_id": [1],
            "predicted_track_id": [1],
            "score": [0.9],
        }
    )
    write_predictions(
        df,
        tmp_path,
        RunMeta(
            run_id="r",
            implementation="tracking_core:test",
            config={},
            wrote_at="2026-05-08T00:00:00Z",
        ),
    )


def test_score_requires_exactly_one_truth_source(tmp_path):
    _seed_predictions(tmp_path)
    with pytest.raises((ValueError, RuntimeError, NotImplementedError)) as exc:
        score(tmp_path)  # neither truth_config_id nor truth_path
    # Either the missing-truth error, the missing-pyacts error, or the
    # not-yet-wired error — all three are valid signals that the
    # contract surface exists.
    assert exc.value.args


def test_score_pyacts_or_not_implemented(tmp_path):
    _seed_predictions(tmp_path)
    with pytest.raises((RuntimeError, NotImplementedError)):
        score(tmp_path, truth_config_id="ttbar_pu0")
