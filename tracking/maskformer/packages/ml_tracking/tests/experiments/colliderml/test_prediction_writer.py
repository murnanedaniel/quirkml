from types import SimpleNamespace

import polars as pl
import pytest
import torch

from ml_tracking.experiments.colliderml.prediction_writer import CanonicalPredictionWriter
from tracking_core.schema import read_predictions


def test_canonical_prediction_writer(tmp_path):
    writer = CanonicalPredictionWriter(str(tmp_path), object_threshold=0.5, mask_threshold=0.5)
    batch = (
        {"sihit_valid": torch.tensor([[True, True, True], [True, True, False]])},
        {"event_id": torch.tensor([[17], [23]])},
    )
    predictions = {
        "preds": {
            "final": {
                "flow_valid": {"flow_valid_prob": torch.tensor([[0.9, 0.4], [0.5, 0.8]])},
                "flow_sihit_assignment": {
                    "flow_sihit_valid_prob": torch.tensor(
                        [
                            [[0.8, 0.4, 0.7], [0.9, 0.9, 0.9]],
                            [[0.6, 0.7, 0.1], [0.7, 0.3, 0.9]],
                        ]
                    )
                },
            }
        }
    }

    writer.add_batch(predictions, batch)
    writer.on_predict_end(SimpleNamespace(ckpt_path="fresh.ckpt"), None)

    frame, meta = read_predictions(tmp_path)
    assert frame.schema == {
        "event_id": pl.UInt64,
        "hit_id": pl.UInt64,
        "predicted_track_id": pl.Int64,
        "score": pl.Float32,
    }
    assert frame["event_id"].to_list() == [17, 17, 17, 23, 23]
    assert frame["hit_id"].to_list() == [0, 1, 2, 0, 1]
    assert frame["predicted_track_id"].to_list() == [0, -1, 0, 1, 0]
    assert frame["score"].to_list() == pytest.approx([0.8, float("nan"), 0.7, 0.7, 0.7], nan_ok=True)
    assert meta.implementation == "ml_tracking:colliderml_maskformer"
    assert meta.config == {"object_threshold": 0.5, "mask_threshold": 0.5, "ckpt_path": "fresh.ckpt"}


def test_canonical_prediction_writer_rejects_multi_rank(tmp_path):
    writer = CanonicalPredictionWriter(str(tmp_path))
    with pytest.raises(NotImplementedError, match="single-process only"):
        writer.on_predict_end(SimpleNamespace(world_size=2), None)


def test_canonical_prediction_writer_restores_original_hit_ids(tmp_path):
    writer = CanonicalPredictionWriter(str(tmp_path))
    outputs = {"preds": {"final": {
        "valid": {"flow_valid_prob": torch.tensor([[0.9]])},
        "mask": {"flow_sihit_valid_prob": torch.tensor([[[0.8, 0.7]]])},
    }}}
    batch = (
        {"sihit_valid": torch.tensor([[True, True]]), "sihit_original_hit_id": torch.tensor([[1, 4]])},
        {"event_id": torch.tensor([17])},
    )
    writer.add_batch(outputs, batch)
    assert writer._frames[0]["hit_id"].to_list() == [1, 4]
