"""Write quirk MaskFormer predictions (event_id, hit_id, predicted_track_id, score) to predictions.parquet.

hit_id is the row position of the hit in the event's per-hit list columns of the
tracker_hits parquet (the dataset never filters or reorders hits). Each hit goes
to the valid query (object prob >= object_threshold) with the highest mask
probability, if that probability is >= mask_threshold; otherwise -1.
"""

from pathlib import Path

from ml_tracking.experiments.colliderml.prediction_writer import CanonicalPredictionWriter


class QuirkPredictionWriter(CanonicalPredictionWriter):
    def __init__(self, out_dir: str, object_threshold: float = 0.1, mask_threshold: float = 0.5):
        super().__init__(out_dir=str(Path(out_dir).resolve()), object_threshold=object_threshold, mask_threshold=mask_threshold)

    def on_predict_end(self, trainer, pl_module) -> None:
        super().on_predict_end(trainer, pl_module)
        n_events = len(self._frames)
        print(f"[QuirkPredictionWriter] wrote {n_events} events to {self.out_dir / 'predictions.parquet'} "
              f"(object_threshold={self.object_threshold}, mask_threshold={self.mask_threshold})")
