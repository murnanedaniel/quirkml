"""Write quirk MaskFormer predictions (event_id, hit_id, predicted_track_id, score) to predictions.parquet.

hit_id is the row position of the hit in the event's per-hit list columns of the tracker_hits parquet (the dataset
never filters or reorders hits). Queries with object prob >= object_threshold are valid. Each hit goes to one valid
query, if that query's mask probability for the hit is >= mask_threshold; otherwise -1. The query is chosen by
assignment="mask": highest mask probability (default, as in the ColliderML writer), or
assignment="object_weighted": highest object prob x mask probability (resolves hits contested by close tracks).
"""

from pathlib import Path

import polars as pl
import torch

from ml_tracking.experiments.colliderml.prediction_writer import CanonicalPredictionWriter


class QuirkPredictionWriter(CanonicalPredictionWriter):
    def __init__(self, out_dir: str, object_threshold: float = 0.1, mask_threshold: float = 0.5, assignment: str = "mask"):
        super().__init__(out_dir=str(Path(out_dir).resolve()), object_threshold=object_threshold, mask_threshold=mask_threshold)
        if assignment not in {"mask", "object_weighted"}:
            raise ValueError(f"unknown assignment {assignment}")
        self.assignment = assignment

    def add_batch(self, outputs: dict, batch) -> None:
        inputs, targets = batch
        object_probs, mask_probs = self._find_probabilities(self._final_predictions(outputs))
        event_ids = targets["event_id"].reshape(-1)
        hit_valid = inputs.get("sihit_valid")
        for b, event_id in enumerate(event_ids):
            n_hits = int(hit_valid[b].sum().item()) if hit_valid is not None else mask_probs.shape[-1]
            masks = mask_probs[b].detach().float().cpu()[:, :n_hits]
            obj = object_probs[b].detach().float().cpu()
            valid = obj >= self.object_threshold
            hit_ids = inputs["sihit_original_hit_id"][b, :n_hits].detach().cpu()
            assignments = torch.full((n_hits,), -1, dtype=torch.int64)
            scores = torch.full((n_hits,), torch.nan, dtype=torch.float32)
            if bool(valid.any()) and n_hits:
                q_idx = torch.nonzero(valid, as_tuple=False).flatten()
                vm = masks[valid]
                rank = vm * obj[valid][:, None] if self.assignment == "object_weighted" else vm
                local = rank.argmax(dim=0)
                win = vm[local, torch.arange(n_hits)]
                ok = win >= self.mask_threshold
                assignments[ok] = q_idx[local[ok]]
                scores[ok] = win[ok]
            self._frames.append(pl.DataFrame({
                "event_id": pl.Series([int(event_id)] * n_hits, dtype=pl.UInt64),
                "hit_id": pl.Series(hit_ids.numpy(), dtype=pl.UInt64),
                "predicted_track_id": pl.Series(assignments.numpy(), dtype=pl.Int64),
                "score": pl.Series(scores.numpy(), dtype=pl.Float32),
            }))

    def on_predict_end(self, trainer, pl_module) -> None:
        super().on_predict_end(trainer, pl_module)
        print(f"[QuirkPredictionWriter] wrote {len(self._frames)} events to {self.out_dir / 'predictions.parquet'} "
              f"(object_threshold={self.object_threshold}, mask_threshold={self.mask_threshold}, assignment={self.assignment})")
