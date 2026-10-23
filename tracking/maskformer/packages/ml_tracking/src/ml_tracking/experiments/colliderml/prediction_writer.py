"""Write ColliderML MaskFormer predictions in tracking_core's canonical format."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import polars as pl
import torch
from lightning import Callback

from tracking_core.schema import RunMeta, write_predictions


class CanonicalPredictionWriter(Callback):
    """Convert query classification and hit-mask probabilities to tracks."""

    def __init__(self, out_dir: str, object_threshold: float = 0.5, mask_threshold: float = 0.5):
        super().__init__()
        if not 0.0 <= object_threshold <= 1.0:
            raise ValueError("object_threshold must be in [0, 1]")
        if not 0.0 <= mask_threshold <= 1.0:
            raise ValueError("mask_threshold must be in [0, 1]")
        self.out_dir = Path(out_dir)
        self.object_threshold = object_threshold
        self.mask_threshold = mask_threshold
        self._frames: list[pl.DataFrame] = []

    def on_predict_start(self, trainer, pl_module) -> None:
        self._frames = []

    @staticmethod
    def _final_predictions(outputs: dict) -> dict:
        predictions = outputs.get("preds", outputs)
        if "final" in predictions:
            predictions = predictions["final"]
        return predictions

    @staticmethod
    def _find_probabilities(predictions: dict) -> tuple[torch.Tensor, torch.Tensor]:
        values = [value for task in predictions.values() if isinstance(task, dict) for value in task.items()]
        object_probs = [value for key, value in values if key.endswith("_valid_prob") and "_sihit_" not in key]
        mask_probs = [value for key, value in values if key.endswith("_sihit_valid_prob")]
        if len(object_probs) != 1 or len(mask_probs) != 1:
            raise KeyError(
                "expected exactly one query validity probability and one sihit mask probability "
                f"in final predictions; found {len(object_probs)} and {len(mask_probs)}"
            )
        return object_probs[0], mask_probs[0]

    def add_batch(self, outputs: dict, batch: tuple[dict, dict]) -> None:
        """Convert and accumulate a prediction batch (public for focused tests)."""
        inputs, targets = batch
        object_probs, mask_probs = self._find_probabilities(self._final_predictions(outputs))
        event_ids = targets["event_id"].reshape(-1)
        hit_valid = inputs.get("sihit_valid")

        for batch_index, event_id in enumerate(event_ids):
            event_masks = mask_probs[batch_index].detach().float().cpu()
            n_hits = int(hit_valid[batch_index].sum().item()) if hit_valid is not None else event_masks.shape[-1]
            event_masks = event_masks[:, :n_hits]
            valid_queries = object_probs[batch_index].detach().float().cpu() >= self.object_threshold
            original_hit_ids = inputs.get("sihit_original_hit_id")
            if original_hit_ids is None:
                event_hit_ids = torch.arange(n_hits)
            else:
                event_hit_ids = original_hit_ids[batch_index, :n_hits].detach().cpu()

            assignments = torch.full((n_hits,), -1, dtype=torch.int64)
            scores = torch.full((n_hits,), torch.nan, dtype=torch.float32)
            if bool(valid_queries.any()) and n_hits:
                query_indices = torch.nonzero(valid_queries, as_tuple=False).flatten()
                winning_scores, local_indices = event_masks[valid_queries].max(dim=0)
                assigned = winning_scores >= self.mask_threshold
                assignments[assigned] = query_indices[local_indices[assigned]]
                scores[assigned] = winning_scores[assigned]

            self._frames.append(
                pl.DataFrame(
                    {
                        "event_id": pl.Series([int(event_id)] * n_hits, dtype=pl.UInt64),
                        "hit_id": pl.Series(
                            event_hit_ids.numpy(),
                            dtype=pl.UInt64,
                        ),
                        "predicted_track_id": pl.Series(assignments.numpy(), dtype=pl.Int64),
                        "score": pl.Series(scores.numpy(), dtype=pl.Float32),
                    }
                )
            )

    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0) -> None:
        self.add_batch(outputs, batch)

    def on_predict_end(self, trainer, pl_module) -> None:
        if getattr(trainer, "world_size", 1) > 1:
            raise NotImplementedError("CanonicalPredictionWriter is single-process only; run predict with devices=1")
        frame = pl.concat(self._frames) if self._frames else pl.DataFrame(
            schema={"event_id": pl.UInt64, "hit_id": pl.UInt64, "predicted_track_id": pl.Int64, "score": pl.Float32}
        )
        ckpt_path = getattr(trainer, "ckpt_path", None)
        config = {"object_threshold": self.object_threshold, "mask_threshold": self.mask_threshold}
        if ckpt_path:
            config["ckpt_path"] = str(ckpt_path)
        meta = RunMeta(
            run_id=str(uuid4()),
            implementation="ml_tracking:colliderml_maskformer",
            config=config,
            wrote_at=datetime.now(timezone.utc).isoformat(),
        )
        write_predictions(frame, self.out_dir, meta)
