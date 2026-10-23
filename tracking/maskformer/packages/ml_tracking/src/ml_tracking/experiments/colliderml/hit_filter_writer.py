"""Prediction callback that writes per-hit ColliderML filter probabilities."""

from pathlib import Path

import polars as pl
import torch
from lightning import Callback


class HitFilterPredictionWriter(Callback):
    """Write ``event_id, hit_id, keep_prob`` parquet rows during ``predict``."""

    def __init__(self, output_path: str):
        super().__init__()
        self.output_path = Path(output_path)
        self._frames: list[pl.DataFrame] = []

    def on_predict_start(self, trainer, pl_module) -> None:
        self._frames = []

    @staticmethod
    def _probabilities(outputs: dict) -> torch.Tensor:
        predictions = outputs.get("preds", outputs)
        predictions = predictions.get("final", predictions)
        found = [
            value
            for task in predictions.values()
            if isinstance(task, dict)
            for key, value in task.items()
            if key.endswith("sihit_on_valid_particle_prob")
        ]
        if len(found) != 1:
            raise KeyError(f"expected one sihit_on_valid_particle_prob tensor, found {len(found)}")
        return found[0]

    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0) -> None:
        inputs, targets = batch
        probabilities = self._probabilities(outputs).detach().float().cpu()
        for index, event_id in enumerate(targets["event_id"].reshape(-1)):
            count = int(inputs["sihit_valid"][index].sum())
            hit_ids = inputs["sihit_original_hit_id"][index, :count].detach().cpu().numpy()
            self._frames.append(pl.DataFrame({
                "event_id": pl.Series([int(event_id)] * count, dtype=pl.UInt64),
                "hit_id": pl.Series(hit_ids, dtype=pl.UInt64),
                "keep_prob": pl.Series(probabilities[index, :count].numpy(), dtype=pl.Float32),
            }))

    def on_predict_end(self, trainer, pl_module) -> None:
        if getattr(trainer, "world_size", 1) > 1:
            raise NotImplementedError("HitFilterPredictionWriter requires predict with devices=1")
        frame = pl.concat(self._frames) if self._frames else pl.DataFrame(
            schema={"event_id": pl.UInt64, "hit_id": pl.UInt64, "keep_prob": pl.Float32}
        )
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        frame.write_parquet(self.output_path)
