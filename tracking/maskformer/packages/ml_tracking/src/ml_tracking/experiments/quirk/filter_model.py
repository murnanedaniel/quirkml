"""Per-hit filter for mu 60 (hit belongs to a target: charged primary pT > 1 GeV with >= 3 hits, or to a quirk).

QuirkHitFilterTask: BCE with positives weighted by n_neg / n_pos per event (as HitFilterTask) and quirk hits by an extra
factor. QuirkFilterModel logs, per stage, the kept-hit count and the recall of target and quirk hits at a few
thresholds, so the operating point can be read off val directly."""

import torch
from torch import Tensor

from ml_tracking.models.task import HitFilterTask
from ml_tracking.models.wrapper import ModelWrapper


class QuirkHitFilterTask(HitFilterTask):
    def __init__(self, *args, quirk_weight: float = 10.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.quirk_weight = quirk_weight

    def loss(self, outputs: dict[str, Tensor], targets: dict[str, Tensor], layer_outputs=None) -> dict[str, Tensor]:
        out = outputs[f"{self.input_object}_logit"].float()
        tgt = targets[f"{self.input_object}_{self.target_field}"].float()
        valid = targets[f"{self.input_object}_valid"]
        q = targets[f"{self.input_object}_is_quirk"].float()
        pos = (tgt * valid).sum().clamp_min(1.0)
        neg = ((1 - tgt) * valid).sum().clamp_min(1.0)
        w = torch.where(tgt > 0, neg / pos, torch.ones_like(tgt)) * (1 + (self.quirk_weight - 1) * q) * valid
        loss = torch.nn.functional.binary_cross_entropy_with_logits(out, tgt, weight=w, reduction="sum") / w.sum()
        return {"bce": loss}


class QuirkFilterModel(ModelWrapper):
    THRESHOLDS = (0.01, 0.05, 0.1, 0.2, 0.5)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for key, value in list(self.lrs_config.items()):
            if isinstance(value, str):
                try:
                    self.lrs_config[key] = float(value)
                except ValueError:
                    pass

    def log_custom_metrics(self, preds, targets, stage):
        p = preds["final"]["hit_filter"]["sihit_on_valid_particle_prob"].float()
        valid = targets["sihit_valid"]
        tgt = targets["sihit_on_valid_particle"] & valid
        q = targets["sihit_is_quirk"] & valid
        for t in self.THRESHOLDS:
            keep = (p >= t) & valid
            self.log(f"{stage}/kept_hits_t{t}", keep.sum(-1).float().mean())
            self.log(f"{stage}/target_recall_t{t}", (keep & tgt).sum() / tgt.sum().clamp_min(1))
            if q.any():
                self.log(f"{stage}/quirk_recall_t{t}", (keep & q).sum() / q.sum())

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        inputs, _targets = batch
        outputs = self.model(inputs)
        return {"outputs": outputs, "preds": self.model.predict(outputs)}
