"""Lightning module for the quirk MaskFormer: the ColliderML wrapper plus quirk metrics."""

import torch

from ml_tracking.models.wrapper import ModelWrapper


class QuirkModel(ModelWrapper):
    """Logs, per stage, hit-mask metrics for all targets and for quirks only.

    Predictions are aligned with targets by the Hungarian matching done inside
    ``model.loss`` (the outputs are permuted), so query i is compared with target i.
    A target counts as efficient at working point ``wp`` when its matched query is
    a valid prediction and both hit efficiency and hit purity are >= wp
    ("double majority" at wp=0.5).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Command-line overrides of dict entries (--model.lrs_config.max 5e-5) arrive as strings.
        for key, value in list(self.lrs_config.items()):
            if isinstance(value, str):
                try:
                    self.lrs_config[key] = float(value)
                except ValueError:
                    pass

    def log_custom_metrics(self, preds, targets, stage):
        preds = preds["final"]
        if "flow_sihit_assignment" not in preds:
            return
        pred_masks = preds["flow_sihit_assignment"]["flow_sihit_valid"].clone()
        true_masks = targets["particle_sihit_valid"].clone()
        hit_valid = targets["sihit_valid"].unsqueeze(1)
        pred_masks &= hit_valid
        pred_valid = preds["flow_valid"]["flow_valid"] & (pred_masks.sum(-1) > 0)
        true_valid = targets["particle_valid"] & (true_masks.sum(-1) > 0)
        pred_masks &= pred_valid.unsqueeze(-1)
        true_masks &= true_valid.unsqueeze(-1)

        tp = (pred_masks & true_masks).sum(-1).float()
        n_pred = pred_masks.sum(-1).float()
        n_true = true_masks.sum(-1).float()
        hit_eff = tp / n_true.clamp_min(1)
        hit_pur = tp / n_pred.clamp_min(1)
        quirk = targets["particle_is_quirk"] & true_valid

        # Hit-level mask accuracy over real hits x real target slots.
        both = true_valid.unsqueeze(-1) & hit_valid
        agree = (pred_masks == true_masks) & both
        self.log(f"{stage}/mask_acc", agree.sum().float() / both.sum().clamp_min(1))
        # Fraction of true hit memberships recovered (insensitive to the many zeros).
        self.log(f"{stage}/hit_recall", tp[true_valid].sum() / n_true[true_valid].sum().clamp_min(1))

        for wp in (0.5, 0.75, 1.0):
            good = (hit_eff >= wp) & (hit_pur >= wp) & pred_valid & true_valid
            self.log(f"{stage}/p{wp}_eff", good.sum().float() / true_valid.sum().clamp_min(1))
            if quirk.any():
                self.log(f"{stage}/p{wp}_quirk_eff", (good & quirk).sum().float() / quirk.sum())
        # Purity / fake proxy: predicted objects whose matched target is not >= 0.5 pure.
        pure = (hit_pur >= 0.5) & pred_valid & true_valid
        self.log(f"{stage}/p0.5_pur", pure.sum().float() / pred_valid.sum().clamp_min(1))
        if quirk.any():
            self.log(f"{stage}/quirk_hit_eff_mean", hit_eff[quirk].mean())
            self.log(f"{stage}/quirk_hit_pur_mean", hit_pur[quirk & pred_valid].mean() if (quirk & pred_valid).any() else torch.zeros((), device=tp.device))
        self.log(f"{stage}/num_pred", pred_valid.sum(-1).float().mean())
        self.log(f"{stage}/num_true", true_valid.sum(-1).float().mean())

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        inputs, _targets = batch
        outputs = self.model(inputs)
        return {"outputs": outputs, "preds": self.model.predict(outputs)}
