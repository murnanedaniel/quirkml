"""Lightning module for the quirk MaskFormer: the ColliderML wrapper plus quirk metrics."""

import math

import torch

from ml_tracking.experiments.quirk.spike_guard import SpikeGuard
from ml_tracking.experiments.quirk.split_losses import split_loss_terms
from ml_tracking.models.wrapper import ModelWrapper


class QuirkModel(ModelWrapper):
    """Logs, per stage, hit-mask metrics for all targets and for quirks only.

    Predictions are aligned with targets by the Hungarian matching done inside
    ``model.loss`` (the outputs are permuted), so query i is compared with target i.
    A target counts as efficient at working point ``wp`` when its matched query is
    a valid prediction and both hit efficiency and hit purity are >= wp
    ("double majority" at wp=0.5).
    """

    def __init__(self, *args, spike_guard: dict | None = None, init_from: str | None = None, log_split_losses: bool = True,
                 **kwargs):
        """spike_guard: None (off) or {factor, window, min_history, max_consecutive} (review thread: 20 / 500 / 100 / 500).
        With the guard, gradients are clipped here (norm 0.1 unless the trainer's gradient_clip_val says otherwise) and
        an update with a non-finite or spiking pre-clip norm is skipped (gradients dropped before the optimizer step)."""
        super().__init__(*args, **kwargs)
        self.spike_guard_cfg = spike_guard
        self.spike_guard = SpikeGuard(**spike_guard) if spike_guard else None
        self.skipped_updates = 0
        self.init_from = init_from
        # monitoring only (no gradient): final-layer loss components split into quirk-pair / ordinary targets
        self.log_split_losses = log_split_losses
        if init_from:
            self._warm_start(init_from)
        # Command-line overrides of dict entries (--model.lrs_config.max 5e-5) arrive as strings.
        for key, value in list(self.lrs_config.items()):
            if isinstance(value, str):
                try:
                    self.lrs_config[key] = float(value)
                except ValueError:
                    pass

    def _warm_start(self, path: str) -> None:
        """Initialise from another quirk MaskFormer checkpoint (mu 60 arm d: the pu0 pair model). Same-name tensors of
        equal shape are copied; the FlexGeo wrapper's encoder.layers.N.layer.* take encoder.layers.N.*; tensors whose
        leading dims differ (initial queries) or whose last dim is wider (input projection with extra input fields)
        take the source block and keep the rest freshly initialised (extra input columns set to zero, so the new
        inputs start without effect). Prints what was copied, partially copied and left fresh."""
        src = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
        tgt = self.state_dict()
        full, part, fresh = [], [], []
        with torch.no_grad():
            for k, v in tgt.items():
                sk = k.replace(".layer.", ".", 1) if ".encoder.layers." in k and ".layer." in k else k
                if sk not in src:
                    fresh.append(k)
                    continue
                s = src[sk]
                if s.shape == v.shape:
                    v.copy_(s)
                    full.append(k)
                elif s.dim() == v.dim() and all(a <= b for a, b in zip(s.shape, v.shape)):
                    if s.dim() == 2 and s.shape[0] == v.shape[0]:  # wider input: copy columns, zero the extra ones
                        v[:, s.shape[1]:] = 0
                    v[tuple(slice(0, n) for n in s.shape)].copy_(s)
                    part.append(f"{k} {tuple(s.shape)} -> {tuple(v.shape)}")
                else:
                    fresh.append(f"{k} (source {tuple(s.shape)} vs {tuple(v.shape)})")
        unused = sorted(set(src) - {k.replace(".layer.", ".", 1) if ".encoder.layers." in k and ".layer." in k else k for k in tgt})
        print(f"[warm start] from {path}: {len(full)} tensors copied, {len(part)} partially, {len(fresh)} fresh, "
              f"{len(unused)} source tensors unused")
        for x in part:
            print("[warm start] partial:", x)
        for x in fresh:
            print("[warm start] fresh:", x)
        for x in unused:
            print("[warm start] unused source:", x)

    def configure_gradient_clipping(self, optimizer, gradient_clip_val=None, gradient_clip_algorithm=None):
        if self.spike_guard is None:
            return super().configure_gradient_clipping(optimizer, gradient_clip_val, gradient_clip_algorithm)
        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip_val or 0.1)
        n = float(norm)
        spike = self.spike_guard.is_spike(n)
        skip = spike or not math.isfinite(n)
        self.spike_guard.record(n, applied=not skip)
        self.log("train/grad_norm_preclip", n if math.isfinite(n) else -1.0, on_step=True, on_epoch=False)
        if skip:
            for p in self.model.parameters():
                p.grad = None
            self.skipped_updates += 1
        self.log("train/skipped_updates", float(self.skipped_updates), on_step=True, on_epoch=False)

    def on_save_checkpoint(self, checkpoint):
        if self.spike_guard is not None:
            checkpoint["spike_guard"] = {**self.spike_guard.state(), "skipped_updates": self.skipped_updates}

    def on_load_checkpoint(self, checkpoint):
        st = checkpoint.get("spike_guard")
        if st and self.spike_guard is not None:
            self.spike_guard = SpikeGuard(st["factor"], st["window"], st["min_history"], st["history"], st["max_consecutive"],
                                          st["skipped"], st["armed"])
            self.spike_guard.consecutive = st.get("consecutive", 0)
            self.skipped_updates = st.get("skipped_updates", 0)

    def log_coupled_strict(self, preds, targets, stage, object_threshold=0.1, mask_threshold=0.5):
        """Coupled quirk efficiency, strict (> 50% / > 50%), with the prediction writer's hit assignment.

        Per event with quirk targets: U = union of the quirk targets' hits (the pair target, or both quirks). Hits go
        to the valid query (object prob >= object_threshold) with the highest mask probability, if >= mask_threshold.
        Matched if some assigned track T has |T & U| > |U| / 2 and |T & U| > |T| / 2. Matching-independent, so it can
        drive early stopping and checkpoint choice (val/coupled_strict)."""
        obj = preds["flow_valid"]["flow_valid_prob"].float()
        mprob = preds["flow_sihit_assignment"]["flow_sihit_valid_prob"].float()
        hit_valid = targets["sihit_valid"]
        quirk = targets["particle_is_quirk"] & targets["particle_valid"]
        union = (targets["particle_sihit_valid"] & quirk.unsqueeze(-1)).any(1) & hit_valid  # [B, H]
        has = union.sum(-1) >= 3
        if not has.any():
            return
        valid_q = obj >= object_threshold  # [B, Q]
        score = mprob.masked_fill(~valid_q.unsqueeze(-1), -1.0)
        best_p, best_q = score.max(1)  # [B, H]
        assigned = (best_p >= mask_threshold) & hit_valid
        nq = mprob.shape[1]
        onehot = torch.nn.functional.one_hot(best_q, nq).bool() & assigned.unsqueeze(-1)  # [B, H, Q]
        size = onehot.sum(1)  # [B, Q]
        inter = (onehot & union.unsqueeze(-1)).sum(1)  # [B, Q]
        u = union.sum(-1, keepdim=True)
        ok = ((2 * inter > u) & (2 * inter > size)).any(-1) & has
        self.log(f"{stage}/coupled_strict", ok[has].float().mean(), batch_size=int(has.sum()), on_step=False, on_epoch=True)

    def log_custom_metrics(self, preds, targets, stage):
        if stage == "val":
            self.log_coupled_strict(preds["final"], targets, stage)
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

    def log_split(self, outputs, targets, stage):
        """Final-layer loss components by target type (split_losses.split_loss_terms), as {stage}/split_<name>."""
        if not self.log_split_losses:
            return
        terms = split_loss_terms(self.model, outputs, targets)
        bs = int(targets["particle_valid"].shape[0])
        for k, v in terms.items():
            self.log(f"{stage}/split_{k}", v, batch_size=bs, on_step=stage == "train", on_epoch=stage != "train")

    def training_step(self, batch, batch_idx):
        """ModelWrapper.training_step plus the split-loss logging (same loss, same optimisation)."""
        inputs, targets = batch
        outputs = self.model(inputs)
        outputs, targets, losses = self.model.loss(outputs, targets)
        if batch_idx % self.trainer.log_every_n_steps == 0:
            preds = self.predict(outputs)
            self.log_metrics(preds, targets, "train")
            self.log_split(outputs, targets, "train")
        if self.mtl:
            self.mlt_opt(losses, outputs)
            return None
        total_loss = self.aggregate_losses(losses, stage="train")
        return {"loss": total_loss}

    def validation_step(self, batch):
        """ModelWrapper.validation_step plus the split-loss logging."""
        inputs, targets = batch
        outputs = self.model(inputs)
        outputs, targets, losses = self.model.loss(outputs, targets)
        total_loss = self.aggregate_losses(losses, stage="val")
        preds = self.model.predict(outputs)
        self.log_metrics(preds, targets, "val")
        self.log_split(outputs, targets, "val")
        return {"loss": total_loss}

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        inputs, _targets = batch
        outputs = self.model(inputs)
        return {"outputs": outputs, "preds": self.model.predict(outputs)}
