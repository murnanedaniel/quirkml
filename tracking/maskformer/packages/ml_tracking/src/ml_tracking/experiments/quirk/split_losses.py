"""Loss components split by target type (quirk-pair target vs ordinary targets), for monitoring only.

Computed after the model's loss (outputs permuted to the target order, so query i is matched to target i), on one
decoder layer (default the final one), with no gradient. For each component (object BCE, mask Dice, mask focal) and
target type it returns

- ``<comp>_<type>``: unweighted mean over targets of that type of the per-target loss (object BCE of the matched query
  with label 1; Dice of the matched query's mask; focal of the matched query's mask, mean over the event's hits, no
  positive-hit weight);
- ``<comp>_<type>_w``: that type's share of the loss actually optimised, i.e. the task's loss weight times the type's
  part of the task's reduction, with the object weights (pair x N), their mean-one normalisation, the null-class weight
  and the positive-hit weight exactly as in task.py. Summing ``_w`` over the target types (and ``object_bce_null_w``
  for unmatched queries, ``mask_*`` have no null part) gives the logged final-layer task losses.

Types: ``pair`` = valid targets with particle_is_quirk (the quirk pair object), ``ord`` = other valid targets;
``null`` (object BCE only) = queries matched to no target. Per-type quantities are means over the batch's targets of
that type; types absent from a batch are left out.
"""

import torch
import torch.nn.functional as F

from ml_tracking.models.task import ObjectClassificationTask, ObjectHitMaskTask


def _types(targets, target_object="particle"):
    valid = targets[f"{target_object}_valid"].bool()
    quirk = targets.get(f"{target_object}_is_quirk")
    quirk = torch.zeros_like(valid) if quirk is None else quirk.bool()
    return {"pair": valid & quirk, "ord": valid & ~quirk}, valid


@torch.no_grad()
def split_loss_terms(net, outputs, targets, layer="final"):
    """net: the MaskFormer (module.model); outputs: after net.loss (permuted); returns {name: float tensor}."""
    out = {}
    lo = outputs[layer]
    for task in net.tasks:
        if isinstance(task, ObjectClassificationTask) and task.name in lo and task.num_classes == 1:
            types, valid = _types(targets, task.target_object)
            logit = lo[task.name][task.output_object + "_logit"].float()
            target = valid.float()
            bce = F.binary_cross_entropy_with_logits(logit, target, reduction="none")  # [B, Q]
            sw = target + task.loss_weights[-1].float() * (1 - target)
            if task.object_weight_field is not None:
                flag = targets[task.target_object + "_" + task.object_weight_field].bool() & valid
                sw = torch.where(flag, sw * task.object_weight_value, sw)
            lw = float(task.losses.get("object_bce", 0.0))
            n = bce.numel()
            for t, m in {**types, "null": ~valid}.items():
                if m.any():
                    out[f"object_bce_{t}"] = bce[m].mean()
                    out[f"object_bce_{t}_w"] = lw * (sw * bce)[m].sum() / n
        elif isinstance(task, ObjectHitMaskTask) and task.name in lo:
            types, valid = _types(targets, task.target_object)
            if not valid.any():
                continue
            logit = lo[task.name][task.output_object_hit + "_logit"].float()  # [B, Q, H]
            target = targets[task.target_object_hit + "_" + task.target_field].float()
            hit_valid = targets[task.input_constituent + "_valid"].bool()
            hv = hit_valid.unsqueeze(1).expand_as(logit).float()
            prob = logit.sigmoid() * hv
            dice = 1 - (2 * (prob * target).sum(-1) + 1) / (prob.sum(-1) + target.sum(-1) + 1)  # [B, Q]
            p_t = prob * target + (1 - prob) * (1 - target)
            nh = hv.sum(-1).clamp_min(1.0)

            def focal(weight):
                ce = F.binary_cross_entropy_with_logits(logit, target, weight=weight, reduction="none") * hv
                return (ce * (1 - p_t) ** 2).sum(-1) / nh  # [B, Q]

            focal_plain = focal(None)
            sw = getattr(task, "pos_hit_weight", 1.0) * target + task.null_weight * (1 - target)
            focal_used = focal(sw)
            if task.object_weight_fn == "binary":
                field = targets[task.object_weight_field].float()
                w = 1 + (task.object_weight_alpha - 1) * (field > 0).float()
                w = w / w[valid].mean()
            else:  # unweighted path (log-radius weighting is not used by the quirk configs)
                w = torch.ones_like(dice)
            nv = valid.sum()
            for comp, plain, used in (("mask_dice", dice, dice), ("mask_focal", focal_plain, focal_used)):
                lw = float(task.losses.get(comp, 0.0))
                for t, m in types.items():
                    if m.any():
                        out[f"{comp}_{t}"] = plain[m].mean()
                        out[f"{comp}_{t}_w"] = lw * (w * used)[m].sum() / nv
    return out


@torch.no_grad()
def own_mask_quality(net, outputs, targets, layer="final", mask_threshold=0.5):
    """Own-mask completeness / purity of the matched query (mask prob >= threshold) per target type, averaged over
    targets of that type: {"compl_pair", "pur_pair", "compl_ord", "pur_ord"}."""
    out = {}
    for task in net.tasks:
        if isinstance(task, ObjectHitMaskTask) and task.name in outputs[layer]:
            types, _ = _types(targets, task.target_object)
            prob = outputs[layer][task.name][task.output_object_hit + "_logit"].float().sigmoid()
            hv = targets[task.input_constituent + "_valid"].bool().unsqueeze(1)
            pred = (prob >= mask_threshold) & hv
            true = targets[task.target_object_hit + "_" + task.target_field].bool() & hv
            tp = (pred & true).sum(-1).float()
            compl = tp / true.sum(-1).clamp_min(1)
            pur = tp / pred.sum(-1).clamp_min(1)
            for t, m in types.items():
                if m.any():
                    out[f"compl_{t}"] = compl[m].mean()
                    out[f"pur_{t}"] = pur[m].mean()
    return out
