"""mu60: per-class edge loss terms, shared by train_gnn60.py logging and class_loss_retro.py.

Classes (live edges only, i.e. base weight > 0, as in the norm3 loss):
  truepair   - quirk-touching edges with y = 1 (true pair edges under --pair-all-true)
  wrongquirk - quirk-touching edges with y = 0
  ordinary   - edges touching no quirk hit
For each class: the unweighted mean BCE, the weighted contribution to the loss actually optimised (the three sum to the
total loss), and on the quirk-touching edges the edge efficiency and purity at score 0.5.
"""
import torch
import torch.nn.functional as F

CLASSES = ("truepair", "wrongquirk", "ordinary")
N_TERMS = 11  # bce_sum x3, n x3, contrib x3, kept_truepair, kept_wrongquirk


@torch.no_grad()
def class_terms(lo, y, w, qt, pw, loss_mode="weighted", norm3_w=(1.0, 1.0, 1.0)):
    """lo logits, y targets, w per-edge weights as used in the loss, qt quirk-touching mask. Returns a 1-D tensor."""
    lo = lo.detach().float()
    live = w > 0
    bce = F.binary_cross_entropy_with_logits(lo, y, reduction="none")
    masks = [qt & (y > 0) & live, qt & (y == 0) & live, ~qt & live]
    sums = [bce[m].sum() for m in masks]
    ns = [m.sum().float() for m in masks]
    if loss_mode == "norm3":
        contrib = [wt * s / n.clamp(min=1.0) * (n > 0) for s, n, wt in zip(sums, ns, norm3_w)]
    else:
        bw = F.binary_cross_entropy_with_logits(lo, y, reduction="none", pos_weight=pw) * w
        W = w.sum().clamp(min=1e-12)
        contrib = [bw[m].sum() / W for m in masks]
    k = lo > 0
    kept = [(masks[0] & k).sum().float(), (masks[1] & k).sum().float()]
    return torch.stack(sums + ns + contrib + kept)


def apply_qw_mode(w, qt, y, mode, qw):
    """Same transform as train_gnn60.py for --qw-mode pos / neg (mode all is applied inside train_gnn.batch)."""
    if mode == "pos":
        return torch.where(qt & (y > 0), w * qw, w)
    if mode == "neg":
        return torch.where(qt & (y == 0), w * qw, w)
    return w


def summarize(acc, n_items, prefix):
    """acc: summed class_terms vectors over n_items steps / events. BCE is pooled over edges; contributions are the mean
    per-step (per-event) contribution; efficiency and purity are pooled."""
    a = acc.detach().double().cpu().tolist()
    out = {}
    for i, c in enumerate(CLASSES):
        out[f"{prefix}bce_{c}"] = a[i] / a[3 + i] if a[3 + i] > 0 else float("nan")
        out[f"{prefix}wloss_{c}"] = a[6 + i] / max(n_items, 1)
        out[f"{prefix}n_{c}_per_item"] = a[3 + i] / max(n_items, 1)
    out[f"{prefix}qedge_eff"] = a[9] / a[3] if a[3] > 0 else float("nan")
    out[f"{prefix}qedge_pur"] = a[9] / (a[9] + a[10]) if a[9] + a[10] > 0 else float("nan")
    return out
