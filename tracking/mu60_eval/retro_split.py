"""Retrospective train-vs-val check of a run's saved checkpoints: final-layer loss components split by target type
(split_losses.split_loss_terms), own-mask completeness / purity of the matched query per type, and the pair found
fraction (the writer's exclusive assignment, strict > 50% / > 50%, as val/coupled_strict), on a set of training events
the model saw and a set of val events. Inference only (GPU).

    python retro_split.py OUT_PARQUET TRAIN_A TRAIN_B VAL_A VAL_B LABEL=RUN_DIR[,RUN_DIR...] [LABEL=...]

For each LABEL the checkpoints are every trunk-step=*.ckpt under the given run dirs plus, for a run dir without
trunk-step files (a decay branch), its newest val checkpoint, labelled with the branch's final step (max_steps).
The model and data config come from the first run dir's config.yaml. Env RETRO_STEPS (e.g. "84000,120000,200000")
keeps only those steps.
"""
import glob
import os
import re
import sys
import time

import polars as pl
import torch
import yaml
from jsonargparse import ArgumentParser

from ml_tracking.experiments.quirk.data import QuirkTrackingDataset
from ml_tracking.experiments.quirk.model import QuirkModel
from ml_tracking.experiments.quirk.split_losses import own_mask_quality, split_loss_terms

out_path = sys.argv[1]
tr_a, tr_b, va_a, va_b = map(int, sys.argv[2:6])
specs = [a.split("=", 1) for a in sys.argv[6:]]


def checkpoints(run_dirs):
    found = []
    for rd in run_dirs:
        trunk = glob.glob(f"{rd}/*/*/version_0/checkpoints/trunk-step=*.ckpt")
        for c in trunk:
            found.append((int(re.search(r"trunk-step=0*(\d+)", c).group(1)), c))
        if not trunk:
            vals = sorted(glob.glob(f"{rd}/*/*/version_0/ckpts/*.ckpt"), key=os.path.getmtime)
            cfg = yaml.safe_load(open(sorted(glob.glob(f"{rd}/*/*/version_0/config.yaml"))[0]))
            found.append((int(cfg["trainer"]["max_steps"]), vals[-1]))
    keep = os.environ.get("RETRO_STEPS")  # optional comma-separated subset of steps
    if keep:
        wanted = {int(x) for x in keep.split(",")}
        found = [(st, c) for st, c in found if st in wanted]
    return sorted(dict(found).items())


def build(cfg_path):
    cfg = yaml.safe_load(open(cfg_path))
    parser = ArgumentParser()
    parser.add_class_arguments(QuirkModel, "model", fail_untyped=False)
    module = parser.instantiate_classes(parser.parse_object({"model": dict(cfg["model"], name="retro", init_from=None)})).model
    return module, cfg["data"]


def load_events(dcfg, a, b):
    sig = QuirkTrackingDataset.__init__.__code__.co_varnames
    kw = {k: v for k, v in dcfg.items() if k in sig and k not in ("event_range", "num_events")}
    kw.update(event_range=[a, b - 1], num_events=-1, hit_mask_dir=None)
    ds = QuirkTrackingDataset(**kw)
    return [QuirkTrackingDataset.collate_fn([ds[i]]) for i in range(len(ds))]


def pair_found(net, outputs, targets, ot=0.1):
    preds = net.predict(outputs)["final"]
    obj = preds["flow_valid"]["flow_valid_prob"].float()[0]
    mprob = preds["flow_sihit_assignment"]["flow_sihit_valid_prob"].float()[0]
    hv = targets["sihit_valid"][0]
    q = targets["particle_is_quirk"][0] & targets["particle_valid"][0]
    u = (targets["particle_sihit_valid"][0] & q.unsqueeze(-1)).any(0) & hv
    nu = int(u.sum())
    if nu < 3:
        return None
    best = mprob.masked_fill(~(obj >= ot).unsqueeze(-1), -1.0).max(0)
    assigned = (best.values >= 0.5) & hv
    onehot = torch.nn.functional.one_hot(best.indices, mprob.shape[0]).bool() & assigned.unsqueeze(-1)
    inter, size = (onehot & u.unsqueeze(-1)).sum(0), onehot.sum(0)
    return float(((2 * inter > nu) & (2 * inter > size)).any())


rows = []
checked = False
for label, dirs in specs:
    run_dirs = dirs.split(",")
    cks = checkpoints(run_dirs)
    module, dcfg = build(sorted(glob.glob(f"{run_dirs[0]}/*/*/version_0/config.yaml"))[0])
    net = module.model.cuda().eval()
    t0 = time.time()
    events = {"train": load_events(dcfg, tr_a, tr_b), "val": load_events(dcfg, va_a, va_b)}
    print(f"{label}: {len(cks)} checkpoints, events loaded in {time.time() - t0:.0f} s", flush=True)
    for step, ck in cks:
        net.load_state_dict({k[len("model."):]: v for k, v in torch.load(ck, map_location="cpu", weights_only=False)["state_dict"].items()
                             if k.startswith("model.")})
        for split, evs in events.items():
            acc = {}
            for inputs, targets in evs:
                inputs = {k: v.cuda(non_blocking=True) for k, v in inputs.items()}
                targets = {k: v.cuda(non_blocking=True) if torch.is_tensor(v) else v for k, v in targets.items()}
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    outputs = net(inputs)
                    outputs, targets, losses = net.loss(outputs, targets)
                vals = {**split_loss_terms(net, outputs, targets), **own_mask_quality(net, outputs, targets)}
                if not checked:  # consistency: the weighted split parts sum to the task losses actually used
                    fl = losses["final"]
                    for comp, task_name, parts in (("object_bce", "flow_valid", ("pair", "ord", "null")),
                                                   ("mask_dice", "flow_sihit_assignment", ("pair", "ord")),
                                                   ("mask_focal", "flow_sihit_assignment", ("pair", "ord"))):
                        tot = sum(float(vals.get(f"{comp}_{t}_w", 0.0)) for t in parts)
                        print(f"check {comp}: split sum {tot:.6g} vs task loss {float(fl[task_name][comp]):.6g}", flush=True)
                    checked = True
                pf = pair_found(net, outputs, targets)
                if pf is not None:
                    vals["pair_found"] = pf
                for k, v in vals.items():
                    acc.setdefault(k, []).append(float(v))
            row = {"run": label, "step": step, "split": split, "checkpoint": ck, "events": len(evs)}
            row.update({k: sum(v) / len(v) for k, v in acc.items()})
            rows.append(row)
            print(f"{label} step {step} {split}: pair found {row.get('pair_found', float('nan')):.3f}, "
                  f"mask_focal pair {row.get('mask_focal_pair', float('nan')):.4f} ord {row.get('mask_focal_ord', float('nan')):.4f}, "
                  f"dice pair {row.get('mask_dice_pair', float('nan')):.3f} ord {row.get('mask_dice_ord', float('nan')):.3f}", flush=True)
        pl.DataFrame(rows).write_parquet(out_path)
    del events, net, module
    torch.cuda.empty_cache()
print("done", flush=True)
