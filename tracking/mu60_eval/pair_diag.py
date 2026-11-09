"""Why is a quirk pair missed? Per val event, for the pair object (pair-target runs):

- indep_any:   some query's own mask (mask prob >= 0.5, before the exclusive hit assignment, any object score)
               strictly double-matches the pair (|T & U| > |U|/2 and > |T|/2);
- indep_valid: the same, among queries with object prob >= OT (the writer's object threshold);
- final:       the writer's exclusive assignment (each hit to the valid query with the highest mask prob >= 0.5)
               gives a track that strictly double-matches the pair (as val/coupled_strict);
- matched_obj / matched_eff / matched_pur: object prob and own-mask efficiency / purity of the query the loss's
               Hungarian matching assigns to the pair.

Missed pairs (final false) fall into: no query represents the pair (indep_any false: representation), a query
represents it but its object score is below OT (indep_any and not indep_valid: object score), or a valid query
represents it but loses hits in the exclusive assignment (indep_valid and not final: query competition).

    python pair_diag.py RUN_DIR A B OUT_PARQUET [OT]     (GPU; inclusive event range)
"""
import glob
import sys

import numpy as np
import polars as pl
import torch
import yaml
from jsonargparse import ArgumentParser

from ml_tracking.experiments.quirk.data import QuirkTrackingDataset
from ml_tracking.experiments.quirk.model import QuirkModel

run, a, b, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
ot = float(sys.argv[5]) if len(sys.argv) > 5 else 0.1
cfg = yaml.safe_load(open(sorted(glob.glob(f"{run}/*/*/version_0/config.yaml"))[0]))
ck = sorted(glob.glob(f"{run}/*/*/version_0/ckpts/*.ckpt"), key=lambda p: __import__("os").path.getmtime(p))[-1]
mcfg = dict(cfg["model"], name=cfg.get("name", "diag"), init_from=None)
parser = ArgumentParser()
parser.add_class_arguments(QuirkModel, "model", fail_untyped=False)
module = parser.instantiate_classes(parser.parse_object({"model": mcfg})).model
module.load_state_dict(torch.load(ck, map_location="cpu", weights_only=False)["state_dict"])
net = module.model.cuda().eval()
d = cfg["data"]
sig = QuirkTrackingDataset.__init__.__code__.co_varnames
kw = {k: v for k, v in d.items() if k in sig and k not in ("event_range", "num_events")}
kw.update(event_range=[a, b], num_events=-1, hit_mask_dir=None)
ds = QuirkTrackingDataset(**kw)
print(f"checkpoint {ck}; {len(ds)} events", flush=True)
rows = []
with torch.no_grad():
    for i in range(len(ds)):
        inputs, targets = QuirkTrackingDataset.collate_fn([ds[i]])
        inputs = {k: v.cuda() for k, v in inputs.items()}
        targets = {k: v.cuda() if torch.is_tensor(v) else v for k, v in targets.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = net(inputs)
            outputs, targets, _ = net.loss(outputs, targets)  # permutes queries to the target order
        preds = net.predict(outputs)["final"]
        obj = preds["flow_valid"]["flow_valid_prob"].float()[0]  # [Q]
        mprob = preds["flow_sihit_assignment"]["flow_sihit_valid_prob"].float()[0]  # [Q, H]
        hv = targets["sihit_valid"][0]
        q = targets["particle_is_quirk"][0] & targets["particle_valid"][0]
        if not q.any():
            continue
        u = (targets["particle_sihit_valid"][0] & q.unsqueeze(-1)).any(0) & hv  # [H]
        nu = int(u.sum())
        if nu < 3:
            continue
        own = (mprob >= 0.5) & hv  # [Q, H]
        inter = (own & u).sum(-1)
        size = own.sum(-1)
        strict = (2 * inter > nu) & (2 * inter > size)
        best = mprob.masked_fill(~(obj >= ot).unsqueeze(-1), -1.0).max(0)
        assigned = (best.values >= 0.5) & hv
        onehot = torch.nn.functional.one_hot(best.indices, mprob.shape[0]).bool() & assigned.unsqueeze(-1)  # [H, Q]
        fin_inter = (onehot & u.unsqueeze(-1)).sum(0)
        fin_size = onehot.sum(0)
        final = bool(((2 * fin_inter > nu) & (2 * fin_inter > fin_size)).any())
        j = int(torch.nonzero(q)[0])  # pair slot = its matched query after the permutation
        rows.append({"event_id": int(targets["event_id"].reshape(-1)[0]), "pair_hits": nu,
                     "indep_any": bool(strict.any()), "indep_valid": bool((strict & (obj >= ot)).any()), "final": final,
                     "matched_obj": float(obj[j]), "matched_eff": float(inter[j]) / nu,
                     "matched_pur": float(inter[j]) / max(int(size[j]), 1),
                     "best_indep_eff": float((inter.float() / nu).max())})
        if (i + 1) % 100 == 0:
            print(f"{i + 1} events", flush=True)
df = pl.DataFrame(rows)
df.write_parquet(out)
n = df.height
miss = df.filter(~pl.col("final"))
print(f"pairs {n}: final {df['final'].mean():.3f}, indep_valid {df['indep_valid'].mean():.3f}, indep_any {df['indep_any'].mean():.3f}")
print(f"missed {miss.height}: representation (no query) {(~miss['indep_any']).sum()}, object score below {ot} "
      f"{(miss['indep_any'] & ~miss['indep_valid']).sum()}, query competition {miss['indep_valid'].sum()}")
print(f"matched query: object prob median {df['matched_obj'].median():.3f} (missed {miss['matched_obj'].median():.3f}), "
      f"own-mask eff median {df['matched_eff'].median():.3f} (missed {miss['matched_eff'].median():.3f}), "
      f"best independent eff median among missed {miss['best_indep_eff'].median():.3f}")
