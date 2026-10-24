"""Stage D diagnostics for a quirk MaskFormer checkpoint (run on one GPU).

For each event: forward pass, final-layer Hungarian matching (scipy, asserted to be a permutation), then per valid
target: class (quirk / ordinary), its matched query, the query's object probability, the target's total matching cost
and mask cost, and the composition of the matched query's predicted mask (sigmoid > 0.5): own hits, hits of the other
quirk, hits of other particles. Writes one parquet row per target and prints class medians.

Usage: python diagnose_matching.py CKPT RUN_CONFIG FIRST LAST OUT_PARQUET [MAX_EVENTS]
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.optimize import linear_sum_assignment

from ml_tracking.experiments.quirk.data import QuirkTrackingDataset
from ml_tracking.experiments.quirk.model import QuirkModel

ckpt, run_cfg, first, last, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
max_events = int(sys.argv[6]) if len(sys.argv) > 6 else -1
cfg = yaml.safe_load(Path(run_cfg).read_text())["data"]
ds = QuirkTrackingDataset(
    cfg["dataset_root"], (first, last), num_events=max_events,
    event_max_num_particles=cfg["event_max_num_particles"], particle_min_pt=cfg["particle_min_pt"],
    particle_max_abs_eta=cfg["particle_max_abs_eta"], min_num_sihit=cfg["min_num_sihit"],
    particle_require_primary=cfg["particle_require_primary"], include_quirks=cfg["include_quirks"],
    quirk_pdg_ids=tuple(cfg["quirk_pdg_ids"]), quirk_min_num_sihit=cfg["quirk_min_num_sihit"],
    cluster_features=cfg["cluster_features"], row_group_cache_size=4,
    direction_dir=cfg.get("direction_dir"), quirk_pair_target=cfg.get("quirk_pair_target", False),
)
module = QuirkModel.load_from_checkpoint(ckpt, map_location="cuda")
module.eval()
net = module.model
dev = torch.device("cuda")

rows = []
nonperm = 0
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    for i in range(len(ds)):
        inputs, targets = QuirkTrackingDataset.collate_fn([ds[i]])
        inputs = {k: v.to(dev) for k, v in inputs.items()}
        targets = {k: v.to(dev) for k, v in targets.items()}
        outputs = net(inputs)
        final = outputs["final"]
        costs = net._compute_decoder_costs({"final": final}, targets)["final"][0].float()  # [Q, P]
        per_task = {t.name: t.cost(final[t.name], targets) for t in net.tasks}
        mask_cost = sum(c for c in per_task["flow_sihit_assignment"].values())[0].float()
        valid = targets["particle_valid"][0]
        tidx = torch.nonzero(valid).flatten().cpu().numpy()
        c = costs[:, tidx].cpu().numpy()
        c = np.nan_to_num(c, nan=1e6, posinf=1e6, neginf=-1e6)
        q_idx, t_pos = linear_sum_assignment(c.T)  # rows targets, cols queries
        if len(set(t_pos.tolist())) != len(t_pos):
            nonperm += 1
        q_for_target = dict(zip(tidx[q_idx], t_pos, strict=True))
        obj_prob = final["flow_valid"]["flow_logit"][0].float().sigmoid().cpu().numpy()
        mask_prob = final["flow_sihit_assignment"]["flow_sihit_logit"][0].float().sigmoid().cpu().numpy()
        true_masks = targets["particle_sihit_valid"][0].cpu().numpy()
        is_quirk = targets["particle_is_quirk"][0].cpu().numpy()
        quirk_rows = [t for t in tidx if is_quirk[t]]
        pt = targets["particle_pt"][0].cpu().numpy()
        any_target_hit = true_masks[tidx].any(0)
        # Writer-style hit assignment at object threshold 0.1 (mask 0.5): plain argmax of the mask probability
        # ("a1") and argmax of object prob x mask prob ("a2", competition resolved by the object score).
        vq = obj_prob >= 0.1
        assign = {}
        for name, score in (("a1", mask_prob), ("a2", mask_prob * obj_prob[:, None])):
            sc = np.where(vq[:, None], score, -1.0)
            best = sc.argmax(0)
            ok = vq.any() & (mask_prob[best, np.arange(mask_prob.shape[1])] >= 0.5)
            assign[name] = np.where(ok, best, -1)
        for t in tidx:
            q = q_for_target[t]
            m = mask_prob[q] > 0.5
            own = true_masks[t]
            other_q = np.zeros_like(own)
            for o in quirk_rows:
                if o != t:
                    other_q |= true_masks[o]
            n_pred = int(m.sum())
            rows.append({
                "event_id": int(targets["event_id"][0]), "target": int(t), "is_quirk": bool(is_quirk[t]),
                "pt": float(pt[t]), "n_hits": int(own.sum()), "query": int(q), "obj_prob": float(obj_prob[q]),
                "cost_total": float(costs[q, t]), "cost_mask": float(mask_cost[q, t]), "n_pred_hits": n_pred,
                "tp": int((m & own).sum()), "pred_other_quirk": int((m & other_q).sum()),
                "pred_other_target": int((m & any_target_hit & ~own & ~other_q).sum()),
                "pred_nontarget": int((m & ~any_target_hit).sum()),
                "other_quirk_hits": int(other_q.sum()),
                "stolen_a1": int((m & own & (assign["a1"] != q)).sum()),
                **{f"tp_{a}": int(((assign[a] == q) & own).sum()) for a in assign},
                **{f"n_{a}": int((assign[a] == q).sum()) for a in assign},
            })
df = pd.DataFrame(rows)
df["eff"] = df.tp / df.n_hits.clip(lower=1)
df["pur"] = df.tp / df.n_pred_hits.clip(lower=1)
df["dm"] = (df.eff >= 0.5) & (df.pur >= 0.5)
for a in ("a1", "a2"):
    e = df[f"tp_{a}"] / df.n_hits.clip(lower=1)
    pu = df[f"tp_{a}"] / df[f"n_{a}"].clip(lower=1)
    df[f"dm_{a}"] = (e >= 0.5) & (pu >= 0.5)
df.to_parquet(out)
print(f"events {df.event_id.nunique()}, targets {len(df)}, non-permutation matchings {nonperm}")
for name, g in df.groupby("is_quirk"):
    lab = "quirk" if name else "ordinary"
    print(f"[{lab}] n={len(g)} hits med {g.n_hits.median():.0f} | obj_prob med {g.obj_prob.median():.3f} "
          f"frac>0.1 {np.mean(g.obj_prob > 0.1):.3f} >0.5 {np.mean(g.obj_prob > 0.5):.3f} | cost_total med "
          f"{g.cost_total.median():.3f} cost_mask med {g.cost_mask.median():.3f} | matched mask: eff med "
          f"{g.eff.median():.2f} pur med {g.pur.median():.2f} empty {np.mean(g.n_pred_hits == 0):.3f} "
          f"double-majority {g.dm.mean():.3f}")
    print(f"   assigned track of the matched query (ot 0.1): double-majority plain argmax {g.dm_a1.mean():.3f}, "
          f"object-weighted argmax {g.dm_a2.mean():.3f}; own hits predicted by the matched query but assigned to "
          f"another query: {g.stolen_a1.sum() / max(g.tp.sum(), 1):.3f}")
    if name:
        wrong = (g.n_pred_hits - g.tp).clip(lower=1)
        print(f"   quirk matched-mask wrong hits: other quirk {g.pred_other_quirk.sum() / wrong.sum():.3f}, other targets "
              f"{g.pred_other_target.sum() / wrong.sum():.3f}, non-target hits {g.pred_nontarget.sum() / wrong.sum():.3f}; "
              f"mask holds >=50% of the other quirk: {np.mean(g.pred_other_quirk >= 0.5 * g.other_quirk_hits.clip(lower=1)):.3f}")
