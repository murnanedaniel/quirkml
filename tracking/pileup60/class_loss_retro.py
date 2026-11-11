"""mu60: per-class loss terms of saved checkpoints on training events vs val events (no training).

Full events (no phi crop) for both splits, the run's own loss weights (config.json), class_loss60.class_terms.
  python class_loss_retro.py RUN_DIR OUT.json [--steps 4000 12000 ...] [--train-blocks 3000 15000 30000 45000]
         [--per-block 50] [--val 2000 2199]
         [--unseen 60000 60999]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "gnn"))
sys.path.insert(0, HERE)
import train_gnn as T  # noqa: E402
from common import DIRECTION, load_events  # noqa: E402
import class_loss60 as CL  # noqa: E402
from train_gnn60 import read_graphs  # noqa: E402


def load_split(cache, graphs, ranges):
    evs, gr = [], {}
    for a, b in ranges:
        g = read_graphs(os.path.join(graphs, f"events{a // 1000 * 1000}-*.parquet"), a, b)
        e = [x for x in load_events(cache, a, b) if x["event_id"] in g]
        evs += e
        gr.update({x["event_id"]: g[x["event_id"]] for x in e})
    return evs, gr


@torch.no_grad()
def per_event_terms(model, evs, gr, pw, c):
    model.eval()
    rows = []
    for ev in evs:
        f, s, t, ef, y, w = T.batch([ev], gr)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lo = model(f, s, t, ef).float()
        hq = torch.as_tensor(ev["hit_quirk"], device=lo.device)
        qt = hq[s] | hq[t]
        mode = c.get("qw_mode") or "all"
        lm = c.get("loss_mode") or "weighted"
        w = CL.apply_qw_mode(w, qt, y, mode if lm == "weighted" else "all", c["quirk_weight"])
        rows.append(CL.class_terms(lo, y, w, qt, pw, lm, c.get("norm3_w") or (1.0, 1.0, 1.0)).cpu())
    return torch.stack(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("out")
    ap.add_argument("--steps", nargs="+", default=["4000", "12000", "20000", "28000", "36000", "44000", "52000", "60000"])
    ap.add_argument("--train-blocks", type=int, nargs="+", default=[3000, 15000, 30000, 45000])
    ap.add_argument("--per-block", type=int, default=50)
    ap.add_argument("--val", type=int, nargs=2, default=[2000, 2199])
    ap.add_argument("--unseen", type=int, nargs=2, default=None, help="training-range events the run never saw")
    ap.add_argument("--splits", nargs="+", default=["train", "val"], help="which of train / val / unseen to score")
    a = ap.parse_args()
    c = json.load(open(os.path.join(a.run, "config.json")))
    mode = c.get("qw_mode") or "all"
    T.QW = c["quirk_weight"] if mode == "all" else 1.0
    T.CLUSTER = c["cluster"]
    T.RECIPE.update(name="all", w_qq=0.1, fake_target=0.10, pair_all_true=c["pair_all_true"])
    DIRECTION["on"] = c["direction"]
    import gnn60_models
    gnn60_models.patch_build_model(T, c.get("agg") or "sum")
    model = T.build_model(c["hidden"], c["n_iter"], c["layers"])
    pw = torch.tensor(c["pos_weight"], device=T.DEV)

    t0 = time.time()
    ranges = {"train": [(b, b + a.per_block - 1) for b in a.train_blocks], "val": [tuple(a.val)],
              "unseen": [tuple(a.unseen)] if a.unseen else None}
    splits = {k: load_split(c["cache"], c["graphs"], ranges[k]) for k in a.splits}
    print({k: len(v[0]) for k, v in splits.items()}, f"loaded in {time.time() - t0:.0f} s", flush=True)
    res = {"run": a.run, "config": {k: c.get(k) for k in ("quirk_weight", "qw_mode", "loss_mode", "agg", "pos_weight")},
           "events": {k: [int(e["event_id"]) for e in v[0]] for k, v in splits.items()},
           "note": "full events, no crop; train events are from the run's training range", "steps": {}}
    for st in a.steps:
        ck = os.path.join(a.run, "final.pt" if st == "final" else f"step{st}.pt")
        model.load_state_dict(torch.load(ck, map_location=T.DEV))
        t1 = time.time()
        res["steps"][st] = {}
        for k, (evs, gr) in splits.items():
            pe = per_event_terms(model, evs, gr, pw, c)
            res["steps"][st][k] = {**CL.summarize(pe.sum(0), len(evs), ""), "per_event": pe.tolist()}
        r = res["steps"][st]
        print(st, " | ".join(f"{sp} " + " ".join(f"{k} {r[sp][k]:.4f}" for k in
                           ("bce_truepair", "bce_wrongquirk", "bce_ordinary", "qedge_eff", "qedge_pur")) for sp in r),
              f"({time.time() - t1:.0f} s)", flush=True)
        json.dump(res, open(a.out + ".tmp", "w"))
        os.replace(a.out + ".tmp", a.out)


if __name__ == "__main__":
    main()
