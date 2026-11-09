"""Edge filter retrained at mu 60, streamed over 1000-event cache blocks (the mu 60 train set does not fit in host
memory), on graphs built on the fly from a frozen metric-learning embedding. Same model, inputs and loss as
gnn/train_filter.py (BCE with pos_weight, edges touching a quirk hit weighted --quirk-weight, same-particle non-chain
edges masked), initialised from a mu 0 filter checkpoint. Per step: one event; every positive and every
quirk-touching edge is kept, other edges are subsampled to --max-edges and up-weighted by 1/fraction (unbiased loss).
Schedule (review-thread recipe): warmup, constant LR, linear decay to 0 over the last 20%; no early stopping, the
final checkpoint is used (final.pt). Val every --val-every steps: edges kept and truth-edge recall at several cuts.

  python train_filter_stream.py --cache C --emb EMB --radius R --init FILTER_DIR --train 3000 12999 --val 2000 2049
         --steps 10000 --out OUT
"""
import argparse
import json
import os
import shutil
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "gnn"))
import train_filter as tf  # noqa: E402
from common import DIRECTION, Embedder, cache_files, edge_counts, hit_features, load_events, node_inputs, radius_graph, summarize_counts  # noqa: E402

DEV = "cuda"


def lr_at(step, total, peak, warm):
    if step < warm:
        return peak * (step + 1) / warm
    d0 = int(0.8 * total)
    return peak if step < d0 else peak * max(0.0, (total - step) / (total - d0))


class Graphs:
    def __init__(self, emb_dir, radius, k_max):
        cfg = json.load(open(os.path.join(emb_dir, "config.json")))
        self.m = Embedder(hidden=cfg["hidden"], n_layers=cfg["layers"], emb_dim=cfg["emb_dim"]).to(DEV)
        self.m.load_state_dict(torch.load(os.path.join(emb_dir, "best.pt"))); self.m.eval()
        self.r, self.k = radius, k_max

    @torch.no_grad()
    def __call__(self, ev):
        s, t, _ = radius_graph(self.m(hit_features(ev, DEV)), self.r, self.k)
        return {"src": s.cpu().numpy(), "dst": t.cpu().numpy()}


@torch.no_grad()
def evaluate(model, graphs, evs, cuts):
    model.eval()
    cs = {c: [] for c in cuts}
    for ev in evs:
        s, t, _, _ = tf.prep(ev, graphs(ev))
        p = torch.sigmoid(tf.scores(model, ev, s, t))
        for c in cuts:
            k = p > c
            cs[c].append(edge_counts(ev, s[k], t[k], DEV))
    model.train()
    return {c: summarize_counts(v) for c, v in cs.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True); ap.add_argument("--emb", required=True)
    ap.add_argument("--radius", type=float, required=True); ap.add_argument("--k-max", type=int, default=512)
    ap.add_argument("--init", required=True, help="filter dir (config.json, best.pt) to start from")
    ap.add_argument("--train", type=int, nargs=2, default=[3000, 12999]); ap.add_argument("--val", type=int, nargs=2, default=[2000, 2049])
    ap.add_argument("--steps", type=int, default=10000); ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=300); ap.add_argument("--max-edges", type=int, default=3_000_000)
    ap.add_argument("--pos-weight", type=float, default=3.0); ap.add_argument("--quirk-weight", type=float, default=50.0)
    ap.add_argument("--val-every", type=int, default=1000); ap.add_argument("--cuts", type=float, nargs="+", default=[0.01, 0.05, 0.1, 0.2, 0.5])
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", required=True); ap.add_argument("--name", default="pu60_filter")
    a = ap.parse_args()
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    os.makedirs(a.out, exist_ok=True)
    icfg = json.load(open(os.path.join(a.init, "config.json")))
    DIRECTION["on"] = bool(icfg.get("direction", False))
    tf.QW = a.quirk_weight
    model = tf.build_model(icfg["hidden"], icfg["layers"])
    model.load_state_dict(torch.load(os.path.join(a.init, "best.pt")))
    cfg = dict(icfg); cfg.update(vars(a)); cfg.update(stage="filter_stream", emb_dir=a.emb, radius=a.radius, k_max=a.k_max)
    json.dump(cfg, open(os.path.join(a.out, "config.json"), "w"), indent=1)
    graphs = Graphs(a.emb, a.radius, a.k_max)
    va = load_events(a.cache, *a.val)
    files = [f for f in cache_files(a.cache) if a.train[0] <= int(os.path.basename(f)[6:].split("-")[0]) <= a.train[1]]
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    pw = torch.tensor(a.pos_weight, device=DEV)
    import wandb
    run = wandb.init(project="quirk-tracking", entity="murnanedaniel", name=a.name, config=cfg, dir=a.out, mode="online")
    log = open(os.path.join(a.out, "log.jsonl"), "a")
    m0 = evaluate(model, graphs, va, a.cuts)
    rec = dict(step=0, **{f"val_c{c:g}_{k}": v for c, mm in m0.items() for k, v in mm.items()})
    log.write(json.dumps(rec) + "\n"); log.flush(); print(json.dumps(rec), flush=True)
    step, ls, gns, t0 = 0, [], [], time.time()
    while step < a.steps:
        for fi in rng.permutation(len(files)):
            evs = load_events(a.cache, *map(int, os.path.basename(files[fi])[6:-4].split("-")))
            for j in rng.permutation(len(evs)):
                ev = evs[j]
                s, t, y, w = tf.prep(ev, graphs(ev))
                hq = torch.as_tensor(ev["hit_quirk"], device=DEV)
                must = (y > 0) | hq[s] | hq[t]
                other = torch.nonzero(~must).flatten()
                n_keep = max(a.max_edges - int(must.sum()), 0)
                if len(other) > n_keep:
                    frac = n_keep / len(other)
                    pick = other[torch.randperm(len(other), device=DEV)[:n_keep]]
                    idx = torch.cat([torch.nonzero(must).flatten(), pick])
                    w = w.clone(); w[pick] = w[pick] / frac
                    s, t, y, w = s[idx], t[idx], y[idx], w[idx]
                for g in opt.param_groups:
                    g["lr"] = lr_at(step, a.steps, a.lr, a.warmup)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    lo = model(node_inputs(ev, DEV), s, t, tf.extra(ev, s, t)).float()
                loss = (F.binary_cross_entropy_with_logits(lo, y, reduction="none", pos_weight=pw) * w).sum() / w.sum()
                opt.zero_grad(set_to_none=True); loss.backward()
                gns.append(float(torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0))); opt.step()
                ls.append(float(loss.detach())); step += 1
                if step % a.val_every == 0 or step == a.steps:
                    m = evaluate(model, graphs, va, a.cuts)
                    rec = dict(step=step, lr=lr_at(step - 1, a.steps, a.lr, a.warmup), train_loss=float(np.mean(ls)),
                               grad_norm=float(np.mean(gns)), grad_norm_max=float(np.max(gns)), elapsed_s=time.time() - t0,
                               **{f"val_c{c:g}_{k}": v for c, mm in m.items() for k, v in mm.items()})
                    run.log(rec); log.write(json.dumps(rec) + "\n"); log.flush(); print(json.dumps(rec), flush=True)
                    torch.save(model.state_dict(), os.path.join(a.out, "last.pt")); ls, gns = [], []
                if step >= a.steps:
                    break
            if step >= a.steps:
                break
    torch.save(model.state_dict(), os.path.join(a.out, "final.pt"))
    shutil.copy(os.path.join(a.out, "final.pt"), os.path.join(a.out, "best.pt"))  # downstream scripts load best.pt
    run.finish()


if __name__ == "__main__":
    main()
