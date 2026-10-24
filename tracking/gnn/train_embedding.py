"""Stage 1: metric learning. MLP hit embedding, hinge loss (truth-chain pairs pulled in, kNN hard negatives and
random pairs pushed beyond the margin). Then a radius scan on val (edge efficiency vs edges/event, overall /
quirk / tech) and graph construction for a list of event ranges.

  train : python train_embedding.py train --cache C --out O --train A B --val A B [--epochs ...]
  graph : python train_embedding.py graph --cache C --out O --radius R --ranges A-B [A-B ...]
"""

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (Embedder, Logger, edge_counts, edge_truth, hit_features, load_events, lr_at,  # noqa: E402
                    radius_graph, summarize_counts, write_graphs)

DEV = "cuda"


RECIPE = {"name": "all", "w_qq": 1.0, "quirk_pos_weight": 1.0}


def event_loss(model, ev, margin, k_hard, n_rand_per_hit):
    """recipe 'all': positives = all truth-chain pairs, negatives = different-particle pairs (weight 1).
    recipe 'lrt' (ROADMAP T1): positives = quirk chain pairs only (other true pairs ignored); negatives =
    different-particle pairs weight 1, plus every pair joining the two quirks, weight w_qq. Each class is a
    weighted mean; loss = pos_weight * pos + neg (as acorn: classes averaged separately, then summed)."""
    f = hit_features(ev, DEV)
    emb = model(f)
    n = f.shape[0]
    ch = torch.as_tensor(ev["chain"], device=DEV).long()
    pid = torch.as_tensor(ev["pid"], device=DEV)
    hq = torch.as_tensor(ev["hit_quirk"], device=DEV)
    lrt = RECIPE["name"] == "lrt"
    if lrt:
        ch = ch[:, hq[ch[0]]]
    with torch.no_grad():
        d = torch.cdist(emb, emb)
        idx = torch.topk(d, min(k_hard + 1, n), dim=1, largest=False).indices[:, 1:]
        hs = torch.arange(n, device=DEV)[:, None].expand_as(idx).reshape(-1)
        hd = idx.reshape(-1)
        rs = torch.randint(0, n, (n * n_rand_per_hit,), device=DEV)
        rd = torch.randint(0, n, (n * n_rand_per_hit,), device=DEV)
        ns = torch.cat([hs, rs])
        nd = torch.cat([hd, rd])
        if lrt:  # all quirk-quirk pairs of different quirks
            qi = torch.nonzero(hq).squeeze(1)
            a, b = torch.meshgrid(qi, qi, indexing="ij")
            a, b = a.reshape(-1), b.reshape(-1)
            ns, nd = torch.cat([ns, a]), torch.cat([nd, b])
        neg = (ns != nd) & (pid[ns] != pid[nd])
        ns, nd = ns[neg], nd[neg]
        wn = torch.ones(ns.numel(), device=DEV)
        if lrt:
            wn[hq[ns] & hq[nd]] = RECIPE["w_qq"]
    dpos = (emb[ch[0]] - emb[ch[1]]).pow(2).sum(1)
    dneg = (emb[ns] - emb[nd]).pow(2).sum(1).clamp_min(1e-12).sqrt()
    if not lrt and RECIPE["quirk_pos_weight"] != 1.0 and dpos.numel():  # quirk chain pairs up-weighted
        wp = torch.where(hq[ch[0]], RECIPE["quirk_pos_weight"], 1.0)
        lpos = (dpos * wp).sum() / wp.sum()
    else:
        lpos = dpos.mean() if dpos.numel() else emb.sum() * 0
    lneg = (torch.relu(margin - dneg).pow(2) * wn).sum() / wn.sum()
    return lpos, lneg


@torch.no_grad()
def evaluate(model, events, radius, k_max):
    model.eval()
    cs, sats = [], []
    for ev in events:
        emb = model(hit_features(ev, DEV))
        s, t, sat = radius_graph(emb, radius, k_max)
        cs.append(edge_counts(ev, s, t, DEV))
        sats.append(sat)
    model.train()
    out = summarize_counts(cs)
    out["kmax_saturated_frac"] = float(np.mean(sats))
    return out


def train(a):
    tr = load_events(a.cache, *a.train)
    va = load_events(a.cache, *a.val)[: a.n_val]
    print(f"train {len(tr)} val {len(va)} events", flush=True)
    torch.manual_seed(0)
    random.seed(0)
    model = Embedder(hidden=a.hidden, n_layers=a.layers, emb_dim=a.emb_dim).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    steps_per_epoch = len(tr) // a.batch
    total = steps_per_epoch * a.epochs
    cfg = vars(a) | {"stage": "embedding", "n_train": len(tr), "n_val": len(va), "total_steps": total}
    log = Logger(a.out, a.name, cfg)
    step, best = 0, -1
    t0 = time.time()
    for epoch in range(a.epochs):
        random.shuffle(tr)
        run = []
        for b in range(steps_per_epoch):
            lr = lr_at(step, total, a.lr, a.warmup)
            for g in opt.param_groups:
                g["lr"] = lr
            lp, ln = 0, 0
            for ev in tr[b * a.batch:(b + 1) * a.batch]:
                p, q = event_loss(model, ev, a.margin, a.k_hard, a.n_rand)
                lp, ln = lp + p / a.batch, ln + q / a.batch
            loss = a.pos_weight * lp + ln
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            run.append([float(loss), float(lp), float(ln)])
            step += 1
        # val loss + graph metrics
        model.eval()
        with torch.no_grad():
            vl = []
            for ev in va:
                p, q = event_loss(model, ev, a.margin, a.k_hard, a.n_rand)
                vl.append(float(a.pos_weight * p + q))
        model.train()
        m = evaluate(model, va, a.margin, a.k_max)
        r = np.mean(run, 0)
        log.log({"epoch": epoch, "step": step, "lr": lr, "batch_events": a.batch, "train_loss": r[0],
                 "train_loss_pos": r[1], "train_loss_neg": r[2], "val_loss": float(np.mean(vl)),
                 **{"val_" + k: v for k, v in m.items()}, "epoch_time_s": time.time() - t0})
        t0 = time.time()
        torch.save(model.state_dict(), os.path.join(a.out, "last.pt"))
        score = m["eff"] if RECIPE["name"] == "all" else epoch  # lrt: last epoch (radius chosen on val quirk eff)
        if score > best:
            best = score
            torch.save(model.state_dict(), os.path.join(a.out, "best.pt"))
    # radius scan on val with the best checkpoint
    model.load_state_dict(torch.load(os.path.join(a.out, "best.pt")))
    scan = []
    for rf in a.scan:
        m = evaluate(model, va, rf * a.margin, a.k_max)
        m["radius"] = rf * a.margin
        scan.append(m)
        print("scan", json.dumps(m), flush=True)
    with open(os.path.join(a.out, "radius_scan.json"), "w") as fh:
        json.dump(scan, fh, indent=1)
    log.finish({"radius_scan": scan})


@torch.no_grad()
def graph(a):
    cfg = json.load(open(os.path.join(a.out, "config.json")))
    model = Embedder(hidden=cfg["hidden"], n_layers=cfg["layers"], emb_dim=cfg["emb_dim"]).to(DEV)
    model.load_state_dict(torch.load(os.path.join(a.out, "best.pt")))
    model.eval()
    gdir = os.path.join(a.out, f"graphs_r{a.radius:g}")
    for rng in a.ranges:
        first, last = map(int, rng.split("-"))
        evs = load_events(a.cache, first, last)
        for c0 in range(0, len(evs), 1000):
            chunk = evs[c0:c0 + 1000]
            rows, cs = [], []
            t0 = time.time()
            for ev in chunk:
                s, t, _ = radius_graph(model(hit_features(ev, DEV)), a.radius, a.k_max)
                cs.append(edge_counts(ev, s, t, DEV))
                s, t = s.cpu().numpy(), t.cpu().numpy()
                y, _ = edge_truth(ev, s, t)
                rows.append(dict(event_id=ev["event_id"], src=s, dst=t, y=y))
            e0, e1 = chunk[0]["event_id"], chunk[-1]["event_id"]
            write_graphs(os.path.join(gdir, f"events{e0}-{e1}.parquet"), rows)
            print(f"graph events {e0}-{e1}: {json.dumps(summarize_counts(cs))} ({time.time() - t0:.1f}s)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train", "graph"])
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="embedding")
    ap.add_argument("--train", type=int, nargs=2, default=[0, 2999])
    ap.add_argument("--val", type=int, nargs=2, default=[3000, 3499])
    ap.add_argument("--n-val", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--emb-dim", type=int, default=12)
    ap.add_argument("--margin", type=float, default=0.1)
    ap.add_argument("--pos-weight", type=float, default=3.0)
    ap.add_argument("--k-hard", type=int, default=32)
    ap.add_argument("--n-rand", type=int, default=4)
    ap.add_argument("--k-max", type=int, default=256)
    ap.add_argument("--scan", type=float, nargs="+", default=[0.5, 0.75, 1.0, 1.25, 1.5, 2.0])
    ap.add_argument("--radius", type=float, default=0.1)
    ap.add_argument("--ranges", nargs="+", default=[])
    ap.add_argument("--recipe", choices=["all", "lrt"], default="all")
    ap.add_argument("--w-qq", type=float, default=1.0, help="lrt: weight of negatives joining the two quirks")
    ap.add_argument("--quirk-pos-weight", type=float, default=1.0, help="recipe all: weight of quirk chain pairs among positives")
    a = ap.parse_args()
    a.emb_dim = a.emb_dim
    os.makedirs(a.out, exist_ok=True)
    RECIPE.update(name=a.recipe, w_qq=a.w_qq, quirk_pos_weight=a.quirk_pos_weight)
    train(a) if a.mode == "train" else graph(a)


if __name__ == "__main__":
    main()
