"""Stage 2: edge filter. MLP on (hit_i features, hit_j features, pair geometry) over the metric-learning graph.
BCE against truth-chain labels; same-particle non-chain pairs masked (weight 0). Then a cut scan on val and
writing of the filtered graph (= the 'quirk-aware graph' for later models) with filter scores.

  train : python train_filter.py train --cache C --graphs 'G/*.parquet' --out O --train A B --val A B
  apply : python train_filter.py apply --cache C --graphs 'G/*.parquet' --out O --cut X --ranges A-B ...
"""

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (EdgeFilter, lrt_weights, DIRECTION, N_DIR_NODE, N_DIR_EDGE, N_FEAT, dir_edge_features, node_inputs, Embedder, radius_graph, Logger, edge_counts, edge_truth, hit_features, load_events, lr_at,  # noqa: E402
                    read_graphs, summarize_counts, write_graphs)

DEV = "cuda"


QW = 1.0
RECIPE = {"name": "all", "w_qq": 1.0}


class LazyGraphs:
    """Metric-learning graph computed on the fly from the embedding checkpoint (no graph files needed)."""

    def __init__(self, emb_dir, events, radius, k_max):
        cfg = json.load(open(os.path.join(emb_dir, "config.json")))
        self.model = Embedder(hidden=cfg["hidden"], n_layers=cfg["layers"], emb_dim=cfg["emb_dim"]).to(DEV)
        self.model.load_state_dict(torch.load(os.path.join(emb_dir, "best.pt")))
        self.model.eval()
        self.ev = {e["event_id"]: e for e in events}
        self.radius, self.k_max = radius, k_max

    def __contains__(self, eid):
        return eid in self.ev

    @torch.no_grad()
    def __getitem__(self, eid):
        s, t, _ = radius_graph(self.model(hit_features(self.ev[eid], DEV)), self.radius, self.k_max)
        return {"src": s.cpu().numpy(), "dst": t.cpu().numpy()}


def get_graphs(a, events, first, last):
    if a.emb_dir:
        return LazyGraphs(a.emb_dir, events, a.radius, a.k_max)
    return read_graphs(a.graphs, first, last)


def prep(ev, g):
    s = torch.as_tensor(g["src"], device=DEV).long()
    t = torch.as_tensor(g["dst"], device=DEV).long()
    y, w = edge_truth(ev, g["src"], g["dst"])
    if RECIPE["name"] == "lrt":  # ROADMAP T1: quirk true 10, other true 0, wrong 1, quirk-quirk wrong w_qq
        w = lrt_weights(ev, g["src"], g["dst"], y, 10.0, 0.0, 1.0, RECIPE["w_qq"])
    elif QW != 1.0:  # up-weight edges touching a quirk hit (quirks are ~1% of truth edges)
        hq = ev["hit_quirk"]
        w = w * np.where(hq[g["src"]] | hq[g["dst"]], QW, 1.0).astype(np.float32)
    return s, t, torch.as_tensor(y, device=DEV).float(), torch.as_tensor(w, device=DEV)


def extra(ev, s, t):
    return dir_edge_features(ev, s, t, DEV) if DIRECTION["on"] else None


def build_model(hidden, layers):
    d = DIRECTION["on"]
    return EdgeFilter(n_in=N_FEAT + (N_DIR_NODE if d else 0), hidden=hidden, n_layers=layers,
                      n_edge_extra=N_DIR_EDGE if d else 0).to(DEV)


def scores(model, ev, s, t, chunk=1_000_000):
    f = node_inputs(ev, DEV)
    out = []
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(0, s.numel(), chunk):
            ss, tt = s[i:i + chunk], t[i:i + chunk]
            out.append(model(f, ss, tt, extra(ev, ss, tt)).float())
    return torch.cat(out) if out else torch.zeros(0, device=DEV)


@torch.no_grad()
def evaluate(model, events, graphs, cuts, pos_weight):
    model.eval()
    cs = {c: [] for c in cuts}
    vl = []
    for ev in events:
        s, t, y, w = prep(ev, graphs[ev["event_id"]])
        lo = scores(model, ev, s, t)
        vl.append(float((F.binary_cross_entropy_with_logits(lo, y, reduction="none",
                                                           pos_weight=torch.tensor(pos_weight, device=DEV)) * w).sum() / w.sum()))
        p = torch.sigmoid(lo)
        for c in cuts:
            k = p > c
            cs[c].append(edge_counts(ev, s[k], t[k], DEV))
    model.train()
    return float(np.mean(vl)), {c: summarize_counts(v) for c, v in cs.items()}


def train(a):
    tr = load_events(a.cache, *a.train)
    va = load_events(a.cache, *a.val)[: a.n_val]
    g = get_graphs(a, tr + va, None, None) if a.emb_dir else (read_graphs(a.graphs, a.train[0], a.train[1]) | read_graphs(a.graphs, a.val[0], a.val[1]))
    tr = [e for e in tr if e["event_id"] in g]
    va = [e for e in va if e["event_id"] in g]
    print(f"train {len(tr)} val {len(va)} events", flush=True)
    torch.manual_seed(a.seed)
    random.seed(a.seed)
    model = build_model(a.hidden, a.layers)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    spe = len(tr) // a.batch
    total = spe * a.epochs
    log = Logger(a.out, a.name, vars(a) | {"stage": "filter", "n_train": len(tr), "n_val": len(va), "total_steps": total})
    pw = torch.tensor(a.pos_weight, device=DEV)
    step, best = 0, 1e9
    for epoch in range(a.epochs):
        random.shuffle(tr)
        run, gns, t0 = [], [], time.time()
        for b in range(spe):
            lr = lr_at(step, total, a.lr, a.warmup)
            for gg in opt.param_groups:
                gg["lr"] = lr
            loss = 0
            for ev in tr[b * a.batch:(b + 1) * a.batch]:
                s, t, y, w = prep(ev, g[ev["event_id"]])
                if a.max_edges and s.numel() > a.max_edges:  # random edge subsample per step
                    k = torch.randperm(s.numel(), device=DEV)[: a.max_edges]
                    s, t, y, w = s[k], t[k], y[k], w[k]
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    lo = model(node_inputs(ev, DEV), s, t, extra(ev, s, t)).float()
                loss = loss + (F.binary_cross_entropy_with_logits(lo, y, reduction="none", pos_weight=pw) * w).sum() / w.sum() / a.batch
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0))
            opt.step()
            run.append(float(loss))
            gns.append(gn)
            step += 1
        vl, m = evaluate(model, va, g, a.cuts, a.pos_weight)
        flat = {f"val_c{c:g}_{k}": v for c, mm in m.items() for k, v in mm.items()}
        log.log({"epoch": epoch, "step": step, "lr": lr, "batch_events": a.batch, "train_loss": float(np.mean(run)),
                 "grad_norm": float(np.mean(gns)), "grad_norm_max": float(np.max(gns)), "val_loss": vl, **flat, "epoch_time_s": time.time() - t0})
        torch.save(model.state_dict(), os.path.join(a.out, "last.pt"))
        if vl < best:
            best = vl
            torch.save(model.state_dict(), os.path.join(a.out, "best.pt"))
    model.load_state_dict(torch.load(os.path.join(a.out, "best.pt")))
    vl, m = evaluate(model, va, g, a.scan, a.pos_weight)
    scan = [dict(cut=c, **mm) for c, mm in m.items()]
    for row in scan:
        print("scan", json.dumps(row), flush=True)
    json.dump(scan, open(os.path.join(a.out, "cut_scan.json"), "w"), indent=1)
    log.finish({"cut_scan": scan})


@torch.no_grad()
def apply(a):
    cfg = json.load(open(os.path.join(a.out, "config.json")))
    DIRECTION["on"] = bool(cfg.get("direction", False))
    model = build_model(cfg["hidden"], cfg["layers"])
    model.load_state_dict(torch.load(os.path.join(a.out, "best.pt")))
    model.eval()
    gdir = os.path.join(a.out, f"graphs_cut{a.cut:g}")
    for rng in a.ranges:
        first, last = map(int, rng.split("-"))
        evs = load_events(a.cache, first, last)
        g = get_graphs(a, evs, first, last)
        for c0 in range(0, len(evs), 1000):
            chunk = evs[c0:c0 + 1000]
            rows, cs, t0 = [], [], time.time()
            for ev in chunk:
                s, t, _, _ = prep(ev, g[ev["event_id"]])
                p = torch.sigmoid(scores(model, ev, s, t))
                k = p > a.cut
                cs.append(edge_counts(ev, s[k], t[k], DEV))
                sn, tn = s[k].cpu().numpy(), t[k].cpu().numpy()
                y, _ = edge_truth(ev, sn, tn)  # noqa
                rows.append(dict(event_id=ev["event_id"], src=sn, dst=tn, y=y, score=p[k].cpu().numpy()))
            e0, e1 = chunk[0]["event_id"], chunk[-1]["event_id"]
            write_graphs(os.path.join(gdir, f"events{e0}-{e1}.parquet"), rows)
            print(f"filtered events {e0}-{e1}: {json.dumps(summarize_counts(cs))} ({time.time() - t0:.1f}s)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train", "apply"])
    ap.add_argument("--cache", required=True)
    ap.add_argument("--graphs", default=None, help="glob of metric-learning graph parquet files")
    ap.add_argument("--emb-dir", default=None, help="embedding run dir: build graphs on the fly instead of --graphs")
    ap.add_argument("--radius", type=float, default=0.07)
    ap.add_argument("--k-max", type=int, default=512)
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="filter")
    ap.add_argument("--train", type=int, nargs=2, default=[0, 2999])
    ap.add_argument("--val", type=int, nargs=2, default=[3000, 3499])
    ap.add_argument("--n-val", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--max-edges", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--pos-weight", type=float, default=2.0)
    ap.add_argument("--quirk-weight", type=float, default=1.0)
    ap.add_argument("--recipe", choices=["all", "lrt"], default="all")
    ap.add_argument("--direction", action="store_true", help="Stage E cluster direction node + edge features")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--w-qq", type=float, default=1.0)
    ap.add_argument("--cuts", type=float, nargs="+", default=[0.05, 0.2, 0.5])
    ap.add_argument("--scan", type=float, nargs="+", default=[0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5])
    ap.add_argument("--cut", type=float, default=0.05)
    ap.add_argument("--ranges", nargs="+", default=[])
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    global QW
    QW = a.quirk_weight
    RECIPE.update(name=a.recipe, w_qq=a.w_qq)
    DIRECTION["on"] = a.direction
    train(a) if a.mode == "train" else apply(a)


if __name__ == "__main__":
    main()
