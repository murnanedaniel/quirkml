"""Shared mu 60 hit filter (GNN and MaskFormer read the same kept hits): edge-aware per-hit classifier.

Graph: each hit's k nearest hits (default 12) in the frozen mu 0 v1 metric-learning embedding (no truth, no time),
symmetrised. Inputs: hit features + cluster-direction node features; edge features: geometric pair features
(dr, dphi, dz, deta, d3), embedding distance, cluster-direction edge features. Model: node and edge encoders, n_iter
residual message-passing steps (sum aggregation), per-hit logit.
Target: keep = hit of a quirk (either quirk) OR hit of a target particle (charge != 0, >= 3 hits) with pT > 1 GeV.
Loss: BCE, quirk hits weighted --w-quirk, pT > 1 hits --w-pt, others 1.
Schedule (review-thread recipe): linear warmup, constant LR, linear decay to 0 over the last 20%; no early stopping;
the final checkpoint is used.

  python hitfilter.py train --cache C --train 3000 12999 --val 2000 2199 --steps 20000 --out OUT
  python hitfilter.py curve --cache C --events 2000 2999 --out OUT          (val trade-off curve, choose the cut)
  python hitfilter.py mask --cache C --r1-set SET --events 0 1999 --out OUT --mask-dir MASKS --split test
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "gnn"))
from common import (DIRECTION, N_DIR_EDGE, N_DIR_NODE, N_FEAT, Embedder, cache_files, dir_edge_features,  # noqa: E402
                    hit_features, load_events, mlp, node_inputs, pair_features)

DEV = "cuda"
EMB = "/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirk_tracking_data/models/gnn/v1/emb"


def labels(ev):
    q = ev["hit_quirk"]
    pt = (ev["hit_pt"] > 1.0) & ev["hit_target"] & ~q
    return q, pt


def knn_graph(e, k, chunk=4096):
    n = e.shape[0]
    src, dst = [], []
    for a in range(0, n, chunk):
        d = torch.cdist(e[a:a + chunk], e)
        idx = torch.topk(d, min(k + 1, n), dim=1, largest=False).indices[:, 1:]
        src.append(torch.arange(a, a + d.shape[0], device=e.device)[:, None].expand_as(idx).reshape(-1))
        dst.append(idx.reshape(-1))
    s, t = torch.cat(src), torch.cat(dst)
    lo, hi = torch.minimum(s, t), torch.maximum(s, t)
    key = torch.unique(lo * n + hi)
    s, t = key // n, key % n
    return torch.cat([s, t]), torch.cat([t, s])  # both directions


def samelayer_graph(ev, k, dmax=50.0):
    """k nearest hits on the same (volume, layer), within dmax mm (3D), both directions: the quirk partner hit is
    usually mm to cm away on the same layer (A0), which a pileup track rarely has along many layers."""
    pos = torch.stack([torch.as_tensor(ev[c], device=DEV) for c in "xyz"], 1)
    key = torch.as_tensor(ev["vol_idx"].astype(np.int64) * 1000 + ev["layer"].astype(np.int64), device=DEV)
    order = torch.argsort(key)
    ks = key[order]
    bounds = torch.nonzero(ks[1:] != ks[:-1]).flatten() + 1
    starts = torch.cat([torch.zeros(1, device=DEV, dtype=torch.long), bounds]).tolist()
    ends = torch.cat([bounds, torch.tensor([len(ks)], device=DEV)]).tolist()
    src, dst = [], []
    for a0, b0 in zip(starts, ends):
        idx = order[a0:b0]
        if len(idx) < 2:
            continue
        d = torch.cdist(pos[idx], pos[idx])
        d.fill_diagonal_(1e9)
        kk = min(k, len(idx) - 1)
        dist, j = torch.topk(d, kk, dim=1, largest=False)
        m = dist < dmax
        src.append(idx[:, None].expand_as(j)[m]); dst.append(idx[j][m])
    if not src:
        z = torch.zeros(0, dtype=torch.long, device=DEV)
        return z, z
    s, t = torch.cat(src), torch.cat(dst)
    n = len(ev["x"])
    lo, hi = torch.minimum(s, t), torch.maximum(s, t)
    key2 = torch.unique(lo * n + hi)
    s, t = key2 // n, key2 % n
    return torch.cat([s, t]), torch.cat([t, s])


class HitGNN(nn.Module):
    def __init__(self, n_in, n_edge_in, hidden=96, n_iter=4, n_layers=2):
        super().__init__()
        self.node_enc = mlp(n_in, hidden, hidden, n_layers)
        self.edge_enc = mlp(2 * hidden + n_edge_in, hidden, hidden, n_layers)
        self.edge_net = nn.ModuleList([mlp(3 * hidden, hidden, hidden, n_layers) for _ in range(n_iter)])
        self.node_net = nn.ModuleList([mlp(2 * hidden, hidden, hidden, n_layers) for _ in range(n_iter)])
        self.out = mlp(hidden, hidden, 1, n_layers)

    def forward(self, f, s, t, ef):
        h = self.node_enc(f)
        e = self.edge_enc(torch.cat([h[s], h[t], ef], 1))
        for en, nn_ in zip(self.edge_net, self.node_net):
            e = e + en(torch.cat([h[s], h[t], e], 1))
            agg = torch.zeros_like(h).index_add_(0, t, e.to(h.dtype))
            h = h + nn_(torch.cat([h, agg], 1))
        return self.out(h).squeeze(-1)


class Prep:
    def __init__(self, k, k_layer=0):
        self.k_layer = k_layer
        cfg = json.load(open(os.path.join(EMB, "config.json")))
        self.emb = Embedder(hidden=cfg["hidden"], n_layers=cfg["layers"], emb_dim=cfg["emb_dim"]).to(DEV)
        self.emb.load_state_dict(torch.load(os.path.join(EMB, "best.pt"))); self.emb.eval()
        self.k = k

    @torch.no_grad()
    def __call__(self, ev):
        DIRECTION["on"] = True
        f = node_inputs(ev, DEV)
        e = self.emb(hit_features(ev, DEV))
        s, t = knn_graph(e, self.k)
        typ = torch.zeros(len(s), 1, device=DEV)
        if self.k_layer:
            s2, t2 = samelayer_graph(ev, self.k_layer)
            s, t = torch.cat([s, s2]), torch.cat([t, t2])
            typ = torch.cat([typ, torch.ones(len(s2), 1, device=DEV)])
        pos = torch.stack([torch.as_tensor(ev[c], device=DEV) for c in "xyz"], 1)
        ef = torch.cat([pair_features(f, s, t), (e[s] - e[t]).norm(dim=1, keepdim=True), dir_edge_features(ev, s, t, DEV)], 1)
        if self.k_layer:
            ef = torch.cat([ef, typ, torch.log1p((pos[s] - pos[t]).norm(dim=1, keepdim=True))], 1)
        return f, s, t, ef


def lr_at(step, total, peak, warm):
    if step < warm:
        return peak * (step + 1) / warm
    d0 = int(0.8 * total)
    return peak if step < d0 else peak * max(0.0, (total - step) / (total - d0))


@torch.no_grad()
def evaluate(model, prep, evs):
    model.eval()
    P, Q, T, N = [], [], [], []
    for ev in evs:
        f, s, t, ef = prep(ev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p = torch.sigmoid(model(f, s, t, ef).float()).cpu().numpy()
        q, pt = labels(ev)
        P.append(p); Q.append(q); T.append(pt); N.append(len(p))
    model.train()
    return P, Q, T, N


def curve(P, Q, T, N, cuts):
    p, q, t = np.concatenate(P), np.concatenate(Q), np.concatenate(T)
    rows = []
    for c in cuts:
        k = p >= c
        rows.append(dict(cut=float(c), quirk_recall=float(k[q].mean()), pt1_recall=float(k[t].mean()),
                         kept_per_event=float(k.sum() / len(N)), kept_frac=float(k.mean())))
    return rows


def train(a):
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    prep = Prep(a.k, a.k_layer)
    files = [f for f in cache_files(a.cache) if a.train[0] <= int(os.path.basename(f)[6:].split("-")[0]) <= a.train[1]]
    va = load_events(a.cache, *a.val)
    model = HitGNN(N_FEAT + N_DIR_NODE, 5 + 1 + N_DIR_EDGE + (2 if a.k_layer else 0), a.hidden, a.n_iter).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    os.makedirs(a.out, exist_ok=True)
    json.dump(vars(a), open(os.path.join(a.out, "config.json"), "w"), indent=1)
    import wandb
    run = wandb.init(project="quirk-tracking", entity="murnanedaniel", name=a.name, config=vars(a), dir=a.out, mode="online")
    log = open(os.path.join(a.out, "log.jsonl"), "a")
    step, t0, ls = 0, time.time(), []
    rng = np.random.default_rng(a.seed)
    while step < a.steps:
        for fi in rng.permutation(len(files)):
            evs = load_events(a.cache, *map(int, os.path.basename(files[fi])[6:-4].split("-")))
            for j in rng.permutation(len(evs)):
                ev = evs[j]
                f, s, t, ef = prep(ev)
                q, pt = labels(ev)
                y = torch.as_tensor(q | pt, device=DEV).float()
                w = torch.as_tensor(np.where(q, a.w_quirk, np.where(pt, a.w_pt, 1.0)), device=DEV).float()
                for g in opt.param_groups:
                    g["lr"] = lr_at(step, a.steps, a.lr, a.warmup)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    lo = model(f, s, t, ef).float()
                loss = (nn.functional.binary_cross_entropy_with_logits(lo, y, reduction="none") * w).sum() / w.sum()
                opt.zero_grad(); loss.backward()
                gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)); opt.step()
                ls.append(float(loss)); step += 1
                if step % a.val_every == 0 or step == a.steps:
                    P, Q, T, N = evaluate(model, prep, va)
                    cv = curve(P, Q, T, N, [0.01, 0.02, 0.05, 0.1])
                    vq = np.concatenate(Q); vt = np.concatenate(T); vp = np.concatenate(P)
                    # working point at 99% quirk-hit recall (val)
                    cq = float(np.quantile(vp[vq], 0.01)); kq = vp >= cq
                    rec = dict(step=step, train_loss=float(np.mean(ls)), lr=lr_at(step - 1, a.steps, a.lr, a.warmup), grad_norm=gn,
                               cut_q99=cq, pt1_recall_at_q99=float(kq[vt].mean()), kept_per_event_at_q99=float(kq.sum() / len(N)),
                               elapsed_s=time.time() - t0, **{f"c{r['cut']:g}_{k}": v for r in cv for k, v in r.items() if k != "cut"})
                    run.log(rec); log.write(json.dumps(rec) + "\n"); log.flush(); print(json.dumps(rec), flush=True)
                    torch.save(model.state_dict(), os.path.join(a.out, "last.pt")); ls = []
                if step >= a.steps:
                    break
            if step >= a.steps:
                break
    torch.save(model.state_dict(), os.path.join(a.out, "final.pt"))
    run.finish()


def load_model(out):
    cfg = json.load(open(os.path.join(out, "config.json")))
    kl = cfg.get("k_layer", 0)
    m = HitGNN(N_FEAT + N_DIR_NODE, 5 + 1 + N_DIR_EDGE + (2 if kl else 0), cfg["hidden"], cfg["n_iter"]).to(DEV)
    m.load_state_dict(torch.load(os.path.join(out, "final.pt"))); m.eval()
    return m, Prep(cfg["k"], kl)


def do_curve(a):
    model, prep = load_model(a.out)
    evs = load_events(a.cache, *a.events)
    P, Q, T, N = evaluate(model, prep, evs)
    cuts = np.unique(np.concatenate([np.geomspace(1e-4, 0.5, 60)]))
    rows = curve(P, Q, T, N, cuts)
    vq, vt, vp = np.concatenate(Q), np.concatenate(T), np.concatenate(P)
    json.dump(dict(events=a.events, n_events=len(N), hits_per_event=float(np.mean(N)), rows=rows,
                   quirk_hits=int(vq.sum()), pt1_hits=int(vt.sum())), open(os.path.join(a.out, f"curve_{a.events[0]}_{a.events[1]}.json"), "w"), indent=1)
    for r in rows[::4]:
        print(json.dumps(r))


@torch.no_grad()
def do_mask(a):
    import pyarrow as pa
    import pyarrow.parquet as pq
    model, prep = load_model(a.out)
    os.makedirs(a.mask_dir, exist_ok=True)
    for b0 in range(a.events[0], a.events[1] + 1, 1000):
        b1 = b0 + 999
        path = os.path.join(a.mask_dir, f"{a.set}.hitmask.events{b0}-{b1}.parquet")
        if os.path.exists(path):
            continue
        evs = load_events(a.cache, b0, b1)
        P, Q, T, N = evaluate(model, prep, evs)
        ids = np.array([e["event_id"] for e in evs], np.uint32)
        offs = pa.array(np.concatenate([[0], np.cumsum(N)]).astype(np.int32))
        p = np.concatenate(P).astype(np.float32)
        tab = pa.table({"event_id": pa.array(ids), "keep": pa.ListArray.from_arrays(offs, pa.array(p >= a.cut)),
                        "keep_prob": pa.ListArray.from_arrays(offs, pa.array(p))})
        pq.write_table(tab, path + ".tmp", compression="zstd"); os.replace(path + ".tmp", path)
        k = p >= a.cut; q, t = np.concatenate(Q), np.concatenate(T)
        print(json.dumps(dict(block=f"{b0}-{b1}", events=len(N), kept_per_event=float(k.sum() / len(N)), quirk_recall=float(k[q].mean()),
                              pt1_recall=float(k[t].mean()))), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train", "curve", "mask"])
    ap.add_argument("--cache", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--train", type=int, nargs=2, default=[3000, 12999]); ap.add_argument("--val", type=int, nargs=2, default=[2000, 2099])
    ap.add_argument("--events", type=int, nargs=2); ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--hidden", type=int, default=96); ap.add_argument("--n-iter", type=int, default=4); ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--k-layer", type=int, default=0, help="add k nearest same-layer hits (< 50 mm) as a second edge type")
    ap.add_argument("--w-quirk", type=float, default=20.0); ap.add_argument("--w-pt", type=float, default=2.0)
    ap.add_argument("--val-every", type=int, default=2000); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default="pu60_hitfilter")
    ap.add_argument("--cut", type=float, default=0.0); ap.add_argument("--mask-dir"); ap.add_argument("--set", default="m200_L774_mu60")
    a = ap.parse_args()
    {"train": train, "curve": do_curve, "mask": do_mask}[a.mode](a)


if __name__ == "__main__":
    main()
