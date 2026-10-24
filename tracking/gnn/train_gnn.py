"""Stage 3: Interaction GNN edge classifier on the filtered graph.

Edge input features: pair geometry (dr, dphi, dz, deta, 3D distance) + filter score. BCE against truth-chain
labels, same-particle non-chain pairs masked. Val metrics at score 0.5 (overall / quirk / tech).

  train : python train_gnn.py train --cache C --graphs 'G/*.parquet' --out O --train A B --val A B
  infer : python train_gnn.py infer --cache C --graphs 'G/*.parquet' --out O --ranges A-B ...
          (writes O/scored/events*.parquet with the GNN score in 'score')
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
from common import (InteractionGNN, lrt_weights, DIRECTION, N_DIR_NODE, N_DIR_EDGE, N_FEAT, dir_edge_features, node_inputs, Logger, edge_counts, edge_truth, hit_features, load_events, lr_at,  # noqa: E402
                    pair_features, read_graphs, summarize_counts, write_graphs)

DEV = "cuda"
QW = 1.0
RECIPE = {"name": "all", "w_qq": 0.1, "fake_target": 0.10}
# cluster-shape inputs: 'node' = the 4 cluster columns of hit_features (size_loc0, size_loc1, n_channels,
# sum_activation; default, as in v1); 'none' = those columns zeroed; 'node_edge' = node + their differences
# across the edge as extra edge features. (local_eta/local_phi/eta_angle/phi_angle are all 0 in the data.)
CLUSTER = "node"
CL = slice(17, 21)
N_EDGE_IN = {"none": 6, "node": 6, "node_edge": 10}


def batch(evs, graphs, with_truth=True):
    """Concatenate events into one disjoint graph."""
    fs, ss, ts, efs, ys, ws, off = [], [], [], [], [], [], 0
    for ev in evs:
        g = graphs[ev["event_id"]]
        f = node_inputs(ev, DEV)
        if CLUSTER == "none":
            f[:, CL] = 0
        s = torch.as_tensor(g["src"], device=DEV).long()
        t = torch.as_tensor(g["dst"], device=DEV).long()
        sc = torch.as_tensor(g["score"], device=DEV).float()
        ef = [pair_features(f, s, t), sc[:, None]]
        if CLUSTER == "node_edge":
            ef.append(f[t, CL] - f[s, CL])
        if DIRECTION["on"]:
            ef.append(dir_edge_features(ev, s, t, DEV))
        efs.append(torch.cat(ef, 1))
        if with_truth:
            y, w = edge_truth(ev, g["src"], g["dst"])
            if RECIPE["name"] == "lrt":  # ROADMAP T1 / LRT recipe S: quirk true 10, other true 0.02, wrong 0.1
                w = lrt_weights(ev, g["src"], g["dst"], y, 10.0, 0.02, 0.1, RECIPE["w_qq"])
            elif QW != 1.0:
                hq = ev["hit_quirk"]
                w = w * np.where(hq[g["src"]] | hq[g["dst"]], QW, 1.0).astype(np.float32)
            ys.append(torch.as_tensor(y, device=DEV).float())
            ws.append(torch.as_tensor(w, device=DEV))
        fs.append(f)
        ss.append(s + off)
        ts.append(t + off)
        off += f.shape[0]
    out = [torch.cat(fs), torch.cat(ss), torch.cat(ts), torch.cat(efs)]
    if with_truth:
        out += [torch.cat(ys), torch.cat(ws)]
    return out


def loss_fn(lo, y, w, pw):
    return (F.binary_cross_entropy_with_logits(lo, y, reduction="none", pos_weight=pw) * w).sum() / w.sum()


@torch.no_grad()
def evaluate(model, events, graphs, pw, cuts=(0.5,)):
    model.eval()
    vl, cs = [], {c: [] for c in cuts}
    for ev in events:
        f, s, t, ef, y, w = batch([ev], graphs)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lo = model(f, s, t, ef).float()
        vl.append(float(loss_fn(lo, y, w, pw)))
        p = torch.sigmoid(lo)
        for c in cuts:
            k = p > c
            cs[c].append(edge_counts(ev, s[k], t[k], DEV))
    model.train()
    return float(np.mean(vl)), {c: summarize_counts(v) for c, v in cs.items()}


@torch.no_grad()
def track_val(model, events, graphs):
    """Quirk efficiency at the matched fake rate on val (ccwalk_stitch + min-hits scan; valtracks.py)."""
    import valtracks as vt
    model.eval()
    labels = {}
    for ev in events:
        f, s, t, ef = batch([ev], graphs, with_truth=False)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p = torch.sigmoid(model(f, s, t, ef).float()).cpu().numpy()
        gg = graphs[ev["event_id"]]
        labels[ev["event_id"]] = vt.labels_for(ev, gg["src"].astype(np.int64), gg["dst"].astype(np.int64), p)
    model.train()
    rows = vt.scan(events, labels)
    return vt.at_fake(rows, RECIPE["fake_target"]), rows


def build_model(hidden, n_iter, layers):
    d = DIRECTION["on"]
    return InteractionGNN(n_in=N_FEAT + (N_DIR_NODE if d else 0), n_edge_in=N_EDGE_IN[CLUSTER] + (N_DIR_EDGE if d else 0),
                          hidden=hidden, n_iter=n_iter, n_layers=layers).to(DEV)


def train(a):
    tr = load_events(a.cache, *a.train)
    va = load_events(a.cache, *a.val)[: a.n_val]
    g = read_graphs(a.graphs, a.train[0], a.train[1]) | read_graphs(a.graphs, a.val[0], a.val[1])
    tr = [e for e in tr if e["event_id"] in g]
    va = [e for e in va if e["event_id"] in g]
    print(f"train {len(tr)} val {len(va)} events", flush=True)
    torch.manual_seed(a.seed)
    random.seed(a.seed)
    model = build_model(a.hidden, a.n_iter, a.layers)
    print("params", sum(p.numel() for p in model.parameters()), flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    spe = len(tr) // a.batch
    total = spe * a.epochs
    state_path = os.path.join(a.out, "state.pt")
    step, best, start = 0, 1e9, 0
    best_trk = -1.0
    prev_rows = []
    if os.path.exists(state_path):  # resume after a time limit (same config, same epoch budget)
        st = torch.load(state_path, weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        step, best, start = st["step"], st["best"], st["epoch"] + 1
        best_trk = st.get("best_trk", -1.0)
        random.setstate(st["py_rng"])
        if os.path.exists(os.path.join(a.out, "log.csv")):
            import csv
            prev_rows = list(csv.DictReader(open(os.path.join(a.out, "log.csv"))))
        print(f"resuming at epoch {start}, step {step}", flush=True)
    log = Logger(a.out, a.name + (f"_resume{start}" if start else ""),
                 vars(a) | {"stage": "gnn", "n_train": len(tr), "n_val": len(va), "total_steps": total, "resume_epoch": start})
    log.rows = prev_rows
    pw = torch.tensor(a.pos_weight, device=DEV)
    for epoch in range(start, a.epochs):
        random.shuffle(tr)
        run, gns, t0 = [], [], time.time()
        for b in range(spe):
            lr = lr_at(step, total, a.lr, a.warmup)
            for gg in opt.param_groups:
                gg["lr"] = lr
            f, s, t, ef, y, w = batch(tr[b * a.batch:(b + 1) * a.batch], g)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lo = model(f, s, t, ef).float()
            loss = loss_fn(lo, y, w, pw)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gns.append(float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)))
            opt.step()
            run.append(float(loss))
            step += 1
        vl, m = evaluate(model, va, g, pw)
        trk = {}
        if a.track_val:
            qeff, rows = track_val(model, va[: a.n_track_val], g)
            trk = {"val_quirk_eff_at_fake": qeff, **{f"val_mh{r['min_hits']}_{k}": v for r in rows for k, v in r.items()
                                                      if k != "min_hits"}}
            torch.save(model.state_dict(), os.path.join(a.out, f"ep{epoch}.pt"))
            if qeff > best_trk:
                best_trk = qeff
                torch.save(model.state_dict(), os.path.join(a.out, "best_trk.pt"))
                json.dump({"epoch": epoch, "quirk_eff_at_fake": qeff, "fake_target": RECIPE["fake_target"], "scan": rows},
                          open(os.path.join(a.out, "best_trk.json"), "w"), indent=1)
        log.log({**trk, "epoch": epoch, "step": step, "lr": lr, "batch_events": a.batch, "train_loss": float(np.mean(run)),
                 "grad_norm": float(np.mean(gns)), "grad_norm_max": float(np.max(gns)), "val_loss": vl, **{"val_" + k: v for k, v in m[0.5].items()}, "epoch_time_s": time.time() - t0})
        torch.save(model.state_dict(), os.path.join(a.out, "last.pt"))
        if vl < best or not os.path.exists(os.path.join(a.out, "best.pt")):
            best = min(best, vl)
            torch.save(model.state_dict(), os.path.join(a.out, "best.pt"))
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "best": best, "best_trk": best_trk, "epoch": epoch,
                    "py_rng": random.getstate()}, state_path + ".tmp")
        os.replace(state_path + ".tmp", state_path)
    model.load_state_dict(torch.load(os.path.join(a.out, "best.pt")))
    vl, m = evaluate(model, va, g, pw, cuts=(0.1, 0.3, 0.5, 0.7, 0.9))
    scan = [dict(cut=c, **mm) for c, mm in m.items()]
    for row in scan:
        print("scan", json.dumps(row), flush=True)
    json.dump(scan, open(os.path.join(a.out, "cut_scan.json"), "w"), indent=1)
    log.finish({"cut_scan": scan})


@torch.no_grad()
def infer(a):
    cfg = json.load(open(os.path.join(a.out, "config.json")))
    global CLUSTER
    CLUSTER = cfg.get("cluster", "node")
    DIRECTION["on"] = bool(cfg.get("direction", False))
    model = build_model(cfg["hidden"], cfg["n_iter"], cfg["layers"])
    model.load_state_dict(torch.load(os.path.join(a.out, a.ckpt)))
    model.eval()
    print("checkpoint", a.ckpt, flush=True)
    for rng in a.ranges:
        first, last = map(int, rng.split("-"))
        evs = load_events(a.cache, first, last)
        g = read_graphs(a.graphs, first, last)
        for c0 in range(0, len(evs), 1000):
            chunk = evs[c0:c0 + 1000]
            rows, cs, t0 = [], [], time.time()
            for ev in chunk:
                f, s, t, ef = batch([ev], g, with_truth=False)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    p = torch.sigmoid(model(f, s, t, ef).float())
                k = p > 0.5
                cs.append(edge_counts(ev, s[k], t[k], DEV))
                gg = g[ev["event_id"]]
                rows.append(dict(event_id=ev["event_id"], src=gg["src"], dst=gg["dst"], y=gg["y"], score=p.cpu().numpy()))
            e0, e1 = chunk[0]["event_id"], chunk[-1]["event_id"]
            write_graphs(os.path.join(a.out, a.scored_dir, f"events{e0}-{e1}.parquet"), rows)
            print(f"gnn events {e0}-{e1} @0.5: {json.dumps(summarize_counts(cs))} ({time.time() - t0:.1f}s)", flush=True)


@torch.no_grad()
def select(a):
    """Re-score every saved epoch checkpoint (ep*.pt) on val with the track-level metric (quirk efficiency at
    the matched fake rate) and copy the best to best_sel.pt (selection independent of val loss)."""
    import glob
    import shutil
    cfg = json.load(open(os.path.join(a.out, "config.json")))
    global CLUSTER
    CLUSTER = cfg.get("cluster", "node")
    DIRECTION["on"] = bool(cfg.get("direction", False))
    model = build_model(cfg["hidden"], cfg["n_iter"], cfg["layers"])
    va = load_events(a.cache, *a.val)[: a.n_track_val]
    g = read_graphs(a.graphs, a.val[0], a.val[1])
    va = [e for e in va if e["event_id"] in g]
    out = []
    for f in sorted(glob.glob(os.path.join(a.out, "ep*.pt")), key=lambda p: int(os.path.basename(p)[2:-3])):
        model.load_state_dict(torch.load(f))
        q, rows = track_val(model, va, g)
        out.append({"ckpt": os.path.basename(f), "quirk_eff_at_fake": q, "scan": rows})
        print(json.dumps({"ckpt": out[-1]["ckpt"], "quirk_eff_at_fake": q,
                          "fake_range": [min(r["fake"] for r in rows), max(r["fake"] for r in rows)]}), flush=True)
    best = max(out, key=lambda d: d["quirk_eff_at_fake"])
    shutil.copy(os.path.join(a.out, best["ckpt"]), os.path.join(a.out, "best_sel.pt"))
    json.dump({"fake_target": RECIPE["fake_target"], "best": best["ckpt"], "all": out},
              open(os.path.join(a.out, "best_sel.json"), "w"), indent=1)
    print("selected", best["ckpt"], best["quirk_eff_at_fake"], flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train", "infer", "select"])
    ap.add_argument("--cache", required=True)
    ap.add_argument("--graphs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="gnn")
    ap.add_argument("--train", type=int, nargs=2, default=[0, 2999])
    ap.add_argument("--val", type=int, nargs=2, default=[3000, 3499])
    ap.add_argument("--n-val", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--n-iter", type=int, default=8)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--pos-weight", type=float, default=1.0)
    ap.add_argument("--quirk-weight", type=float, default=1.0)
    ap.add_argument("--recipe", choices=["all", "lrt"], default="all")
    ap.add_argument("--direction", action="store_true", help="Stage E cluster direction node + edge features")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--w-qq", type=float, default=0.1, help="lrt: weight of wrong edges joining the two quirks")
    ap.add_argument("--track-val", action="store_true", help="per-epoch track-level val; selects best_trk.pt")
    ap.add_argument("--n-track-val", type=int, default=200)
    ap.add_argument("--fake-target", type=float, default=0.10)
    ap.add_argument("--ckpt", default="best.pt", help="infer: checkpoint file in --out")
    ap.add_argument("--scored-dir", default="scored")
    ap.add_argument("--cluster", choices=["none", "node", "node_edge"], default="node")
    ap.add_argument("--ranges", nargs="+", default=[])
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    global QW, CLUSTER
    QW = a.quirk_weight
    CLUSTER = a.cluster
    RECIPE.update(name=a.recipe, w_qq=a.w_qq, fake_target=a.fake_target)
    DIRECTION["on"] = a.direction
    {"train": train, "infer": infer, "select": select}[a.mode](a)


if __name__ == "__main__":
    main()
