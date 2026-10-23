"""Shared pieces of the quirk GNN chain: cache I/O, features, models, graph utilities, metrics, logging.

Pure torch (no torch_geometric / torch_scatter): message passing uses index_add_, graphs use chunked
torch.cdist.
"""

import csv
import glob
import json
import math
import os
import pickle
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn as nn

VOLUMES = [16, 17, 18, 23, 24, 25, 28, 29, 30]
QUIRK_PDG = 10000100
OUT_ROOT = os.path.join(os.environ.get("PSCRATCH", "/pscratch/sd/d/danieltm"), "quirk_tracking", "gnn")
N_FEAT = 21


# ----------------------------------------------------------------------------- cache
def cache_files(cache_dir):
    return sorted(glob.glob(os.path.join(cache_dir, "events*.pkl")), key=lambda f: int(os.path.basename(f)[6:].split("-")[0]))


def load_events(cache_dir, first, last):
    """Events with first <= event_id <= last, in event_id order (list of dicts of numpy arrays)."""
    out = []
    for f in cache_files(cache_dir):
        a, b = os.path.basename(f)[6:-4].split("-")
        if int(b) < first or int(a) > last:
            continue
        with open(f, "rb") as fh:
            evs = pickle.load(fh)
        for ev in evs:
            if first <= ev["event_id"] <= last:
                out.append(ev)
    out.sort(key=lambda e: e["event_id"])
    return out


def hit_features(ev, device):
    """[N, 21] float tensor. Inputs: positions, volume one-hot, layer, cluster shape. No truth, no time."""
    x = torch.as_tensor(ev["x"], device=device)
    y = torch.as_tensor(ev["y"], device=device)
    z = torch.as_tensor(ev["z"], device=device)
    r = torch.hypot(x, y)
    phi = torch.atan2(y, x)
    eta = torch.asinh(z / r.clamp_min(1e-3))
    vol = torch.as_tensor(ev["vol_idx"], device=device).long()
    onehot = torch.nn.functional.one_hot(vol, len(VOLUMES)).float()
    lay = torch.as_tensor(ev["layer"], device=device).float() / 20.0
    cl = torch.as_tensor(ev["cluster"], device=device).float()  # size0, size1, nch, sum_act
    clf = torch.stack([torch.log1p(cl[:, 0]), torch.log1p(cl[:, 1]), torch.log1p(cl[:, 2]), torch.log1p(cl[:, 3])], 1)
    f = torch.cat([torch.stack([r / 1000.0, torch.cos(phi), torch.sin(phi), z / 3000.0, eta / 3.0, x / 1000.0, y / 1000.0], 1),
                   onehot, lay[:, None], clf], 1)
    return f


def pair_features(f, src, dst):
    """Hand-made geometric pair features: dr, dphi (wrapped), dz, deta, 3D distance."""
    r1, r2 = f[src, 0], f[dst, 0]
    phi1 = torch.atan2(f[src, 2], f[src, 1])
    phi2 = torch.atan2(f[dst, 2], f[dst, 1])
    dphi = torch.remainder(phi2 - phi1 + math.pi, 2 * math.pi) - math.pi
    dz = f[dst, 3] - f[src, 3]
    deta = f[dst, 4] - f[src, 4]
    dxy = torch.hypot(f[dst, 5] - f[src, 5], f[dst, 6] - f[src, 6])
    d3 = torch.sqrt(dxy ** 2 + (3 * dz) ** 2)
    return torch.stack([r2 - r1, dphi, dz, deta, d3], 1)


def edge_truth(ev, src, dst):
    """Labels for undirected edge list (numpy). y=1: truth chain edge of a target particle.
    w=0: same-particle but not a chain edge (masked in losses, acorn convention)."""
    n = len(ev["pid"])
    chain = ev["chain"].astype(np.int64)
    key = np.minimum(chain[0], chain[1]) * n + np.maximum(chain[0], chain[1])
    ekey = np.minimum(src, dst).astype(np.int64) * n + np.maximum(src, dst)
    y = np.isin(ekey, key)
    pid = ev["pid"]
    same = pid[src] == pid[dst]
    w = np.where(same & ~y, 0.0, 1.0).astype(np.float32)
    return y, w


def chain_categories(ev):
    """Masks over chain edges: quirk, and 'tech' (non-quirk particle with pT>1 GeV and >=7 hits)."""
    c = ev["chain"]
    q = ev["hit_quirk"][c[0]]
    tech = (~q) & (ev["hit_pt"][c[0]] > 1.0) & (ev["hit_nhits"][c[0]] >= 7)
    return q, tech


# ----------------------------------------------------------------------------- models
def mlp(i, h, o, n_layers, act=nn.SiLU, norm=True, out_act=False):
    layers, d = [], i
    for _ in range(n_layers):
        layers.append(nn.Linear(d, h))
        if norm:
            layers.append(nn.LayerNorm(h))
        layers.append(act())
        d = h
    layers.append(nn.Linear(d, o))
    if out_act:
        layers.append(act())
    return nn.Sequential(*layers)


class Embedder(nn.Module):
    def __init__(self, n_in=N_FEAT, hidden=256, n_layers=4, emb_dim=12):
        super().__init__()
        self.net = mlp(n_in, hidden, emb_dim, n_layers)

    def forward(self, f):
        return self.net(f)


class EdgeFilter(nn.Module):
    def __init__(self, n_in=N_FEAT, hidden=512, n_layers=3):
        super().__init__()
        self.net = mlp(2 * n_in + 5, hidden, 1, n_layers)

    def forward(self, f, src, dst):
        return self.net(torch.cat([f[src], f[dst], pair_features(f, src, dst)], 1)).squeeze(-1)


class InteractionGNN(nn.Module):
    """acorn InteractionGNN2-style: node/edge encoders, n_iter residual message-passing steps with
    separate in/out aggregation (sum), edge classifier on the final edge state. Undirected edges are
    used in both directions through the in/out aggregation."""

    def __init__(self, n_in=N_FEAT, n_edge_in=6, hidden=128, n_iter=8, n_layers=2):
        super().__init__()
        self.n_iter = n_iter
        self.node_enc = mlp(n_in, hidden, hidden, n_layers)
        self.edge_enc = mlp(2 * hidden + n_edge_in, hidden, hidden, n_layers)
        self.edge_net = mlp(3 * hidden, hidden, hidden, n_layers)
        self.node_net = mlp(3 * hidden, hidden, hidden, n_layers)
        self.out = mlp(3 * hidden, hidden, 1, n_layers)

    def forward(self, f, src, dst, efeat):
        h = self.node_enc(f)
        e = self.edge_enc(torch.cat([h[src], h[dst], efeat], 1))
        n = h.shape[0]
        for _ in range(self.n_iter):
            agg_in = torch.zeros_like(h).index_add_(0, dst, e)
            agg_out = torch.zeros_like(h).index_add_(0, src, e)
            h = h + self.node_net(torch.cat([h, agg_in, agg_out], 1))
            e = e + self.edge_net(torch.cat([h[src], h[dst], e], 1))
        return self.out(torch.cat([h[src], h[dst], e], 1)).squeeze(-1)


# ----------------------------------------------------------------------------- graphs
@torch.no_grad()
def radius_graph(emb, r, k_max, chunk=4096):
    """Undirected i<j pairs with |emb_i - emb_j| < r, keeping at most k_max nearest per node (each node's
    list; union over both ends). Returns src, dst (long), and the fraction of nodes whose k_max was saturated."""
    n = emb.shape[0]
    srcs, dsts, sat = [], [], 0
    k = min(k_max + 1, n)
    for a in range(0, n, chunk):
        d = torch.cdist(emb[a:a + chunk], emb)
        dist, idx = torch.topk(d, k, dim=1, largest=False)
        ok = dist < r
        sat += int(ok[:, -1].sum()) if k == k_max + 1 else 0
        rows = torch.arange(a, a + d.shape[0], device=emb.device)[:, None].expand_as(idx)
        m = ok & (idx != rows)
        srcs.append(rows[m])
        dsts.append(idx[m])
    s = torch.cat(srcs)
    t = torch.cat(dsts)
    lo, hi = torch.minimum(s, t), torch.maximum(s, t)
    key = torch.unique(lo * n + hi)
    return key // n, key % n, sat / max(n, 1)


# ----------------------------------------------------------------------------- graph files
def write_graphs(path, rows):
    """rows: list of dicts with event_id, src, dst, score (optional), y. One parquet row per event."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cols = {"event_id": pa.array([r["event_id"] for r in rows], pa.uint32()),
            "src": pa.array([r["src"].astype(np.int32) for r in rows], pa.list_(pa.int32())),
            "dst": pa.array([r["dst"].astype(np.int32) for r in rows], pa.list_(pa.int32())),
            "y": pa.array([r["y"].astype(bool) for r in rows], pa.list_(pa.bool_()))}
    if "score" in rows[0]:
        cols["score"] = pa.array([r["score"].astype(np.float32) for r in rows], pa.list_(pa.float32()))
    pq.write_table(pa.table(cols), path)


def read_graphs(path_glob, first=None, last=None):
    """{event_id: {src, dst, y[, score]} numpy arrays} (zero-copy list flattening, no Python lists)."""
    out = {}
    for f in sorted(glob.glob(path_glob)):
        t = pq.read_table(f)
        eids = t.column("event_id").to_numpy()
        sel = np.ones(len(eids), bool) if first is None else (eids >= first) & (eids <= last)
        if not sel.any():
            continue
        cols = {}
        for k in t.column_names:
            if k == "event_id":
                continue
            arr = t.column(k).combine_chunks()
            cols[k] = (arr.offsets.to_numpy(), arr.values.to_numpy(zero_copy_only=False))
        for i in np.flatnonzero(sel):
            out[int(eids[i])] = {k: v[o[i]:o[i + 1]] for k, (o, v) in cols.items()}
    return out


# ----------------------------------------------------------------------------- logging
class Logger:
    """CSV + W&B (online; offline fallback, never disabled)."""

    def __init__(self, out_dir, name, config):
        os.makedirs(out_dir, exist_ok=True)
        self.csv_path = os.path.join(out_dir, "log.csv")
        self.fields = None
        self.rows = []
        self.t0 = time.time()
        with open(os.path.join(out_dir, "config.json"), "w") as fh:
            json.dump(config, fh, indent=1)
        import wandb
        os.environ.setdefault("WANDB_MODE", "online")
        try:
            self.run = wandb.init(project="quirk-tracking", entity="murnanedaniel", name=name, config=config,
                                  dir=out_dir, mode="online")
        except Exception as exc:  # offline fallback (allowed); never disabled
            print("wandb online failed, falling back to offline:", exc, flush=True)
            self.run = wandb.init(project="quirk-tracking", entity="murnanedaniel", name=name, config=config,
                                  dir=out_dir, mode="offline")

    def log(self, d):
        d = dict(d)
        d["wall_s"] = round(time.time() - self.t0, 1)
        self.run.log(d)
        self.rows.append(d)
        keys = []
        for r in self.rows:
            keys += [k for k in r if k not in keys]
        with open(self.csv_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(self.rows)
        print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in d.items()}), flush=True)

    def finish(self, summary=None):
        if summary:
            for k, v in summary.items():
                self.run.summary[k] = v
        self.run.finish()


def lr_at(step, total, base, warmup):
    if step < warmup:
        return base * (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    return base * 0.5 * (1 + math.cos(math.pi * min(1.0, p)))


# ----------------------------------------------------------------------------- metrics
def edge_counts(ev, src, dst, device):
    """Counts for one event's undirected graph (src, dst torch long on device) against the truth chain.
    Returns dict of ints: edges, true (chain), samepid, chain_all/quirk/tech totals and found."""
    n = len(ev["pid"])
    ch = torch.as_tensor(ev["chain"], device=device).long()
    ck = torch.minimum(ch[0], ch[1]) * n + torch.maximum(ch[0], ch[1])
    ek = torch.minimum(src, dst) * n + torch.maximum(src, dst)
    found = torch.isin(ck, ek)
    is_true = torch.isin(ek, ck)
    pid = torch.as_tensor(ev["pid"], device=device)
    same = pid[src] == pid[dst]
    q, tech = chain_categories(ev)
    q = torch.as_tensor(q, device=device)
    tech = torch.as_tensor(tech, device=device)
    hq = torch.as_tensor(ev["hit_quirk"], device=device)
    return dict(edges=int(src.numel()), true=int(is_true.sum()), samepid=int(same.sum()),
                chain_all=int(ck.numel()), found_all=int(found.sum()),
                chain_quirk=int(q.sum()), found_quirk=int((found & q).sum()),
                chain_tech=int(tech.sum()), found_tech=int((found & tech).sum()),
                quirk_touch=int((hq[src] | hq[dst]).sum()), quirk_true=int((is_true & hq[src] & hq[dst]).sum()))


def summarize_counts(cs, prefix=""):
    s = {k: sum(c[k] for c in cs) for k in cs[0]}
    ne = max(len(cs), 1)
    return {prefix + "edges_per_event": s["edges"] / ne,
            prefix + "eff": s["found_all"] / max(s["chain_all"], 1),
            prefix + "eff_quirk": s["found_quirk"] / max(s["chain_quirk"], 1),
            prefix + "eff_tech": s["found_tech"] / max(s["chain_tech"], 1),
            prefix + "purity": s["true"] / max(s["edges"], 1),
            prefix + "pid_purity": s["samepid"] / max(s["edges"], 1),
            prefix + "purity_quirk": s["quirk_true"] / max(s["quirk_touch"], 1),
            prefix + "quirk_touch_edges_per_event": s["quirk_touch"] / ne}


# ----------------------------------------------------------------------------- LRT-style quirk-as-signal weights
def lrt_weights(ev, src, dst, y, w_quirk, w_true, w_wrong, w_qq):
    """Per-edge loss weights (numpy) for the quirk-as-signal recipe (ROADMAP T1):
    quirk true (chain) edges w_quirk, other true edges w_true, wrong edges w_wrong, wrong edges joining the two
    quirks w_qq; same-particle non-chain pairs 0 (masked, as before)."""
    pid, hq = ev["pid"], ev["hit_quirk"]
    same = pid[src] == pid[dst]
    q = hq[src] & hq[dst]
    w = np.full(len(src), w_wrong, np.float32)
    w[q & ~same] = w_qq
    w[same] = 0.0
    w[y & ~q] = w_true
    w[y & q] = w_quirk
    return w
