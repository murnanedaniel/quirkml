"""A1 learned splitter, relational version: a transformer over the hits of one track whose attention carries a
learned bias from pair geometry (displacement, distance, angle of the displacement to each hit's cluster-axis
candidates, axis-axis angle, same layer), and a symmetric pair head (same particle or not). The affinity loss covers
each hit's k nearest hits (weight 1) and all other pairs (weight 0.1); a per-hit head flags "neither" hits. The split:
"neither" hits (P > 0.5) dropped, the rest cut in two by the sign of the leading eigenvector of the signed affinity
2 sigma(L) - 1 restricted to the k-nearest-neighbour pairs. Data from split_data.py (feature layout below).

Time arms (--time): none (headline), smear30 / smear100 (true sim-hit time + Gaussian 30 / 100 ps per hit),
true (unsmeared: upper-bound reference only). Selection: best val strict loose efficiency.

  python splitter2.py train --train T.pkl --val V.pkl --out DIR --time none
  python splitter2.py predict --data X.pkl --out DIR --base BASE/predictions.parquet --pred-out P
"""
import argparse, json, math, os, pickle, random, time
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F

DEV = "cuda"
F0 = 38  # base feature count (split_data.py); time column appended after
POS = slice(0, 3)   # x/1000, y/1000, z/3000
LAYERKEY = slice(14, 24)  # volume one-hot + layer
AXA, AXB, AXOK = slice(32, 35), slice(35, 38), 31
KNN = 8
PAIR_BUDGET = 400_000  # sum over a batch of n^2 (hit pairs) per forward pass


def budget_batches(order, lengths, cap=None):
    """Consecutive batches of the (length-sorted) order with B * n_max^2 <= PAIR_BUDGET (and <= cap tracks)."""
    out, cur = [], []
    for j in order:
        n = max([lengths[k] for k in cur] + [lengths[j]])
        if cur and ((len(cur) + 1) * n * n > PAIR_BUDGET or (cap and len(cur) >= cap)):
            out.append(cur); cur = []
        cur.append(j)
    if cur:
        out.append(cur)
    return out


def pair_features(X):
    """[B, n, n, 8] from the per-hit features."""
    p = X[..., POS] * torch.tensor([1000.0, 1000.0, 3000.0], device=X.device)
    d = p[:, None, :, :] - p[:, :, None, :]                       # d[i, j] = p_j - p_i
    dist = d.norm(dim=-1).clamp_min(1e-6)
    u = d / dist[..., None]
    A, B, ok = X[..., AXA], X[..., AXB], X[..., AXOK]

    def ang(ax):  # line angle between d and each hit's axis, for the hit at i
        c = (u * ax[:, :, None, :]).sum(-1).abs().clamp(max=1)
        return torch.arccos(c)
    ai = torch.minimum(ang(A), ang(B)) * ok[:, :, None]
    aj = ai.transpose(1, 2)
    cab = torch.stack([(A[:, :, None] * A[:, None]).sum(-1), (A[:, :, None] * B[:, None]).sum(-1),
                       (B[:, :, None] * A[:, None]).sum(-1), (B[:, :, None] * B[:, None]).sum(-1)], -1).abs().amax(-1).clamp(max=1)
    aab = torch.arccos(cab) * (ok[:, :, None] * ok[:, None, :])
    k = X[..., LAYERKEY]
    same = ((k[:, :, None] - k[:, None]).abs().sum(-1) < 1e-4).float()
    return torch.stack([(d / 100).clamp(-5, 5)[..., 0], (d / 100).clamp(-5, 5)[..., 1], (d / 100).clamp(-5, 5)[..., 2],
                        torch.log1p(dist), ai, aj, aab, same], -1), dist


class Block(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.h = heads; self.n1 = nn.LayerNorm(d); self.qkv = nn.Linear(d, 3 * d); self.o = nn.Linear(d, d)
        self.n2 = nn.LayerNorm(d); self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.SiLU(), nn.Linear(2 * d, d))

    def forward(self, x, bias, pad):
        Bn, n, d = x.shape
        q, k, v = self.qkv(self.n1(x)).view(Bn, n, 3, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        att = (q @ k.transpose(-1, -2)) / math.sqrt(d // self.h) + bias.permute(0, 3, 1, 2)
        att = att.masked_fill(pad[:, None, None, :], float("-inf")).softmax(-1)
        x = x + self.o((att @ v).transpose(1, 2).reshape(Bn, n, d))
        return x + self.ff(self.n2(x))


class Splitter(nn.Module):
    def __init__(self, n_in, d=128, layers=4, heads=4, npf=8):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(n_in, d), nn.SiLU(), nn.Linear(d, d))
        self.pb = nn.Sequential(nn.Linear(npf, 64), nn.SiLU(), nn.Linear(64, layers * heads))
        self.blocks = nn.ModuleList([Block(d, heads) for _ in range(layers)])
        self.heads, self.layers = heads, layers
        self.norm = nn.LayerNorm(d)
        self.neither = nn.Linear(d, 1)
        self.pair = nn.Sequential(nn.Linear(2 * d + npf, 128), nn.SiLU(), nn.Linear(128, 1))

    def forward(self, x, pad, G=None):
        pf, dist = pair_features(x)
        if G is not None:  # GNN v3 edge score + edge-present channels
            pf = torch.cat([pf, G], -1)
        bias = self.pb(pf).view(*pf.shape[:3], self.layers, self.heads)
        h = self.inp(x)
        for i, b in enumerate(self.blocks):
            h = b(h, bias[..., i, :], pad)
        h = self.norm(h)
        hi, hj = h[:, :, None].expand(-1, -1, h.shape[1], -1), h[:, None].expand(-1, h.shape[1], -1, -1)
        pl = self.pair(torch.cat([hi * hj, (hi - hj).abs(), pf], -1)).squeeze(-1)
        return self.neither(h).squeeze(-1), 0.5 * (pl + pl.transpose(1, 2)), dist


def knn_mask(dist, pad, k=KNN):
    big = dist.masked_fill(pad[:, None, :] | pad[:, :, None], 1e9)
    big = big + torch.eye(dist.shape[1], device=dist.device)[None] * 1e9
    kk = min(k, dist.shape[1] - 1)
    idx = big.topk(kk, dim=-1, largest=False).indices
    m = torch.zeros_like(dist, dtype=torch.bool).scatter_(-1, idx, True)
    return (m | m.transpose(1, 2)) & ~(pad[:, None, :] | pad[:, :, None])


def loss_fn(out, y, pad):
    nl, aff, dist = out
    valid = y >= 0
    core = valid & (y < 2)
    pm = core[:, :, None] & core[:, None, :] & ~torch.eye(y.shape[1], dtype=torch.bool, device=y.device)[None]
    w = torch.where(knn_mask(dist, pad), 1.0, 0.1) * pm
    tgt = (y[:, :, None] == y[:, None, :]).float()
    lp = (F.binary_cross_entropy_with_logits(aff, tgt, reduction="none") * w).sum((1, 2)) / w.sum((1, 2)).clamp(min=1e-6)
    ln = (F.binary_cross_entropy_with_logits(nl, (y == 2).float(), reduction="none") * valid).sum(1) / valid.sum(1)
    return (lp + ln).mean()


DECODE = {"mode": "knn"}


def viterbi_labels(S, pos, key, maxk=6):
    """Two-path assignment by dynamic programming over layer groups: hits ordered by R^2 = r^2 + z^2, consecutive hits
    on the same layer form a group; state = A/B labels of the group's hits; score = sum of S_ij x_i x_j within the
    group and between the group and the previous group (x = +-1). Groups larger than maxk keep their eigenvector labels."""
    n = len(pos)
    R = (pos ** 2).sum(1)
    o = np.argsort(R)
    groups, cur = [], [o[0]]
    for i in o[1:]:
        if key[i] == key[cur[-1]]:
            cur.append(i)
        else:
            groups.append(cur); cur = [i]
    groups.append(cur)
    ev_, v = np.linalg.eigh(S)
    x0 = np.where(v[:, -1] >= 0, 1.0, -1.0)
    states = []
    for g in groups:
        if len(g) > maxk:
            states.append(np.array([x0[g]]))
        else:
            k = len(g)
            states.append(np.array([[1.0 if (b >> j) & 1 else -1.0 for j in range(k)] for b in range(2 ** k)]))
    # DP
    prev_score = None
    back = []
    for gi, g in enumerate(groups):
        st = states[gi]
        within = np.einsum("si,ij,sj->s", st, S[np.ix_(g, g)], st) / 2
        if gi == 0:
            prev_score = within; back.append(None); continue
        gp = groups[gi - 1]; sp = states[gi - 1]
        cross = sp @ S[np.ix_(gp, g)] @ st.T  # [n_prev_states, n_states]
        tot = prev_score[:, None] + cross
        bi = tot.argmax(0)
        prev_score = tot[bi, np.arange(len(st))] + within
        back.append(bi)
    x = np.zeros(n)
    si = int(prev_score.argmax())
    for gi in range(len(groups) - 1, -1, -1):
        x[groups[gi]] = states[gi][si]
        if gi > 0:
            si = int(back[gi][si])
    return x


def split_labels(nl, aff, km, n, X=None):
    """decode modes: knn (sign of the leading eigenvector of the signed affinity on k-NN pairs), full (all pairs, k-NN
    pairs weight 1 and others 0.1), knn_refine / full_refine (then single-hit flips while they raise sum S_ij x_i x_j)."""
    lab = np.full(n, 2, np.int64)
    keep = np.flatnonzero(nl[:n] <= 0)
    if len(keep) == 0:
        return lab
    mode = DECODE["mode"]
    w = km[:n, :n] if mode.startswith("knn") else np.where(km[:n, :n] > 0, 1.0, 0.1)
    S = ((2 / (1 + np.exp(-aff[:n, :n])) - 1) * w)[np.ix_(keep, keep)]
    np.fill_diagonal(S, 0)
    if mode.startswith("viterbi"):
        Sf = ((2 / (1 + np.exp(-aff[:n, :n])) - 1))[np.ix_(keep, keep)]
        np.fill_diagonal(Sf, 0)
        pos = X[:n][keep][:, POS] * np.array([1000.0, 1000.0, 3000.0])
        key = np.argmax(X[:n][keep][:, 14:23], 1) * 100 + np.round(X[:n][keep][:, 23] * 20).astype(int)
        x = viterbi_labels(Sf, pos, key)
        S = Sf
    else:
        ev, v = np.linalg.eigh(S)
        x = np.where(v[:, -1] >= 0, 1.0, -1.0)
    if mode.endswith("refine"):
        for _ in range(50):
            f = S @ x
            bad = np.flatnonzero(np.sign(f) * x < 0)
            if len(bad) == 0:
                break
            i = bad[np.argmax(np.abs(f[bad]))]
            x[i] = -x[i]
    if x.sum() < 0:
        x = -x
    lab[keep] = np.where(x > 0, 0, 1)
    return lab


def time_feature(tr, mode, rng):
    if mode == "none":
        return None
    t = tr["time"].astype(np.float64)
    if mode.startswith("smear"):
        t = t + rng.normal(0, float(mode[5:]) * 1e-3, size=len(t))
    return ((t - t.min()) / 10.0).astype(np.float32)[:, None]


AUG = {"on": False}
GNNF = {"on": False}  # attach the GNN v3 edge-score pair channels (tracks carry gs_ij / gs_s from add_gnn_scores.py)


def augment(x, rng):
    """Random rotation about the beam axis and random z reflection (the detector is close to symmetric in both)."""
    x = x.copy()
    a = rng.uniform(0, 2 * np.pi); c, s_ = np.cos(a), np.sin(a)
    for i, j in ((0, 1), (4, 5), (7, 8), (10, 11), (32, 33), (35, 36)):
        xi, xj = x[:, i].copy(), x[:, j].copy()
        x[:, i], x[:, j] = c * xi - s_ * xj, s_ * xi + c * xj
    if rng.random() < 0.5:
        for k in (2, 6, 9, 12, 34, 37):
            x[:, k] = -x[:, k]
    return x


def collate(tracks, mode, rng):
    n = max(len(t["hits"]) for t in tracks)
    fx = F0 + (0 if mode == "none" else 1)
    X = np.zeros((len(tracks), n, fx), np.float32); Y = np.full((len(tracks), n), -1, np.int64); pad = np.ones((len(tracks), n), bool)
    for i, t in enumerate(tracks):
        m = len(t["hits"]); tf = time_feature(t, mode, rng)
        xx = augment(t["x"], rng) if AUG["on"] else t["x"]
        X[i, :m] = xx if tf is None else np.concatenate([xx, tf], 1)
        Y[i, :m] = t["y"]; pad[i, :m] = False
    G = None
    if GNNF["on"]:
        G = np.zeros((len(tracks), n, n, 2), np.float32)
        for i, t in enumerate(tracks):
            ij, s = t["gs_ij"], t["gs_s"]
            for a_, b_ in ((0, 1), (1, 0)):
                G[i, ij[:, a_], ij[:, b_], 0] = np.maximum(G[i, ij[:, a_], ij[:, b_], 0], s)
                G[i, ij[:, a_], ij[:, b_], 1] = 1.0
        G = torch.from_numpy(G).to(DEV)
    return torch.from_numpy(X).to(DEV), torch.from_numpy(Y).to(DEV), torch.from_numpy(pad).to(DEV), G


def track_metrics(tracks, preds):
    nq, ok5, ok9 = 0, 0, 0
    for t, p in zip(tracks, preds):
        sets = [np.flatnonzero(p == c) for c in (0, 1)]
        for q in np.unique(t["pid"][t["quirk"]]):
            nq += 1; isq = t["pid"] == q
            ntot = t["ntot"][isq][0] if "ntot" in t else isq.sum()
            b5 = b9 = False
            for s in sets:
                if len(s):
                    k = isq[s].sum(); b5 |= (k / len(s) > 0.5) and (k / ntot > 0.5); b9 |= (k / len(s) > 0.9) and (k / ntot > 0.9)
            ok5 += b5; ok9 += b9
    return dict(quirks=nq, loose_strict=ok5 / max(nq, 1), tight_strict=ok9 / max(nq, 1))


@torch.no_grad()
def predict_tracks(model, tracks, mode, bs=128):
    model.eval(); rng = np.random.default_rng(12345)
    L = [len(t["hits"]) for t in tracks]
    res = [None] * len(tracks)
    for ids in budget_batches(np.argsort(L), L, cap=bs):
        b = [tracks[j] for j in ids]
        X, _, pad, G = collate(b, mode, rng)
        nl, aff, dist = model(X, pad, G)
        km = knn_mask(dist, pad).float().cpu().numpy()
        nl, aff, Xn = nl.float().cpu().numpy(), aff.float().cpu().numpy(), X.cpu().numpy()
        for jj, j in enumerate(ids):
            res[j] = split_labels(nl[jj], aff[jj], km[jj], len(tracks[j]["hits"]), Xn[jj])
    model.train()
    return res


def train(a):
    import glob as _g
    tr = []
    for f in sorted(_g.glob(a.train)):
        tr += pickle.load(open(f, "rb"))
        if a.n_train and len(tr) >= a.n_train:
            break
    if a.n_train:
        tr = tr[: a.n_train]
    va = pickle.load(open(a.val, "rb"))
    print("train tracks", len(tr), "val", len(va), flush=True)
    torch.manual_seed(a.seed); random.seed(a.seed); rng = np.random.default_rng(a.seed)
    n_in = F0 + (0 if a.time == "none" else 1)
    GNNF["on"] = a.gnn_score
    model = Splitter(n_in, a.d, a.layers, npf=10 if a.gnn_score else 8).to(DEV)
    if a.init:
        model.load_state_dict(torch.load(a.init)); print("init from", a.init, flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    steps = a.epochs * (len(budget_batches(np.argsort([len(t['hits']) for t in tr]), [len(t['hits']) for t in tr], cap=a.batch)) + 5)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.05)
    os.makedirs(a.out, exist_ok=True)
    json.dump(vars(a) | {"n_in": n_in}, open(os.path.join(a.out, "config.json"), "w"), indent=1)
    import wandb
    run = wandb.init(project="quirk-tracking", entity="murnanedaniel", name=a.name, config=vars(a), dir=a.out, mode="online")
    log = open(os.path.join(a.out, "log.jsonl"), "w"); best = -1
    # length-sorted buckets for speed
    for ep in range(a.epochs):
        L = [len(t["hits"]) for t in tr]
        idx = np.argsort([l + random.random() * 6 for l in L])
        batches = budget_batches(idx, L, cap=a.batch)
        random.shuffle(batches); t0 = time.time(); ls = []; gns = []
        for bi in batches:
            AUG["on"] = a.augment
            X, Y, pad, G = collate([tr[j] for j in bi], a.time, rng)
            AUG["on"] = False
            loss = loss_fn(model(X, pad, G), Y, pad)
            opt.zero_grad(); loss.backward()
            gns.append(float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))); opt.step(); ls.append(float(loss))
            if sched.last_epoch + 1 < steps:  # batch count varies a little per epoch (length jitter)
                sched.step()
        with torch.no_grad():
            vl = []
            Lv = [len(t["hits"]) for t in va]
            for ids in budget_batches(np.argsort(Lv), Lv, cap=128):
                X, Y, pad, G = collate([va[j] for j in ids], a.time, np.random.default_rng(1)); vl.append(float(loss_fn(model(X, pad, G), Y, pad)))
        m = track_metrics(va, predict_tracks(model, va, a.time))
        rec = dict(epoch=ep, train_loss=float(np.mean(ls)), val_loss=float(np.mean(vl)), lr=sched.get_last_lr()[0], batch=a.batch,
                   grad_norm=float(np.mean(gns)), **{"val_" + k: v for k, v in m.items()}, epoch_s=time.time() - t0)
        run.log(rec); log.write(json.dumps(rec) + "\n"); log.flush(); print(json.dumps(rec), flush=True)
        if m["loose_strict"] + 0.001 * m["tight_strict"] > best:
            best = m["loose_strict"] + 0.001 * m["tight_strict"]; torch.save(model.state_dict(), os.path.join(a.out, "best.pt"))
    run.finish()


def predict(a):
    cfg = json.load(open(os.path.join(a.out, "config.json")))
    GNNF["on"] = bool(cfg.get("gnn_score", False))
    model = Splitter(cfg["n_in"], cfg["d"], cfg["layers"], npf=10 if GNNF["on"] else 8).to(DEV)
    model.load_state_dict(torch.load(os.path.join(a.out, "best.pt")))
    tracks = pickle.load(open(a.data, "rb"))
    if a.decode_scan:
        for mode in ["knn_refine", "viterbi", "viterbi_refine"]:
            DECODE["mode"] = mode
            print("decode", mode, json.dumps(track_metrics(tracks, predict_tracks(model, tracks, cfg["time"]))), flush=True)
        return
    DECODE["mode"] = a.decode
    preds = predict_tracks(model, tracks, cfg["time"])
    print("track-level", json.dumps(track_metrics(tracks, preds)), flush=True)
    base = pq.read_table(a.base).to_pandas()
    new = {int(e): g.sort_values("hit_id").predicted_track_id.to_numpy().copy() for e, g in base.groupby("event_id")}
    nxt = {e: int(v.max()) + 1 for e, v in new.items()}
    gate = None
    if a.gate_file:
        gs = pq.read_table(a.gate_file, columns=["event_id", "track_id", "quirk_tag"]).to_pandas()
        gate = set(zip(gs.event_id[gs.quirk_tag >= a.gate_cut].astype(int), gs.track_id[gs.quirk_tag >= a.gate_cut].astype(int)))
    nsplit = 0
    for t, p in zip(tracks, preds):
        e = t["event_id"]; h = t["hits"]
        orig = new[e][h].copy()
        if gate is not None and (e, int(orig[0])) not in gate:
            continue
        sets = [h[p == c] for c in (0, 1)]
        if a.keep_neither:
            if min(len(sets[0]), len(sets[1])) < a.min_hits:
                continue  # nothing to split: the track stays as it was
            nsplit += 1
            for s in sets:  # neither hits keep the original track id (the remainder)
                new[e][s] = nxt[e]; nxt[e] += 1
        else:
            new[e][h] = -1
            for s in sets:
                if len(s) >= a.min_hits:
                    new[e][s] = nxt[e]; nxt[e] += 1
    print("tracks split", nsplit, flush=True)
    eids = np.concatenate([np.full(len(v), e, np.uint64) for e, v in new.items()])
    hids = np.concatenate([np.arange(len(v), dtype=np.uint64) for v in new.values()])
    tids = np.concatenate(list(new.values())).astype(np.int64)
    os.makedirs(a.pred_out, exist_ok=True)
    pq.write_table(pa.table({"event_id": eids, "hit_id": hids, "predicted_track_id": tids, "score": np.full(len(tids), np.nan, np.float32)}),
                   os.path.join(a.pred_out, "predictions.parquet"))
    json.dump({"run_id": os.path.basename(a.pred_out), "implementation": "quirkml:pair_splitter", "config": cfg | vars(a),
               "wrote_at": time.strftime("%Y-%m-%dT%H:%M:%S")}, open(os.path.join(a.pred_out, "meta.json"), "w"), indent=1, default=str)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train", "predict"])
    ap.add_argument("--train"); ap.add_argument("--val"); ap.add_argument("--data"); ap.add_argument("--out", required=True)
    ap.add_argument("--base"); ap.add_argument("--pred-out"); ap.add_argument("--min-hits", type=int, default=3)
    ap.add_argument("--time", default="none", choices=["none", "smear30", "smear100", "true"])
    ap.add_argument("--epochs", type=int, default=40); ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4); ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default="splitter2")
    ap.add_argument("--gate-file", default=None, help="predict: parquet with event_id, track_id, quirk_tag (B3 tagger)")
    ap.add_argument("--gate-cut", type=float, default=0.0)
    ap.add_argument("--keep-neither", action="store_true", help="predict: split only if both parts have >= min-hits; neither hits stay")
    ap.add_argument("--gnn-score", action="store_true", help="train: GNN v3 edge-score pair channels (data from add_gnn_scores.py)")
    ap.add_argument("--init", default=None, help="train: initialise from this checkpoint")
    ap.add_argument("--n-train", type=int, default=0, help="train: cap on training tracks (files read in order)")
    ap.add_argument("--augment", action="store_true", help="train: random phi rotation and z reflection")
    ap.add_argument("--decode", default="knn", choices=["knn", "full", "knn_refine", "full_refine", "viterbi", "viterbi_refine"])
    ap.add_argument("--decode-scan", action="store_true", help="predict: print val-style metrics for every decode mode on --data")
    a = ap.parse_args()
    train(a) if a.mode == "train" else predict(a)


if __name__ == "__main__":
    main()
