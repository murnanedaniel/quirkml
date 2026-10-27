"""A1 learned splitter: a small transformer over the hits of one track. Two heads: per-hit "neither" (hit belongs to
neither of the track's two main particles) and a symmetric pair affinity (same particle or not) between hits; the
affinity loss is permutation invariant by construction. The split: hits with P(neither) > 0.5 are dropped, the rest are
cut in two by the sign of the leading eigenvector of the affinity matrix 2 sigma(L) - 1 (one particle -> one set).
(A first version with per-hit A/B logits and a min-over-swap loss did not learn: val loss stayed near ln 2.)
Inputs from split_data.py.

Time arms (--time): none (headline), smear30 / smear100 (true sim-hit time + Gaussian 30 / 100 ps per hit, the
assumed detector resolution), true (unsmeared sim-hit time: upper-bound reference only).
Selection: best val per-quirk strict loose efficiency (> 50% purity and completeness) within the track set.

  python splitter.py train --train T.pkl --val V.pkl --out DIR --time none
  python splitter.py predict --data X.pkl --out DIR --base BASE/predictions.parquet --pred-out P [--oracle]
"""
import argparse, json, os, pickle, random, time
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F

DEV = "cuda"


class Splitter(nn.Module):
    def __init__(self, n_in, d=128, layers=4, heads=4):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(n_in, d), nn.SiLU(), nn.Linear(d, d))
        enc = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(enc, layers)
        self.norm = nn.LayerNorm(d)
        self.neither = nn.Linear(d, 1)
        self.q = nn.Linear(d, 64)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x, pad):
        h = self.norm(self.enc(self.inp(x), src_key_padding_mask=pad))
        q = self.q(h)
        aff = q @ q.transpose(1, 2) / 8.0 + self.bias  # symmetric pair logits
        return self.neither(h).squeeze(-1), aff


def time_feature(tr, mode, rng):
    t = tr["time"].astype(np.float64)
    if mode == "none":
        return None
    if mode.startswith("smear"):
        t = t + rng.normal(0, float(mode[5:]) * 1e-3, size=len(t))
    return ((t - t.min()) / 10.0).astype(np.float32)[:, None]


def collate(tracks, mode, rng):
    n = max(len(t["hits"]) for t in tracks)
    fx = tracks[0]["x"].shape[1] + (0 if mode == "none" else 1)
    X = np.zeros((len(tracks), n, fx), np.float32)
    Y = np.full((len(tracks), n), -1, np.int64)
    pad = np.ones((len(tracks), n), bool)
    for i, t in enumerate(tracks):
        m = len(t["hits"])
        tf = time_feature(t, mode, rng)
        X[i, :m] = t["x"] if tf is None else np.concatenate([t["x"], tf], 1)
        Y[i, :m] = t["y"]
        pad[i, :m] = False
    return torch.from_numpy(X).to(DEV), torch.from_numpy(Y).to(DEV), torch.from_numpy(pad).to(DEV)


def perm_loss(out, y):
    """Pair affinity BCE over valid hit pairs whose hits both belong to A or B (target: same particle), plus BCE for the
    per-hit neither head; both averaged per track."""
    nl, aff = out
    valid = y >= 0
    core = valid & (y < 2)
    pm = core[:, :, None] & core[:, None, :]
    eye = torch.eye(y.shape[1], dtype=torch.bool, device=y.device)[None]
    pm = pm & ~eye
    tgt = (y[:, :, None] == y[:, None, :]).float()
    lp = F.binary_cross_entropy_with_logits(aff, tgt, reduction="none")
    lp = (lp * pm).sum((1, 2)) / pm.sum((1, 2)).clamp(min=1)
    ln = F.binary_cross_entropy_with_logits(nl, (y == 2).float(), reduction="none")
    ln = (ln * valid).sum(1) / valid.sum(1)
    return (lp + ln).mean()


def split_labels(nl, aff, n):
    """Labels 0 / 1 / 2 for one track from its head outputs (numpy)."""
    lab = np.full(n, 2, np.int64)
    keep = np.flatnonzero(nl[:n] <= 0)
    if len(keep) == 0:
        return lab
    S = 2 / (1 + np.exp(-aff[:n, :n][np.ix_(keep, keep)])) - 1
    np.fill_diagonal(S, 0)
    w, v = np.linalg.eigh(S)
    lead = v[:, -1]
    lead = lead if lead.sum() >= 0 else -lead
    lab[keep] = np.where(lead >= 0, 0, 1)
    return lab


def track_metrics(tracks, preds):
    """Per-quirk strict efficiency (> 50% / > 90% purity and completeness) of the split sets of each track; completeness
    uses the quirk's hits in the whole event (ntot)."""
    nq, ok5, ok9 = 0, 0, 0
    for t, p in zip(tracks, preds):
        sets = [np.flatnonzero(p == c) for c in (0, 1)]
        for q in np.unique(t["pid"][t["quirk"]]):
            nq += 1
            isq = t["pid"] == q
            ntot = t["ntot"][isq][0] if "ntot" in t else isq.sum()
            best5 = best9 = False
            for s in sets:
                if len(s) == 0:
                    continue
                k = isq[s].sum()
                best5 |= (k / len(s) > 0.5) and (k / ntot > 0.5)
                best9 |= (k / len(s) > 0.9) and (k / ntot > 0.9)
            ok5 += best5; ok9 += best9
    return dict(quirks=nq, loose_strict=ok5 / max(nq, 1), tight_strict=ok9 / max(nq, 1))


@torch.no_grad()
def predict_tracks(model, tracks, mode, bs=256):
    model.eval()
    rng = np.random.default_rng(12345)
    out = []
    for i in range(0, len(tracks), bs):
        b = tracks[i:i + bs]
        X, _, pad = collate(b, mode, rng)
        nl, aff = model(X, pad)
        nl, aff = nl.float().cpu().numpy(), aff.float().cpu().numpy()
        out += [split_labels(nl[j], aff[j], len(t["hits"])) for j, t in enumerate(b)]
    model.train()
    return out


def train(a):
    tr = pickle.load(open(a.train, "rb")); va = pickle.load(open(a.val, "rb"))
    torch.manual_seed(a.seed); random.seed(a.seed); rng = np.random.default_rng(a.seed)
    n_in = tr[0]["x"].shape[1] + (0 if a.time == "none" else 1)
    model = Splitter(n_in, a.d, a.layers).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    steps = a.epochs * (len(tr) // a.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.05)
    os.makedirs(a.out, exist_ok=True)
    json.dump(vars(a) | {"n_in": n_in}, open(os.path.join(a.out, "config.json"), "w"), indent=1)
    import wandb
    run = wandb.init(project="quirk-tracking", entity="murnanedaniel", name=a.name, config=vars(a), dir=a.out, mode="online")
    log = open(os.path.join(a.out, "log.jsonl"), "w")
    best = -1
    for ep in range(a.epochs):
        random.shuffle(tr); t0 = time.time(); ls = []
        for i in range(0, len(tr) - a.batch + 1, a.batch):
            X, Y, pad = collate(tr[i:i + a.batch], a.time, rng)
            loss = perm_loss(model(X, pad), Y)
            opt.zero_grad(); loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); ls.append(float(loss))
        with torch.no_grad():
            vl = []
            for i in range(0, len(va), 256):
                X, Y, pad = collate(va[i:i + 256], a.time, np.random.default_rng(1))
                vl.append(float(perm_loss(model(X, pad), Y)))
        m = track_metrics(va, predict_tracks(model, va, a.time))
        rec = dict(epoch=ep, train_loss=float(np.mean(ls)), val_loss=float(np.mean(vl)), lr=sched.get_last_lr()[0], batch=a.batch,
                   grad_norm=float(gn), **{"val_" + k: v for k, v in m.items()}, epoch_s=time.time() - t0)
        run.log(rec); log.write(json.dumps(rec) + "\n"); log.flush(); print(json.dumps(rec), flush=True)
        if m["loose_strict"] + 0.001 * m["tight_strict"] > best:
            best = m["loose_strict"] + 0.001 * m["tight_strict"]
            torch.save(model.state_dict(), os.path.join(a.out, "best.pt"))
    run.finish()


def predict(a):
    cfg = json.load(open(os.path.join(a.out, "config.json")))
    model = Splitter(cfg["n_in"], cfg["d"], cfg["layers"]).to(DEV)
    model.load_state_dict(torch.load(os.path.join(a.out, "best.pt")))
    tracks = pickle.load(open(a.data, "rb"))
    preds = predict_tracks(model, tracks, cfg["time"])
    print("track-level", json.dumps(track_metrics(tracks, preds)), flush=True)
    base = pq.read_table(a.base).to_pandas()
    lab = {int(e): g.sort_values("hit_id") for e, g in base.groupby("event_id")}
    new = {e: g.predicted_track_id.to_numpy().copy() for e, g in lab.items()}
    nxt = {e: int(v.max()) + 1 for e, v in new.items()}
    for t, p in zip(tracks, preds):
        e = t["event_id"]; h = t["hits"]
        new[e][h] = -1
        for c in (0, 1):
            s = h[p == c]
            if len(s) >= a.min_hits:
                new[e][s] = nxt[e]; nxt[e] += 1
    eids = np.concatenate([np.full(len(v), e, np.uint64) for e, v in new.items()])
    hids = np.concatenate([np.arange(len(v), dtype=np.uint64) for v in new.values()])
    tids = np.concatenate(list(new.values())).astype(np.int64)
    os.makedirs(a.pred_out, exist_ok=True)
    pq.write_table(pa.table({"event_id": eids, "hit_id": hids, "predicted_track_id": tids,
                             "score": np.full(len(tids), np.nan, np.float32)}), os.path.join(a.pred_out, "predictions.parquet"))
    json.dump({"run_id": os.path.basename(a.pred_out), "implementation": "quirkml:gnn_pair_splitter", "config": cfg | vars(a),
               "wrote_at": time.strftime("%Y-%m-%dT%H:%M:%S")}, open(os.path.join(a.pred_out, "meta.json"), "w"), indent=1, default=str)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train", "predict"])
    ap.add_argument("--train"); ap.add_argument("--val"); ap.add_argument("--data"); ap.add_argument("--out", required=True)
    ap.add_argument("--base"); ap.add_argument("--pred-out"); ap.add_argument("--min-hits", type=int, default=3)
    ap.add_argument("--time", default="none", choices=["none", "smear30", "smear100", "true"])
    ap.add_argument("--epochs", type=int, default=60); ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default="splitter")
    a = ap.parse_args()
    train(a) if a.mode == "train" else predict(a)


if __name__ == "__main__":
    main()
