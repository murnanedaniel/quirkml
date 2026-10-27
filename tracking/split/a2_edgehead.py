"""A2: edge head on the pair GNN. The pair GNN (e2_pair_dir_s0, pair truth, direction features) gets a second edge
output: same quirk vs other quirk, trained on edges whose two hits both belong to the event's quirk pair (target from
the per-quirk labels); the first output keeps its pair-target loss. Fine-tuned from the pair GNN checkpoint.
Split: tracks from components on the first output (cut --cc-cut); in each track the within-track edges carry the signed
affinity 2 p_same - 1; the track is cut in two by the sign of the leading eigenvector plus single-hit flips (as A1);
one-particle tracks stay whole. No truth at inference.

  python a2_edgehead.py train --out DIR [--epochs 4]
  python a2_edgehead.py infer --out DIR --ranges 16000-16999 17000-19999
  python a2_edgehead.py split --out DIR --events 17000 19999 --pred-out P [--cc-cut 0.5] [--min-hits 4]
"""
import argparse, json, os, random, sys, time
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

G = "/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/gnn"
sys.path.insert(0, G)
import train_gnn as tg  # noqa: E402
from common import DIRECTION, N_DIR_EDGE, N_DIR_NODE, N_FEAT, InteractionGNN, edge_truth, load_events, lr_at, mlp, read_graphs  # noqa: E402

O = "/pscratch/sd/d/danieltm/quirk_tracking/gnn"
PAIRCACHE, DIRCACHE = O + "/cache/m200_L774_mu0_dir_pair", O + "/cache/m200_L774_mu0_dir"
GRAPHS = O + "/main/e2_pair_dir_s0/filter/graphs_cut0.005/*.parquet"
BASE = O + "/main/e2_pair_dir_s0/gnn"
DEV = "cuda"


class EdgeHeadGNN(torch.nn.Module):
    def __init__(self, base, hidden=128, n_layers=2):
        super().__init__()
        self.base = base
        self.head2 = mlp(3 * hidden, hidden, 1, n_layers)

    def forward(self, f, src, dst, efeat):
        b = self.base
        h = b.node_enc(f)
        e = b.edge_enc(torch.cat([h[src], h[dst], efeat], 1))
        for _ in range(b.n_iter):
            agg_in = torch.zeros_like(h).index_add_(0, dst, e)
            agg_out = torch.zeros_like(h).index_add_(0, src, e)
            h = h + b.node_net(torch.cat([h, agg_in, agg_out], 1))
            e = e + b.edge_net(torch.cat([h[src], h[dst], e], 1))
        z = torch.cat([h[src], h[dst], e], 1)
        return b.out(z).squeeze(-1), self.head2(z).squeeze(-1)


def build(cfg):
    DIRECTION["on"] = True
    tg.CLUSTER = cfg.get("cluster", "node")
    base = InteractionGNN(n_in=N_FEAT + N_DIR_NODE, n_edge_in=tg.N_EDGE_IN[tg.CLUSTER] + N_DIR_EDGE, hidden=cfg["hidden"],
                          n_iter=cfg["n_iter"], n_layers=cfg["layers"]).to(DEV)
    return base


def load_both(first, last):
    """pair-cache events with the per-quirk particle ids (pid_orig) of the direction cache, loaded in 1000-event
    chunks to keep memory down."""
    out = []
    for a in range(first, last + 1, 1000):
        b = min(a + 999, last)
        ep = {e["event_id"]: e for e in load_events(PAIRCACHE, a, b)}
        for e in load_events(DIRCACHE, a, b):
            ep[e["event_id"]]["pid_orig"] = e["pid"]
        out += [ep[k] for k in sorted(ep)]
    return out


def quirk_targets(ev, s, t):
    """mask of within-pair edges and target 1 = same quirk."""
    hq = ev["hit_quirk"]
    m = hq[s] & hq[t]
    same = ev["pid_orig"][s] == ev["pid_orig"][t]
    return m, same


def train(a):
    cfg = json.load(open(BASE + "/config.json"))
    base = build(cfg); base.load_state_dict(torch.load(BASE + "/best.pt"))
    model = EdgeHeadGNN(base, cfg["hidden"], cfg["layers"]).to(DEV)
    tg.RECIPE["name"] = "all"; tg.QW = cfg.get("quirk_weight", 10.0); tg.RECIPE["pair_all_true"] = False
    tr = load_both(0, a.n_train - 1); va = load_both(16000, 16199)
    g = read_graphs(GRAPHS, 0, a.n_train - 1) | read_graphs(GRAPHS, 16000, 16199)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    spe = len(tr) // 4; total = spe * a.epochs; step = 0
    os.makedirs(a.out, exist_ok=True); json.dump(cfg | vars(a), open(a.out + "/config.json", "w"), indent=1)
    import wandb
    run = wandb.init(project="quirk-tracking", entity="murnanedaniel", name=a.name, config=vars(a), dir=a.out, mode="online")
    pw = torch.tensor(1.0, device=DEV); best = -1
    for ep_i in range(a.epochs):
        random.shuffle(tr); t0 = time.time(); ls = []; l2s = []
        for b in range(spe):
            for gg in opt.param_groups:
                gg["lr"] = lr_at(step, total, a.lr, 200)
            evs = tr[4 * b:4 * b + 4]
            f, s, t, ef, y, w = tg.batch(evs, g)
            m_all, same_all = [], []
            for ev in evs:
                gg2 = g[ev["event_id"]]
                m, same = quirk_targets(ev, gg2["src"].astype(np.int64), gg2["dst"].astype(np.int64))
                m_all.append(m); same_all.append(same)
            m2 = torch.as_tensor(np.concatenate(m_all), device=DEV); y2 = torch.as_tensor(np.concatenate(same_all), device=DEV).float()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lo, lo2 = model(f, s, t, ef)
            l1 = tg.loss_fn(lo.float(), y, w, pw)
            l2 = F.binary_cross_entropy_with_logits(lo2.float()[m2], y2[m2]) if m2.any() else lo2.sum() * 0
            loss = l1 + a.w2 * l2
            opt.zero_grad(); loss.backward(); gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); step += 1
            ls.append(float(l1)); l2s.append(float(l2))
        acc = val_acc(model, va, g)
        rec = dict(epoch=ep_i, train_pair_loss=float(np.mean(ls)), train_quirk_loss=float(np.mean(l2s)), lr=opt.param_groups[0]["lr"],
                   batch=4, grad_norm=float(gn), **acc, epoch_s=time.time() - t0)
        run.log(rec); print(json.dumps(rec), flush=True)
        torch.save(model.state_dict(), a.out + f"/ep{ep_i}.pt")
        if acc["val_quirk_head_acc"] > best:
            best = acc["val_quirk_head_acc"]; torch.save(model.state_dict(), a.out + "/best.pt")
    run.finish()


@torch.no_grad()
def val_acc(model, va, g):
    model.eval(); c = n = 0; l2 = []
    for ev in va:
        f, s, t, ef = tg.batch([ev], g, with_truth=False)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lo, lo2 = model(f, s, t, ef)
        gg = g[ev["event_id"]]
        m, same = quirk_targets(ev, gg["src"].astype(np.int64), gg["dst"].astype(np.int64))
        p = lo2.float().cpu().numpy()[m] > 0
        c += int((p == same[m]).sum()); n += int(m.sum())
    model.train()
    return dict(val_quirk_head_acc=c / max(n, 1), val_within_pair_edges=n / len(va))


@torch.no_grad()
def infer(a):
    cfg = json.load(open(BASE + "/config.json"))
    model = EdgeHeadGNN(build(cfg), cfg["hidden"], cfg["layers"]).to(DEV)
    model.load_state_dict(torch.load(a.out + "/" + a.ckpt)); model.eval()
    os.makedirs(a.out + "/scored", exist_ok=True)
    for rng in a.ranges:
        first, last = map(int, rng.split("-"))
        evs = load_events(DIRCACHE, first, last); g = read_graphs(GRAPHS, first, last)
        for c0 in range(0, len(evs), 1000):
            ch = evs[c0:c0 + 1000]; rows = []
            for ev in ch:
                f, s, t, ef = tg.batch([ev], g, with_truth=False)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    lo, lo2 = model(f, s, t, ef)
                gg = g[ev["event_id"]]
                rows.append(dict(event_id=ev["event_id"], src=gg["src"], dst=gg["dst"], score=torch.sigmoid(lo.float()).cpu().numpy(),
                                 score2=torch.sigmoid(lo2.float()).cpu().numpy()))
            e0, e1 = ch[0]["event_id"], ch[-1]["event_id"]
            pq.write_table(pa.table({"event_id": pa.array([r["event_id"] for r in rows], pa.uint32()),
                                     "src": pa.array([r["src"].astype(np.int32) for r in rows], pa.list_(pa.int32())),
                                     "dst": pa.array([r["dst"].astype(np.int32) for r in rows], pa.list_(pa.int32())),
                                     "score": pa.array([r["score"].astype(np.float32) for r in rows], pa.list_(pa.float32())),
                                     "score2": pa.array([r["score2"].astype(np.float32) for r in rows], pa.list_(pa.float32()))}),
                           a.out + f"/scored/events{e0}-{e1}.parquet")
            print("scored", e0, e1, flush=True)


def split_track(idx, s, t, p2, refine=True):
    """Labels 0/1 for the hits idx from within-track edges (s, t local indices) with same-quirk prob p2."""
    n = len(idx)
    if n < 2 or len(s) == 0:
        return np.zeros(n, int)
    S = np.zeros((n, n))
    S[s, t] = 2 * p2 - 1; S[t, s] = 2 * p2 - 1
    ev_, v = np.linalg.eigh(S)
    x = np.where(v[:, -1] >= 0, 1.0, -1.0)
    if refine:
        for _ in range(100):
            fld = S @ x
            bad = np.flatnonzero(np.sign(fld) * x < 0)
            if len(bad) == 0:
                break
            i = bad[np.argmax(np.abs(fld[bad]))]; x[i] = -x[i]
    if x.sum() < 0:
        x = -x
    return np.where(x > 0, 0, 1)


def split(a):
    first, last = a.events
    evs = load_events(DIRCACHE, first, last)
    g = read_graphs(a.out + "/scored/*.parquet", first, last)
    eids, hids, tids = [], [], []
    nsplit = 0
    gmask = None
    if a.gate_file:  # gate on the B3 quirk tag of the pair GNN + components tracks (--base): a component is a pair
        # candidate if most of its hits sit on a tagged base track
        gs = pq.read_table(a.gate_file, columns=["event_id", "track_id", "quirk_tag"]).to_pandas()
        gs = gs[gs.quirk_tag >= a.gate_cut]
        tagged = {}
        for e, tr in zip(gs.event_id.astype(int), gs.track_id.astype(int)):
            tagged.setdefault(e, set()).add(tr)
        base = pq.read_table(a.base, filters=[("event_id", ">=", first), ("event_id", "<", last)]).to_pandas()
        gmask = {int(e): np.isin(gg.sort_values("hit_id").predicted_track_id.to_numpy(), list(tagged.get(int(e), ())))
                 for e, gg in base.groupby("event_id")}
    for ev in evs:
        n = len(ev["x"]); gg = g[ev["event_id"]]
        s, t, sc, s2 = gg["src"].astype(np.int64), gg["dst"].astype(np.int64), gg["score"], gg["score2"]
        k = sc > a.cc_cut
        _, lab = connected_components(coo_matrix((np.ones(k.sum()), (s[k], t[k])), shape=(n, n)), directed=False)
        out = np.full(n, -1, np.int64); nxt = 0
        for c in np.unique(lab):
            idx = np.flatnonzero(lab == c)
            if len(idx) < a.min_hits:
                continue
            loc = -np.ones(n, np.int64); loc[idx] = np.arange(len(idx))
            m = k & (loc[s] >= 0) & (loc[t] >= 0)
            if gmask is not None and gmask[int(ev["event_id"])][idx].mean() <= 0.5:
                out[idx] = nxt; nxt += 1
                continue
            parts = split_track(idx, loc[s[m]], loc[t[m]], s2[m])
            sizes = [np.sum(parts == q) for q in (0, 1)]
            if min(sizes) >= a.min_hits:
                nsplit += 1
                for q in (0, 1):
                    out[idx[parts == q]] = nxt; nxt += 1
            else:
                out[idx] = nxt; nxt += 1
        eids.append(np.full(n, ev["event_id"], np.uint64)); hids.append(np.arange(n, dtype=np.uint64)); tids.append(out)
    os.makedirs(a.pred_out, exist_ok=True)
    tids = np.concatenate(tids)
    pq.write_table(pa.table({"event_id": np.concatenate(eids), "hit_id": np.concatenate(hids), "predicted_track_id": tids,
                             "score": np.full(len(tids), np.nan, np.float32)}), a.pred_out + "/predictions.parquet")
    json.dump({"run_id": os.path.basename(a.pred_out), "implementation": "quirkml:pair_gnn_edgehead_split", "config": vars(a),
               "wrote_at": time.strftime("%Y-%m-%dT%H:%M:%S")}, open(a.pred_out + "/meta.json", "w"), indent=1)
    print("tracks split", nsplit, "events", len(evs), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train", "infer", "split"])
    ap.add_argument("--out", required=True); ap.add_argument("--epochs", type=int, default=4); ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--n-train", type=int, default=8000)
    ap.add_argument("--w2", type=float, default=1.0); ap.add_argument("--name", default="a2_edgehead"); ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--ranges", nargs="+", default=[]); ap.add_argument("--events", type=int, nargs=2)
    ap.add_argument("--gate-file"); ap.add_argument("--gate-cut", type=float, default=0.0); ap.add_argument("--base")
    ap.add_argument("--pred-out"); ap.add_argument("--cc-cut", type=float, default=0.5); ap.add_argument("--min-hits", type=int, default=4)
    a = ap.parse_args()
    {"train": train, "infer": infer, "split": split}[a.mode](a)


if __name__ == "__main__":
    main()
