"""GNN data-scaling trainer: train_gnn.py's model, features, loss and validation, with the training events streamed
from disk in 1000-event shards (the in-memory loader of train_gnn.py needs ~1.6 GB of RAM per 1000 events of graph v3,
too much for 50k-150k events on a shared GPU node).

Training budget (common rule for the GNN and MaskFormer scaling studies, 2026-10-27): train until the STRICT
validation efficiency has not improved for --patience validation points, capped at --steps optimizer steps (the cap
is 4x the 16k control's 64k steps); the cosine schedule of train_gnn.py runs over the cap. Every --interval steps:
val loss and edge metrics on the first --n-val events of --val (as train_gnn.py), and the strict track-level
efficiency on all events of --track-val: --stop-metric perquirk = walk + stitch 0.7, >= 4 hits, per-quirk efficiency
(normal cache); coupled = connected components at --cc-cut, >= 4 hits, on a pair cache (the pair is one particle).
Strict = purity > 50% and completeness > 50% (ACTS very_loose_strict); tight > 90% is logged too.
best_sel.pt / best_sel.json = best strict val point (the checkpoint to test); best.pt = lowest val loss; step<N>.pt
per validation point; state.pt (resume every --save-every steps); cut_scan.json (on best_sel.pt) at the end. Data order: shards in a random order per pass, --buffer shards loaded at a time (prefetched in a thread),
events shuffled inside the buffer, batches of --batch events.

  python train_gnn_stream.py --cache C --graphs G --out O --train-ranges 0-15999 20000-53999 --val 16000 16199 \
      --steps 64000 --interval 4000 [--direction] [--pair-all-true] --quirk-weight 10 --warmup 1000 --seed S
Cache and graph directories hold events<a>-<b>.pkl / events<a>-<b>.parquet shards (symlinks are fine).
"""
import argparse
import glob
import json
import os
import pickle
import random
import sys
import threading
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import train_gnn as T  # noqa: E402
from common import DIRECTION, Logger, load_events, lr_at, read_graphs  # noqa: E402


def shard_list(cache, graphs, ranges):
    out = []
    for f in sorted(glob.glob(os.path.join(cache, "events*.pkl"))):
        a, b = map(int, os.path.basename(f)[6:-4].split("-"))
        if not any(a >= lo and b <= hi for lo, hi in ranges):
            continue
        g = os.path.join(graphs, f"events{a}-{b}.parquet")
        if not os.path.exists(g):
            raise FileNotFoundError(g)
        out.append((f, g))
    return out


def load_shard(pair):
    f, g = pair
    with open(f, "rb") as fh:
        evs = pickle.load(fh)
    gr = read_graphs(g)
    return [e for e in evs if e["event_id"] in gr], gr


class Stream:
    """Infinite stream of training batches; shard order reshuffled every pass (seeded by seed and pass number)."""

    def __init__(self, shards, buffer, batch, seed, start_pass):
        self.shards, self.buffer, self.batch, self.seed = shards, buffer, batch, seed
        self.pass_no = start_pass
        self.queue = []
        self._next = None
        self._thread = None
        self._fill_order()
        self._prefetch()

    def _fill_order(self):
        order = list(self.shards)
        random.Random(self.seed * 100003 + self.pass_no).shuffle(order)
        self.queue += [order[i:i + self.buffer] for i in range(0, len(order), self.buffer)]
        self.pass_no += 1

    def _prefetch(self):
        if not self.queue:
            self._fill_order()
        group = self.queue.pop(0)
        res = {}

        def work():
            evs, gr = [], {}
            for p in group:
                e, g = load_shard(p)
                evs += e
                gr.update(g)
            res["v"] = (evs, gr)
        self._thread = threading.Thread(target=work, daemon=True)
        self._thread.start()
        self._next = res

    def buffers(self):
        while True:
            self._thread.join()
            if "v" not in self._next:
                raise RuntimeError("shard loading failed")
            evs, gr = self._next["v"]
            self._prefetch()
            random.shuffle(evs)
            yield evs, gr


def strict_metrics(evs, labels, f=0.5):
    """Strict (>) double matching, as quick_eval.metrics but '>' instead of '>='. Quirks: particles with a quirk hit
    and >= 3 hits (on a pair cache the pair is one particle). Returns quirk eff at f, tracks, strict fake rate."""
    nq = nqm = ntr = nfake = 0
    for ev in evs:
        lab, pid = labels[ev["event_id"]], ev["pid"]
        u, inv, pcnt = np.unique(pid, return_inverse=True, return_counts=True)
        nh = pcnt[inv]
        quirks = np.unique(pid[ev["hit_quirk"] & (nh >= 3)])
        nq += len(quirks)
        m = lab >= 0
        if not m.any():
            continue
        P = int(pid.max()) + 1
        k, c = np.unique(lab[m].astype(np.int64) * P + pid[m], return_counts=True)
        trk, par = k // P, k % P
        tsize = np.bincount(lab[m])
        psize = dict(zip(u.tolist(), pcnt.tolist()))
        psz = np.array([psize[int(p)] for p in par])
        ok = (c > f * tsize[trk]) & (c > f * psz)
        matched = set(par[ok].tolist())
        nqm += sum(int(q) in matched for q in quirks)
        ok5 = (c > 0.5 * tsize[trk]) & (c > 0.5 * psz)
        good = np.zeros(len(tsize), bool)
        good[trk[ok5]] = True
        ntr += int((tsize > 0).sum())
        nfake += int((~good & (tsize > 0)).sum())
    return nqm / max(nq, 1), nq, nfake / max(ntr, 1)


_BT = {}


def _build(item):
    import build_tracks as bt
    bt.ARGS = _BT["args"]
    return item[0]["event_id"], bt.build_event(item)[1]


@torch.no_grad()
def track_val(model, evs, graphs, metric, cc_cut, workers):
    """Strict val efficiency of the current model (builder as in the scaling eval)."""
    from multiprocessing import get_context
    model.eval()
    items = []
    for ev in evs:
        f, s, t, ef = T.batch([ev], graphs, with_truth=False)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p = torch.sigmoid(model(f, s, t, ef).float()).cpu().numpy()
        g = graphs[ev["event_id"]]
        items.append((ev, {"src": g["src"], "dst": g["dst"], "score": p}))
    model.train()
    _BT["args"] = argparse.Namespace(method="ccwalk_stitch" if metric == "perquirk" else "cc", score_cut_cc=0.01,
                                     th_min=0.1, th_add=0.6, th_stitch=0.7, min_hits=4, cc_cut=cc_cut)
    with get_context("fork").Pool(workers) as pool:
        labels = dict(pool.map(_build, items, chunksize=8))
    loose, nq, fake = strict_metrics(evs, labels, 0.5)
    tight, _, _ = strict_metrics(evs, labels, 0.9)
    return {"val_strict_loose": loose, "val_strict_tight": tight, "val_strict_fake": fake, "val_strict_n": nq}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--graphs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="gnn")
    ap.add_argument("--train-ranges", nargs="+", required=True)
    ap.add_argument("--val", type=int, nargs=2, default=[16000, 16199])
    ap.add_argument("--n-val", type=int, default=200)
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--interval", type=int, default=4000)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--buffer", type=int, default=3)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--n-iter", type=int, default=8)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--pos-weight", type=float, default=1.0)
    ap.add_argument("--quirk-weight", type=float, default=10.0)
    ap.add_argument("--direction", action="store_true")
    ap.add_argument("--pair-all-true", action="store_true")
    ap.add_argument("--cluster", choices=["none", "node", "node_edge"], default="node")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stop-metric", choices=["perquirk", "coupled"], required=True)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--track-val", type=int, nargs=2, default=[16000, 16999])
    ap.add_argument("--cc-cut", type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--deadline", type=float, default=0, help="unix time: save state and exit 3 before it")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    # train_gnn.py globals (same meaning as its CLI)
    T.QW = a.quirk_weight
    T.CLUSTER = a.cluster
    T.RECIPE.update(name="all", w_qq=0.1, fake_target=0.10, pair_all_true=a.pair_all_true)
    DIRECTION["on"] = a.direction
    # config.json keys that train_gnn.py infer reads
    a.recipe, a.epochs, a.train = "all", None, None

    ranges = [tuple(map(int, r.split("-"))) for r in a.train_ranges]
    shards = shard_list(a.cache, a.graphs, ranges)
    n_train = 1000 * len(shards)
    va = load_events(a.cache, *a.val)[: a.n_val]
    gv = read_graphs(os.path.join(a.graphs, f"events{a.val[0] // 1000 * 1000}-*.parquet"), a.val[0], a.val[1])
    va = [e for e in va if e["event_id"] in gv]
    vt = load_events(a.cache, *a.track_val)
    gt = read_graphs(os.path.join(a.graphs, "events1[6-9]???-*.parquet"), a.track_val[0], a.track_val[1])
    vt = [e for e in vt if e["event_id"] in gt]
    print(f"train shards {len(shards)} (~{n_train} events), val {len(va)} events, track val {len(vt)} events", flush=True)

    torch.manual_seed(a.seed)
    random.seed(a.seed)
    model = T.build_model(a.hidden, a.n_iter, a.layers)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    state_path = os.path.join(a.out, "state.pt")
    step, best, seen = 0, 1e9, 0
    es = {"best": -1.0, "best_step": 0, "bad": 0, "history": [], "stopped": False}
    prev_rows = []
    if os.path.exists(state_path):
        st = torch.load(state_path, weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        step, best, seen = st["step"], st["best"], st["seen"]
        es = st["es"]
        random.setstate(st["py_rng"])
        if os.path.exists(os.path.join(a.out, "log.csv")):
            import csv
            prev_rows = list(csv.DictReader(open(os.path.join(a.out, "log.csv"))))
        print(f"resuming at step {step}", flush=True)
    if (step >= a.steps or es["stopped"]) and os.path.exists(os.path.join(a.out, "cut_scan.json")):
        print("already finished", flush=True)
        return
    cfg = vars(a) | {"stage": "gnn_stream", "n_train": n_train, "n_val": len(va), "total_steps": a.steps,
                     "resume_step": step, "epochs_equiv": a.steps * a.batch / n_train}
    log = Logger(a.out, a.name + (f"_resume{step}" if step else ""), cfg)
    log.rows = prev_rows
    pw = torch.tensor(a.pos_weight, device=T.DEV)
    # a resumed run starts a fresh pass (new shard order, seeded by the resume step)
    stream = Stream(shards, a.buffer, a.batch, a.seed, start_pass=step + 1)

    def save_state():
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "best": best, "seen": seen,
                    "py_rng": random.getstate(), "es": es}, state_path + ".tmp")
        os.replace(state_path + ".tmp", state_path)

    run, gns, t0, t_step = [], [], time.time(), []
    for evs, gr in (stream.buffers() if not es["stopped"] else iter(())):
        if step >= a.steps or es["stopped"]:
            break
        for b0 in range(0, len(evs) - a.batch + 1, a.batch):
            if step >= a.steps or es["stopped"]:
                break
            ts = time.time()
            lr = lr_at(step, a.steps, a.lr, a.warmup)
            for gg in opt.param_groups:
                gg["lr"] = lr
            f, s, t, ef, y, w = T.batch(evs[b0:b0 + a.batch], gr)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lo = model(f, s, t, ef).float()
            loss = T.loss_fn(lo, y, w, pw)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gns.append(float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)))
            opt.step()
            run.append(float(loss))
            step += 1
            seen += a.batch
            t_step.append(time.time() - ts)
            if not np.isfinite(run[-1]):
                print(f"non-finite loss at step {step}", flush=True)
                sys.exit(2)
            if step % a.interval == 0 or step == a.steps:
                vl, m = T.evaluate(model, va, gv, pw)
                stream._thread.join()  # no fork while the prefetch thread is inside pyarrow / pickle
                tv = track_val(model, vt, gt, a.stop_metric, a.cc_cut, a.workers)
                torch.save(model.state_dict(), os.path.join(a.out, f"step{step}.pt"))
                es["history"].append({"step": step, **tv, "val_loss": vl})
                if tv["val_strict_loose"] > es["best"]:
                    es.update(best=tv["val_strict_loose"], best_step=step, bad=0)
                    torch.save(model.state_dict(), os.path.join(a.out, "best_sel.pt"))
                else:
                    es["bad"] += 1
                if es["bad"] >= a.patience:
                    es["stopped"] = True
                json.dump({"rule": f"strict val {a.stop_metric} loose, patience {a.patience}, cap {a.steps} steps",
                           **{k: v for k, v in es.items()}}, open(os.path.join(a.out, "best_sel.json"), "w"), indent=1)
                log.log({**tv, "best_strict": es["best"], "best_step": es["best_step"], "bad": es["bad"],"step": step, "epoch_equiv": seen / n_train, "lr": lr, "train_loss": float(np.mean(run)),
                         "grad_norm": float(np.mean(gns)), "grad_norm_max": float(np.max(gns)), "val_loss": vl,
                         **{"val_" + k: v for k, v in m[0.5].items()}, "interval_time_s": time.time() - t0,
                         "step_time_s": float(np.mean(t_step))})
                run, gns, t0, t_step = [], [], time.time(), []
                torch.save(model.state_dict(), os.path.join(a.out, "last.pt"))
                if vl < best or not os.path.exists(os.path.join(a.out, "best.pt")):
                    best = min(best, vl)
                    torch.save(model.state_dict(), os.path.join(a.out, "best.pt"))
                save_state()
            elif step % a.save_every == 0:
                save_state()
                if a.deadline and time.time() > a.deadline:
                    print(f"deadline reached at step {step}; state saved", flush=True)
                    log.finish()
                    sys.exit(3)
    save_state()
    print(f"stopped at step {step} (early stop: {es['stopped']}); best strict val {es['best']:.4f} at step {es['best_step']}",
          flush=True)
    model.load_state_dict(torch.load(os.path.join(a.out, "best_sel.pt")))
    vl, m = T.evaluate(model, va, gv, pw, cuts=(0.1, 0.3, 0.5, 0.7, 0.9))
    scan = [dict(cut=c, **mm) for c, mm in m.items()]
    for row in scan:
        print("scan", json.dumps(row), flush=True)
    json.dump(scan, open(os.path.join(a.out, "cut_scan.json"), "w"), indent=1)
    log.finish({"cut_scan": scan})


if __name__ == "__main__":
    main()
