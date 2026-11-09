"""Filtered graphs for mu 60 in a compact per-block format (the parquet graph files of train_filter.py apply are
~29 GB per 1000 events at 1.4M edges/event and slow to write): embedding -> radius graph -> filter -> edges with
score > cut, saved as events<a>-<b>.d/ (one .npy per array, memory-mappable) with event_id, off (edge offsets per event), src/dst (int32), score
(float16). read_compact() returns {event_id: {"src", "dst", "score"}} like common.read_graphs.

  python graph_apply60.py --cache C --emb EMB --radius R --filter F --cut CUT --out G --blocks 0 1 2
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "gnn"))
import train_filter as tf  # noqa: E402
from common import DIRECTION, Embedder, edge_counts, hit_features, load_events, radius_graph, summarize_counts  # noqa: E402


def read_compact(path, first=None, last=None):
    """path: events<a>-<b>.npz (loaded fully) or events<a>-<b>.d/ (one .npy per array, memory-mapped: only the
    events used are read). Scores stay float16."""
    if path.endswith(".d") or os.path.isdir(path):
        z = {k: np.load(os.path.join(path, k + ".npy"), mmap_mode="r") for k in ["event_id", "off", "src", "dst", "score"]}
    else:
        z = np.load(path)
    out = {}
    eids, off = np.asarray(z["event_id"]), np.asarray(z["off"])
    for i, e in enumerate(eids):
        e = int(e)
        if first is not None and not (first <= e <= last):
            continue
        a, b = off[i], off[i + 1]
        out[e] = {"src": z["src"][a:b], "dst": z["dst"][a:b], "score": z["score"][a:b]}
    return out


def npz_to_dir(path):
    """Convert events<a>-<b>.npz to the memory-mappable events<a>-<b>.d/ layout."""
    d = path[:-4] + ".d"
    os.makedirs(d + ".tmp", exist_ok=True)
    z = np.load(path)
    for k in ["event_id", "off", "src", "dst", "score"]:
        np.save(os.path.join(d + ".tmp", k + ".npy"), z[k])
    os.replace(d + ".tmp", d)
    return d


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True); ap.add_argument("--emb", required=True); ap.add_argument("--radius", type=float, required=True)
    ap.add_argument("--k-max", type=int, default=512); ap.add_argument("--filter", required=True); ap.add_argument("--cut", type=float, required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--blocks", type=int, nargs="+", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    ecfg = json.load(open(os.path.join(a.emb, "config.json")))
    emb = Embedder(hidden=ecfg["hidden"], n_layers=ecfg["layers"], emb_dim=ecfg["emb_dim"]).cuda()
    emb.load_state_dict(torch.load(os.path.join(a.emb, "best.pt"))); emb.eval()
    fcfg = json.load(open(os.path.join(a.filter, "config.json")))
    DIRECTION["on"] = bool(fcfg.get("direction", False))
    filt = tf.build_model(fcfg["hidden"], fcfg["layers"])
    filt.load_state_dict(torch.load(os.path.join(a.filter, "best.pt"))); filt.eval()
    for b in a.blocks:
        path = os.path.join(a.out, f"events{1000 * b}-{1000 * b + 999}.d")
        if os.path.exists(path) or os.path.exists(path[:-2] + ".npz"):
            print("exists", path, flush=True); continue
        t0 = time.time()
        evs = load_events(a.cache, 1000 * b, 1000 * b + 999)
        eids, offs, S, D, P, cs = [], [0], [], [], [], []
        for ev in evs:
            s, t, _ = radius_graph(emb(hit_features(ev, "cuda")), a.radius, a.k_max)
            p = torch.sigmoid(tf.scores(filt, ev, s, t))
            k = p > a.cut
            s, t, p = s[k], t[k], p[k]
            if len(cs) < 50:
                cs.append(edge_counts(ev, s, t, "cuda"))
            S.append(s.int().cpu().numpy()); D.append(t.int().cpu().numpy()); P.append(p.half().cpu().numpy())
            eids.append(ev["event_id"]); offs.append(offs[-1] + len(S[-1]))
        os.makedirs(path + ".tmp", exist_ok=True)
        for k, v in dict(event_id=np.array(eids, np.int64), off=np.array(offs, np.int64), src=np.concatenate(S),
                         dst=np.concatenate(D), score=np.concatenate(P)).items():
            np.save(os.path.join(path + ".tmp", k + ".npy"), v)
        os.replace(path + ".tmp", path)
        print(f"block {b}: {len(evs)} events, {offs[-1] / len(evs) / 1e6:.2f}M edges/event, {time.time() - t0:.0f} s,",
              json.dumps(summarize_counts(cs)), flush=True)


if __name__ == "__main__":
    main()
