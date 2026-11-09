"""Zero-shot inference at mu 60 in one pass per event (no graph files: at the mu 0 filter cuts the filtered graphs
are ~10M edges/event): embedding -> radius graph -> filter (cut) -> GNN, then only edges with GNN score > --keep are
written (scored files in the train_gnn.py infer format). --keep 0.01 = the lowest threshold any track builder uses
(ccwalk components at 0.01, walk 0.1, components 0.5), so built tracks are the same as from the full scored graph.
Per-event stats (edges per stage, time, peak GPU memory) to <out>/stats.jsonl.

  python zs_fused.py --cache C --emb EMB_DIR --radius R --k-max K --filter FILTER_DIR --cut CUT --gnn GNN_DIR
                     --ranges 0-999 1000-1999 --out OUT_DIR [--keep 0.01]
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
import train_gnn as tg  # noqa: E402
from common import DIRECTION, Embedder, edge_counts, edge_truth, hit_features, load_events, radius_graph, write_graphs  # noqa: E402


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True); ap.add_argument("--emb", required=True)
    ap.add_argument("--radius", type=float, required=True); ap.add_argument("--k-max", type=int, default=512)
    ap.add_argument("--filter", required=True); ap.add_argument("--cut", type=float, required=True)
    ap.add_argument("--gnn", required=True); ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--ranges", nargs="+", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--keep", type=float, default=0.01); ap.add_argument("--max-events", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    ecfg = json.load(open(os.path.join(a.emb, "config.json")))
    emb = Embedder(hidden=ecfg["hidden"], n_layers=ecfg["layers"], emb_dim=ecfg["emb_dim"]).cuda()
    emb.load_state_dict(torch.load(os.path.join(a.emb, "best.pt"))); emb.eval()
    fcfg = json.load(open(os.path.join(a.filter, "config.json")))
    DIRECTION["on"] = bool(fcfg.get("direction", False))
    filt = tf.build_model(fcfg["hidden"], fcfg["layers"])
    filt.load_state_dict(torch.load(os.path.join(a.filter, "best.pt"))); filt.eval()
    gcfg = json.load(open(os.path.join(a.gnn, "config.json")))
    tg.CLUSTER = gcfg.get("cluster", "node")
    DIRECTION["on"] = bool(gcfg.get("direction", False))
    gnn = tg.build_model(gcfg["hidden"], gcfg["n_iter"], gcfg["layers"])
    gnn.load_state_dict(torch.load(os.path.join(a.gnn, a.ckpt))); gnn.eval()
    stats = open(os.path.join(a.out, "stats.jsonl"), "a")
    for rng in a.ranges:
        first, last = map(int, rng.split("-"))
        path = os.path.join(a.out, f"events{first}-{last}.parquet")
        if os.path.exists(path):
            print("exists", path, flush=True); continue
        evs = load_events(a.cache, first, last)
        if a.max_events:
            evs = evs[: a.max_events]
        rows = []
        for ev in evs:
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t0 = time.time()
            s, t, sat = radius_graph(emb(hit_features(ev, "cuda")), a.radius, a.k_max)
            n_graph = s.numel()
            DIRECTION["on"] = bool(fcfg.get("direction", False))
            p = torch.sigmoid(tf.scores(filt, ev, s, t))
            k = p > a.cut
            s, t, p = s[k], t[k], p[k]
            del k
            torch.cuda.synchronize(); t1 = time.time(); m1 = torch.cuda.max_memory_allocated() / 2**30
            torch.cuda.reset_peak_memory_stats()
            DIRECTION["on"] = bool(gcfg.get("direction", False))
            g = {ev["event_id"]: {"src": s, "dst": t, "score": p}}
            f, ss, tt, ef = tg.batch([ev], g, with_truth=False)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                q = torch.sigmoid(gnn(f, ss, tt, ef).float())
            torch.cuda.synchronize(); t2 = time.time(); m2 = torch.cuda.max_memory_allocated() / 2**30
            c05 = edge_counts(ev, s[q > 0.5], t[q > 0.5], "cuda")
            keep = q > a.keep
            sn, tn, qn = s[keep].cpu().numpy(), t[keep].cpu().numpy(), q[keep].cpu().numpy()
            y, _ = edge_truth(ev, sn, tn)
            rows.append(dict(event_id=ev["event_id"], src=sn, dst=tn, y=y, score=qn))
            r = dict(event_id=ev["event_id"], hits=len(ev["x"]), sat=sat, graph_edges=n_graph, filtered_edges=int(s.numel()),
                     kept_edges=int(len(sn)), t_graph_filter=t1 - t0, t_gnn=t2 - t1, mem_graph_filter_gb=m1, mem_gnn_gb=m2,
                     gnn05_eff_quirk=c05["found_quirk"] / max(c05["chain_quirk"], 1), gnn05_eff=c05["found_all"] / max(c05["chain_all"], 1),
                     gnn05_edges=c05["edges"])
            stats.write(json.dumps(r) + "\n"); stats.flush()
            del s, t, p, q, f, ss, tt, ef, g
        write_graphs(path, rows)
        print("wrote", path, len(rows), flush=True)


if __name__ == "__main__":
    main()
