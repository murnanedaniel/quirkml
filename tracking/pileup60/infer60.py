"""GNN inference on the compact mu 60 graphs (graph_apply60.py): writes scored files in the train_gnn.py infer format,
keeping only edges with GNN score > --keep (0.01 = the lowest threshold any track builder uses).

  python infer60.py --run RUN_DIR --ckpt final.pt --cache C --graphs G --blocks 0 1 --out SCORED_DIR
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "gnn"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_gnn as T  # noqa: E402
from common import DIRECTION, edge_truth, load_events, write_graphs  # noqa: E402
from graph_apply60 import read_compact  # noqa: E402


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True); ap.add_argument("--ckpt", default="final.pt")
    ap.add_argument("--cache", required=True); ap.add_argument("--graphs", required=True)
    ap.add_argument("--blocks", type=int, nargs="+", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--keep", type=float, default=0.01)
    a = ap.parse_args()
    cfg = json.load(open(os.path.join(a.run, "config.json")))
    T.CLUSTER = cfg.get("cluster", "node")
    DIRECTION["on"] = bool(cfg.get("direction", False))
    import gnn60_models
    gnn60_models.patch_build_model(T, cfg.get("agg", "sum"))
    model = T.build_model(cfg["hidden"], cfg["n_iter"], cfg["layers"])
    model.load_state_dict(torch.load(os.path.join(a.run, a.ckpt))); model.eval()
    os.makedirs(a.out, exist_ok=True)
    for b in a.blocks:
        e0, e1 = 1000 * b, 1000 * b + 999
        path = os.path.join(a.out, f"events{e0}-{e1}.parquet")
        if os.path.exists(path):
            continue
        t0 = time.time()
        evs = load_events(a.cache, e0, e1)
        g = read_compact(os.path.join(a.graphs, f"events{e0}-{e1}.d"))
        rows = []
        for ev in evs:
            f, s, t, ef = T.batch([ev], g, with_truth=False)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                p = torch.sigmoid(model(f, s, t, ef).float())
            k = p > a.keep
            sn, tn = s[k].cpu().numpy(), t[k].cpu().numpy()
            y, _ = edge_truth(ev, sn, tn)
            rows.append(dict(event_id=ev["event_id"], src=sn, dst=tn, y=y, score=p[k].cpu().numpy()))
        write_graphs(path + ".tmp", rows)  # a run killed mid-write must not leave a file that looks finished
        os.replace(path + ".tmp", path)
        print(f"block {b}: {len(rows)} events, {np.mean([len(r['src']) for r in rows]):.0f} kept edges/event, {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
