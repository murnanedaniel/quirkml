"""Step 1 size check at mu 60 with the existing (mu 0 trained) graph-construction models, per event:
hits, quirk hits, metric-learning graph edges (and k_max saturation), filtered edges, truth-chain recall (all / quirk /
technical) before and after the filter, time and peak GPU memory per stage.

Graphs: graph v3 (g1_qw10 embedding, r 0.075, k <= 512; filter g1f_qw10r075_q50 with direction, cut 0.001) on the
direction cache; pair graph (v1 embedding, r 0.07, k <= 512; filter e2_pair_dir_s0, cut 0.005) on the pair cache.

  python sizecheck.py --cache-dir C --cache-pair CP --events 0 49 --out OUT.json [--filter-cuts ...]
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
from common import DIRECTION, Embedder, edge_counts, hit_features, load_events, radius_graph  # noqa: E402

M = "/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirk_tracking_data/models/gnn"
GRAPHS = {
    "v3": dict(cache="dir", emb=f"{M}/g1_qw10/emb", r=0.075, k=512, filt=f"{M}/g1f_qw10r075_q50/filter", cut=0.001),
    "pair": dict(cache="pair", emb=f"{M}/v1/emb", r=0.07, k=512, filt=f"{M}/e2_pair_dir_s0/filter", cut=0.005),
}


def load_emb(d):
    cfg = json.load(open(os.path.join(d, "config.json")))
    m = Embedder(hidden=cfg["hidden"], n_layers=cfg["layers"], emb_dim=cfg["emb_dim"]).cuda()
    m.load_state_dict(torch.load(os.path.join(d, "best.pt"))); m.eval()
    return m


def load_filter(d):
    cfg = json.load(open(os.path.join(d, "config.json")))
    DIRECTION["on"] = bool(cfg.get("direction", False))
    m = tf.build_model(cfg["hidden"], cfg["layers"])
    m.load_state_dict(torch.load(os.path.join(d, "best.pt"))); m.eval()
    return m


def rec(c):
    return dict(edges=c["edges"], eff=c["found_all"] / max(c["chain_all"], 1), eff_quirk=c["found_quirk"] / max(c["chain_quirk"], 1),
                eff_tech=c["found_tech"] / max(c["chain_tech"], 1), found_quirk=c["found_quirk"], chain_quirk=c["chain_quirk"],
                found_all=c["found_all"], chain_all=c["chain_all"], found_tech=c["found_tech"], chain_tech=c["chain_tech"])


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True); ap.add_argument("--cache-pair", required=True)
    ap.add_argument("--events", type=int, nargs=2, required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--extra-cuts", type=float, nargs="*", default=[0.01, 0.05, 0.1])
    a = ap.parse_args()
    res = {}
    for name, g in GRAPHS.items():
        evs = load_events(a.cache_dir if g["cache"] == "dir" else a.cache_pair, *a.events)
        emb, filt = load_emb(g["emb"]), load_filter(g["filt"])
        rows = []
        for ev in evs:
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            e = emb(hit_features(ev, "cuda"))
            torch.cuda.synchronize(); t1 = time.time()
            s, t, sat = radius_graph(e, g["r"], g["k"])
            torch.cuda.synchronize(); t2 = time.time()
            mem_graph = torch.cuda.max_memory_allocated() / 2**30
            torch.cuda.reset_peak_memory_stats()
            p = torch.sigmoid(tf.scores(filt, ev, s, t))
            torch.cuda.synchronize(); t3 = time.time()
            mem_filter = torch.cuda.max_memory_allocated() / 2**30
            r = dict(event_id=ev["event_id"], hits=len(ev["x"]), quirk_hits=int(ev["hit_quirk"].sum()), sat=sat,
                     t_emb=t1 - t0, t_graph=t2 - t1, t_filter=t3 - t2, mem_graph_gb=mem_graph, mem_filter_gb=mem_filter,
                     graph=rec(edge_counts(ev, s, t, "cuda")))
            for c in [g["cut"]] + a.extra_cuts:
                k = p > c
                r[f"filter_{c:g}"] = rec(edge_counts(ev, s[k], t[k], "cuda"))
            rows.append(r)
            print(name, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if not isinstance(v, dict)}),
                  "graph", r["graph"]["edges"], "%.3f" % r["graph"]["eff_quirk"], "filt", r[f"filter_{g['cut']:g}"]["edges"],
                  "%.3f" % r[f"filter_{g['cut']:g}"]["eff_quirk"], flush=True)
        tot = lambda key, f: sum(x[key][f] for x in rows)  # noqa: E731
        summ = dict(events=len(rows), hits_mean=float(np.mean([x["hits"] for x in rows])), hits_max=int(max(x["hits"] for x in rows)),
                    quirk_hits_mean=float(np.mean([x["quirk_hits"] for x in rows])), sat_mean=float(np.mean([x["sat"] for x in rows])),
                    t_emb=float(np.mean([x["t_emb"] for x in rows[1:]])), t_graph=float(np.mean([x["t_graph"] for x in rows[1:]])),
                    t_filter=float(np.mean([x["t_filter"] for x in rows[1:]])),
                    mem_graph_gb_max=float(max(x["mem_graph_gb"] for x in rows)), mem_filter_gb_max=float(max(x["mem_filter_gb"] for x in rows)))
        for key in ["graph"] + [f"filter_{c:g}" for c in [g["cut"]] + a.extra_cuts]:
            summ[key] = dict(edges_mean=float(np.mean([x[key]["edges"] for x in rows])), edges_max=int(max(x[key]["edges"] for x in rows)),
                             eff=tot(key, "found_all") / tot(key, "chain_all"), eff_quirk=tot(key, "found_quirk") / tot(key, "chain_quirk"),
                             eff_tech=tot(key, "found_tech") / max(tot(key, "chain_tech"), 1))
        res[name] = dict(summary=summ, cut=g["cut"], rows=rows)
        print(name, json.dumps(summ), flush=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
