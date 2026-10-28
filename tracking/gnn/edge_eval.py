"""Edge-level efficiency and purity of the quirk GNNs on saved scored graphs (no inference; CPU only).

Inputs per model: <run>/gnn/scored/events*.parquet (train_gnn.py infer: src, dst, score for every candidate edge of
the model's own graph). Truth for every model comes from the per-quirk direction cache (pid = R1 particle_id, i.e.
the first entry of the packaged particle_ids; hit_quirk; time-ordered chain) plus the pair cache's chain (one chain
over the union of both quirks' hits, in true time order). Hit order is the same in every cache and graph.

Edge classes (mutually exclusive), "chain" = consecutive hits of one particle in true time order (make_cache.py,
the training truth of the per-quirk GNNs), "pair chain" = consecutive hits of the quirk pair's merged time-ordered
sequence (make_pair_cache.py, the truth of the pair GNN e2):
  ord_chain_tech / ord_chain_low : chain edge of an ordinary (non-quirk) particle; tech = pT > 1 GeV and >= 7 hits
  ord_same_nonchain : two hits of one ordinary particle, not consecutive (masked, weight 0, in training)
  q_chain_pc / q_chain_npc : chain edge of one quirk (same quirk, consecutive), also / not a pair-chain edge
  q_same_nonchain_pc / _npc : two hits of one quirk, not consecutive in that quirk's chain
  cross_pc / cross_npc : one hit of quirk A, one of quirk B (same event), on / not on the pair chain
  quirk_ord : one quirk hit, one ordinary hit
  other : two ordinary hits of different particles

Quirk-rule variant (tracking/quirk_rule.py rule_labels on the packaged particle_ids): a hit belongs to quirk q if q is
anywhere in its particle_ids (a hit of both quirks belongs to both); a hit whose particles are all quirk descendants
(quirk-secondary) is in no particle's truth, and every edge touching one is put in class 'sec' (excluded). Quirk
membership by the rule, everything else as above; chains unchanged (an edge is same-quirk if its two hits share a
quirk, cross if both are quirk hits and share none).

Output per (model, event range): score histograms per class (1000 bins on [0, 1]), exact counts above the cuts in
CUTS, truth denominators (all chain / pair-chain edges, including those missing from the graph), and the
training-log quantities (common.edge_counts at 0.5) for the consistency check.

  python edge_eval.py run --out DIR --workers 28        # all models x blocks -> DIR/raw/*.npz
  python edge_eval.py summarize --out DIR               # -> DIR/edge_eval.json
"""

import argparse
import glob
import json
import os
import pickle
import sys
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from common import read_graphs  # noqa: E402

G = "/pscratch/sd/d/danieltm/quirk_tracking/gnn"
CACHE = f"{G}/cache/m200_L774_mu0_dir"
PAIR_CACHE = f"{G}/cache/m200_L774_mu0_dir_pair"
V2_CACHE = f"{G}/cache/m200_L774_mu0"
PACKAGED = "/global/cfs/cdirs/m4958/data/ColliderML/simulation/quirks/m200_L774_mu0/v1/parquet"

# model key -> (scored dir, target kind, label, operating cuts used for track building)
MODELS = {
    "v2": (f"{G}/main/v2/gnn_node/scored", "perquirk", "per-quirk GNN, graph v2", {"walk": [0.1, 0.6]}),
    "v3_s0": (f"{G}/main/g2_q50_dir_s0/gnn/scored", "perquirk", "per-quirk GNN, graph v3 (seed 0)", {"walk": [0.1, 0.6]}),
    "v3_s1": (f"{G}/main/g2_q50_dir_s1/gnn/scored", "perquirk", "per-quirk GNN, graph v3 (seed 1)", {"walk": [0.1, 0.6]}),
    "e2_s0": (f"{G}/main/e2_pair_dir_s0/gnn/scored", "pairchain", "pair GNN, own graph (seed 0)", {"components": [0.5]}),
    "e2_s1": (f"{G}/main/e2_pair_dir_s1/gnn/scored", "pairchain", "pair GNN, own graph (seed 1)", {"components": [0.5]}),
    "e2_s2": (f"{G}/main/e2_pair_dir_s2/gnn/scored", "pairchain", "pair GNN, own graph (seed 2)", {"components": [0.5]}),
    "p3": (f"{G}/main/p3_pairv3_s0/gnn/scored", "pairall", "pair GNN, graph v3 (seed 0)", {"components": [0.7]}),
}
CLASSES = ["ord_chain_tech", "ord_chain_low", "ord_same_nonchain", "q_chain_pc", "q_chain_npc", "q_same_nonchain_pc",
           "q_same_nonchain_npc", "cross_pc", "cross_npc", "quirk_ord", "other", "sec"]
CI = {c: i for i, c in enumerate(CLASSES)}
NB = 1000
CUTS = [0.0, 0.1, 0.3, 0.5, 0.6, 0.7, 0.9]
BLOCKS = [(16000, 16999), (17000, 17999), (18000, 18999), (19000, 19999)]


def tag_of(eid):
    if eid < 16200:
        return "val200"
    if eid < 17000:
        return "val800"
    return "test"


def load_cache(path, first, last):
    f = os.path.join(path, f"events{first}-{last}.pkl")
    with open(f, "rb") as fh:
        return {e["event_id"]: e for e in pickle.load(fh)}


def rule_hit_labels(first, last):
    """{event_id: (qa, qb, sec)} boolean arrays per hit under the quirk rule; quirk A/B = quirk particle_ids sorted."""
    import polars as pl
    import quirk_rule as qr
    hits, parts = qr.read_packaged(PACKAGED, first, last + 1)
    kinds, _ = qr.quirk_descendants(parts)
    labels, _ = qr.rule_labels(hits, kinds)
    nh = hits.group_by("event_id").agg(pl.col("hit_id").max().alias("n"))
    nmap = dict(zip(nh["event_id"].to_list(), (np.asarray(nh["n"].to_list()) + 1).tolist()))
    qk = kinds.filter(pl.col("is_quirk")).group_by("event_id").agg(pl.col("particle_id").sort())
    qmap = dict(zip(qk["event_id"].to_list(), qk["particle_id"].to_list()))
    out = {}
    lab = labels.partition_by("event_id", as_dict=True)
    for eid, n in nmap.items():
        d = lab.get((eid,))
        qa, qb, has = np.zeros(n, bool), np.zeros(n, bool), np.zeros(n, bool)
        if d is not None:
            h = d["hit_id"].to_numpy()
            l = d["label"].to_numpy()
            has[h] = True
            q = qmap.get(eid, [])
            if len(q) > 0:
                qa[h[l == q[0]]] = True
            if len(q) > 1:
                qb[h[l == q[1]]] = True
        out[eid] = (qa, qb, ~has)
    return out


def keyset(a, b, n):
    return np.minimum(a, b).astype(np.int64) * n + np.maximum(a, b)


def classify(ev, evp, src, dst, rule=None):
    """Class index per edge (primary truth; with rule=(qa, qb, sec) the quirk-rule variant)."""
    n = len(ev["pid"])
    ek = keyset(src, dst, n)
    ch = ev["chain"].astype(np.int64)
    inC = np.isin(ek, keyset(ch[0], ch[1], n))
    pc = evp["chain"].astype(np.int64)
    pq_ = ev["hit_quirk"][pc[0]] & ev["hit_quirk"][pc[1]]
    inPC = np.isin(ek, keyset(pc[0, pq_], pc[1, pq_], n))
    pid = ev["pid"]
    same = pid[src] == pid[dst]
    if rule is None:
        hq = ev["hit_quirk"]
        qs, qd = hq[src], hq[dst]
        sameq = qs & qd & same
        cross = qs & qd & ~same
        sec = np.zeros(len(src), bool)
    else:
        qa, qb, sh = rule
        hq = qa | qb
        qs, qd = hq[src], hq[dst]
        sameq = (qa[src] & qa[dst]) | (qb[src] & qb[dst])
        cross = qs & qd & ~sameq
        sec = sh[src] | sh[dst]
    tech = (ev["hit_pt"][src] > 1.0) & (ev["hit_nhits"][src] >= 7)
    ordn = ~qs & ~qd
    c = np.full(len(src), CI["other"], np.int8)
    c[qs ^ qd] = CI["quirk_ord"]
    c[ordn & same & inC & tech] = CI["ord_chain_tech"]
    c[ordn & same & inC & ~tech] = CI["ord_chain_low"]
    c[ordn & same & ~inC] = CI["ord_same_nonchain"]
    c[sameq & inC & inPC] = CI["q_chain_pc"]
    c[sameq & inC & ~inPC] = CI["q_chain_npc"]
    c[sameq & ~inC & inPC] = CI["q_same_nonchain_pc"]
    c[sameq & ~inC & ~inPC] = CI["q_same_nonchain_npc"]
    c[cross & inPC] = CI["cross_pc"]
    c[cross & ~inPC] = CI["cross_npc"]
    c[sec] = CI["sec"]
    return c


def truth_denoms(ev, evp, rule):
    """All truth edges, including those not in any graph."""
    hq = ev["hit_quirk"]
    ch = ev["chain"]
    q = hq[ch[0]]
    tech = (~q) & (ev["hit_pt"][ch[0]] > 1.0) & (ev["hit_nhits"][ch[0]] >= 7)
    pc = evp["chain"]
    pq_ = hq[pc[0]] & hq[pc[1]]
    cross_pc = pq_ & (ev["pid"][pc[0]] != ev["pid"][pc[1]])
    qa, qb, sh = rule
    secc = sh[ch[0]] | sh[ch[1]]
    rq = (qa[ch[0]] & qa[ch[1]]) | (qb[ch[0]] & qb[ch[1]])
    rqh = qa | qb
    rord = ~rqh[ch[0]] & ~rqh[ch[1]] & ~secc
    return dict(chain_all=ch.shape[1], chain_quirk=int(q.sum()), chain_ord=int((~q).sum()), chain_tech=int(tech.sum()),
                pc_quirk=int(pq_.sum()), pc_cross=int(cross_pc.sum()), pc_same=int((pq_ & ~cross_pc).sum()),
                rule_chain_quirk=int(rq.sum()), rule_chain_ord=int(rord.sum()),
                rule_chain_ord_tech=int((rord & tech).sum()))


def logdef_counts(ev_log, src, dst, k):
    """common.edge_counts on numpy (the training-log quantities) for kept edges k, against the cache the model used."""
    n = len(ev_log["pid"])
    ch = ev_log["chain"].astype(np.int64)
    ck = keyset(ch[0], ch[1], n)
    s, d = src[k], dst[k]
    ek = keyset(s, d, n)
    found = np.isin(ck, ek)
    is_true = np.isin(ek, ck)
    same = ev_log["pid"][s] == ev_log["pid"][d]
    hq = ev_log["hit_quirk"]
    q = hq[ch[0]]
    tech = (~q) & (ev_log["hit_pt"][ch[0]] > 1.0) & (ev_log["hit_nhits"][ch[0]] >= 7)
    return dict(edges=int(k.sum()), true=int(is_true.sum()), samepid=int(same.sum()), chain_all=int(len(ck)),
                found_all=int(found.sum()), chain_quirk=int(q.sum()), found_quirk=int((found & q).sum()),
                chain_tech=int(tech.sum()), found_tech=int((found & tech).sum()),
                quirk_touch=int((hq[s] | hq[d]).sum()), quirk_true=int((is_true & hq[s] & hq[d]).sum()))


def task(args):
    model, (first, last), out = args
    path = os.path.join(out, "raw", f"{model}_{first}.npz")
    if os.path.exists(path):
        return path, "skip"
    sdir, kind = MODELS[model][:2]
    ev = load_cache(CACHE, first, last)
    evp = load_cache(PAIR_CACHE, first, last)
    log_cache = None
    if model == "v2":
        log_cache = load_cache(V2_CACHE, first, last)
    rule = rule_hit_labels(first, last)
    g = read_graphs(os.path.join(sdir, f"events{first}-{last}.parquet"))
    tags = ["val200", "val800", "test"]
    H = {t: np.zeros((2, len(CLASSES), NB), np.int64) for t in tags}       # [primary/rule, class, bin]
    X = {t: np.zeros((2, len(CLASSES), len(CUTS)), np.int64) for t in tags}  # exact counts with score > cut
    D = {t: {} for t in tags}
    L = {t: {} for t in tags}
    nev = {t: 0 for t in tags}
    checks = {"chain_equal_v2": True, "quirk_hits_cache": 0, "quirk_hits_rule_first": 0}
    for eid in sorted(g):
        e, ep, r = ev[eid], evp[eid], rule[eid]
        assert len(r[0]) == len(e["pid"]), (eid, len(r[0]), len(e["pid"]))
        t = tag_of(eid)
        nev[t] += 1
        src = g[eid]["src"].astype(np.int64)
        dst = g[eid]["dst"].astype(np.int64)
        sc = g[eid]["score"]
        b = np.clip((sc * NB).astype(np.int64), 0, NB - 1)
        for v, rr in enumerate([None, r]):
            c = classify(e, ep, src, dst, rr).astype(np.int64)
            H[t][v] += np.bincount(c * NB + b, minlength=len(CLASSES) * NB).reshape(len(CLASSES), NB)
            for j, cut in enumerate(CUTS):
                X[t][v][:, j] += np.bincount(c[sc > cut], minlength=len(CLASSES))
        for kk, vv in truth_denoms(e, ep, r).items():
            D[t][kk] = D[t].get(kk, 0) + vv
        elog = ep if kind != "perquirk" else (log_cache[eid] if log_cache is not None else e)
        if log_cache is not None and not np.array_equal(log_cache[eid]["chain"], e["chain"]):
            checks["chain_equal_v2"] = False
        for cut in (0.5,):
            for kk, vv in logdef_counts(elog, src, dst, sc > cut).items():
                L[t][kk] = L[t].get(kk, 0) + vv
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, **{f"H_{t}": H[t] for t in tags}, **{f"X_{t}": X[t] for t in tags},
             meta=json.dumps({"model": model, "block": [first, last], "D": D, "L": L, "nev": nev, "checks": checks}))
    return path, nev


# ----------------------------------------------------------------------------- summary
def true_classes(kind, which):
    """Classes counted as true. which: 'target' (the model's training target), 'consec' (consecutive-hit truth of the
    cache the model's log used: per-quirk chain, or the pair chain), 'same' (any two hits of one particle; for pair
    models the pair is one particle)."""
    ordc = ["ord_chain_tech", "ord_chain_low"]
    if which == "same":
        base = ordc + ["ord_same_nonchain", "q_chain_pc", "q_chain_npc", "q_same_nonchain_pc", "q_same_nonchain_npc"]
        return base + (["cross_pc", "cross_npc"] if kind != "perquirk" else [])
    if kind == "perquirk":
        return ordc + ["q_chain_pc", "q_chain_npc"]
    if kind == "pairchain" or which == "consec":
        return ordc + ["q_chain_pc", "q_same_nonchain_pc", "cross_pc"]
    return ordc + ["q_chain_pc", "q_chain_npc", "q_same_nonchain_pc", "q_same_nonchain_npc", "cross_pc", "cross_npc"]


QTOUCH = ["q_chain_pc", "q_chain_npc", "q_same_nonchain_pc", "q_same_nonchain_npc", "cross_pc", "cross_npc", "quirk_ord"]


def metrics(cnt, D, kind, rule=False):
    """cnt: [class, ncut] counts above each cut (np array). Returns dict of arrays over the cuts."""
    s = lambda names: sum(cnt[CI[c]] for c in names)  # noqa: E731
    excl = ["sec"] if rule else []
    allc = [c for c in CLASSES if c not in excl]
    out = {}
    pre = "rule_" if rule else ""
    out["eff_ordinary"] = s(["ord_chain_tech", "ord_chain_low"]) / D[pre + "chain_ord"]
    out["eff_ordinary_tech"] = s(["ord_chain_tech"]) / D[pre + ("chain_ord_tech" if rule else "chain_tech")]
    out["eff_samequirk"] = s(["q_chain_pc", "q_chain_npc"]) / D[pre + "chain_quirk"]
    out["eff_pairchain_cross"] = cnt[CI["cross_pc"]] / D["pc_cross"]
    out["eff_pairchain"] = s(["q_chain_pc", "q_same_nonchain_pc", "cross_pc"]) / D["pc_quirk"]
    out["n_cross_kept"] = s(["cross_pc", "cross_npc"])
    out["n_samequirk_any_kept"] = s(["q_chain_pc", "q_chain_npc", "q_same_nonchain_pc", "q_same_nonchain_npc"])
    out["n_edges"] = s(allc)
    out["n_quirk_touch"] = s(QTOUCH)
    for which in ["target", "consec", "same"]:
        tc = [c for c in true_classes(kind, which) if c not in excl]
        out[f"purity_all_{which}"] = s(tc) / np.maximum(out["n_edges"], 1)
        out[f"purity_quirk_{which}"] = s([c for c in tc if c in QTOUCH]) / np.maximum(out["n_quirk_touch"], 1)
    return out


def summarize(a):
    res = {"definitions": __doc__, "cuts": CUTS, "models": {}}
    hists = {}
    for model, (sdir, kind, label, op) in MODELS.items():
        files = sorted(glob.glob(os.path.join(a.out, "raw", f"{model}_*.npz")))
        assert len(files) == 4, (model, files)
        agg = {}
        for f in files:
            z = np.load(f)
            m = json.loads(str(z["meta"]))
            for t in ["val200", "val800", "test"]:
                if m["nev"][t] == 0:
                    continue
                A = agg.setdefault(t, {"H": 0, "X": 0, "D": {}, "L": {}, "nev": 0, "checks": []})
                A["H"] = A["H"] + z[f"H_{t}"]
                A["X"] = A["X"] + z[f"X_{t}"]
                A["nev"] += m["nev"][t]
                for k in ["D", "L"]:
                    for kk, vv in m[k][t].items():
                        A[k][kk] = A[k].get(kk, 0) + vv
                A["checks"].append(m["checks"])
        agg["val"] = {"H": agg["val200"]["H"] + agg["val800"]["H"], "X": agg["val200"]["X"] + agg["val800"]["X"],
                      "nev": agg["val200"]["nev"] + agg["val800"]["nev"],
                      "D": {k: agg["val200"]["D"][k] + agg["val800"]["D"][k] for k in agg["val200"]["D"]},
                      "L": {k: agg["val200"]["L"][k] + agg["val800"]["L"][k] for k in agg["val200"]["L"]}}
        mres = {"label": label, "target": kind, "operating_cuts": op, "scored": sdir}
        for t in ["test", "val", "val200"]:
            A = agg[t]
            ex = {}
            for v, name in enumerate(["primary", "rule"]):
                mm = metrics(A["X"][v].astype(float), A["D"], kind, rule=(v == 1))
                ex[name] = {k: [float(x) for x in np.atleast_1d(vv)] for k, vv in mm.items()}
                # curves: cumulative counts above bin edges k/NB (score >= k/NB)
                cum = np.cumsum(A["H"][v][:, ::-1], axis=1)[:, ::-1].astype(float)
                cm = metrics(cum, A["D"], kind, rule=(v == 1))
                hists[(model, t, name)] = {k: np.atleast_1d(vv) for k, vv in cm.items()}
                ex[name]["cross_in_graph"] = int(A["H"][v][[CI["cross_pc"], CI["cross_npc"]]].sum())
                ex[name]["samequirk_any_in_graph"] = int(A["H"][v][[CI[c] for c in QTOUCH[:4]]].sum())
                ex[name]["class_counts_in_graph"] = {c: int(A["H"][v][CI[c]].sum()) for c in CLASSES}
            L = A["L"]
            logdef = {"eff": L["found_all"] / L["chain_all"], "eff_quirk": L["found_quirk"] / L["chain_quirk"],
                      "eff_tech": L["found_tech"] / L["chain_tech"], "purity": L["true"] / L["edges"],
                      "pid_purity": L["samepid"] / L["edges"], "purity_quirk": L["quirk_true"] / L["quirk_touch"],
                      "edges_per_event": L["edges"] / A["nev"]}
            mres[t] = {"n_events": A["nev"], "denominators": A["D"], "at_cuts": ex, "logdef_at_0.5": logdef}
        cs = json.load(open(os.path.join(os.path.dirname(sdir), "cut_scan.json")))
        mres["train_cut_scan_at_0.5"] = [r for r in cs if abs(r["cut"] - 0.5) < 1e-9][0]
        res["models"][model] = mres
    json.dump(res, open(os.path.join(a.out, "edge_eval.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(a.out, "curves.npz"),
                        **{f"{m}|{t}|{v}|{k}": arr for (m, t, v), d in hists.items() for k, arr in d.items()})
    print("wrote", os.path.join(a.out, "edge_eval.json"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["run", "summarize"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=28)
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--blocks", nargs="+", type=int, default=[b[0] for b in BLOCKS])
    a = ap.parse_args()
    if a.mode == "summarize":
        return summarize(a)
    jobs = [(m, b, a.out) for m in a.models for b in BLOCKS if b[0] in a.blocks]
    with Pool(min(a.workers, len(jobs))) as p:
        for r in p.imap_unordered(task, jobs):
            print(*r, flush=True)


if __name__ == "__main__":
    main()
