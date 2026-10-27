"""Event-display gallery of quirk pairs, coupled task, strict matching (> 50% purity and completeness), test events
17000-19999 of m200_L774_mu0.

Models (predictions as scored in the strict ACTS reruns, /pscratch/sd/d/danieltm/quirk_tracking/strict50/acts):
  gnn: pair GNN gnn-e2-pair-dir-s0, connected components at 0.5, >= 4 hits  (acts/e2_s0_cc_pair)
  mf:  MaskFormer mf-E-pairdir-long, final checkpoint, object threshold 0.1    (acts/mf_long_ot0.1_pair)
Found / not found per pair = the ACTS strict per-event flags (per_event.parquet: quirks_selected,
quirks_matched_very_loose_strict) on the pair-truth copy; only selected pairs (the efficiency denominator) are used.
A Python re-implementation of the matching (pair = union of both quirks' Release 1 hits; majority particle,
purity and completeness > 0.5) is checked against ACTS on every selected pair and must agree.

Sampling: numpy default_rng(SEED), 20 found and 20 not found per model, sorted by event id.
Panels: x-y (top) and r-z (bottom), zoomed to the pair; pair hits on a third or later track ringed grey; quirk A dark blue circles, quirk B light orange squares;
hits of the best-matching track (most pair hits) ringed teal; for not-found pairs the second track ringed magenta
when it holds >= 25% of the pair's hits; hits of other tracks near the pair light grey.
Also: an edge figure for three pairs, graph v3 candidate edges among the pair's hits, coloured by the per-quirk GNN
(gnn-g2-v3-dir-s0) and the pair GNN on graph v3 (gnn-p3-pairv3-s0) scores.

    ~/.conda/envs/collider-env/bin/python fig_pair_gallery.py
Writes figs/gallery_<model>_<found|notfound>_<k>.png, figs/gallery_edges.png, figs/pair_gallery_table.csv,
numbers in figs/numbers.json (key pair_gallery).
"""
import csv
import os
import pickle
import sys
from collections import Counter

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qstyle import C, FIGS, plt, record, save, setup  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

SEED = 20261026
QS = "/pscratch/sd/d/danieltm/quirk_tracking"
CACHE = f"{QS}/gnn/cache/m200_L774_mu0/events{{a}}-{{b}}.pkl"
R1P = "/pscratch/sd/d/danieltm/quirk_pilots/r1/m200_L774_mu0/truth/particles"
MODELS = {
    "gnn": dict(name="Pair GNN", acts=f"{QS}/strict50/acts/e2_s0_cc_pair",
                pred=f"{QS}/gnn/main/e2_pair_dir_s0/test/pred_cc0.5_mh4/predictions.parquet"),
    "mf": dict(name="MaskFormer", acts=f"{QS}/strict50/acts/mf_long_ot0.1_pair",
               pred=f"{QS}/maskformer/predictions/t2e_pairdir_long_e049_test_ot0.1/predictions.parquet"),
}
QPDG = 10000100
RING = {"best": "#0f7c78", "second": "#c2185b"}


def load_cache():
    ev = {}
    for a in (17000, 18000, 19000):
        for e in pickle.load(open(CACHE.format(a=a, b=a + 999), "rb")):
            ev[e["event_id"]] = {k: e[k] for k in ("x", "y", "z", "pid")}
    return ev


def quirk_ids():
    out = {}
    for f in sorted(os.listdir(R1P)):
        a, b = f.split(".events")[1][:-8].split("-")
        if int(b) < 17000 or int(a) > 19999:
            continue
        t = pq.read_table(f"{R1P}/{f}", columns=["event_id", "particle_id", "pdg_id"]).to_pydict()
        for e, pid, pdg in zip(t["event_id"], t["particle_id"], t["pdg_id"]):
            if 17000 <= e <= 19999:
                out[int(e)] = [int(p) for p, g in zip(pid, pdg) if abs(g) == QPDG]
    return out


def tracks_of(pred, e):
    d = pred.get(e)
    tr = {}
    if d is None:
        return tr
    for h, t in zip(d[0], d[1]):
        if t >= 0:
            tr.setdefault(int(t), set()).add(int(h))
    return tr


def load_pred(path):
    t = pq.read_table(path, columns=["event_id", "hit_id", "predicted_track_id"]).to_pandas()
    return {int(e): (g.hit_id.to_numpy(), g.predicted_track_id.to_numpy()) for e, g in t.groupby("event_id")}


def score_pair(ev, qids, trs):
    pid = ev["pid"]
    A = set(np.flatnonzero(pid == qids[0]).tolist()) if len(qids) > 0 else set()
    B = set(np.flatnonzero(pid == qids[1]).tolist()) if len(qids) > 1 else set()
    P = A | B
    rows = []
    for t, H in trs.items():
        npair = len(H & P)
        if npair == 0:
            continue
        others = Counter(int(pid[h]) for h in H - P)
        maj_other = max(others.values()) if others else 0
        rows.append(dict(track=t, n=len(H), npair=npair, pur=npair / len(H), comp=npair / max(len(P), 1),
                         majority=npair >= maj_other, hits=H))
    rows.sort(key=lambda r: (-r["npair"], -r["pur"]))
    found = any(r["majority"] and r["pur"] > 0.5 and r["comp"] > 0.5 for r in rows)
    return A, B, rows, found


def panel(ax_xy, ax_rz, ev, A, B, rows, title, notfound):
    x, y, z = ev["x"], ev["y"], ev["z"]
    r = np.hypot(x, y)
    P = sorted(A | B)
    best = rows[0]["hits"] if rows else set()
    second = rows[1]["hits"] if (notfound and len(rows) > 1 and rows[1]["npair"] >= 0.25 * len(P)) else set()
    other = set()
    for rr in rows[2 if second else 1:]:
        other |= rr["hits"]
    on_other = other & set(P)
    for ax, (u, w) in ((ax_xy, (x, y)), (ax_rz, (z, r))):
        lo_u, hi_u = u[P].min(), u[P].max()
        lo_w, hi_w = w[P].min(), w[P].max()
        pu, pw = max(15, 0.12 * (hi_u - lo_u)), max(15, 0.12 * (hi_w - lo_w))
        lo_u, hi_u, lo_w, hi_w = lo_u - pu, hi_u + pu, lo_w - pw, hi_w + pw
        box = (u > lo_u) & (u < hi_u) & (w > lo_w) & (w < hi_w)
        o = np.array(sorted(h for h in other | (best - set(P)) | (second - set(P)) if box[h]), int)
        if len(o):
            ax.scatter(u[o], w[o], s=7, color=C["other"], lw=0, zorder=1)
        for H, col, s in ((best, RING["best"], 70), (second, RING["second"], 105), (on_other, C["muted"], 70)):
            hh = np.array(sorted(h for h in H if box[h]), int)
            if len(hh):
                ax.scatter(u[hh], w[hh], s=s, facecolors="none", edgecolors=col, lw=1.6, zorder=2)
        for H, col, mk in ((A, C["A"], "o"), (B, C["B"], "s")):
            hh = np.array(sorted(H), int)
            if len(hh):
                ax.scatter(u[hh], w[hh], s=18, color=col, marker=mk, lw=0, zorder=3)
        ax.set_xlim(lo_u, hi_u)
        ax.set_ylim(lo_w, hi_w)
        ax.set_xticks([])
        ax.set_yticks([])
        for s_ in ax.spines.values():
            s_.set_color(C["axis"])
    ax_xy.set_title(title, fontsize=12.5, loc="left", pad=3)
    ax_xy.text(0.01, 0.97, "x-y", transform=ax_xy.transAxes, va="top", fontsize=10.5, color=C["muted"])
    ax_rz.text(0.01, 0.97, "r-z", transform=ax_rz.transAxes, va="top", fontsize=10.5, color=C["muted"])


def gallery(model, kind, picks, cache, qids, trs_all, k):
    setup(12)
    fig = plt.figure(figsize=(15.6, 6.9))
    gs = fig.add_gridspec(4, 5, hspace=0.45, wspace=0.06, left=0.005, right=0.995, top=0.915, bottom=0.01)
    for i, e in enumerate(picks):
        rI, cI = divmod(i, 5)
        ax1 = fig.add_subplot(gs[2 * rI, cI])
        ax2 = fig.add_subplot(gs[2 * rI + 1, cI])
        A, B, rows, _ = score_pair(cache[e], qids[e], trs_all[e])
        if rows:
            title = f"{e}  {len(A)}/{len(B)} hits\npur {rows[0]['pur']:.2f}  comp {rows[0]['comp']:.2f}"
        else:
            title = f"{e}  {len(A)}/{len(B)} hits\nno track holds pair hits"
        panel(ax1, ax2, cache[e], A, B, rows, title, kind == "notfound")
    return save(fig, f"gallery_{model}_{kind}_{k}")


def main():
    rng = np.random.default_rng(SEED)
    cache = load_cache()
    qids = quirk_ids()
    table, numbers = [], {"seed": SEED}
    sel = {}
    for model, cfg in MODELS.items():
        pe = pq.read_table(f"{cfg['acts']}/per_event.parquet").to_pandas()
        pe = pe[pe.quirks_selected_very_loose_strict == 1]
        pred = load_pred(cfg["pred"])
        trs = {int(e): tracks_of(pred, int(e)) for e in pe.event_id}
        agree = 0
        for e, m in zip(pe.event_id.astype(int), pe.quirks_matched_very_loose_strict.astype(int)):
            A, B, rows, found = score_pair(cache[e], qids[e], trs[e])
            agree += int(found == bool(m))
            b = rows[0] if rows else None
            sec = rows[1] if len(rows) > 1 else None
            table.append(dict(model=model, event_id=e, found_acts=int(m), found_python=int(found), nA=len(A),
                              nB=len(B), n_tracks_with_pair_hits=len(rows),
                              best_pur=round(b["pur"], 3) if b else 0, best_comp=round(b["comp"], 3) if b else 0,
                              best_n=b["n"] if b else 0, best_other=(b["n"] - b["npair"]) if b else 0,
                              best_A=len(b["hits"] & A) if b else 0, best_B=len(b["hits"] & B) if b else 0,
                              second_comp=round(sec["comp"], 3) if sec else 0))
        found_ev = pe.event_id[pe.quirks_matched_very_loose_strict == 1].astype(int).to_numpy()
        nf_ev = pe.event_id[pe.quirks_matched_very_loose_strict == 0].astype(int).to_numpy()
        numbers[model] = dict(selected=len(pe), found=len(found_ev), not_found=len(nf_ev), python_agrees=agree)
        print(model, numbers[model])
        for kind, pool in (("found", found_ev), ("notfound", nf_ev)):
            picks = sorted(rng.choice(pool, size=20, replace=False).tolist())
            sel[(model, kind)] = picks
            numbers[model][f"picks_{kind}"] = picks
            for k in (1, 2):
                gallery(model, kind, picks[10 * (k - 1):10 * k], cache, qids, trs, k)
    with open(os.path.join(FIGS, "pair_gallery_table.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
    record("pair_gallery", numbers)
    edges(cache, qids, numbers)


def edges(cache, qids, numbers):
    """Graph v3 candidate edges among the pair's hits for three pairs (deterministic: the 25th, 50th and 75th
    percentile of the number of A-B candidate edges among pairs with both quirks >= 10 hits in 17000-17999)."""
    G = f"{QS}/gnn/main"
    sq = pq.read_table(f"{G}/g2_q50_dir_s0/gnn/scored/events17000-17999.parquet").to_pydict()
    sp = pq.read_table(f"{G}/p3_pairv3_s0/gnn/scored/events17000-17999.parquet").to_pydict()
    cand = []
    for i, e in enumerate(sq["event_id"]):
        e = int(e)
        q = qids.get(e, [])
        if len(q) != 2:
            continue
        pid = cache[e]["pid"]
        a, b = pid == q[0], pid == q[1]
        if a.sum() < 10 or b.sum() < 10:
            continue
        s, d = np.asarray(sq["src"][i]), np.asarray(sq["dst"][i])
        ab = (a[s] & b[d]) | (b[s] & a[d])
        cand.append((int(ab.sum()), e, i))
    cand.sort()
    picks = [cand[int(p * (len(cand) - 1))] for p in (0.25, 0.5, 0.75)]
    cmap = LinearSegmentedColormap.from_list("s", ["#e8e6df", "#f39a3c", "#7b3f9e"])
    setup(13)
    fig, axs = plt.subplots(2, 3, figsize=(15.6, 6.9))
    info = []
    for j, (nab, e, i) in enumerate(picks):
        ev = cache[e]
        pid = ev["pid"]
        a, b = pid == qids[e][0], pid == qids[e][1]
        xyz = np.stack([ev["x"], ev["y"], ev["z"]], 1).astype(float)
        P_ = np.flatnonzero(a | b)
        cen = xyz[P_].mean(0)
        _, _, vt = np.linalg.svd(xyz[P_] - cen)
        z = (xyz - cen) @ vt[0]   # along the pair
        r = (xyz - cen) @ vt[1]   # transverse offset
        s, d = np.asarray(sq["src"][i]), np.asarray(sq["dst"][i])
        assert np.array_equal(s, np.asarray(sp["src"][i])) and int(sp["event_id"][i]) == e
        same = (a[s] & a[d]) | (b[s] & b[d])
        ab = (a[s] & b[d]) | (b[s] & a[d])
        for k, (scores, lab) in enumerate(((np.asarray(sq["score"][i]), "per-quirk GNN"),
                                           (np.asarray(sp["score"][i]), "pair GNN"))):
            ax = axs[k, j]
            for m_, col, lw in ((same, C["ink2"], 1.0),):
                seg = [[(z[u], r[u]), (z[v], r[v])] for u, v in zip(s[m_], d[m_])]
                ax.add_collection(LineCollection(seg, colors=col, linewidths=lw, alpha=0.5, zorder=1))
            seg = [[(z[u], r[u]), (z[v], r[v])] for u, v in zip(s[ab], d[ab])]
            lc = LineCollection(seg, cmap=cmap, linewidths=1.8, zorder=2)
            lc.set_array(scores[ab])
            lc.set_clim(0, 1)
            ax.add_collection(lc)
            ax.scatter(z[a], r[a], s=22, color=C["A"], zorder=3)
            ax.scatter(z[b], r[b], s=22, color=C["B"], marker="s", zorder=3)
            P = a | b
            pu = 0.04 * np.ptp(z[P]) + 10
            pw = 1.3 * np.abs(r[P]).max() + 2
            ax.set_xlim(z[P].min() - pu, z[P].max() + pu)
            ax.set_ylim(-pw, pw)
            ax.set_xticks([])
            ax.set_yticks([])
            hi = (scores[ab] > 0.5).sum()
            ax.set_title(f"{lab}: {hi} of {int(ab.sum())} A-B edges > 0.5", fontsize=13, loc="left")
            if j == 0:
                ax.set_ylabel(lab + "\n(transverse offset)", fontsize=13)
        info.append(dict(event_id=e, nA=int(a.sum()), nB=int(b.sum()), same_quirk_edges=int(same.sum()),
                         ab_edges=nab, ab_above_05_perquirk=int((np.asarray(sq["score"][i])[ab] > 0.5).sum()),
                         ab_above_05_pair=int((np.asarray(sp["score"][i])[ab] > 0.5).sum())))
        axs[0, j].text(0.0, 1.17, f"event {e}: {int(same.sum())} same-quirk, {nab} A-B edges",
                       transform=axs[0, j].transAxes, fontsize=13.5, fontweight="bold")
    cax = fig.add_axes([0.93, 0.15, 0.012, 0.7])
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1)), cax=cax)
    cb.set_label("A-B edge score")
    fig.subplots_adjust(left=0.04, right=0.92, top=0.88, bottom=0.02, wspace=0.05, hspace=0.25)
    save(fig, "gallery_edges")
    numbers["edges"] = info
    numbers["edges_candidates"] = len(cand)
    record("pair_gallery", numbers)
    print(info)


if __name__ == "__main__":
    main()
