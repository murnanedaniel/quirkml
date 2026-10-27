"""Event display: one interleaved quirk pair (m200_L774_mu0 test event) with the hits of three track finders.

Panels: truth; per-quirk GNN on graph v3 (gnn-g2-v3-dir-s0, walk + stitch 0.7, >= 4 hits); pair GNN
(gnn-e2-pair-dir-s0-cc, components at 0.5, >= 4 hits); MaskFormer with the pair as one target
(mf-E-pairdir-long, ot 0.1); MaskFormer with per-quirk targets (mf-t2fix, ot 0.1).
Coordinates: the quirk hits projected on their principal axis (s, along the pair) and on the second axis
(transverse offset), so the A/B interleaving is visible.

Event choice (deterministic, first match in 17000-17999): each quirk 12-16 hits; the per-quirk GNN puts the pair on
exactly two tracks, each with >= 3 hits of both quirks; the pair GNN puts >= 90% of the pair's hits on one track.
Hit truth is the cache's first particle (Release 1 label).
"""
import os
import pickle
import sys

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qstyle import C, TRACK_COLS, plt, record, save, setup  # noqa: E402

G = "/pscratch/sd/d/danieltm/quirk_tracking/gnn"
CACHE = f"{G}/cache/m200_L774_mu0/events17000-17999.pkl"
PRED = {
    "per-quirk GNN, graph v3": f"{G}/main/g2_q50_dir_s0/test/pred_stitch0.7_mh4/predictions.parquet",
    "pair GNN, connected components": f"{G}/main/e2_pair_dir_s0/test/pred_cc0.5_mh4/predictions.parquet",
    "MaskFormer, pair as one target": "/pscratch/sd/d/danieltm/quirk_tracking/maskformer/predictions/"
                                      "t2e_pairdir_long_test_ot0.1/predictions.parquet",
    "MaskFormer, per-quirk targets": "/pscratch/sd/d/danieltm/quirk_tracking/maskformer/predictions/"
                                     "t2fix_lr5e-5_test_ot0.1/predictions.parquet",
}


def labels(path, lo, hi):
    t = pq.read_table(path, columns=["event_id", "hit_id", "predicted_track_id"],
                      filters=[("event_id", ">=", lo), ("event_id", "<=", hi)]).to_pandas()
    return {e: g for e, g in t.groupby("event_id")}


evs = pickle.load(open(CACHE, "rb"))
lab = {k: labels(p, 17000, 17999) for k, p in PRED.items()}
kg, kp = list(PRED)[:2]


def tracks_of(df, hits):
    """{track: set(hit)} restricted to tracks touching the given hits."""
    d = df[df.predicted_track_id >= 0]
    on = d[d.hit_id.isin(hits)].predicted_track_id.unique()
    return {int(t): set(d[d.predicted_track_id == t].hit_id.astype(int)) for t in on}


chosen = None
for ev in evs:
    e = ev["event_id"]
    q = np.flatnonzero(ev["hit_quirk"])
    pids = np.unique(ev["pid"][q])
    if len(pids) != 2:
        continue
    A = set(np.flatnonzero(ev["pid"] == pids[0]).tolist())
    B = set(np.flatnonzero(ev["pid"] == pids[1]).tolist())
    if not (12 <= len(A) <= 16 and 12 <= len(B) <= 16):
        continue
    tg = tracks_of(lab[kg][e], A | B)
    if len(tg) != 2 or not all(len(t & A) >= 3 and len(t & B) >= 3 for t in tg.values()):
        continue
    tp = tracks_of(lab[kp][e], A | B)
    if not any(len(t & (A | B)) >= 0.9 * len(A | B) for t in tp.values()):
        continue
    chosen = (ev, A, B)
    break

ev, A, B = chosen
e = ev["event_id"]
xyz = np.stack([ev["x"], ev["y"], ev["z"]], 1).astype(float)
Q = sorted(A | B)
cen = xyz[Q].mean(0)
_, _, vt = np.linalg.svd(xyz[Q] - cen)
s = (xyz - cen) @ vt[0]
d = (xyz - cen) @ vt[1]
smin, smax = s[Q].min() - 60, s[Q].max() + 60
dpad = max(25.0, 1.5 * np.abs(d[Q]).max())
near = (s > smin) & (s < smax) & (np.abs(d) < dpad) & (np.abs((xyz - cen) @ vt[2]) < dpad)

setup(14)
fig, axs = plt.subplots(5, 1, figsize=(15.6, 8.6), sharex=True)
info = {"event_id": int(e), "nA": len(A), "nB": len(B)}
for k, ax in enumerate(axs):
    oth = near.copy()
    oth[Q] = False
    ax.scatter(s[oth], d[oth], s=10, color=C["other"], lw=0, zorder=1)
    if k == 0:
        for H, col in ((A, C["A"]), (B, C["B"])):
            h = sorted(H, key=lambda i: s[i])
            ax.plot(s[h], d[h], "-", color=col, lw=2, alpha=0.7, zorder=2)
        ax.set_title(f"truth, test event {e}: quirk A {len(A)} hits, quirk B {len(B)} hits", loc="left", fontsize=15)
    else:
        key = list(PRED)[k - 1]
        tr = tracks_of(lab[key][e], A | B)
        desc = []
        for j, (t, H) in enumerate(sorted(tr.items(), key=lambda kv: -len(kv[1] & (A | B)))):
            h = sorted(H, key=lambda i: s[i])
            col = TRACK_COLS[j % len(TRACK_COLS)]
            ax.plot(s[h], d[h], "-", color=col, lw=6, alpha=0.35, solid_capstyle="round", zorder=2)
            na, nb, no = len(H & A), len(H & B), len(H - A - B)
            desc.append(f"{na}A+{nb}B" + (f"+{no} other" if no else ""))
        unas = [i for i in Q if not any(i in H for H in tr.values())]
        if unas:
            ax.scatter(s[unas], d[unas], s=150, facecolors="none", edgecolors=C["ink"], lw=1.4, zorder=4)
            desc.append(f"{len(unas)} quirk hits on no track")
        ax.set_title(f"{key}: " + ", ".join(desc[:5]), loc="left", fontsize=15)
        info[key] = desc
    ax.scatter(s[sorted(A)], d[sorted(A)], s=60, color=C["A"], marker="o", edgecolor="white", lw=0.8, zorder=3)
    ax.scatter(s[sorted(B)], d[sorted(B)], s=60, color=C["B"], marker="s", edgecolor="white", lw=0.8, zorder=3)
    ax.set_ylim(-dpad, dpad)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
axs[-1].set_xlim(smin, smax)
axs[-1].set_xlabel(f"distance along the pair [mm]   (vertical: transverse offset, ±{dpad:.0f} mm in every row)")
fig.subplots_adjust(left=0.01, right=0.995, top=0.95, bottom=0.09, hspace=0.48)
save(fig, "evd_tracks")
info["transverse_halfrange_mm"] = float(dpad)
record("evd_tracks", info)
print(info)
