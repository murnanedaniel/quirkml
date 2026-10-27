"""A0 crossing ambiguity picture (ledger finding 2026-10-26 "A0 crossing ambiguity"; tracking/split/a0_crossing.py).

Left: one test pair (event 17003, the pair of the event-display slide): the A-B separation along the trajectory,
positions linearly interpolated in time between each quirk's hits (as a0_crossing.py does), against time; dots: for
every hit, the distance to the nearest hit of the other quirk on the same layer.
Right: over all 2973 test pairs (a0_pairs.json), the minimum separation along the trajectory and the number of
A-B hit pairs on one layer closer than 1 / 3 / 10 mm.
"""
import json
import os
import pickle
import sys

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qstyle import C, plt, record, save, setup  # noqa: E402

EV = 17003
CACHE = "/pscratch/sd/d/danieltm/quirk_tracking/gnn/cache/m200_L774_mu0/events17000-17999.pkl"
R1 = ("/global/cfs/cdirs/m4958/data/ColliderML/simulation/quirks/m200_L774_mu0/v1/parquet_r1/reco/tracker_hits/")
A0 = "/pscratch/sd/d/danieltm/quirk_tracking/split/a0"

ev = [e for e in pickle.load(open(CACHE, "rb")) if e["event_id"] == EV][0]
f = [x for x in os.listdir(R1) if "events17000-17999" in x][0]
h = pq.read_table(R1 + f, columns=["event_id", "time", "volume_id", "layer_id"],
                  filters=[("event_id", "=", EV)]).to_pydict()
t = np.asarray(h["time"][0], float)
vol, lay = np.asarray(h["volume_id"][0]), np.asarray(h["layer_id"][0])
pid = ev["pid"]
qs = [q for q in np.unique(pid[ev["hit_quirk"]]) if (pid == q).sum() >= 3]
ia, ib = np.flatnonzero(pid == qs[0]), np.flatnonzero(pid == qs[1])
pos = np.stack([ev["x"], ev["y"], ev["z"]], 1).astype(float)
lo, hi = max(t[ia].min(), t[ib].min()), min(t[ia].max(), t[ib].max())
ts = np.linspace(lo, hi, 2000)
oa, ob = np.argsort(t[ia]), np.argsort(t[ib])
pa = np.stack([np.interp(ts, t[ia][oa], pos[ia][oa, k]) for k in range(3)], 1)
pb = np.stack([np.interp(ts, t[ib][ob], pos[ib][ob, k]) for k in range(3)], 1)
sep = np.linalg.norm(pa - pb, axis=1)


def layer_nn(src, oth):
    out = []
    for i in src:
        m = oth[(vol[oth] == vol[i]) & (lay[oth] == lay[i])]
        out.append(np.linalg.norm(pos[m] - pos[i], axis=1).min() if len(m) else np.nan)
    return np.array(out)


dA, dB = layer_nn(ia, ib), layer_nn(ib, ia)
rows = json.load(open(f"{A0}/a0_pairs.json"))
ms = np.array([r["min_sep"] for r in rows], float)
setup(15)
fig, axs = plt.subplots(1, 2, figsize=(15.6, 5.8), gridspec_kw={"width_ratios": [1.6, 1]})
ax = axs[0]
ax.plot(ts, sep, color=C["ink2"], lw=1.6, label="A-B separation between layers (interpolated)")
ax.scatter(t[ia], dA, color=C["A"], s=60, zorder=3, label="A hit: nearest B hit on its layer")
ax.scatter(t[ib], dB, color=C["B"], marker="s", s=60, zorder=3, label="B hit: nearest A hit on its layer")
ax.set_yscale("log")
ax.set_ylim(0.05, 5000)
ax.set_xlabel("time [ns]")
ax.set_ylabel("distance [mm]")
ax.set_title(f"test event {EV}", fontsize=16, loc="left")
ax.legend(loc="upper right", fontsize=12.5)
ax.grid(True, axis="y")
ax = axs[1]
bins = np.logspace(-3, 2.5, 40)
ax.hist(ms[np.isfinite(ms)], bins=bins, color=C["ink2"], histtype="step", lw=2, label="minimum separation, trajectory")
ax.set_xscale("log")
ax.set_xlabel("closest A-B approach [mm]")
ax.set_ylabel("pairs")
ax.set_title("2973 test pairs", fontsize=16, loc="left")
fig.subplots_adjust(left=0.06, right=0.99, top=0.91, bottom=0.13, wspace=0.22)
save(fig, "a0_crossing")
mins = np.flatnonzero((sep[1:-1] < sep[:-2]) & (sep[1:-1] <= sep[2:])) + 1
record("a0_crossing", {"event": EV, "nA": int(len(ia)), "nB": int(len(ib)), "event_min_sep_mm": float(sep.min()),
                       "event_minima_below_3mm": int((sep[mins] < 3).sum()),
                       "event_layer_nn_median_mm": float(np.nanmedian(np.r_[dA, dB])),
                       "all_min_sep_median_mm": float(np.nanmedian(ms))})
print(sep.min(), int((sep[mins] < 3).sum()), np.nanmedian(np.r_[dA, dB]), np.nanmedian(ms))
