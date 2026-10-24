"""Separability ceiling, part 2: can hit timing tell the two quirks apart where they are close?

m200_L774_mu0 test events. Hit time = packaged tracker_simhits true_time (ns) of the hit's first simhit, which is
the Release 1 tracker_hits `time` column (checked equal on event 17000; the per-run files store c*t in mm, not
used here). Positions are the reconstructed hit positions (cache = R1 x, y, z). Quirk labels: R1 first particle,
quirks with >= 3 hits, events with both quirks selected (as gnn/dig.py).

1. Close pairs: every (hit of quirk A, hit of quirk B) pair with 3D distance < --close-mm; |t_A - t_B|.
2. Switch points of a prediction set (finder as gnn/dig.py: per track, its quirk hits ordered by true time; a switch
   is two consecutive ones from different quirks). For a switch a (quirk X) -> b (quirk Y):
   - step: |t_b - t_a|, |x_b - x_a|;
   - close pair at the switch: the nearest X hit to b and the nearest Y hit to a; the closer of the two pairs, if
     within --close-mm (the confusable alternative), and its |dt|;
   - successor: X's next hit in time after a (where the track should have gone), distance and |dt| to b.
3. Same-quirk consecutive hits (time order): |dt| and distance, all steps and steps between different surfaces.
4. Toy: hit time measured with resolution sigma_t; two hits are told apart at 3 sigma if |dt| > 3 sqrt(2) sigma_t
   (threshold on the true dt; a smeared version, |dt + N(0, sqrt(2) sigma_t)| > 3 sqrt(2) sigma_t, is also given).

  python timing.py --pred P/predictions.parquet --events 17000 19999 --out DIR --fig-dir FIGDIR
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "gnn"))
from common import load_events  # noqa: E402

R1 = "/global/cfs/cdirs/m4958/data/ColliderML/simulation/quirks/m200_L774_mu0/v1/parquet_r1"
CACHE = "/pscratch/sd/d/danieltm/quirk_tracking/gnn/cache/m200_L774_mu0"
SIGMAS_PS = [10, 30, 50, 100, 300]


def read_r1(first, last):
    out = {}
    for f in glob.glob(f"{R1}/reco/tracker_hits/*.parquet"):
        a, b = os.path.basename(f).split(".events")[1].split(".")[0].split("-")
        if int(b) < first or int(a) > last:
            continue
        t = pq.read_table(f, columns=["event_id", "time", "x", "volume_id", "layer_id", "surface_id"]).to_pydict()
        for k, e in enumerate(t["event_id"]):
            if first <= e <= last:
                surf = (np.asarray(t["volume_id"][k], np.int64) * 10**8 + np.asarray(t["layer_id"][k], np.int64) * 10**5
                        + np.asarray(t["surface_id"][k], np.int64))
                out[int(e)] = (np.asarray(t["time"][k], np.float64), np.asarray(t["x"][k], np.float32), surf)
    return out


def interp_pos(t, ts, pos):
    o = np.argsort(ts)
    return np.array([np.interp(t, ts[o], pos[o, k]) for k in range(3)])


def toy(dt_ns, rng):
    dt = np.abs(np.asarray(dt_ns)) * 1000.0  # ps
    res = {}
    for s in SIGMAS_PS:
        thr = 3 * np.sqrt(2) * s
        meas = np.abs(dt + rng.normal(0, np.sqrt(2) * s, len(dt)))
        res[str(s)] = dict(threshold_ps=round(thr, 1), frac=float((dt > thr).mean()) if len(dt) else None,
                           frac_smeared=float((meas > thr).mean()) if len(dt) else None)
    return res


def q(a, ps=(10, 25, 50, 75, 90)):
    return np.percentile(a, ps).round(4).tolist() if len(a) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--events", type=int, nargs=2, default=[17000, 19999])
    ap.add_argument("--close-mm", type=float, default=10.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fig-dir", required=True)
    ap.add_argument("--figs-only", action="store_true", help="redraw the figures from <out>/timing_arrays.npz")
    a = ap.parse_args()
    if a.figs_only:
        z = np.load(os.path.join(a.out, "timing_arrays.npz"))
        sw = {k[3:]: z[k] for k in z.files if k.startswith("sw_")}
        figures(z["close_dt"], sw, z["same_dt_diffsurf"], a.fig_dir, a.close_mm)
        return
    os.makedirs(a.out, exist_ok=True)
    os.makedirs(a.fig_dir, exist_ok=True)
    evs = load_events(CACHE, *a.events)
    r1 = read_r1(*a.events)
    pred = pq.read_table(a.pred, filters=[("event_id", ">=", a.events[0]), ("event_id", "<=", a.events[1])]).to_pandas()
    pred = {int(e): g.sort_values("hit_id").predicted_track_id.to_numpy() for e, g in pred.groupby("event_id")}
    close_dt, close_d = [], []
    nearest_dt = []  # per quirk hit: nearest other-quirk hit, if within close-mm
    sw = dict(step_dt=[], step_d=[], cp_dt=[], cp_d=[], succ_dt=[], succ_d=[], sep_mid=[], has_cp=[])
    same_dt, same_d, same_dt_diffsurf = [], [], []
    n_pairs = n_sw = 0
    for ev in evs:
        eid = ev["event_id"]
        t, xr1, surf = r1[eid]
        assert np.allclose(xr1, ev["x"]), eid
        pid = ev["pid"]
        u, cnt = np.unique(pid[ev["hit_quirk"]], return_counts=True)
        qq = u[cnt >= 3]
        if len(qq) != 2 or eid not in pred:
            continue
        n_pairs += 1
        pos = np.stack([ev["x"], ev["y"], ev["z"]], 1).astype(np.float64)
        hq = {int(qv): np.flatnonzero(pid == qv) for qv in qq}
        A, B = (int(v) for v in qq)
        hA, hB = hq[A], hq[B]
        D = np.linalg.norm(pos[hA][:, None] - pos[hB][None], axis=2)
        DT = np.abs(t[hA][:, None] - t[hB][None])
        m = D < a.close_mm
        close_dt += DT[m].tolist()
        close_d += D[m].tolist()
        for Dm, DTm in ((D, DT), (D.T, DT.T)):
            j = Dm.argmin(1)
            ok = Dm[np.arange(len(j)), j] < a.close_mm
            nearest_dt += DTm[np.arange(len(j)), j][ok].tolist()
        for h in (hA, hB):
            o = h[np.argsort(t[h], kind="stable")]
            dts = np.diff(t[o])
            ds = np.linalg.norm(np.diff(pos[o], axis=0), axis=1)
            same_dt += dts.tolist()
            same_d += ds.tolist()
            same_dt_diffsurf += dts[surf[o[1:]] != surf[o[:-1]]].tolist()
        lab = pred[eid]
        for tid in np.unique(lab[np.concatenate([hA, hB])]):
            if tid < 0:
                continue
            th = np.flatnonzero(lab == tid)
            qh = th[np.isin(th, hA) | np.isin(th, hB)]
            qh = qh[np.argsort(t[qh], kind="stable")]
            isA = np.isin(qh, hA)
            for j in np.flatnonzero(isA[1:] != isA[:-1]):
                ha, hb = qh[j], qh[j + 1]
                X, Y = (hA, hB) if isA[j] else (hB, hA)
                n_sw += 1
                sw["step_dt"].append(abs(t[hb] - t[ha]))
                sw["step_d"].append(float(np.linalg.norm(pos[hb] - pos[ha])))
                dbX = np.linalg.norm(pos[X] - pos[hb], axis=1)
                daY = np.linalg.norm(pos[Y] - pos[ha], axis=1)
                i1, i2 = dbX.argmin(), daY.argmin()
                if dbX[i1] <= daY[i2]:
                    cd, cdt = dbX[i1], abs(t[X[i1]] - t[hb])
                else:
                    cd, cdt = daY[i2], abs(t[Y[i2]] - t[ha])
                sw["cp_d"].append(float(cd))
                sw["cp_dt"].append(float(cdt))
                sw["has_cp"].append(bool(cd < a.close_mm))
                later = X[t[X] > t[ha]]
                if len(later):
                    s = later[np.argmin(t[later])]
                    sw["succ_d"].append(float(np.linalg.norm(pos[s] - pos[hb])))
                    sw["succ_dt"].append(abs(t[s] - t[hb]))
                tm = 0.5 * (t[ha] + t[hb])
                sw["sep_mid"].append(float(np.linalg.norm(interp_pos(tm, t[hA], pos[hA]) - interp_pos(tm, t[hB], pos[hB]))))
    rng = np.random.default_rng(0)
    sw = {k: np.asarray(v) for k, v in sw.items()}
    hc = sw["has_cp"]
    S = dict(pred=a.pred, events=a.events, pairs=n_pairs, close_mm=a.close_mm, time_units="ns (quantiles in ns)",
             close_pairs=dict(n=len(close_dt), per_event=len(close_dt) / max(n_pairs, 1), dt_ns_q10_25_50_75_90=q(close_dt),
                              frac_dt_lt=dict({f"{x}ps": float((np.array(close_dt) < x / 1000).mean()) for x in (10, 50, 100, 300, 1000)}),
                              toy=toy(close_dt, rng)),
             close_nearest_per_hit=dict(n=len(nearest_dt), dt_ns_q10_25_50_75_90=q(nearest_dt), toy=toy(nearest_dt, rng)),
             switches=dict(n=n_sw, sep_mid_mm_q50=float(np.median(sw["sep_mid"])),
                           step_dt_ns_q=q(sw["step_dt"]), step_d_mm_q=q(sw["step_d"]),
                           frac_with_close_pair=float(hc.mean()), close_pair_d_mm_q=q(sw["cp_d"][hc]),
                           close_pair_dt_ns_q=q(sw["cp_dt"][hc]), close_pair_toy=toy(sw["cp_dt"][hc], rng),
                           step_toy=toy(sw["step_dt"], rng),
                           succ_d_mm_q=q(sw["succ_d"]), succ_dt_ns_q=q(sw["succ_dt"]), succ_toy=toy(sw["succ_dt"], rng)),
             same_quirk_consecutive=dict(n=len(same_dt), dt_ns_q=q(same_dt), d_mm_q=q(same_d),
                                         n_diff_surface=len(same_dt_diffsurf), dt_ns_diff_surface_q=q(same_dt_diffsurf)))
    json.dump(S, open(os.path.join(a.out, "timing_summary.json"), "w"), indent=1)
    np.savez(os.path.join(a.out, "timing_arrays.npz"), close_dt=close_dt, close_d=close_d, same_dt=same_dt,
             same_dt_diffsurf=same_dt_diffsurf, **{f"sw_{k}": v for k, v in sw.items()})
    print(json.dumps(S, indent=1))
    figures(close_dt, sw, same_dt_diffsurf, a.fig_dir, a.close_mm)


def figures(close_dt, sw, same_dt, fig_dir, close_mm):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, ink2, grid, surf = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    c1, c2, c3 = "#2a78d6", "#eb6834", "#1baf7a"
    hc = sw["has_cp"]
    series = [(np.asarray(close_dt) * 1000, c1, f"quirk A-B hit pairs within {close_mm:g} mm"),
              (sw["cp_dt"][hc] * 1000, c2, f"close pair at a GNN v2 switch (within {close_mm:g} mm)"),
              (np.asarray(same_dt) * 1000, c3, "consecutive hits of one quirk (different surfaces)")]
    fig, ax = plt.subplots(figsize=(8, 4.8), facecolor=surf)
    ax.set_facecolor(surf)
    xs = np.logspace(-2, 5, 400)
    for v, c, nm in series:
        v = np.sort(v)
        ax.plot(xs, 1 - np.searchsorted(v, xs, side="right") / len(v), color=c, lw=2, label=f"{nm} (n = {len(v)})")
    for s in SIGMAS_PS:
        thr = 3 * np.sqrt(2) * s
        ax.axvline(thr, color=ink2, lw=0.8, ls=":")
        ax.text(thr * 1.08, 0.02, f"sigma_t = {s} ps", rotation=90, va="bottom", fontsize=8, color=ink2)
    ax.set_xscale("log")
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("|true time difference| threshold [ps]  (dotted: 3 sqrt(2) sigma_t)", color=ink2)
    ax.set_ylabel("fraction with |dt| above threshold", color=ink2)
    ax.grid(color=grid, lw=0.6)
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    for sp in ax.spines.values():
        sp.set_color("#c3c2b7")
    ax.tick_params(colors=ink2)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "dt_close_pairs.png"), dpi=140, facecolor=surf)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.4, 4.4), facecolor=surf)
    ax.set_facecolor(surf)
    sig = np.array(SIGMAS_PS)
    for v, c, nm in series[:2]:
        fr = [(v > 3 * np.sqrt(2) * s).mean() for s in sig]
        ax.plot(sig, fr, "-o", color=c, lw=2, ms=8, mec=surf, mew=2, label=nm)
    ax.set_xscale("log")
    ax.set_ylim(0, 1)
    ax.set_xlabel("hit time resolution sigma_t [ps]", color=ink2)
    ax.set_ylabel("fraction with |dt| > 3 sqrt(2) sigma_t", color=ink2)
    ax.grid(color=grid, lw=0.6)
    ax.legend(fontsize=8, frameon=False)
    for sp in ax.spines.values():
        sp.set_color("#c3c2b7")
    ax.tick_params(colors=ink2)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "frac_separable_vs_sigma_t.png"), dpi=140, facecolor=surf)
    plt.close(fig)


if __name__ == "__main__":
    main()
