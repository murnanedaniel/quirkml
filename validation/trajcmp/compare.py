"""Geant4 per-step quirk positions vs quirk_sim at the same lab times (pair-only runs, EM off).
quirk_sim: RK4, 0.02 mm/c step, uniform 3 T; positions linearly interpolated to the Geant4 step times.
Output per event: <=300 points per quirk inside r < 1150 mm, |z| < 3000 mm."""
import os, sys
import numpy as np, pandas as pd
Q = "/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirk_erc"
sys.path.insert(0, f"{Q}/fits"); sys.path.insert(0, f"{Q}/ana")
import q2
from quirk_sim import QuirkPair
C = 299.792458
HERE = os.path.dirname(os.path.abspath(__file__))
for tag, p, evn, grid in (("L500", "m400_L500", 1, None), ("L1000", "m400_L1000", 3, (10.0, 0.05))):
    ev = pd.read_parquet(f"{Q}/out/pilot/compact/{p}_events.parquet").set_index("event").loc[evn]
    G = pd.read_csv(f"{HERE}/g4_{tag}_steps.csv")
    print(tag, "tracks", G.groupby(["trackID", "pdg"]).size().to_dict())
    m = ev.m_GeV * 1e3; sig = q2.sigma_of(ev.lambda_eV); dt = 0.02
    v = np.array([ev.vx, ev.vy, ev.vz]); p1 = np.array([ev.qp_px, ev.qp_py, ev.qp_pz]) * 1e3; p2 = np.array([ev.qm_px, ev.qm_py, ev.qm_pz]) * 1e3
    res = QuirkPair(m, sig, b_tesla=3.0, string_mode=1).integrate(v, p1, +1, v, p2, -1, dt=dt, max_steps=int(6e4 / dt), r_stop_mm=1150.0)
    tq = (ev.vt + dt * np.arange(len(res["traj1"]))) / C
    rows = []
    for q, pdg, T in ((1, 4900101, res["traj1"]), (-1, -4900101, res["traj2"])):
        g = G[G.pdg == pdg].sort_values("t_ns")
        g = g[(np.hypot(g.x_mm, g.y_mm) < 1150) & (g.z_mm.abs() < 3000) & (g.t_ns <= tq[-1])]
        t = g.t_ns.values
        if grid:  # fine grid early, coarse late
            t_end, step = grid
            want = np.r_[np.arange(t[0], min(t_end, t[-1]), step), np.linspace(min(t_end, t[-1]), t[-1], max(300 - int((min(t_end, t[-1]) - t[0]) / step), 2))]
        else:
            want = np.linspace(t[0], t[-1], 300)
        idx = np.unique(np.clip(np.searchsorted(t, want), 0, len(t) - 1))[:300]
        for i in idx:
            r = g.iloc[i]
            qs = [np.interp(r.t_ns, tq, T[:, k]) for k in range(3)]
            d = np.sqrt((r.x_mm - qs[0]) ** 2 + (r.y_mm - qs[1]) ** 2 + (r.z_mm - qs[2]) ** 2)
            rows.append((q, r.t_ns, r.x_mm, r.y_mm, r.z_mm, *qs, d))
        print(tag, q, "g4 steps in tracker", len(g), "median step dt ns", np.median(np.diff(t)), "sent", len(idx))
    out = pd.DataFrame(rows, columns=["quirk", "t_ns", "g4_x", "g4_y", "g4_z", "qs_x", "qs_y", "qs_z", "d3_mm"])
    out.to_csv(f"{HERE}/cmp_{tag}.csv", index=False, float_format="%.3f")
    for q in (1, -1):
        o = out[out.quirk == q]
        print(tag, q, " ".join(f"t<{tt}:{o[o.t_ns < tt].d3_mm.max():.3f}" for tt in (1, 2, 5, 10, 20, 40) if (o.t_ns < tt).any()))
