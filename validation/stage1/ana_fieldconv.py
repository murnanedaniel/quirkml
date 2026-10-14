"""Field on (3 T), quirk ionisation off: Geant4 at MaxBoost 1e-2/1e-3/1e-4 vs converged quirk_sim (dt 0.00125 mm)."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "../trajcmp/fieldoff")
from closed_form import load, at, C
import q2
from quirk_sim import QuirkPair
WIN = (2, 5, 10, 20, 40)
def g4(path, pdg):
    G = pd.read_csv(path); g = G[G.pdg == pdg].sort_values("t_ns")
    g = g[(np.hypot(g.x_mm, g.y_mm) < 1150) & (g.z_mm.abs() < 3000)]
    return g.t_ns.values, g[["x_mm", "y_mm", "z_mm"]].values
for tag in ("L500", "L1000"):
    ev = load(tag); m = ev.m_GeV * 1e3; sig = q2.sigma_of(ev.lambda_eV); dt = 0.00125
    v = np.array([ev.vx, ev.vy, ev.vz]); p1 = np.array([ev.qp_px, ev.qp_py, ev.qp_pz]) * 1e3; p2 = np.array([ev.qm_px, ev.qm_py, ev.qm_pz]) * 1e3
    r = QuirkPair(m, sig, b_tesla=3.0, string_mode=1).integrate(v, p1, +1, v, p2, -1, dt=dt, max_steps=int(4.5e4 / dt), r_stop_mm=1150.0)
    tq = (ev.vt + dt * np.arange(len(r["traj1"]))) / C; qs = {1: (tq, r["traj1"]), -1: (tq, r["traj2"])}
    g4ref = {}
    for mb in ("", "_mb1e-3", "_mb1e-4"):
        t, X = g4(f"steps_noioni_fieldon_{tag}{mb}.csv", 4900101); keep = t <= tq[-1]; t, X = t[keep], X[keep]
        g4ref[mb] = (t, X)
        d = np.linalg.norm(X - at(qs[1], t), axis=1)
        print(f"{tag} Geant4 MaxBoost {mb[3:] or '1e-2':5s} vs quirk_sim(conv):", "  ".join(f"t<{w}:{d[t < w].max():.4f}" for w in WIN if (t < w).any()))
    t, X = g4ref["_mb1e-4"]
    for mb in ("", "_mb1e-3"):
        d = np.linalg.norm(at(g4ref[mb], t) - X, axis=1)
        print(f"{tag} Geant4 MaxBoost {mb[3:] or '1e-2':5s} vs Geant4 1e-4:      ", "  ".join(f"t<{w}:{d[t < w].max():.4f}" for w in WIN if (t < w).any()))
