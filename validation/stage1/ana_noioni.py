"""ODD with quirk ionisation off: field off vs closed form; field on (3 T) vs quirk_sim."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "../trajcmp/fieldoff")
from closed_form import load, closed_form, at, C
import q2
from quirk_sim import QuirkPair
WIN = (1, 2, 5, 10, 20, 40)
def g4(path, pdg):
    G = pd.read_csv(path); g = G[G.pdg == pdg].sort_values("t_ns")
    g = g[(np.hypot(g.x_mm, g.y_mm) < 1150) & (g.z_mm.abs() < 3000)]
    return g.t_ns.values, g[["x_mm", "y_mm", "z_mm"]].values
def line(name, d, t):
    return f"{name:44s} " + "  ".join(f"t<{w}:{d[t < w].max():.4f}" for w in WIN if (t < w).any())
rows = []
for tag in ("L500", "L1000"):
    ev = load(tag); cf, _ = closed_form(ev)
    m = ev.m_GeV * 1e3; sig = q2.sigma_of(ev.lambda_eV); dt = 0.02
    v = np.array([ev.vx, ev.vy, ev.vz]); p1 = np.array([ev.qp_px, ev.qp_py, ev.qp_pz]) * 1e3; p2 = np.array([ev.qm_px, ev.qm_py, ev.qm_pz]) * 1e3
    res = QuirkPair(m, sig, b_tesla=3.0, string_mode=1).integrate(v, p1, +1, v, p2, -1, dt=dt, max_steps=int(6e4 / dt), r_stop_mm=1150.0)
    tq = (ev.vt + dt * np.arange(len(res["traj1"]))) / C; qs = {1: (tq, res["traj1"]), -1: (tq, res["traj2"])}
    for q, pdg in ((1, 4900101), (-1, -4900101)):
        for kind, path, ref in (("ODD B=0, no ionisation vs exact", f"steps_noioni_fieldoff_{tag}.csv", cf),
                                ("ODD B=0, ionisation on vs exact", f"../trajcmp/fieldoff/g4_odd_{tag}_mb1e-2_steps.csv", cf),
                                ("ODD 3T, no ionisation vs quirk_sim", f"steps_noioni_fieldon_{tag}.csv", qs),
                                ("ODD 3T, ionisation on vs quirk_sim", f"../trajcmp/g4_{tag}_steps.csv", qs)):
            t, X = g4(path, pdg)
            if ref is qs: keep = t <= tq[-1]; t, X = t[keep], X[keep]
            d = np.linalg.norm(X - at(ref[q], t), axis=1)
            print(line(f"{tag} q={q:+d} {kind}", d, t))
            for w in WIN:
                if (t < w).any(): rows.append((tag, kind, q, w, d[t < w].max()))
pd.DataFrame(rows, columns=["event", "comparison", "quirk", "t_lt_ns", "max_d3_mm"]).to_csv("noioni_summary.csv", index=False, float_format="%.5f")
