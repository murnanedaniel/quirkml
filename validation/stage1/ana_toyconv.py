"""quirk_sim step convergence with the 3 T field (dt 0.02 vs 0.005 vs 0.00125 mm)."""
import sys, numpy as np
sys.path.insert(0, "../trajcmp/fieldoff")
from closed_form import load, at, C
import q2
from quirk_sim import QuirkPair
for tag in ("L500", "L1000"):
    ev = load(tag); m = ev.m_GeV * 1e3; sig = q2.sigma_of(ev.lambda_eV)
    v = np.array([ev.vx, ev.vy, ev.vz]); p1 = np.array([ev.qp_px, ev.qp_py, ev.qp_pz]) * 1e3; p2 = np.array([ev.qm_px, ev.qm_py, ev.qm_pz]) * 1e3
    tr = {}
    for dt in (0.02, 0.005, 0.00125):
        r = QuirkPair(m, sig, b_tesla=3.0, string_mode=1).integrate(v, p1, +1, v, p2, -1, dt=dt, max_steps=int(4.5e4 / dt), r_stop_mm=1150.0)
        tr[dt] = ((ev.vt + dt * np.arange(len(r["traj1"]))) / C, r["traj1"])
    t, X = tr[0.00125]; t_end = min(tr[d][0][-1] for d in tr)
    tt = np.linspace(t[0], min(t_end, 40), 4000)
    ref = at((t, X), tt)
    for dt in (0.02, 0.005):
        d = np.linalg.norm(at(tr[dt], tt) - ref, axis=1)
        print(tag, f"dt={dt} vs dt=0.00125:", "  ".join(f"t<{w}:{d[tt < w].max():.4f}" for w in (2, 5, 10, 20, 40) if (tt < w).any()))
