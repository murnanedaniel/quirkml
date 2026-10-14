"""Field-off, energy-loss-off quirk pair: exact closed-form trajectory, compared with quirk_sim and Geant4 (QuirkDDG4).

With B = 0 and no material, the pair rest frame is inertial and the two quirks move back to back along one line n.
Each feels a constant force sigma towards the other, so in the rest frame (c = 1, mm, MeV):
    p(u) = k - sigma*u                      for 0 <= u < 2k/sigma   (separating, then returning)
    p(u) = -k + sigma*(u - 2k/sigma)         for 2k/sigma <= u < 4k/sigma
    s(u) = +-(E0 - E(p))/sigma,   E(p) = sqrt(m^2 + p^2),   u = tau mod 4k/sigma
with quirk 1 at +s n and quirk 2 at -s n. Events are boosted to the lab and sampled in lab time.

Usage: python closed_form.py [g4 tags...]   (run from quirk_erc/out/trajcmp/fieldoff)
"""
import os, sys
import numpy as np, pandas as pd
Q = "/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirk_erc"
sys.path.insert(0, f"{Q}/fits"); sys.path.insert(0, f"{Q}/ana")
import q2
from quirk_sim import QuirkPair
C = 299.792458  # mm/ns
HERE = os.path.dirname(os.path.abspath(__file__))
WIN = (1, 2, 5, 10, 20, 40)
EVENTS = {"L500": ("m400_L500", 1), "L1000": ("m400_L1000", 3)}


def load(tag):
    p, evn = EVENTS[tag]
    return pd.read_parquet(f"{Q}/out/pilot/compact/{p}_events.parquet").set_index("event").loc[evn]


def closed_form(ev, n_tau=400000, t_max_ns=60.0):
    """Lab-frame trajectories (t_ns, xyz) of both quirks; quirk 1 = +pdg (qp), quirk 2 = -pdg (qm)."""
    m = ev.m_GeV * 1e3; sig = q2.sigma_of(ev.lambda_eV)
    p1 = np.array([ev.qp_px, ev.qp_py, ev.qp_pz]) * 1e3; p2 = np.array([ev.qm_px, ev.qm_py, ev.qm_pz]) * 1e3
    E1 = np.sqrt(m ** 2 + p1 @ p1); E2 = np.sqrt(m ** 2 + p2 @ p2)
    P = p1 + p2; E = E1 + E2; beta = P / E; b = np.linalg.norm(beta); bh = beta / b; g = 1 / np.sqrt(1 - b * b)
    # quirk 1 momentum in the rest frame
    p1s = p1 + ((g - 1) * (p1 @ bh) - g * b * E1) * bh
    k = np.linalg.norm(p1s); n = p1s / k; E0 = np.sqrt(m ** 2 + k ** 2)
    tau = np.linspace(0, t_max_ns * C / g * 1.05, n_tau)  # rest-frame time, mm
    T = 4 * k / sig; u = np.mod(tau, T); first = u < 2 * k / sig
    p = np.where(first, k - sig * u, -k + sig * (u - 2 * k / sig))
    s = np.where(first, 1.0, -1.0) * (E0 - np.sqrt(m ** 2 + p ** 2)) / sig
    v = np.array([ev.vx, ev.vy, ev.vz]); out = {}
    for q, sgn in ((1, 1.0), (-1, -1.0)):
        r = (sgn * s)[:, None] * n[None, :]
        rb = r @ bh
        t_lab = g * (tau + b * rb)
        x_lab = r + ((g - 1) * rb + g * b * tau)[:, None] * bh[None, :]
        out[q] = ((ev.vt + t_lab) / C, x_lab + v)
    info = dict(k_GeV=k / 1e3, gamma=g, beta=b, period_rest_ns=T / C, first_crossing_rest_ns=2 * k / sig / C)
    return out, info


def at(traj, t):
    tt, X = traj
    return np.stack([np.interp(t, tt, X[:, i]) for i in range(3)], 1)


def toy(ev, dt):
    m = ev.m_GeV * 1e3; sig = q2.sigma_of(ev.lambda_eV)
    v = np.array([ev.vx, ev.vy, ev.vz]); p1 = np.array([ev.qp_px, ev.qp_py, ev.qp_pz]) * 1e3; p2 = np.array([ev.qm_px, ev.qm_py, ev.qm_pz]) * 1e3
    res = QuirkPair(m, sig, b_tesla=0.0, string_mode=1).integrate(v, p1, +1, v, p2, -1, dt=dt, max_steps=int(6e4 / dt), r_stop_mm=1150.0)
    tq = (ev.vt + dt * np.arange(len(res["traj1"]))) / C
    return {1: (tq, res["traj1"]), -1: (tq, res["traj2"])}


def summary(name, d, t):
    return name + "  " + "  ".join(f"t<{w}:{d[t < w].max():.4f}" for w in WIN if (t < w).any())


def main():
    tags = sys.argv[1:]
    rows = []
    for tag in EVENTS:
        ev = load(tag); cf, info = closed_form(ev)
        print(f"== {tag}: " + ", ".join(f"{a}={b:.4g}" for a, b in info.items()))
        t_end = {}
        for dt in (0.02, 0.005):
            ty = toy(ev, dt)
            for q in (1, -1):
                tq, X = ty[q]
                inside = (np.hypot(X[:, 0], X[:, 1]) < 1150) & (np.abs(X[:, 2]) < 3000)
                tq, X = tq[inside], X[inside]; t_end[q] = tq[-1]
                d = np.linalg.norm(X - at(cf[q], tq), axis=1)
                print(summary(f"quirk_sim dt={dt} q={q:+d} (max mm vs closed form)", d, tq))
                for w in WIN:
                    if (tq < w).any(): rows.append((tag, f"quirk_sim_dt{dt}", q, w, d[tq < w].max()))
        for g4 in tags:
            f = f"{HERE}/g4_{g4.replace('TAG', tag)}_steps.csv"  # tags like odd_TAG_mb1e-2
            if not os.path.exists(f) or os.path.getsize(f) == 0:
                print(f"(missing {f})"); continue
            G = pd.read_csv(f)
            for q, pdg in ((1, 4900101), (-1, -4900101)):
                g = G[G.pdg == pdg].sort_values("t_ns")
                g = g[(np.hypot(g.x_mm, g.y_mm) < 1150) & (g.z_mm.abs() < 3000)]
                t = g.t_ns.values; X = g[["x_mm", "y_mm", "z_mm"]].values
                d = np.linalg.norm(X - at(cf[q], t), axis=1)
                name = g4.replace("TAG", tag)
                print(summary(f"geant4 {name} q={q:+d} (max mm vs closed form, {len(t)} steps)", d, t))
                for w in WIN:
                    if (t < w).any(): rows.append((tag, f"geant4_{name}", q, w, d[t < w].max()))
    pd.DataFrame(rows, columns=["event", "code", "quirk", "t_lt_ns", "max_d3_mm"]).to_csv(f"{HERE}/closed_form_summary.csv", index=False, float_format="%.5f")


if __name__ == "__main__":
    main()
