"""Per-event best-fit string tension, as a cross-check of the Lambda mapping.

For each event we scan the string tension and pick the value minimising the
spatial residual to the truth hits.  The best-fit values should cluster around
the a-priori tension sigma = Lambda^2 / hbar*c, confirming the toy needs no
free parameter.

Run:  python -m quirk_sim.fit_sigma
"""
import numpy as np

from . import QuirkPair, Barrel
from . import dataio
from .units import CM, string_tension_from_lambda_eV


def hit_xyz(hd):
    return np.c_[hd.r * np.cos(hd.phi), hd.r * np.sin(hd.phi), hd.z] * CM


def residual(traj, data_xyz):
    if len(data_xyz) == 0:
        return np.nan, np.array([])
    d = np.empty(len(data_xyz))
    for i, h in enumerate(data_xyz):
        dd = traj - h
        d[i] = np.sqrt((dd * dd).sum(axis=1)).min()
    return np.median(d) / CM, d / CM


def fit_event(mass, lam, ev, sigmas, string_mode=1, dt=1.0):
    try:
        parts, hits = dataio.load_event(mass, lam, ev)
    except Exception:
        return None
    qs = dataio.quirks_from_particles(parts)
    if len(qs) != 2:
        return None
    q1, q2 = qs
    v0 = 0.5 * (q1["r0"] + q2["r0"])
    barrel = Barrel()
    pid1, pid2 = parts.iloc[0]["particle_id"], parts.iloc[1]["particle_id"]
    d1 = hit_xyz(hits[hits.particle_id == pid1])
    d2 = hit_xyz(hits[hits.particle_id == pid2])
    best = None
    for sig in sigmas:
        sim = QuirkPair(q1["mass"], sig, b_tesla=2.0, string_mode=string_mode)
        res = sim.integrate(v0, q1["p"], q1["q"], v0, q2["p"], q2["q"],
                            dt=dt, max_steps=40000, r_stop_mm=barrel.r_max * 1.2)
        r1, _ = residual(res["traj1"], d1)
        r2, _ = residual(res["traj2"], d2)
        score = 0.5 * (r1 + r2)
        if best is None or score < best[1]:
            best = (sig, score)
    return best


def main(mass=1000, lam=1000, nmax=40):
    theory = string_tension_from_lambda_eV(lam)
    sigmas = np.geomspace(0.2 * theory, 5 * theory, 25)
    evs = dataio.list_events(mass, lam)[:nmax]
    best = []
    for ev in evs:
        b = fit_event(mass, lam, ev, sigmas)
        if b:
            best.append(b[0])
    best = np.array(best)
    print(f"mass={mass} Lambda={lam}: theory sigma = {theory:.0f} MeV/mm")
    print(f"per-event best-fit sigma over {len(best)} events: "
          f"median={np.median(best):.0f}  mean={np.mean(best):.0f}  "
          f"[{np.percentile(best,16):.0f}, {np.percentile(best,84):.0f}]")
    print(f"median best-fit / theory = {np.median(best)/theory:.2f}")


if __name__ == "__main__":
    main()
