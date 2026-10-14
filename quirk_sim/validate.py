"""Per-event validation of the toy quirk simulator against For_Daniel truth.

For every event in mass_1000_Lambda_1000 (the config with full per-event truth)
we take the two quirks' exact initial conditions (vertex, momentum, charge),
integrate the coupled infracolor-string trajectory, intersect the cylindrical
barrel, and compare the simulated hits to the recorded truth hits:

  * spatial residual: median 3D distance [cm] from each truth hit to the
    nearest point on the simulated trajectory,
  * hit-count ratio: n_simulated_crossings / n_truth_hits,
  * max-layer agreement.

Run:  python -m quirk_sim.validate            (uses theory sigma)
      python -m quirk_sim.validate --fit-sigma  (also fit a global sigma)
"""
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import QuirkPair, Barrel
from . import dataio
from .units import CM, string_tension_from_lambda_eV, lambda_eV_from_string_tension

HERE = os.path.dirname(os.path.abspath(__file__))


def hit_xyz(hd):
    return np.c_[hd.r * np.cos(hd.phi), hd.r * np.sin(hd.phi), hd.z] * CM


def nearest_resid(traj, data_xyz):
    """median + per-hit nearest-point distance [cm]."""
    if len(data_xyz) == 0:
        return np.nan, np.array([])
    d = np.empty(len(data_xyz))
    for i, h in enumerate(data_xyz):
        dd = traj - h
        d[i] = np.sqrt((dd * dd).sum(axis=1)).min()
    return np.median(d) / CM, d / CM


def simulate_event(mass, lam, ev, sigma, string_mode=1, dt=1.0):
    try:
        parts, hits = dataio.load_event(mass, lam, ev)
    except Exception:
        return None
    if len(parts) != 2 or len(hits) == 0:
        return None
    qs = dataio.quirks_from_particles(parts)
    if len(qs) != 2:
        return None
    q1, q2 = qs
    v0 = 0.5 * (q1["r0"] + q2["r0"])
    barrel = Barrel()
    sim = QuirkPair(q1["mass"], sigma, b_tesla=2.0, string_mode=string_mode)
    res = sim.integrate(v0, q1["p"], q1["q"], v0, q2["p"], q2["q"],
                        dt=dt, max_steps=40000, r_stop_mm=barrel.r_max * 1.2)
    out = []
    for k, pid in [(0, parts.iloc[0]["particle_id"]),
                   (1, parts.iloc[1]["particle_id"])]:
        d = hit_xyz(hits[hits.particle_id == pid])
        traj = res["traj1"] if k == 0 else res["traj2"]
        med, allr = nearest_resid(traj, d)
        sim_hits = barrel.crossings(traj)
        out.append(dict(
            n_data=len(d), n_sim=len(sim_hits),
            median_resid=med, resid=allr,
            data_maxlayer=int(hits[hits.particle_id == pid].layer_id.max()),
            sim_maxlayer=max([h["layer_id"] for h in sim_hits], default=0),
        ))
    return out, res, parts, hits


def run(mass=1000, lam=1000, string_mode=1, nmax=None):
    sigma = string_tension_from_lambda_eV(lam)
    evs = dataio.list_events(mass, lam)
    if nmax:
        evs = evs[:nmax]
    rows, all_resid = [], []
    for ev in evs:
        r = simulate_event(mass, lam, ev, sigma, string_mode)
        if r is None:
            continue
        per, *_ = r
        for q in per:
            rows.append(dict(event=ev, **{k: q[k] for k in
                        ("n_data", "n_sim", "median_resid",
                         "data_maxlayer", "sim_maxlayer")}))
            all_resid.append(q["resid"])
    med = np.array([r["median_resid"] for r in rows])
    ratio = np.array([r["n_sim"] / max(r["n_data"], 1) for r in rows])
    dmax = np.array([r["data_maxlayer"] for r in rows])
    smax = np.array([r["sim_maxlayer"] for r in rows])
    allr = np.concatenate(all_resid) if all_resid else np.array([])
    print(f"\n=== mass={mass} Lambda={lam}  (sigma_theory="
          f"{sigma:.0f} MeV/mm, string_mode={string_mode}) ===")
    print(f"events={len(evs)}  quirks={len(rows)}")
    print(f"per-quirk median spatial residual [cm]: "
          f"median={np.median(med):.3f}  mean={np.mean(med):.3f}  "
          f"90%={np.percentile(med,90):.3f}")
    print(f"all-hit residual [cm]: median={np.median(allr):.3f}  "
          f"90%={np.percentile(allr,90):.3f}  95%={np.percentile(allr,95):.3f}")
    print(f"hit-count ratio n_sim/n_data: median={np.median(ratio):.3f}  "
          f"mean={np.mean(ratio):.3f}  [{np.percentile(ratio,10):.2f},"
          f"{np.percentile(ratio,90):.2f}]")
    print(f"max-layer |sim-data|: median={np.median(np.abs(smax-dmax)):.1f}  "
          f"mean={np.mean(np.abs(smax-dmax)):.2f}")
    frac1 = np.mean(med < 1.0); frac2 = np.mean(med < 2.0)
    print(f"fraction of quirks with median residual < 1cm: {frac1:.2%}, "
          f"< 2cm: {frac2:.2%}")
    return dict(rows=rows, med=med, ratio=ratio, allr=allr,
                dmax=dmax, smax=smax, sigma=sigma)


def make_plots(summary, mass=1000, lam=1000):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    ax[0].hist(summary["allr"], bins=60, range=(0, 5), color="C0")
    ax[0].set_xlabel("hit residual [cm]"); ax[0].set_ylabel("hits")
    ax[0].axvline(np.median(summary["allr"]), color="k", ls="--",
                  label=f"median {np.median(summary['allr']):.2f} cm")
    ax[0].legend(); ax[0].set_title("per-hit spatial residual")
    ax[1].hist(summary["ratio"], bins=40, range=(0, 3), color="C1")
    ax[1].axvline(1.0, color="k", ls="--")
    ax[1].set_xlabel("n_sim / n_data hits"); ax[1].set_ylabel("quirks")
    ax[1].set_title("hit-count ratio")
    ax[2].plot(summary["dmax"], summary["smax"], ".", alpha=0.5)
    lim = [summary["dmax"].min() - 1, summary["dmax"].max() + 1]
    ax[2].plot(lim, lim, "k--", lw=0.8)
    ax[2].set_xlabel("data max layer"); ax[2].set_ylabel("sim max layer")
    ax[2].set_title("max layer reached")
    fig.suptitle(f"Toy quirk sim vs For_Daniel truth  "
                 f"(mass={mass} GeV, Lambda={lam} eV, "
                 f"sigma={summary['sigma']:.0f} MeV/mm)")
    fig.tight_layout()
    out = os.path.join(HERE, "validation_summary.png")
    fig.savefig(out, dpi=120)
    print("wrote", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mass", type=int, default=1000)
    ap.add_argument("--lam", type=int, default=1000)
    ap.add_argument("--mode", type=int, default=1)
    ap.add_argument("--nmax", type=int, default=None)
    a = ap.parse_args()
    s = run(a.mass, a.lam, a.mode, a.nmax)
    make_plots(s, a.mass, a.lam)
