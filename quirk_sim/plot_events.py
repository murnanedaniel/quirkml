"""Overlay simulated quirk trajectories on the For_Daniel truth hits.

Produces quirk_sim/event_overlays.png: a row per event, x-y and r-z views,
data hits as markers and the toy trajectory as lines, at the theory string
tension with the relativistic string force.
"""
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import QuirkPair, Barrel
from . import dataio
from .units import CM, string_tension_from_lambda_eV

HERE = os.path.dirname(os.path.abspath(__file__))


def _xy(hd):
    return hd.r * np.cos(hd.phi), hd.r * np.sin(hd.phi)


def main(mass=1000, lam=1000, events=None):
    sigma = string_tension_from_lambda_eV(lam)
    barrel = Barrel()
    if events is None:
        events = dataio.list_events(mass, lam)[:3]
    n = len(events)
    fig, axes = plt.subplots(n, 2, figsize=(11, 4.0 * n))
    if n == 1:
        axes = axes[None, :]
    for i, ev in enumerate(events):
        parts, hits = dataio.load_event(mass, lam, ev)
        qs = dataio.quirks_from_particles(parts)
        q1, q2 = qs
        v0 = 0.5 * (q1["r0"] + q2["r0"])
        sim = QuirkPair(q1["mass"], sigma, b_tesla=2.0, string_mode=1)
        res = sim.integrate(v0, q1["p"], q1["q"], v0, q2["p"], q2["q"],
                            dt=1.0, max_steps=40000, r_stop_mm=barrel.r_max * 1.2)
        t1, t2 = res["traj1"] / CM, res["traj2"] / CM
        pid1, pid2 = parts.iloc[0]["particle_id"], parts.iloc[1]["particle_id"]
        hd1, hd2 = hits[hits.particle_id == pid1], hits[hits.particle_id == pid2]

        ax = axes[i, 0]
        for R in barrel.radii / CM:
            ax.add_patch(plt.Circle((0, 0), R, fill=False, ec="0.9", lw=0.4))
        ax.plot(t1[:, 0], t1[:, 1], "-", lw=0.8, color="C0", label="sim quirk")
        ax.plot(t2[:, 0], t2[:, 1], "-", lw=0.8, color="C3", label="sim antiquirk")
        x1, y1 = _xy(hd1); x2, y2 = _xy(hd2)
        ax.plot(x1, y1, "o", ms=4, color="C0", mfc="none")
        ax.plot(x2, y2, "s", ms=4, color="C3", mfc="none")
        ax.set_aspect("equal"); ax.set_xlim(-66, 66); ax.set_ylim(-66, 66)
        ax.set_xlabel("x [cm]"); ax.set_ylabel("y [cm]")
        ax.set_title(f"event {ev}: x-y")
        if i == 0:
            ax.legend(fontsize=8, loc="upper right")

        ax = axes[i, 1]
        r1 = np.hypot(t1[:, 0], t1[:, 1]); r2 = np.hypot(t2[:, 0], t2[:, 1])
        ax.plot(t1[:, 2], r1, "-", lw=0.8, color="C0")
        ax.plot(t2[:, 2], r2, "-", lw=0.8, color="C3")
        ax.plot(hd1.z, hd1.r, "o", ms=4, color="C0", mfc="none")
        ax.plot(hd2.z, hd2.r, "s", ms=4, color="C3", mfc="none")
        ax.set_ylim(0, 66); ax.set_xlabel("z [cm]"); ax.set_ylabel("r [cm]")
        ax.set_title(f"event {ev}: r-z  (markers=truth, lines=toy)")

    fig.suptitle(f"Toy quirk simulation vs For_Daniel truth  "
                 f"(mass={mass} GeV, Lambda={lam} eV, sigma={sigma:.0f} MeV/mm, "
                 f"relativistic string)", y=1.0)
    fig.tight_layout()
    out = os.path.join(HERE, "event_overlays.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", type=int, nargs="*", default=None)
    a = ap.parse_args()
    main(events=a.events)
