"""Fig 3: Geant4 simulation cost per event vs string tension Lambda (cost sets, 100 events, 8 threads).
core-s/event = (simulation wall seconds - 40 s start-up) / 100 events * 8 threads.
Source: cost/times.txt; lines '<dataset> simulation rc=0 <s>s'; 0-1 s lines after '--- resumed' are skips."""

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qcommon import C, MASS_C, record, save, setup, plt, np  # noqa: E402

TIMES = "/pscratch/sd/d/danieltm/quirk_pilots/cost/times.txt"
STARTUP, NEV, NTHR = 40.0, 100, 8


def parse():
    sim = {}
    for line in open(TIMES):
        m = re.match(r"(\S+) simulation rc=0 (\d+)s", line.strip())
        if m and int(m.group(2)) > 1:
            sim[m.group(1)] = int(m.group(2))
    return sim


def main():
    setup()
    sim = parse()
    cost = {k: (v - STARTUP) / NEV * NTHR for k, v in sim.items()}
    pts = {}
    for k, v in cost.items():
        m = re.match(r"m(\d+)_L(\d+)_mu(\d+)", k)
        pts.setdefault((int(m.group(1)), int(m.group(3))), []).append((int(m.group(2)), v))
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    # regimes: separated <= 278, interleaved 464-1292, unresolved >= 2154 (boundaries at geometric midpoints)
    b1, b2 = np.sqrt(278 * 464), np.sqrt(1292 * 2154)
    ax.axvspan(b1, b2, color="#f3f2ee", lw=0, zorder=0)
    for xm, lab in ((np.sqrt(70 * b1), "separated"), (np.sqrt(b1 * b2), "interleaved"), (np.sqrt(b2 * 14000), "unresolved")):
        ax.text(xm, 0.97, lab, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=11, color=C["ink2"])
    order = [(50, 0), (175, 0), (300, 0)]
    marker = {175: "o", 50: "s", 300: "D"}
    dodge = {50: 0.94, 175: 1.0, 300: 1.06}  # small horizontal offset so overlapping points stay visible
    for (m, mu) in order:
        L, v = np.array(sorted(pts[(m, mu)])).T
        ax.plot(L * dodge[m], v, ls="-" if m == 175 else "none", marker=marker[m], ms=8, color=MASS_C[m],
                mec="white", mew=1, lw=2, label=f"m = {m} GeV", zorder=3)
    mu60 = cost["m200_L774_mu60"]
    ax.scatter([774], [mu60], marker="_", s=500, lw=2.5, color=C["ink"], zorder=4)
    ax.annotate(f"m = 200 GeV, pile-up μ = 60\n{mu60:.0f} core-s/event", (774, mu60), xytext=(1500, mu60 * 3.0),
                fontsize=10, color=C["ink2"], ha="left", arrowprops=dict(arrowstyle="-", color=C["muted"], lw=0.8))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(70, 14000)
    ax.set_ylim(1, 1500)
    ax.set_xlabel("Λ [eV]")
    ax.set_ylabel("Simulation cost [core-s / event]")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.9))
    save(fig, "cost_vs_lambda")
    record("cost_vs_lambda", {
        "formula": "core-s/event = (sim_seconds - 40) / 100 * 8",
        "sim_wall_seconds": sim, "core_s_per_event": {k: round(v, 2) for k, v in cost.items()},
        "ratio_m175_L10000_over_L100": cost["m175_L10000_mu0"] / cost["m175_L100_mu0"],
    }, {"times": TIMES})
    for k in sorted(cost):
        print(k, sim[k], round(cost[k], 1))


if __name__ == "__main__":
    main()
