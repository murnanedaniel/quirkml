"""Fig 8: m = 200 GeV, Lambda = 774 eV, test events 17000-19999 (Release-1 style copy, one particle per hit).
For every quirk hit: 3D distance to the nearest hit of the OTHER quirk, and 3D distance to the next hit of the
SAME quirk in true-time order. Measured hit positions (x, y, z)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qcommon import C, R1, r1_events, record, save, setup, plt, np  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

DS, FIRST, LAST = "m200_L774_mu0", 17000, 19999


def main():
    setup()
    d_other, d_same = [], []
    n_ev = 0
    for e in r1_events(DS, FIRST, LAST):
        q = e["quirks"]
        if len(q) != 2 or min(len(v) for v in q.values()) < 2:
            continue
        n_ev += 1
        pos = np.stack([e["x"], e["y"], e["z"]], 1).astype(np.float64)
        a, b = q["qplus"], q["qminus"]
        for s, o in ((a, b), (b, a)):
            d_other.append(cKDTree(pos[o]).query(pos[s])[0])
            d_same.append(np.linalg.norm(np.diff(pos[s], axis=0), axis=1))
    d_other, d_same = np.concatenate(d_other), np.concatenate(d_same)
    med_o, med_s = np.median(d_other), np.median(d_same)
    frac_closer = float(np.mean(d_other < med_s))

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    bins = np.logspace(-1, 3.5, 70)
    ax.hist(d_other, bins=bins, histtype="step", lw=2, color=C["qminus"],
            label="to nearest hit of the other quirk")
    ax.hist(d_same, bins=bins, histtype="step", lw=2, color=C["qplus"],
            label="to next hit of the same quirk (true time order)")
    ax.set_xscale("log")
    ax.set_xlabel("3D distance between hits [mm]")
    ax.set_ylabel("Quirk hits per bin")
    ymax = ax.get_ylim()[1]
    for m, k, ha in ((med_o, "qminus", "right"), (med_s, "qplus", "left")):
        ax.axvline(m, ymax=0.78, color=C[k], lw=1, alpha=0.8)
        ax.text(m * (0.92 if ha == "right" else 1.08), ymax * 1.0, f"median {m:.0f} mm", color=C["ink2"],
                fontsize=10, va="bottom", ha=ha)
    ax.set_ylim(0, ymax * 1.28)
    ax.set_xlim(0.1, 4000)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.0))
    ax.text(0.02, 0.62, f"m = 200 GeV, Λ = 774 eV\n{n_ev} events, no pile-up", transform=ax.transAxes,
            ha="left", va="top", fontsize=10, color=C["ink2"])
    save(fig, "quirk_separation")
    record("quirk_separation", {
        "events": n_ev, "quirk_hits_other": int(len(d_other)), "pairs_same": int(len(d_same)),
        "median_nearest_other_quirk_mm": float(med_o), "median_consecutive_same_quirk_mm": float(med_s),
        "q10_q90_nearest_other_mm": np.percentile(d_other, [10, 90]).tolist(),
        "q10_q90_consecutive_same_mm": np.percentile(d_same, [10, 90]).tolist(),
        "frac_hits_with_other_quirk_closer_than_median_same_step": frac_closer,
    }, {"hits": f"{R1}/{DS}/reco/tracker_hits (events {FIRST}-{LAST})", "particles": f"{R1}/{DS}/truth/particles"})
    print(med_o, med_s, frac_closer, n_ev)


if __name__ == "__main__":
    main()
