"""Fig 2: one m = 200 GeV, Lambda = 774 eV event (Release-1 style copy, test range 17000-19999), zoomed so the
side-by-side interleaving of the two quirks is visible. Hits of each quirk joined in true-time order.

Left: x-y view around the quirk pair. Right: the same hits unrolled, transverse offset r * dphi from the
pair's mean direction vs r (stretches the small transverse separation).

Event choice (deterministic): first event in the range where both quirks have 13-20 hits, all quirk hits
have |z| < 1000 mm and r reaches > 700 mm for both, and the median distance to the nearest other-quirk hit
is 15-30 mm (close to the sample median of about 22 mm).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qcommon import C, QLABEL, R1, r1_events, record, save, setup, plt, np  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

DS, FIRST, LAST = "m200_L774_mu0", 17000, 19999


def pick():
    for e in r1_events(DS, FIRST, LAST):
        q = e["quirks"]
        if len(q) != 2:
            continue
        n = [len(v) for v in q.values()]
        if min(n) < 13 or max(n) > 20:
            continue
        x, y, z = e["x"], e["y"], e["z"]
        allq = np.concatenate(list(q.values()))
        if np.abs(z[allq]).max() > 1000:
            continue
        r = np.hypot(x, y)
        if min(r[v].max() for v in q.values()) < 700:
            continue
        pos = np.stack([x, y, z], 1)
        dd = np.concatenate([cKDTree(pos[q["qminus"]]).query(pos[q["qplus"]])[0],
                             cKDTree(pos[q["qplus"]]).query(pos[q["qminus"]])[0]])
        med = float(np.median(dd))
        if 15 <= med <= 30:
            return e, med
    raise RuntimeError("no event matched")


def main():
    setup()
    e, med = pick()
    q = e["quirks"]
    x, y, z = e["x"], e["y"], e["z"]
    r = np.hypot(x, y)
    allq = np.concatenate([q["qplus"], q["qminus"]])
    phi0 = np.arctan2(y[allq].mean(), x[allq].mean())
    dphi = np.angle(np.exp(1j * (np.arctan2(y, x) - phi0)))
    off = r * dphi

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(13, 5.6), gridspec_kw={"width_ratios": [1, 1.25]})
    pad = 40
    x0, x1 = x[allq].min() - pad, x[allq].max() + pad
    y0, y1 = y[allq].min() - pad, y[allq].max() + pad
    inwin = (x > x0) & (x < x1) & (y > y0) & (y < y1)
    isq = np.zeros(len(x), bool)
    isq[allq] = True
    axa.scatter(x[inwin & ~isq], y[inwin & ~isq], s=8, color=C["other"], lw=0, label="other hits")
    wb = (np.abs(off) < 120) & (r < r[allq].max() + 50)
    axb.scatter(r[wb & ~isq], off[wb & ~isq], s=8, color=C["other"], lw=0)
    for k in ("qplus", "qminus"):
        v = q[k]
        for ax, (u, w) in ((axa, (x, y)), (axb, (r, off))):
            ax.plot(u[v], w[v], "-", color=C[k], lw=1.2, alpha=0.8, zorder=2)
            ax.scatter(u[v], w[v], s=28, color=C[k], edgecolor="white", lw=0.6, zorder=3,
                       label=QLABEL[k] if ax is axa else None)
        # mark the first hit in time
        axb.scatter(r[v[0]], off[v[0]], s=70, facecolor="none", edgecolor=C[k], lw=1.2, zorder=4)
    axa.set_xlim(x0, x1)
    axa.set_ylim(y0, y1)
    axa.set_aspect("equal")
    axa.set_xlabel("x [mm]")
    axa.set_ylabel("y [mm]")
    axa.legend(loc="best", handletextpad=0.2)
    axb.set_xlabel("r [mm]")
    axb.set_ylabel("transverse offset r·Δφ [mm]")
    axb.axhline(0, color=C["grid"], lw=0.8, zorder=0)
    axb.text(0.02, 0.97, f"event {e['event_id']}, m = 200 GeV, Λ = 774 eV\n"
             f"{len(q['qplus'])} + {len(q['qminus'])} quirk hits, lines follow true time\n"
             f"median distance to nearest other-quirk hit: {med:.0f} mm\nopen circle = first hit",
             transform=axb.transAxes, va="top", ha="left", fontsize=10, color=C["ink2"])
    ylim = np.abs(off[allq]).max() * 1.6 + 10
    axb.set_ylim(-ylim, ylim * 1.35)
    fig.tight_layout(w_pad=2)
    save(fig, "evd_interleaved_zoom")
    record("evd_interleaved_zoom", {"event_id": e["event_id"], "quirk_plus_hits": len(q["qplus"]),
                                    "quirk_minus_hits": len(q["qminus"]), "median_nearest_other_quirk_mm": med},
           {"hits": f"{R1}/{DS}/reco/tracker_hits", "particles": f"{R1}/{DS}/truth/particles"})
    print(e["event_id"], med)


if __name__ == "__main__":
    main()
