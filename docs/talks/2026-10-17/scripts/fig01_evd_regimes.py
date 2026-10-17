"""Fig 1: event displays of the two quirks' hits at m = 175 GeV for three string tensions
(Lambda = 100, 774, 10000 eV; cost sets, 100 events each). Top row x-y, bottom row r-z.

Event choice: among events where both quirks have at least the 75th-percentile number of hits (of the
smaller quirk), take the one whose larger quirk has the fewest hits (avoids loopers). Deterministic.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qcommon import C, QLABEL, SIM, ev, hit_quirk_pids, quirks_of_event, read, record, save, setup, plt, np  # noqa: E402

LAMBDAS = [100, 774, 10000]
REGIME = {100: "separated", 774: "interleaved", 10000: "unresolved"}


def load(ds):
    d = f"{SIM}/{ds}/cost/runs/0"
    p = read(d + "/particles", ["event_id", "particle_id", "pdg_id", "charge"])
    h = read(d + "/tracker_hits", ["event_id", "x", "y", "z", "volume_id", "layer_id", "particle_ids", "simhit_ids"])
    s = read(d + "/tracker_simhits", ["event_id", "true_time"])
    return d, p, h, s


def quirk_hits(p, h, s, i):
    """{key: (hit indices sorted by true time)} for quirk+ / quirk- in event i (row i of all tables)."""
    q = quirks_of_event(p, i)
    hidx, fl = hit_quirk_pids(h["particle_ids"], i)
    _, sids = hit_quirk_pids(h["simhit_ids"], i)
    t = ev(s["true_time"], i)[sids]
    out = {}
    for (qid, ch) in q:
        m = fl == qid
        hh, tt = hidx[m], t[m]
        order = np.argsort(tt, kind="stable")
        hh = hh[order]
        _, first = np.unique(hh, return_index=True)
        out["qplus" if ch > 0 else "qminus"] = hh[np.sort(first)]
    return out


def layers(h):
    """Layer outlines from the hits themselves: barrel = (r, zmin, zmax); endcap = (z, rmin, rmax)."""
    x, y, z = h["x"][1], h["y"][1], h["z"][1]
    vol, lay = h["volume_id"][1], h["layer_id"][1]
    r = np.hypot(x, y)
    key = vol.astype(np.int64) * 10000 + lay
    bar, ec = [], []
    for k in np.unique(key):
        m = key == k
        if m.sum() < 200:
            continue
        rr, zz = r[m], z[m]
        if np.ptp(zz) > 2 * np.ptp(rr):
            bar.append((np.median(rr), np.percentile(zz, 0.5), np.percentile(zz, 99.5)))
        else:
            ec.append((np.median(zz), np.percentile(rr, 0.5), np.percentile(rr, 99.5)))
    return bar, ec


def main():
    setup()
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.2), gridspec_kw={"height_ratios": [1.25, 1]})
    numbers, sources = {}, {}
    geo = None
    for j, L in enumerate(LAMBDAS):
        ds = f"m175_L{L}_mu0"
        d, p, h, s = load(ds)
        if geo is None:
            geo = layers(h)
        n_ev = len(p["event_id"])
        per = [quirk_hits(p, h, s, i) for i in range(n_ev)]
        nmin = np.array([min(len(v) for v in q.values()) for q in per])
        nmax = np.array([max(len(v) for v in q.values()) for q in per])
        cand = np.flatnonzero(nmin >= np.percentile(nmin, 75))
        i = int(cand[np.argmin(nmax[cand])])
        qh = per[i]
        x, y, z = ev(h["x"], i), ev(h["y"], i), ev(h["z"], i)
        r = np.hypot(x, y)
        isq = np.zeros(len(x), bool)
        for v in qh.values():
            isq[v] = True
        axa, axb = axes[0, j], axes[1, j]
        bar, ec = geo
        for (rb, z0, z1) in bar:
            axa.add_patch(plt.Circle((0, 0), rb, fill=False, color=C["layer"], lw=0.8, zorder=0))
            axb.plot([z0, z1], [rb, rb], color=C["layer"], lw=0.8, zorder=0)
        for (ze, r0, r1) in ec:
            axb.plot([ze, ze], [r0, r1], color=C["layer"], lw=0.8, zorder=0)
        axa.scatter(x[~isq], y[~isq], s=2, color=C["other"], lw=0, zorder=1, rasterized=True)
        axb.scatter(z[~isq], r[~isq], s=2, color=C["other"], lw=0, zorder=1, rasterized=True)
        # quirk (+) drawn larger underneath, quirk (-) smaller on top: a shared hit shows as a blue ring
        size = {"qplus": 34, "qminus": 11}
        for k in ("qplus", "qminus"):
            v = qh[k]
            axa.plot(x[v], y[v], "-", color=C[k], lw=0.8, alpha=0.6, zorder=2)
            axa.scatter(x[v], y[v], s=size[k], color=C[k], edgecolor="white", lw=0.4, zorder=3, label=QLABEL[k])
            axb.plot(z[v], r[v], "-", color=C[k], lw=0.8, alpha=0.6, zorder=2)
            axb.scatter(z[v], r[v], s=size[k], color=C[k], edgecolor="white", lw=0.4, zorder=3)
        shared = len(np.intersect1d(qh["qplus"], qh["qminus"]))
        if L >= 774:
            # zoom inset around the middle quirk (+) hit
            v = qh["qplus"]
            c = v[len(v) // 2]
            w = 75.0 if L == 774 else 40.0
            ins = axa.inset_axes([0.56, 0.04, 0.4, 0.4])
            ins.scatter(x[~isq], y[~isq], s=6, color=C["other"], lw=0)
            for k in ("qplus", "qminus"):
                vv = qh[k]
                ins.plot(x[vv], y[vv], "-", color=C[k], lw=0.8, alpha=0.6)
                ins.scatter(x[vv], y[vv], s=size[k] * 2, color=C[k], edgecolor="white", lw=0.4, zorder=3)
            ins.set_xlim(x[c] - w, x[c] + w)
            ins.set_ylim(y[c] - w, y[c] + w)
            ins.set_aspect("equal")
            ins.tick_params(labelsize=8, length=2)
            for sp in ins.spines.values():
                sp.set_visible(True)
                sp.set_color(C["axis"])
            ins.text(0.05, 0.93, f"{2 * w:.0f} mm \u00d7 {2 * w:.0f} mm", transform=ins.transAxes, fontsize=8, va="top", color=C["ink2"])
            axa.indicate_inset_zoom(ins, edgecolor=C["muted"], lw=0.8)
        axa.set_xlim(-1150, 1150)
        axa.set_ylim(-1150, 1150)
        axa.set_aspect("equal")
        axa.set_xlabel("x [mm]")
        axb.set_xlim(-3100, 3100)
        axb.set_ylim(0, 1150)
        axb.set_xlabel("z [mm]")
        if j == 0:
            axa.set_ylabel("y [mm]")
            axb.set_ylabel("r [mm]")
        axa.text(0.03, 0.97, f"Λ = {L} eV\n{REGIME[L]}", transform=axa.transAxes, va="top", ha="left",
                 fontsize=12, color=C["ink"])
        nq = {k: len(v) for k, v in qh.items()}
        axb.text(0.03, 0.96, f"event {int(p['event_id'][i])}: {nq['qplus']} + {nq['qminus']} quirk hits" + (f" ({shared} shared)" if shared else ""),
                 transform=axb.transAxes, va="top", ha="left", fontsize=10.5, color=C["ink2"])
        numbers[ds] = {"event_id": int(p["event_id"][i]), "quirk_plus_hits": nq["qplus"], "quirk_minus_hits": nq["qminus"],
                       "all_hits": int(len(x)), "shared_quirk_hits": int(shared), "p75_min_quirk_hits": float(np.percentile(nmin, 75))}
        sources[ds] = d
    axes[0, 0].legend(loc="lower left", handletextpad=0.2, markerscale=1.3)
    fig.tight_layout(w_pad=1.5, h_pad=1.0)
    save(fig, "evd_regimes")
    numbers["mass_GeV"] = 175
    record("evd_regimes", numbers, sources)


if __name__ == "__main__":
    main()
