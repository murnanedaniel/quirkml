"""Fig 6: multi-thread validation of the quirk Geant4 plugin. The same 400 input events (m = 400 GeV,
Lambda = 1000 eV) simulated with 1 and 8 Geant4 threads; per-event quirk hits (simulated tracker hits whose
MC particle is a quirk), paired by eventNumber. Logic follows qml_img/scripts/mt_stats.py.
Wall times from mt_test/run_big.out ('<run> rc=0 <s> s')."""

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qcommon import C, QUIRK_PDG, record, save, setup, plt, np  # noqa: E402
import uproot  # noqa: E402
import awkward as ak  # noqa: E402

MT = "/pscratch/sd/d/danieltm/qml_img/quirks/mt_test"
RO = ["PixelBarrelReadout", "PixelEndcapReadout", "ShortStripBarrelReadout", "ShortStripEndcapReadout",
      "LongStripBarrelReadout", "LongStripEndcapReadout"]


def load(path):
    t = uproot.open(path)["events"]
    evn = ak.to_numpy(t["EventHeader/EventHeader.eventNumber"].array()[:, 0])
    pdg = t["MCParticles/MCParticles.PDG"].array()
    nq = np.zeros(len(evn))
    for ro in RO:
        idx = t[f"_{ro}_particle/_{ro}_particle.index"].array()
        nq += ak.to_numpy(ak.sum(np.abs(pdg[idx]) == QUIRK_PDG, axis=1))
    o = np.argsort(evn)
    return evn[o], nq[o]


def walltimes():
    out = {}
    for line in open(f"{MT}/run_big.out"):
        m = re.match(r"(big_t\d) rc=0 (\d+) s", line.strip())
        if m:
            out[m.group(1)] = int(m.group(2))
    return out


def main():
    setup()
    e1, a = load(f"{MT}/big_t1/edm4hep.root")
    e8, b = load(f"{MT}/big_t8/edm4hep.root")
    common = np.intersect1d(e1, e8)
    a, b = a[np.isin(e1, common)], b[np.isin(e8, common)]
    d = b - a
    dm, de = d.mean(), d.std(ddof=1) / np.sqrt(len(d))
    wt = walltimes()
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(11.5, 4.8))
    lim = np.percentile(np.concatenate([a, b]), 99.5) * 1.1
    axa.plot([0, lim], [0, lim], color=C["axis"], lw=1, zorder=0)
    axa.scatter(a, b, s=14, color=C["qplus"], alpha=0.6, lw=0)
    axa.set_xlim(0, lim)
    axa.set_ylim(0, lim)
    axa.set_aspect("equal")
    axa.set_xlabel("Simulated quirk hits per event, 1 thread")
    axa.set_ylabel("Simulated quirk hits per event, 8 threads")
    frac_same = float(np.mean(d == 0))
    lo, hi = np.percentile(d, [0.5, 99.5])
    w = max(abs(lo), abs(hi))
    bins = np.arange(-np.ceil(w) - 0.5, np.ceil(w) + 1.5, 1)
    axb.hist(np.clip(d, bins[0] + 0.5, bins[-1] - 0.5), bins=bins, color=C["qplus"], edgecolor="white", lw=0.8)
    axb.set_xlabel("Paired difference in quirk hits (8 threads − 1 thread)")
    axb.set_ylabel("Events")
    axb.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    txt = (f"{len(d)} events, same input\n"
           f"mean quirk hits: {a.mean():.2f} (1 thr) vs {b.mean():.2f} (8 thr)\n"
           f"paired difference: {dm:+.2f} ± {de:.2f}\n"
           f"wall time: {wt['big_t1']} s vs {wt['big_t8']} s ({wt['big_t1'] / wt['big_t8']:.1f}×)")
    axb.text(0.02, 0.97, txt, transform=axb.transAxes, va="top", ha="left", fontsize=10, color=C["ink"],
             bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=C["grid"]))
    axb.set_ylim(0, axb.get_ylim()[1] * 1.45)
    fig.tight_layout(w_pad=2.5)
    save(fig, "mt_validation")
    record("mt_validation", {"events": int(len(d)), "mean_quirk_hits_t1": float(a.mean()), "mean_quirk_hits_t8": float(b.mean()),
                             "paired_diff_mean": float(dm), "paired_diff_sem": float(de), "frac_events_identical": frac_same,
                             "wall_s_t1": wt["big_t1"], "wall_s_t8": wt["big_t8"], "speedup": wt["big_t1"] / wt["big_t8"]},
           {"edm4hep_t1": f"{MT}/big_t1/edm4hep.root", "edm4hep_t8": f"{MT}/big_t8/edm4hep.root",
            "wall": f"{MT}/run_big.out", "logic": "/pscratch/sd/d/danieltm/qml_img/scripts/mt_stats.py"})
    print(len(d), a.mean(), b.mean(), dm, de, frac_same, wt)


if __name__ == "__main__":
    main()
