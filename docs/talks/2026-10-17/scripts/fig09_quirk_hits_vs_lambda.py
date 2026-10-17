"""Fig 9: mean digitised tracker hits per quirk vs Lambda (cost sets, 100 events = 200 quirks each) and the
fraction of quirks with >= 3 hits. A hit counts for a quirk if the quirk is among its particle_ids
(a hit shared by both quirks counts for both). m = 175 GeV scan plus m = 50 and 300 GeV at 100 and 10000 eV."""

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qcommon import C, MASS_C, SIM, hit_quirk_pids, quirks_of_event, read, record, save, setup, plt, np  # noqa: E402

LAM175 = [100, 167, 278, 464, 774, 1292, 2154, 3594, 5995, 10000]
SETS = [(175, L) for L in LAM175] + [(m, L) for m in (50, 300) for L in (100, 10000)]


def per_quirk_hits(ds):
    d = f"{SIM}/{ds}/cost/runs/0"
    p = read(d + "/particles", ["event_id", "particle_id", "pdg_id", "charge"])
    h = read(d + "/tracker_hits", ["event_id", "particle_ids"])
    prow = {int(e): i for i, e in enumerate(p["event_id"])}
    n = []
    for i, e in enumerate(h["event_id"]):
        hidx, fl = hit_quirk_pids(h["particle_ids"], i)
        for qid, _ in quirks_of_event(p, prow[int(e)]):
            n.append(len(np.unique(hidx[fl == qid])))
    return np.array(n), d


def main():
    setup()
    res, srcs = {}, {}
    for m, L in SETS:
        ds = f"m{m}_L{L}_mu0"
        n, d = per_quirk_hits(ds)
        res[(m, L)] = n
        srcs[ds] = d
        print(ds, len(n), n.mean(), (n >= 3).mean())
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(12, 4.6))
    marker = {175: "o", 50: "s", 300: "D"}
    dodge = {50: 0.94, 175: 1.0, 300: 1.06}
    b1, b2 = np.sqrt(278 * 464), np.sqrt(1292 * 2154)
    for ax in (axa, axb):
        ax.axvspan(b1, b2, color="#f3f2ee", lw=0, zorder=0)
    for m in (50, 175, 300):
        Ls = np.array(sorted(L for mm, L in res if mm == m))
        mean = np.array([res[(m, L)].mean() for L in Ls])
        sem = np.array([res[(m, L)].std(ddof=1) / np.sqrt(len(res[(m, L)])) for L in Ls])
        k = np.array([(res[(m, L)] >= 3).sum() for L in Ls])
        tot = np.array([len(res[(m, L)]) for L in Ls])
        f = k / tot
        fe = np.sqrt(f * (1 - f) / tot)
        ls = "-" if m == 175 else "none"
        kw = dict(marker=marker[m], ms=8, color=MASS_C[m], mec="white", mew=1, lw=2, ls=ls, elinewidth=1.5,
                  capsize=0, label=f"m = {m} GeV", zorder=3)
        axa.errorbar(Ls * dodge[m], mean, yerr=sem, **kw)
        axb.errorbar(Ls * dodge[m], f, yerr=fe, **kw)
    for ax in (axa, axb):
        ax.set_xscale("log")
        ax.set_xlim(70, 14000)
        ax.set_xlabel("Λ [eV]")
        for xm, lab in ((np.sqrt(70 * b1), "separated"), (np.sqrt(b1 * b2), "interleaved"),
                        (np.sqrt(b2 * 14000), "unresolved")):
            ax.text(xm, 0.97, lab, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=10.5,
                    color=C["ink2"])
    axa.set_ylabel("Mean tracker hits per quirk")
    axa.set_ylim(0, 20)
    axb.set_ylabel("Fraction of quirks with ≥ 3 hits")
    axb.set_ylim(0.9, 1.012)
    axa.legend(loc="lower left")
    fig.tight_layout(w_pad=2.5)
    save(fig, "quirk_hits_vs_lambda")
    vals = {f"m{m}_L{L}": {"quirks": int(len(n)), "mean_hits_per_quirk": float(n.mean()),
                           "sem": float(n.std(ddof=1) / np.sqrt(len(n))), "frac_ge3": float((n >= 3).mean()),
                           "median": float(np.median(n))} for (m, L), n in res.items()}
    record("quirk_hits_vs_lambda", vals, srcs)


if __name__ == "__main__":
    main()
