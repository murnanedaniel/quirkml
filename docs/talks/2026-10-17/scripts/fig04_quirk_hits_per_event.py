"""Fig 4: quirk hits per event, m = 400 GeV, Lambda = 1000 eV, no pile-up vs mu = 10 (packaged, 8 runs x 500).
A hit counts as a quirk hit if any of its particle_ids is a quirk (pdg +-10000100). Right panel: per-run means
with standard errors (run = event_id // 500)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qcommon import C, QUIRK_PDG, SIM, read, record, save, setup, plt, np  # noqa: E402

DS = {"mu0": "m400_L1000_mu0", "mu10": "m400_L1000_mu10"}
LABEL = {"mu0": "no pile-up", "mu10": "pile-up μ = 10"}
EV_PER_RUN = 500


def quirk_hits(ds):
    base = f"{SIM}/{ds}/v1/parquet"
    p = read(f"{base}/truth/particles", ["event_id", "particle_id", "pdg_id"])
    h = read(f"{base}/reco/tracker_hits", ["event_id", "particle_ids"])
    o, pid = p["particle_id"]
    _, pdg = p["pdg_id"]
    isq = np.abs(pdg) == QUIRK_PDG
    ev_of = np.repeat(p["event_id"], np.diff(o))
    qkeys = set(zip(ev_of[isq].tolist(), pid[isq].tolist()))
    hev = h["event_id"]
    o1, o2, v = h["particle_ids"]
    n_entries_per_ev = o2[o1[1:]] - o2[o1[:-1]]
    ev_entry = np.repeat(hev, n_entries_per_ev)
    hit_entry = np.repeat(np.arange(len(o2) - 1), np.diff(o2))
    # vectorised membership: encode (event, pid) pairs
    qev = np.array([k[0] for k in qkeys], np.int64)
    qpid = np.array([k[1] for k in qkeys], np.uint64)
    key_q = qev.astype(np.uint64) * np.uint64(1 << 40) + qpid
    key_e = ev_entry.astype(np.uint64) * np.uint64(1 << 40) + v
    m = np.isin(key_e, key_q)
    qh = np.unique(hit_entry[m])  # global hit indices that carry a quirk
    hit_ev = np.repeat(hev, np.diff(o1))
    n = np.bincount(np.searchsorted(np.sort(hev), hit_ev[qh]), minlength=len(hev))
    return np.sort(hev), n, base, int(len(qkeys))


def main():
    setup()
    res = {}
    for k, ds in DS.items():
        ev, n, base, nq = quirk_hits(ds)
        res[k] = (ev, n, base, nq)
        print(k, len(ev), n.mean(), nq, ev.min(), ev.max())
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1.5, 1]})
    hi = max(np.percentile(r[1], 99.5) for r in res.values())
    bins = np.arange(-0.5, hi + 1.5, 1)
    vals, srcs = {}, {}
    for j, k in enumerate(("mu0", "mu10")):
        ev, n, base, nq = res[k]
        axa.hist(n, bins=bins, histtype="step", lw=2, color=C[k], label=f"{LABEL[k]}: mean {n.mean():.2f}")
        run = ev // EV_PER_RUN
        runs = np.unique(run)
        mean = np.array([n[run == r].mean() for r in runs])
        err = np.array([n[run == r].std(ddof=1) / np.sqrt((run == r).sum()) for r in runs])
        xoff = -0.12 if k == "mu0" else 0.12
        axb.errorbar(runs + xoff, mean, yerr=err, fmt="o", ms=7, color=C[k], mec="white", mew=1, elinewidth=1.5,
                     capsize=0, label=LABEL[k])
        axb.axhline(n.mean(), color=C[k], lw=1, alpha=0.5, zorder=0)
        vals[k] = {"events": int(len(n)), "quirks": nq, "mean_quirk_hits_per_event": float(n.mean()),
                   "sem": float(n.std(ddof=1) / np.sqrt(len(n))), "median": float(np.median(n)),
                   "events_with_zero_quirk_hits": int((n == 0).sum()),
                   "per_run_mean": dict(zip(runs.tolist(), np.round(mean, 3).tolist())),
                   "per_run_sem": dict(zip(runs.tolist(), np.round(err, 3).tolist()))}
        srcs[k] = base
    d = vals["mu10"]["mean_quirk_hits_per_event"] - vals["mu0"]["mean_quirk_hits_per_event"]
    de = np.hypot(vals["mu10"]["sem"], vals["mu0"]["sem"])
    vals["difference_mu10_minus_mu0"] = [float(d), float(de)]
    axa.set_xlabel("Quirk hits per event")
    axa.set_ylabel("Events")
    axa.set_xlim(bins[0], bins[-1])
    axa.legend(loc="upper right")
    axa.text(0.98, 0.72, "m = 400 GeV, Λ = 1000 eV\n4000 events each", transform=axa.transAxes, ha="right",
             va="top", fontsize=10, color=C["ink2"])
    axb.set_xlabel("Run")
    axb.set_ylabel("Mean quirk hits per event")
    axb.set_xticks(np.arange(8))
    lo = min(min(v["per_run_mean"].values()) for v in (vals["mu0"], vals["mu10"]))
    hi2 = max(max(v["per_run_mean"].values()) for v in (vals["mu0"], vals["mu10"]))
    axb.set_ylim(lo - 1.5, hi2 + 2.0)
    axb.legend(loc="upper left", ncol=2)
    fig.tight_layout(w_pad=2.5)
    save(fig, "quirk_hits_per_event")
    record("quirk_hits_per_event", vals, srcs)
    print(vals["difference_mu10_minus_mu0"])


if __name__ == "__main__":
    main()
