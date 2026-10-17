"""Fig 5: all tracker hits per event, m = 400 GeV, Lambda = 1000 eV, no pile-up vs mu = 10 (packaged, 4000 events each)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qcommon import C, SIM, read, record, save, setup, plt, np  # noqa: E402

DS = {"mu0": "m400_L1000_mu0", "mu10": "m400_L1000_mu10"}
LABEL = {"mu0": "no pile-up", "mu10": "pile-up μ = 10"}


def main():
    setup()
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    bins = np.logspace(np.log10(400), np.log10(50000), 60)
    vals, srcs = {}, {}
    for k, ds in DS.items():
        path = f"{SIM}/{ds}/v1/parquet/reco/tracker_hits"
        h = read(path, ["event_id", "volume_id"])
        n = np.diff(h["volume_id"][0])
        med = np.median(n)
        ax.hist(n, bins=bins, histtype="step", lw=2, color=C[k], label=f"{LABEL[k]}: median {med:,.0f}")
        vals[k] = {"events": int(len(n)), "mean": float(n.mean()), "median": float(med),
                   "q10_q90": np.percentile(n, [10, 90]).tolist(), "min": int(n.min()), "max": int(n.max())}
        srcs[k] = path
        print(k, vals[k])
    vals["ratio_median_mu10_over_mu0"] = vals["mu10"]["median"] / vals["mu0"]["median"]
    ax.set_xscale("log")
    ax.set_xlabel("Tracker hits per event")
    ax.set_ylabel("Events per bin")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.25)
    ax.legend(loc="upper left")
    ax.text(0.98, 0.97, "m = 400 GeV, Λ = 1000 eV\n4000 events each", transform=ax.transAxes, ha="right",
            va="top", fontsize=10, color=C["ink2"])
    save(fig, "hits_per_event")
    record("hits_per_event", vals, srcs)


if __name__ == "__main__":
    main()
