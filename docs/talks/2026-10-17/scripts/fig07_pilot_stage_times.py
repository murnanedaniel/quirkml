"""Fig 7: wall time per pipeline stage for the m = 400 GeV, Lambda = 1000 eV pilots (8 runs x 500 events),
no pile-up vs mu = 10. Numbers (minutes) as supplied with the talk brief (pilot logs pilot_gen.out / pilot_pu10.out)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qcommon import C, record, save, setup, plt, np  # noqa: E402

STAGES = ["generation", "stage 1 (merge + smear)", "simulation", "digitization", "package"]
KEYS = ["generation", "stage1", "simulation", "digitization", "package"]
MIN = {"mu0": [1.7, 0.33, 7.3, 1.0, 0.57], "mu10": [1.6, 0.9, 14.2, 2.0, 1.2]}
LABEL = {"mu0": "no pile-up", "mu10": "pile-up μ = 10"}


def main():
    setup()
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    y = np.arange(len(STAGES))[::-1]
    h = 0.36
    for j, k in enumerate(("mu0", "mu10")):
        yy = y + (h / 2 + 0.02) * (1 if k == "mu0" else -1)
        ax.barh(yy, MIN[k], height=h, color=C[k], label=f"{LABEL[k]} (total {sum(MIN[k]):.1f} min)")
        for v, yv in zip(MIN[k], yy):
            ax.text(v + 0.15, yv, f"{v:g}", va="center", ha="left", fontsize=10, color=C["ink2"])
    ax.set_yticks(y)
    ax.set_yticklabels(STAGES)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Wall time [min] (8 runs × 500 events in parallel)")
    ax.set_xlim(0, 16.5)
    ax.legend(loc="lower right")
    ax.text(0.98, 0.3, "m = 400 GeV, Λ = 1000 eV", transform=ax.transAxes, ha="right", fontsize=10, color=C["ink2"])
    save(fig, "pilot_stage_times")
    record("pilot_stage_times", {k: dict(zip(KEYS, v)) | {"total": round(sum(v), 2)} for k, v in MIN.items()},
           {"all": "talk brief (pilot logs /pscratch/sd/d/danieltm/quirk_pilots/pilot_gen.out, pilot_pu10.out)"})


if __name__ == "__main__":
    main()
