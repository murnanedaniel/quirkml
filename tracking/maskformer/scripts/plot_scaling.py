"""MaskFormer data scaling (pair target): test coupled efficiency (strict, pair truth, ACTS) vs training events.
Writes tracking/maskformer/scaling_coupled.png. Numbers from eval/pairtruth_<run>_chosen_test_ot<t>/summary.json."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

E = Path("/pscratch/sd/d/danieltm/quirk_tracking/maskformer/eval")
OUT = Path("/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/maskformer/scaling_coupled.png")


def eff(run, t, wp):
    q = json.loads((E / f"pairtruth_{run}_chosen_test_ot{t}" / "summary.json").read_text())["quirk_track_efficiency"][wp]
    return q["eff"], q["err"]


fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=False)
for ax, wp, title in ((axes[0], "very_loose_strict", "coupled efficiency, > 50%"), (axes[1], "tight_strict", "coupled efficiency, > 90%")):
    for t, color, marker in (("0.1", "#2a6fb0", "o"), ("0.2", "#d9822b", "s")):
        n = [16000, 50000, 150000]
        v = [eff(r, t, wp) for r in ("s2_pair_n16k", "s2_pair_n50k", "s2_pair_n150k")]
        ax.errorbar(n, [x[0] for x in v], yerr=[x[1] for x in v], color=color, marker=marker, lw=2, ms=7, capsize=3,
                    label=f"plateau rule, object threshold {t}")
        seeds = [eff(r, t, wp)[0] for r in ("t2d_pair_lr5e-5", "t2e_pair_lr5e-5_s1", "t2e_pair_lr5e-5_s2")]
        ax.errorbar([16000 * 1.08], [np.mean(seeds)], yerr=[np.std(seeds, ddof=1)], color=color, marker=marker,
                    mfc="white", ms=7, capsize=3, ls="none", label=f"16k, 30k steps incl. cooldown (3 seeds), {t}")
    ax.set_xscale("log")
    ax.set_xticks([16000, 50000, 150000], ["16k", "50k", "150k"])
    ax.minorticks_off()
    ax.set_xlabel("training events")
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)
axes[0].set_ylabel("test efficiency (pair truth, ACTS)")
axes[0].legend(fontsize=7, loc="lower right")
fig.tight_layout()
fig.savefig(OUT, dpi=150)
print(OUT)
