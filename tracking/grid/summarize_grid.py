"""Stage F table and maps: the long pair+direction MaskFormer on every grid test set (runs 0-3, events 0-1999).

Reads quirk_tracking_data/grid_eval/mf_pairdir_long/<ds>/{pair,truth}_ot<t>/summary.json (ACTS) and python_ot<t>.json.
Writes tracking/grid/stageF_results.csv, stageF_results.md, and maps stageF_<metric>_ot<t>.png over (m, Lambda).
"""
import csv
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
EV = Path("/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirk_tracking_data/grid_eval/mf_pairdir_long")
SETS = (HERE / "grid_sets_mu0.txt").read_text().split()
THRS = ("0.1", "0.2")


def load(p):
    return json.loads(p.read_text()) if p.exists() else None


rows = []
for ds in SETS:
    m, lam = (int(x) for x in re.match(r"m(\d+)_L(\d+)_mu0", ds).groups())
    row = {"dataset": ds, "m_GeV": m, "Lambda_eV": lam}
    for t in THRS:
        pair, truth, py = (load(EV / ds / f"pair_ot{t}" / "summary.json"), load(EV / ds / f"truth_ot{t}" / "summary.json"),
                           load(EV / ds / f"python_ot{t}.json"))
        row[f"status_ot{t}"] = "ok" if (pair and truth) else "missing"
        # Headline: strict matching (> 50% / > 90%, ATLAS convention); "_ge" columns: ACTS >= 50% / >= 90%.
        if pair:
            q = pair["quirk_track_efficiency"]
            row[f"coupled_vl_ot{t}"], row[f"coupled_tight_ot{t}"] = q["very_loose_strict"]["eff"], q["tight_strict"]["eff"]
            row[f"coupled_vl_ge_ot{t}"], row[f"coupled_tight_ge_ot{t}"] = q["very_loose"]["eff"], q["tight"]["eff"]
            row[f"n_pairs_ot{t}"] = int(q["very_loose"]["total"])
        if truth:
            for wp, sfx in (("very_loose_strict", ""), ("very_loose", "_ge")):
                row[f"per_quirk{sfx}_ot{t}"] = truth["quirk_track_efficiency"][wp]["eff"]
                row[f"tech_sm{sfx}_ot{t}"] = truth["technical_sm_efficiency"][wp]["eff"]
                row[f"fake{sfx}_ot{t}"] = truth["fake_rate"][wp]
        if py:
            row[f"coupled_python_ot{t}"] = py["coupled_eff"]
    rows.append(row)

keys = ["dataset", "m_GeV", "Lambda_eV"] + [f"{k}_ot{t}" for t in THRS for k in
        ("coupled_vl", "coupled_tight", "per_quirk", "tech_sm", "fake", "coupled_vl_ge", "coupled_tight_ge",
         "per_quirk_ge", "tech_sm_ge", "fake_ge", "coupled_python", "n_pairs", "status")]
with open(HERE / "stageF_results.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=keys)
    w.writeheader()
    for r in sorted(rows, key=lambda r: (r["m_GeV"], r["Lambda_eV"])):
        w.writerow({k: r.get(k, "") for k in keys})


def f(v):
    return "" if v is None or v == "" else f"{v:.3f}"


with open(HERE / "stageF_results.md", "w") as fh:
    fh.write("Strict matching (> 50% very loose, > 90% tight; ATLAS convention) at object thresholds 0.1 / 0.2; "
             "in brackets the ACTS >= value at 0.1. Coupled = pair truth; per quirk, tech_sm, fake = normal truth.\n\n")
    fh.write("| dataset | m (GeV) | Lambda (eV) | coupled vl 0.1 / 0.2 | coupled tight 0.1 / 0.2 | per quirk 0.1 / 0.2 | "
             "tech_sm 0.1 / 0.2 | fake 0.1 / 0.2 | pairs |\n|---|---|---|---|---|---|---|---|---|\n")
    for r in sorted(rows, key=lambda r: (r["m_GeV"], r["Lambda_eV"])):
        g = lambda k: f"{f(r.get(k + '_ot0.1'))} / {f(r.get(k + '_ot0.2'))} ({f(r.get(k + '_ge_ot0.1'))})"  # noqa: E731
        fh.write(f"| {r['dataset']} | {r['m_GeV']} | {r['Lambda_eV']} | {g('coupled_vl')} | {g('coupled_tight')} | "
                 f"{g('per_quirk')} | {g('tech_sm')} | {g('fake')} | {r.get('n_pairs_ot0.1', '')} |\n")

masses = sorted({r["m_GeV"] for r in rows})
lams = sorted({r["Lambda_eV"] for r in rows})
for metric, label, cmap, vmax in (("coupled_vl", "coupled quirk efficiency (> 50%)", "Blues", 1.0),
                                  ("coupled_tight", "coupled quirk efficiency (> 90%)", "Blues", 1.0),
                                  ("fake", "fake rate, all tracks (> 50%)", "Oranges", None)):
    for t in THRS:
        grid = np.full((len(lams), len(masses)), np.nan)
        for r in rows:
            v = r.get(f"{metric}_ot{t}")
            if v is not None:
                grid[lams.index(r["Lambda_eV"]), masses.index(r["m_GeV"])] = v
        fig, ax = plt.subplots(figsize=(1.0 + 0.62 * len(masses), 1.2 + 0.42 * len(lams)))
        ax.set_facecolor("#d9d9d9")
        vm = vmax if vmax is not None else max(np.nanmax(grid), 1e-3)
        im = ax.imshow(np.ma.masked_invalid(grid), origin="lower", cmap=cmap, vmin=0, vmax=vm, aspect="auto")
        for i in range(len(lams)):
            for j in range(len(masses)):
                if np.isfinite(grid[i, j]):
                    ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center", fontsize=7,
                            color="white" if grid[i, j] > 0.6 * vm else "#222222")
        ax.set_xticks(range(len(masses)), [str(m) for m in masses])
        ax.set_yticks(range(len(lams)), [str(lam) for lam in lams])
        ax.set_xlabel("quirk mass m (GeV)")
        ax.set_ylabel("Lambda (eV)")
        ax.set_title(f"{label}, object threshold {t}", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.04)
        fig.tight_layout()
        fig.savefig(HERE / f"stageF_{metric}_ot{t}.png", dpi=150)
        plt.close(fig)
missing = [r["dataset"] for r in rows if r["status_ot0.1"] != "ok" or r["status_ot0.2"] != "ok"]
print(f"{len(rows)} sets, {len(rows) - len(missing)} complete; missing: {missing}")
