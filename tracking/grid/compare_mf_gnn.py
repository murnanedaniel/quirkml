"""Stage F: MaskFormer (long pair+direction, ot 0.1) vs GNN grid (gnn_grid_strict.csv), strict matching, per set.
Writes stageF_vs_gnn.csv and stageF_vs_gnn_coupled.png (difference map MF - GNN of coupled efficiency, > 50%)."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

H = Path(__file__).resolve().parent
mf = pd.read_csv(H / "stageF_results.csv")
g = pd.read_csv(H / "gnn_grid_strict.csv").rename(columns={"set": "dataset"})
d = mf.merge(g, on="dataset", how="outer", indicator=True)
out = pd.DataFrame({
    "dataset": d.dataset, "m_GeV": d.m_GeV, "Lambda_eV": d.Lambda_eV,
    "mf_coupled_vl": d["coupled_vl_ot0.1"], "gnn_coupled_vl": d.coupled_eff_strict,
    "mf_coupled_tight": d["coupled_tight_ot0.1"], "gnn_coupled_tight": d.coupled_eff_tight_strict,
    "mf_per_quirk": d["per_quirk_ot0.1"], "gnn_per_quirk": d.quirk_eff_strict,
    "mf_tech_sm": d["tech_sm_ot0.1"], "gnn_tech_sm": d.technical_sm,
    "mf_fake": d["fake_ot0.1"], "gnn_fake": d.fake, "in": d._merge})
out["d_coupled_vl"] = out.mf_coupled_vl - out.gnn_coupled_vl
out["d_coupled_tight"] = out.mf_coupled_tight - out.gnn_coupled_tight
out.to_csv(H / "stageF_vs_gnn.csv", index=False)
print(out["in"].value_counts().to_dict())
b = out[out["in"] == "both"]
for c in ("coupled_vl", "coupled_tight", "per_quirk", "tech_sm", "fake"):
    print(f"{c}: MF median {b['mf_' + c].median():.3f} [{b['mf_' + c].min():.3f}, {b['mf_' + c].max():.3f}]  "
          f"GNN median {b['gnn_' + c].median():.3f} [{b['gnn_' + c].min():.3f}, {b['gnn_' + c].max():.3f}]")
print("MF > GNN coupled vl in", int((b.d_coupled_vl > 0).sum()), "of", len(b), "sets; tight in", int((b.d_coupled_tight > 0).sum()))
masses, lams = sorted(b.m_GeV.unique()), sorted(b.Lambda_eV.unique())
grid = np.full((len(lams), len(masses)), np.nan)
for _, r in b.iterrows():
    grid[lams.index(r.Lambda_eV), masses.index(r.m_GeV)] = r.d_coupled_vl
fig, ax = plt.subplots(figsize=(1.0 + 0.62 * len(masses), 1.2 + 0.42 * len(lams)))
ax.set_facecolor("#d9d9d9")
v = np.nanmax(np.abs(grid))
im = ax.imshow(np.ma.masked_invalid(grid), origin="lower", cmap="RdBu", vmin=-v, vmax=v, aspect="auto")
for i in range(len(lams)):
    for j in range(len(masses)):
        if np.isfinite(grid[i, j]):
            ax.text(j, i, f"{grid[i, j]:+.2f}", ha="center", va="center", fontsize=7)
ax.set_xticks(range(len(masses)), [str(m) for m in masses]); ax.set_yticks(range(len(lams)), [str(x) for x in lams])
ax.set_xlabel("quirk mass m (GeV)"); ax.set_ylabel("Lambda (eV)")
ax.set_title("coupled efficiency (> 50%): MaskFormer minus GNN, object threshold 0.1", fontsize=9)
fig.colorbar(im, ax=ax, fraction=0.04); fig.tight_layout(); fig.savefig(H / "stageF_vs_gnn_coupled.png", dpi=150)
