"""MaskFormer grid (Stage F, ledger row mf-F-grid): the long pair + direction MaskFormer (mf-E-pairdir-long, 100k
checkpoint), no retraining, on the mu = 0 grid test sets (runs 0-3, 2000 events), object threshold 0.1, STRICT
matching (> 50% / > 90%), ACTS on the pair-truth copy. Reads tracking/grid/stageF_results.csv and stageF_vs_gnn.csv.

Writes:
  figs/mf_grid.png          coupled efficiency > 50% and > 90% maps
  figs/mf_grid_fake.png     fake rate (strict, normal truth) and technical_sm maps
  figs/mf_grid_vs_gnn.png   MaskFormer minus GNN (pair GNN e2_pair_dir_s0, components 0.5; gnn_grid_strict.csv),
                            coupled > 50% and > 90%
and prints the by-Lambda medians and win counts used on the slides.
m400_L1000 (development point) is in the grid run but off the (m, Lambda) grid: left out of the maps, counted
separately. The training point m200_L774 is not a grid test set: grey cell, orange box (every map filled from its own test set: MaskFormer strict coupled 0.907 / 0.553, normal-truth fake 0.071,
technical_sm 0.944; pair GNN e2 s0 components strict coupled 0.941 / 0.692 for the differences).
"""
import csv
import os
import statistics as st
import sys
from collections import defaultdict

import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qstyle import C, plt, record, save, setup  # noqa: E402

GRID = "/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/grid"
RES = list(csv.DictReader(open(f"{GRID}/stageF_results.csv")))
VS = list(csv.DictReader(open(f"{GRID}/stageF_vs_gnn.csv")))
M = list(range(50, 301, 25))
L = [100, 167, 278, 464, 774, 1292, 2154, 3594, 5995, 10000]
PURP = LinearSegmentedColormap.from_list("q", ["#f7f6f2", "#c9b8ea", "#5b3f9e", "#2e1d5c"])
DIV = LinearSegmentedColormap.from_list("d", ["#0f7c78", "#f7f6f2", "#5b3f9e"])
GREY = LinearSegmentedColormap.from_list("g", [C["grid"], C["grid"]])


def grid(rows, key, mk="m_GeV", lk="Lambda_eV"):
    g = np.full((len(L), len(M)), np.nan)
    for r in rows:
        m, lam = int(r[mk]), int(r[lk])
        if m in M and lam in L and r.get(key):
            g[L.index(lam), M.index(m)] = float(r[key])
    return g


def heat(ax, g, title, cmap, vmin, vmax, fmt, dark):
    ax.imshow(np.zeros_like(g), origin="lower", cmap=GREY, aspect="auto")
    ax.imshow(np.ma.masked_invalid(g), origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    for i in range(len(L)):
        for j in range(len(M)):
            v = g[i, j]
            ax.text(j, i, "·" if np.isnan(v) else fmt.format(v), ha="center", va="center", fontsize=12.5,
                    color=C["muted"] if np.isnan(v) else ("white" if dark(v) else C["ink"]))
    ax.add_patch(Rectangle((M.index(200) - 0.5, L.index(774) - 0.5), 1, 1, fill=False, ec=C["B"], lw=3))
    ax.set_xticks(range(len(M)), M)
    ax.set_yticks(range(len(L)), L)
    ax.set_xlabel("quirk mass m [GeV]")
    ax.set_title(title, fontsize=16)
    ax.tick_params(length=0)
    for s_ in ax.spines.values():
        s_.set_visible(False)


def two(panels, name):
    setup(14)
    fig, axs = plt.subplots(1, 2, figsize=(15.6, 6.6))
    for ax, p in zip(axs, panels):
        heat(ax, *p)
    axs[0].set_ylabel("Λ [eV]")
    fig.subplots_adjust(left=0.06, right=0.995, top=0.93, bottom=0.10, wspace=0.10)
    save(fig, name)


eff = lambda v: v > 0.55  # noqa: E731
# training point m200_L774 is not a grid set: fill its cell (every map) from its own test set (events 17000-19999,
# mf-E-pairdir-long ot 0.1, strict, ledger strict table: coupled 0.907 / 0.553), boxed as on the GNN maps.
def with_train(g, v):
    g = g.copy()
    g[L.index(774), M.index(200)] = v
    return g


two([(with_train(grid(RES, "coupled_vl_ot0.1"), 0.907), "MaskFormer coupled efficiency, loose (> 50%)", PURP, 0, 1, "{:.2f}", eff),
     (with_train(grid(RES, "coupled_tight_ot0.1"), 0.553), "MaskFormer coupled efficiency, tight (> 90%)", PURP, 0, 1, "{:.2f}", eff)],
    "mf_grid")
two([(with_train(grid(RES, "fake_ot0.1"), 0.071), "MaskFormer fake rate (loose matching)", PURP, 0, 0.15, "{:.3f}", lambda v: v > 0.09),
     (with_train(grid(RES, "tech_sm_ot0.1"), 0.944), "MaskFormer ordinary-particle efficiency (loose)", PURP, 0.85, 1.0, "{:.3f}", lambda v: v > 0.93)],
    "mf_grid_fake")
two([(with_train(grid(VS, "d_coupled_vl"), 0.907 - 0.941), "MaskFormer − GNN, coupled, loose (> 50%)", DIV, -0.3, 0.3, "{:+.2f}", lambda v: abs(v) > 0.18),
     (with_train(grid(VS, "d_coupled_tight"), 0.553 - 0.692), "MaskFormer − GNN, coupled, tight (> 90%)", DIV, -0.3, 0.3, "{:+.2f}", lambda v: abs(v) > 0.18)],
    "mf_grid_vs_gnn")

# numbers for the slides
on = [r for r in VS if int(r["m_GeV"]) in M]
out = {"sets_total": len(VS), "sets_on_grid": len(on),
       "wins_vl_all": sum(float(r["d_coupled_vl"]) > 0 for r in VS),
       "wins_tight_all": sum(float(r["d_coupled_tight"]) > 0 for r in VS),
       "wins_vl_grid": sum(float(r["d_coupled_vl"]) > 0 for r in on),
       "wins_tight_grid": sum(float(r["d_coupled_tight"]) > 0 for r in on)}
for k in ("mf_fake", "gnn_fake", "mf_tech_sm", "gnn_tech_sm", "mf_coupled_vl", "mf_coupled_tight"):
    out[f"median_{k}"] = round(st.median(float(r[k]) for r in VS), 3)
by = defaultdict(list)
for r in on:
    by[int(r["Lambda_eV"])].append(r)
out["by_lambda"] = {lam: {k: round(st.median(float(r[k]) for r in rs), 3)
                          for k in ("mf_coupled_vl", "gnn_coupled_vl", "d_coupled_vl", "mf_coupled_tight",
                                    "gnn_coupled_tight", "d_coupled_tight")}
                    | {"wins_vl": sum(float(r["d_coupled_vl"]) > 0 for r in rs),
                       "wins_tight": sum(float(r["d_coupled_tight"]) > 0 for r in rs), "n": len(rs)}
                    for lam, rs in sorted(by.items())}
record("mf_grid", out)
for k, v in out.items():
    print(k, v)


# ---- ratio maps: MaskFormer / pair GNN coupled efficiency, log2 colour centred at 1 (Daniel, 2026-10-27).
# Cells where either model finds fewer than 20 pairs (efficiency x selected pairs) are greyed: the ratio is set
# by counts there. Training point (not a grid set) from its own test set, strict: MaskFormer 0.907 / 0.553,
# pair GNN e2_pair_dir_s0 components 0.941 / 0.692 (EXPERIMENTS.md strict section, seed 0).
NP = {(int(r["m_GeV"]), int(r["Lambda_eV"])): float(r["n_pairs_ot0.1"]) for r in RES}
TRAIN = {"vl": (0.907, 0.941), "tight": (0.553, 0.692)}


def ratio_panel(ax, key, title):
    g = np.full((len(L), len(M)), np.nan)
    low = np.zeros((len(L), len(M)), bool)
    for r in VS:
        m, lam = int(r["m_GeV"]), int(r["Lambda_eV"])
        if m not in M or lam not in L:
            continue
        mf, gn = float(r[f"mf_coupled_{key}"]), float(r[f"gnn_coupled_{key}"])
        n = NP.get((m, lam), np.nan)
        i, j = L.index(lam), M.index(m)
        g[i, j] = mf / gn if gn > 0 else np.nan
        low[i, j] = (mf * n < 20) or (gn * n < 20)
    i, j = L.index(774), M.index(200)
    g[i, j] = TRAIN[key][0] / TRAIN[key][1]
    lg = np.log2(np.clip(g, 1 / 4, 4))
    ax.imshow(np.zeros_like(g), origin="lower", cmap=GREY, aspect="auto")
    ax.imshow(np.ma.masked_where(low | np.isnan(g), lg), origin="lower", cmap=DIV, vmin=-2, vmax=2, aspect="auto")
    for i in range(len(L)):
        for j in range(len(M)):
            v = g[i, j]
            txt = "·" if np.isnan(v) else f"{v:#.2g}"
            col = C["muted"] if (np.isnan(v) or low[i, j]) else ("white" if abs(np.log2(v)) > 1.2 else C["ink"])
            ax.text(j, i, txt, ha="center", va="center", fontsize=12.5, color=col)
    ax.add_patch(Rectangle((M.index(200) - 0.5, L.index(774) - 0.5), 1, 1, fill=False, ec=C["B"], lw=3))
    ax.set_xticks(range(len(M)), M)
    ax.set_yticks(range(len(L)), L)
    ax.set_xlabel("quirk mass m [GeV]")
    ax.set_title(title, fontsize=16)
    ax.tick_params(length=0)
    for s_ in ax.spines.values():
        s_.set_visible(False)
    return g, low


setup(14)
fig, axs = plt.subplots(1, 2, figsize=(15.6, 6.6))
gv, lv = ratio_panel(axs[0], "vl", "MaskFormer / pair GNN, coupled eff., loose (> 50%)")
gt, lt = ratio_panel(axs[1], "tight", "MaskFormer / pair GNN, coupled eff., tight (> 90%)")
axs[0].set_ylabel("Λ [eV]")
fig.subplots_adjust(left=0.06, right=0.995, top=0.93, bottom=0.10, wspace=0.10)
save(fig, "mf_grid_ratio")
rat = {}
for lam, rs in sorted(by.items()):
    mv = st.median(float(r["mf_coupled_vl"]) for r in rs); gvv = st.median(float(r["gnn_coupled_vl"]) for r in rs)
    mt = st.median(float(r["mf_coupled_tight"]) for r in rs); gtt = st.median(float(r["gnn_coupled_tight"]) for r in rs)
    rat[lam] = {"ratio_vl": round(mv / gvv, 3), "ratio_tight": round(mt / gtt, 3) if gtt > 0 else None,
                "low_vl": int(sum(lv[L.index(lam)])), "low_tight": int(sum(lt[L.index(lam)]))}
on_v = [x for x in gv[~lv].ravel() if np.isfinite(x)]
on_t = [x for x in gt[~lt].ravel() if np.isfinite(x)]
out2 = {"by_lambda_ratio": rat, "cells_low_vl": int(lv.sum()), "cells_low_tight": int(lt.sum()),
        "ratio_vl_min": round(min(on_v), 3), "ratio_vl_max": round(max(on_v), 3),
        "ratio_tight_min": round(min(on_t), 3), "ratio_tight_max": round(max(on_t), 3),
        "n_ratio_gt1_vl": int(sum(x > 1 for x in on_v)), "n_ratio_gt1_tight": int(sum(x > 1 for x in on_t))}
record("mf_grid_ratio", out2)
for k, v in out2.items():
    print(k, v)
