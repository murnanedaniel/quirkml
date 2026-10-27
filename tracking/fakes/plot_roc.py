"""Coupled efficiency vs background rejection (1 / fake rate) and a bar chart of rejection at fixed coupled
efficiency, from coupled_roc.py outputs (test 17000-19999, pair-truth copy, strict matching).

Rejection at efficiency e = the best rejection over score cuts that keep coupled efficiency >= e (read off the test
curve, as for any efficiency-vs-rejection curve). Seeds: mean line, band = min..max over seeds; bars: mean +- sample
std over seeds.

    python plot_roc.py ROC_ROOT FIG_DIR [--effs 0.5 0.8 0.9]
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

BG, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = [  # fixed categorical order (dataviz reference palette, slots 1-7)
    ("Pair GNN, components", [f"pair_gnn_s{s}_cc" for s in range(3)], "#2a78d6"),
    ("Pair GNN on graph v3, components", ["pair_gnn_v3_p3_cc07"], "#eb6834"),
    ("Pair GNN, walk + stitch", ["pair_gnn_s0_walk"], "#1baf7a"),
    ("MaskFormer pair", [f"mf_pair_s{s}_ot0.1" for s in range(3)], "#eda100"),
    ("MaskFormer pair + direction", [f"mf_pairdir_s{s}_ot0.1" for s in range(3)], "#e87ba4"),
    ("MaskFormer pair + direction, long", ["mf_long_ot0.1"], "#008300"),
    ("Per-quirk GNN v3 (reference)", ["gnn_v3_s0_walk"], "#6250d6"),
]


def rejection_at(c, eff_col, fake_col, grid):
    e, f = c[eff_col].to_numpy(), c[fake_col].to_numpy()
    out = np.full(len(grid), np.nan)
    for k, g in enumerate(grid):
        ok = (e >= g) & (f > 0)
        if ok.any():
            out[k] = 1 / f[ok].min()
    return out


def point_at(c, e):
    """Row of the best-rejection cut that keeps loose coupled efficiency >= e (None if not reached)."""
    ok = c.filter((pl.col("coupled_eff") >= e) & (pl.col("fake") > 0))
    return None if ok.height == 0 else ok.sort("fake").row(0, named=True)


SHORT = {"Pair GNN, components": "Pair GNN\ncomponents", "Pair GNN on graph v3, components": "Pair GNN, graph v3\ncomponents",
         "Pair GNN, walk + stitch": "Pair GNN\nwalk + stitch", "MaskFormer pair": "MaskFormer\npair",
         "MaskFormer pair + direction": "MaskFormer\npair + direction",
         "MaskFormer pair + direction, long": "MaskFormer pair +\ndirection, long", "Per-quirk GNN v3 (reference)": "Per-quirk GNN v3\n(reference)"}


def load(root):
    data = {}
    for label, models, color in SERIES:
        cs = [pl.read_csv(root / m / "roc_test.csv") for m in models if (root / m / "roc_test.csv").exists()]
        data[label] = (cs, color, len(models))
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("figdir", type=Path)
    ap.add_argument("--effs", type=float, nargs="+", default=[0.5, 0.8, 0.9])
    a = ap.parse_args()
    data = load(a.root)
    fig, axs = plt.subplots(1, 2, figsize=(16, 6.4), facecolor=BG)
    table = {}
    for ax, (eff_col, fake_col, title) in zip(axs, [("coupled_eff", "fake", "Loose matching (> 50% of hits)"),
                                                    ("coupled_eff_tight", "fake_tight", "Tight matching (> 90% of hits)")]):
        for label, (cs, color, n) in data.items():
            if not cs:
                continue
            hi = min(c[eff_col][0] for c in cs)
            grid = np.linspace(0.02, hi, 300)
            R = np.array([rejection_at(c, eff_col, fake_col, grid) for c in cs])
            lab = f"{label} ({len(cs)} seeds)" if len(cs) > 1 else label
            ax.plot(grid, np.nanmean(R, 0), color=color, lw=2, label=lab)
            if len(cs) > 1:
                ax.fill_between(grid, np.nanmin(R, 0), np.nanmax(R, 0), color=color, alpha=0.18, lw=0)
            ax.plot([np.mean([c[eff_col][0] for c in cs])], [np.mean([1 / c[fake_col][0] for c in cs])], "o", ms=8,
                    color=color, mec=BG, mew=2)
        ax.set_yscale("log")
        ax.set_xlabel("coupled efficiency (both quirks found as one track)", color=MUTED, fontsize=11)
        ax.set_ylabel("background rejection = 1 / fake rate", color=MUTED, fontsize=11)
        ax.set_title(title, color=INK, fontsize=12)
        ax.set_xlim(0, 1)
        ax.grid(color=GRID, lw=0.8, which="major")
        ax.tick_params(colors=MUTED)
        for s in ax.spines.values():
            s.set_visible(False)
    axs[0].legend(frameon=False, fontsize=9.5, loc="upper right")
    fig.suptitle("Coupled quirk efficiency vs background rejection, swept with a per-track quality score "
                 "(test events, dots: no cut)", color=INK, fontsize=12.5)
    fig.tight_layout()
    a.figdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.figdir / "eff_vs_rejection_coupled.png", dpi=160, facecolor=BG)
    plt.close(fig)
    # bar chart at fixed loose coupled efficiency
    rows = []
    for label, (cs, color, n) in data.items():
        for e in a.effs:
            vals = [rejection_at(c, "coupled_eff", "fake", [e])[0] for c in cs]
            reached = [v for v in vals if np.isfinite(v)]
            rows.append(dict(model=label, eff=e, seeds=len(cs), reached=len(reached),
                             mean=float(np.mean(reached)) if len(reached) == len(cs) else None,
                             std=float(np.std(reached, ddof=1)) if len(reached) == len(cs) and len(cs) > 1 else None,
                             values=vals, max_eff=[float(c["coupled_eff"][0]) for c in cs],
                             no_cut_rejection=[float(1 / c["fake"][0]) for c in cs],
                             points=[point_at(c, e) for c in cs]))
    table["bars"] = rows
    fig, ax = plt.subplots(figsize=(14, 6.2), facecolor=BG)
    labels = [s[0] for s in SERIES if data[s[0]][0]]
    width = 0.8 / len(a.effs)
    shades = ["#86b6ef", "#2a78d6", "#104281"][: len(a.effs)] if len(a.effs) <= 3 else None
    for k, e in enumerate(a.effs):
        xs, ys, es = [], [], []
        for i, lab in enumerate(labels):
            r = next(x for x in rows if x["model"] == lab and x["eff"] == e)
            x = i + (k - (len(a.effs) - 1) / 2) * width
            if r["mean"] is None:
                ax.text(x, 1.05, "not\nreached", ha="center", va="bottom", fontsize=7.5, color=MUTED)
                continue
            xs.append(x), ys.append(r["mean"]), es.append(r["std"] if r["std"] else np.nan)
            ax.text(x, r["mean"] * 1.08 + (r["std"] or 0), f"{r['mean']:.0f}", ha="center", va="bottom", fontsize=8,
                    color=INK)
        ax.bar(xs, ys, width * 0.92, yerr=es, color=shades[k] if shades else None, ecolor=INK, capsize=3,
               label=f"coupled efficiency {e:.2f}")
    ax.set_yscale("log")
    ax.set_xticks(range(len(labels)), [SHORT.get(l, l) for l in labels], fontsize=9.5, color=MUTED)
    ax.set_ylabel("background rejection = 1 / fake rate", color=MUTED, fontsize=11)
    ax.set_ylim(1, None)
    ax.grid(axis="y", color=GRID, lw=0.8, which="major")
    ax.tick_params(colors=MUTED)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.legend(frameon=False, fontsize=10, loc="upper right")
    ax.set_title("Background rejection at fixed coupled efficiency (loose matching, > 50%; error bars: seed spread)",
                 color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(a.figdir / "rejection_bar_coupled.png", dpi=160, facecolor=BG)
    plt.close(fig)
    (a.root / "rejection_table.json").write_text(json.dumps(table, indent=1, default=float) + "\n")
    for r in rows:
        pts = [p for p in r["points"] if p]
        print(f"{r['model']:36s} eff {r['eff']:.2f} reached {r['reached']}/{r['seeds']} rejection "
              f"{'-' if r['mean'] is None else f'{r.get(chr(109)+chr(101)+chr(97)+chr(110)):.0f}'}"
              f"{'' if not r['std'] else f' +- {r[chr(115)+chr(116)+chr(100)]:.0f}'} | fakes kept "
              f"{[int(p['n_fake']) for p in pts]} tech_sm {[round(p['technical_sm_eff'], 3) for p in pts]} "
              f"tracks/ev {[round(p['tracks_per_event'], 1) for p in pts]} | max eff {[round(x, 3) for x in r['max_eff']]} "
              f"no-cut rej {[round(x, 1) for x in r['no_cut_rejection']]}")


if __name__ == "__main__":
    main()
