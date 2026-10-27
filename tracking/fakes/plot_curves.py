"""Efficiency vs fake-rate plots from quality.py outputs (B2): for one model, the quality-score cuts and the
hit-count-only cut on test, one panel per efficiency (per-quirk, coupled, technical_sm, quirk event);
curves drawn while >= 50% of tracks are kept; the no-cut point (black ring) and the operating points chosen on val at fake 0.05 / 0.02 (squares) are marked.

    python plot_curves.py B2_ROOT MODEL [MODEL ...]     -> B2_ROOT/<model>/eff_vs_fake.png
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import polars as pl  # noqa: E402

BG, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
COL = {"quality": "#2a78d6", "n_hits": "#eb6834", "quality_qw": "#1baf7a", "quality_or_tag": "#6250d6"}
LAB = {"quality": "quality score cut", "n_hits": "hit-count cut", "quality_qw": "quality, quirk tracks weighted x30",
       "quality_or_tag": "quality OR quirk tag"}
PANELS = [("quirk_eff", "Per-quirk efficiency"), ("coupled_eff", "Coupled efficiency"),
          ("technical_sm_eff", "technical_sm efficiency"), ("quirk_event_eff", "Quirk event efficiency")]


def plot(root: Path, model: str):
    d = root / model
    res = json.loads((d / "result.json").read_text())
    fake_col = "fake_rate" if res["own_truth"] == "normal" else "fake_rate_pair"
    fig, axs = plt.subplots(1, 4, figsize=(19, 4.6), facecolor=BG)
    for ax, (col, title) in zip(axs, PANELS):
        for name in [n for n in ("n_hits", "quality", "quality_qw", "quality_or_tag") if n in res["b2"]]:
            c = pl.read_csv(d / f"curve_{name}_test.csv").filter(pl.col(fake_col).is_not_null())
            # drawn while at least half of the tracks are kept (the tail beyond is cut into everything)
            c = c.filter(pl.col("tracks_per_event") >= 0.5 * c["tracks_per_event"][0])
            ax.plot(c[fake_col], c[col], color=COL[name], lw=2, label=LAB[name],
                    drawstyle="steps-post" if name == "n_hits" else "default")
            for op in res["b2"][name]["op"]:
                if op.get("test"):
                    ax.plot(op["test"][fake_col], op["test"][col], "s", ms=8, color=COL[name], mec=BG, mew=2)
        nc = res["b2"]["quality"]["no_cut_test"]
        ax.plot(nc[fake_col], nc[col], "o", ms=9, mfc="none", mec=INK, mew=1.5, label="no cut")
        ax.set_title(title, color=INK, fontsize=11)
        ax.set_xlabel(f"fake rate ({'pair' if res['own_truth'] == 'pair' else 'normal'} truth, > 50%)", color=MUTED)
        ax.set_xlim(0, max(0.13, nc[fake_col] * 1.1))
        ax.grid(color=GRID, lw=0.8)
        ax.tick_params(colors=MUTED)
        for s in ax.spines.values():
            s.set_visible(False)
    axs[0].legend(frameon=False, fontsize=9, loc="lower right")
    fig.suptitle(f"{model}: test 17000-19999, cuts chosen on val 16500-16999 (squares: fake 0.05 / 0.02)",
                 color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(d / "eff_vs_fake.png", dpi=120, facecolor=BG)
    plt.close(fig)


if __name__ == "__main__":
    for m in sys.argv[2:]:
        plot(Path(sys.argv[1]), m)
