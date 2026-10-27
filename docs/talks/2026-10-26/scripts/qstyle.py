"""Shared style for the 2026-10-26 quirk tracking talk figures (Claude Slides deck, 1920x1080 canvas).

Figures are PNGs sized for the slide: width in inches x DPI = pixels on the slide, fonts >= 24 px at that size
(24 px at 120 dpi = 14.4 pt). Run with ~/.conda/envs/collider-env/bin/python.
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(os.path.dirname(HERE), "figs")
DPI = 120

# one colour per entity, the same in every figure and on the slides
C = {
    "A": "#1f5fbf",       # quirk A (dark blue)
    "B": "#f39a3c",       # quirk B (light orange)
    "other": "#a9a7a0",   # other hits / tracks
    "fake": "#7b3f9e",    # fakes
    "ink": "#1b1f24",
    "ink2": "#4a5058",
    "muted": "#8a8f96",
    "grid": "#e4e2dc",
    "axis": "#b9b7af",
    "bg": "#ffffff",
    "gnn": "#0f7c78",     # GNN family (teal)
    "gnn2": "#63bdb6",
    "mf": "#5b3f9e",      # MaskFormer family (purple)
    "mf2": "#b6a2dd",
    "acts": "#4a5058",    # ACTS baselines
    "ceil": "#1b1f24",
}
TRACK_COLS = ["#0f7c78", "#c2185b", "#6d4c41", "#5b3f9e", "#7cb342", "#00838f"]


def setup(fs=15):
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": fs,
        "axes.labelsize": fs + 1,
        "axes.titlesize": fs + 1,
        "xtick.labelsize": fs - 1,
        "ytick.labelsize": fs - 1,
        "legend.fontsize": fs - 1,
        "legend.frameon": False,
        "axes.edgecolor": C["axis"],
        "axes.labelcolor": C["ink"],
        "axes.linewidth": 0.9,
        "xtick.color": C["ink2"],
        "ytick.color": C["ink2"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": C["grid"],
        "grid.linewidth": 0.7,
        "lines.linewidth": 2.2,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.dpi": DPI,
        "pdf.fonttype": 42,
        "mathtext.fontset": "dejavusans",
    })


def save(fig, name):
    os.makedirs(FIGS, exist_ok=True)
    p = os.path.join(FIGS, f"{name}.png")
    fig.savefig(p, dpi=DPI)
    fig.savefig(os.path.join(FIGS, f"{name}.pdf"))
    plt.close(fig)
    print("wrote", p)
    return p


def record(name, values):
    """Merge {name: values} into figs/numbers.json (numbers shown in each figure, for tracing)."""
    path = os.path.join(FIGS, "numbers.json")
    d = json.load(open(path)) if os.path.exists(path) else {}
    d[name] = values
    with open(path, "w") as f:
        json.dump(d, f, indent=1, default=float)
