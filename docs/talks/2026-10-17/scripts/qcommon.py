"""Shared style, colours, I/O and numbers-file helpers for the 2026-10-17 quirk talk figures.

Heavy reads: run the figure scripts on a compute node (srun), not on a login node.
Python: ~/.conda/envs/collider-env/bin/python (pyarrow + numpy; awkward is not used because the
installed awkward/pyarrow pair is incompatible).
"""

import glob
import json
import os

import numpy as np
import pyarrow.parquet as pq
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(os.path.dirname(HERE), "figs")
NUMBERS = os.path.join(FIGS, "numbers.json")

SIM = "/global/cfs/cdirs/m4958/data/ColliderML/simulation/quirks"
R1 = "/pscratch/sd/d/danieltm/quirk_pilots/r1"
GNN = "/pscratch/sd/d/danieltm/quirk_tracking/gnn"
QUIRK_PDG = 10000100

# ----------------------------------------------------------------------------- colours (validated, light surface)
# One colour per entity, the same in every figure.
C = {
    "qplus": "#2a78d6",    # quirk with charge +1
    "qminus": "#eb6834",   # quirk with charge -1
    "mu0": "#1baf7a",      # no pile-up
    "mu10": "#6250d6",     # mu = 10 pile-up
    "v1": "#008300",       # GNN chain v1
    "v2": "#e34948",       # GNN chain v2
    "other": "#b9b8b0",    # other hits / other edges
    "layer": "#e6e5df",    # detector layers
    "ink": "#0b0b0b",
    "ink2": "#52514e",
    "muted": "#898781",
    "grid": "#e1e0d9",
    "axis": "#c3c2b7",
}
# Ordinal blue ramp for quirk mass (light -> dark = 50 -> 175 -> 300 GeV)
MASS_C = {50: "#6da7ec", 175: "#2a78d6", 200: "#1c5cab", 300: "#104281"}
QLABEL = {"qplus": "quirk (+)", "qminus": "quirk (−)"}


def setup():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "font.size": 12,
        "axes.labelsize": 13,
        "axes.titlesize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "legend.frameon": False,
        "axes.edgecolor": C["axis"],
        "axes.labelcolor": C["ink"],
        "axes.linewidth": 0.8,
        "xtick.color": C["ink2"],
        "ytick.color": C["ink2"],
        "xtick.direction": "out",
        "ytick.direction": "out",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "grid.color": C["grid"],
        "grid.linewidth": 0.6,
        "lines.linewidth": 2.0,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.dpi": 200,
        "pdf.fonttype": 42,
        "mathtext.fontset": "dejavusans",
    })


def save(fig, name):
    os.makedirs(FIGS, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGS, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print("wrote", os.path.join(FIGS, name + ".{pdf,png}"))


def record(fig_name, values, sources):
    """Merge {fig_name: {values, sources}} into figs/numbers.json (run figure scripts one at a time)."""
    d = {}
    if os.path.exists(NUMBERS):
        with open(NUMBERS) as f:
            d = json.load(f)
    d[fig_name] = {"values": values, "sources": sources}
    with open(NUMBERS, "w") as f:
        json.dump(d, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))


# ----------------------------------------------------------------------------- parquet I/O (no awkward)
def files(path_dir):
    return sorted(glob.glob(os.path.join(path_dir, "*.parquet")))


def read(path_or_dir, columns, filters=None):
    """Dict of columns. List columns become (offsets, values); list<list> become (off1, off2, values)."""
    paths = files(path_or_dir) if os.path.isdir(path_or_dir) else [path_or_dir]
    out = {}
    tabs = [pq.read_table(p, columns=columns, filters=filters) for p in paths]
    import pyarrow as pa
    t = pa.concat_tables(tabs)
    for c in columns:
        arr = t.column(c).combine_chunks()
        if hasattr(arr, "offsets"):
            o1 = arr.offsets.to_numpy()
            v = arr.values
            if hasattr(v, "offsets"):
                o2 = v.offsets.to_numpy()
                out[c] = (o1, o2, v.values.to_numpy(zero_copy_only=False))
            else:
                out[c] = (o1, v.to_numpy(zero_copy_only=False))
        else:
            out[c] = arr.to_numpy(zero_copy_only=False)
    return out


def ev(col, i):
    """Event i of a (offsets, values) list column."""
    o, v = col
    return v[o[i]:o[i + 1]]


def hit_quirk_pids(pids_col, i):
    """For event i of a list<list> particle_ids column: (hit index per entry, particle id per entry)."""
    o1, o2, v = pids_col
    a, b = o2[o1[i]], o2[o1[i + 1]]
    counts = np.diff(o2[o1[i]:o1[i + 1] + 1])
    hidx = np.repeat(np.arange(len(counts)), counts)
    return hidx, v[a:b]


def quirks_of_event(parts, i):
    """[(particle_id, charge)] of the quirks in event i (pdg +-10000100), charge-sorted (+ first)."""
    pid, pdg, ch = ev(parts["particle_id"], i), ev(parts["pdg_id"], i), ev(parts["charge"], i)
    m = np.abs(pdg) == QUIRK_PDG
    q = sorted(zip(pid[m].tolist(), ch[m].tolist()), key=lambda t: -t[1])
    return q


# ----------------------------------------------------------------------------- Release-1 style copies (one particle per hit)
def r1_paths(ds, table, first, last):
    """Files of r1/<ds>/<reco|truth>/<table> overlapping event range [first, last]."""
    sub = "reco" if table == "tracker_hits" else "truth"
    out = []
    for f in files(f"{R1}/{ds}/{sub}/{table}"):
        a, b = os.path.basename(f).split(".events")[1][:-8].split("-")
        if int(b) >= first and int(a) <= last:
            out.append(f)
    return out


def r1_events(ds, first, last):
    """Yield per-event dicts for r1 events first..last: hit arrays + quirk hit indices in true-time order.
    keys: event_id, x, y, z, time, pid, quirks = {'qplus': idx, 'qminus': idx}."""
    import pyarrow as pa
    filt = [("event_id", ">=", first), ("event_id", "<=", last)]
    hp = r1_paths(ds, "tracker_hits", first, last)
    pp = r1_paths(ds, "particles", first, last)
    h = read_many(hp, ["event_id", "x", "y", "z", "time", "particle_id"], filt)
    p = read_many(pp, ["event_id", "particle_id", "pdg_id", "charge"], filt)
    prow = {int(e): i for i, e in enumerate(p["event_id"])}
    for i, e in enumerate(h["event_id"]):
        j = prow[int(e)]
        q = quirks_of_event(p, j)
        pid = ev(h["particle_id"], i)
        t = ev(h["time"], i)
        quirks = {}
        for qid, ch in q:
            idx = np.flatnonzero(pid == qid)
            quirks["qplus" if ch > 0 else "qminus"] = idx[np.argsort(t[idx], kind="stable")]
        yield {"event_id": int(e), "x": ev(h["x"], i), "y": ev(h["y"], i), "z": ev(h["z"], i), "time": t, "pid": pid,
               "quirks": quirks}


def read_many(paths, columns, filters=None):
    import pyarrow as pa
    t = pa.concat_tables([pq.read_table(p, columns=columns, filters=filters) for p in paths])
    out = {}
    for c in columns:
        arr = t.column(c).combine_chunks()
        if hasattr(arr, "offsets"):
            out[c] = (arr.offsets.to_numpy(), arr.values.to_numpy(zero_copy_only=False))
        else:
            out[c] = arr.to_numpy(zero_copy_only=False)
    return out
