"""Loading of For_Daniel truth events."""

import os
import glob
import numpy as np
import pandas as pd

from .units import CM, GEV

DATA_ROOT = os.path.join(
    os.path.dirname(__file__), "..", "data", "For_Daniel"
)


def config_dir(mass, lam):
    return os.path.join(DATA_ROOT, f"mass_{mass}_Lambda_{lam}")


def truth_dir(mass, lam):
    return os.path.join(config_dir(mass, lam), "truth",
                        f"mass_{mass}_Lambda_{lam}_mix_sel", "Quirk")


def list_events(mass, lam):
    d = truth_dir(mass, lam)
    return sorted(
        int(os.path.basename(f).split("-")[0].replace("event", ""))
        for f in glob.glob(os.path.join(d, "event*-hits.csv"))
    )


def load_event(mass, lam, event):
    """Return (particles_df, hits_df) for one event, in raw data units
    (cm, GeV)."""
    d = truth_dir(mass, lam)
    stem = os.path.join(d, f"event{event:09d}")
    parts = pd.read_csv(stem + "-particles.csv")
    hits = pd.read_csv(stem + "-hits.csv")
    return parts, hits


def quirks_from_particles(parts):
    """Convert the two-row particles dataframe into initial conditions in
    internal units (mm, MeV).  Returns a list of dicts with r0, p, q, pid."""
    out = []
    for _, row in parts.iterrows():
        out.append(dict(
            pid=int(row["PID"]),
            q=float(row["charge"]),
            r0=np.array([row["vx"], row["vy"], row["vz"]], float) * CM,
            p=np.array([row["px"], row["py"], row["pz"]], float) * GEV,
            mass=float(row["Mass"]) * GEV,
            n_hits=int(row["n_hits"]),
            max_layer=int(row["max_layer_id"]),
        ))
    return out
