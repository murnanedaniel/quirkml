"""Ionisation baseline (after Li et al., arXiv:1911.02223): quirks are heavy and often slow, so they
deposit more energy per hit than relativistic tracks.

ColliderML reco tracker hits carry no charge; this uses the Geant4 energy deposit per sim hit
(`tracker_simhits.dE`, truth level), so it is an idealised upper bound on separation. Per particle
with >= min_hits sim hits: the truncated mean of dE (lowest 70% of hits, against Landau tails).

    python ionisation.py <parquet_root> [--quirk-pdg 10000100]
"""
import argparse, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from quirk_loader import QuirkDataset, QUIRK_PDG, _explode


def truncated_mean(x, frac=0.7):
    x = np.sort(x); return float(x[: max(1, int(np.ceil(frac * len(x))))].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("parquet_root"); ap.add_argument("--quirk-pdg", type=int, default=QUIRK_PDG)
    ap.add_argument("--min-hits", type=int, default=5)
    a = ap.parse_args()
    ds = QuirkDataset(a.parquet_root, a.quirk_pdg)
    q_vals, sm_vals = [], []
    for eid in ds.event_ids:
        parts = ds.event(eid)["particles"]
        quirks = set(parts.particle_id[parts.is_quirk]); charged = set(parts.particle_id[parts.charge != 0])
        sim = _explode(ds.simhits.loc[eid])
        for pid, g in sim.groupby("particle_id"):
            if len(g) < a.min_hits or pid not in charged:
                continue
            v = truncated_mean(g.dE.values) * 1e6     # dE in GeV (ACTS units) -> keV
            (q_vals if pid in quirks else sm_vals).append(v)
    q, s = np.array(q_vals), np.array(sm_vals)
    thr = np.percentile(s, 99.9) if len(s) else np.nan
    print(f"quirks        n={len(q):5d}  truncated-mean dE per hit (keV): median {np.median(q):.4g}  [16%, 84%] [{np.percentile(q,16):.4g}, {np.percentile(q,84):.4g}]")
    print(f"other charged n={len(s):5d}  median {np.median(s):.4g}  [16%, 84%] [{np.percentile(s,16):.4g}, {np.percentile(s,84):.4g}]  99.9% {thr:.4g}")
    print(f"quirks above the 99.9% point of other charged particles: {np.mean(q > thr):.2f}")


if __name__ == "__main__":
    main()
