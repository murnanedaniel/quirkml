"""Coplanarity baseline (after Knapen et al., arXiv:1708.02243): the two quirks of a pair move close
to one plane through the production vertex, so their hits fit such a plane well; hits of two
unrelated tracks generally do not.

For each event: fit a plane through the true production vertex to a set of hits (smallest singular
vector of the vertex-subtracted positions) and report the RMS hit distance to it. Compared for
  quirk pair : all hits of the two quirks;
  SM pair    : all hits of the two highest-pT non-quirk charged particles with >= min_hits hits.
Long-strip hits (ODD volumes 28-30) are left out: their position along the strip is nominal.
Truth hit assignment; this measures the separating power, not a selection efficiency.

    python coplanar.py <parquet_root> [--quirk-pdg 10000100]
"""
import argparse, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from quirk_loader import QuirkDataset, QUIRK_PDG

LONG_STRIP_VOLUMES = (28, 29, 30)


def plane_rms(xyz, vtx):
    d = xyz - vtx
    _, _, vt = np.linalg.svd(d, full_matrices=False)
    return float(np.sqrt(np.mean((d @ vt[-1]) ** 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("parquet_root"); ap.add_argument("--quirk-pdg", type=int, default=QUIRK_PDG)
    ap.add_argument("--min-hits", type=int, default=5)
    a = ap.parse_args()
    ds = QuirkDataset(a.parquet_root, a.quirk_pdg)
    rq, rs = [], []
    for eid in ds.event_ids:
        ev = ds.event(eid); h, p = ev["hits"], ev["particles"]
        h = h[~np.isin(h.volume_id, LONG_STRIP_VOLUMES)]
        q = p[p.is_quirk]
        if len(q) != 2:
            continue
        vtx = q[["vx", "vy", "vz"]].values[0]
        mq = np.isin(h.particle_id, q.particle_id)
        if mq.sum() >= 2 * a.min_hits:
            rq.append(plane_rms(h.loc[mq, ["x", "y", "z"]].values, vtx))
        cnt = h.particle_id.value_counts()
        sm = p[(~p.is_quirk) & (p.charge != 0) & p.particle_id.isin(cnt[cnt >= a.min_hits].index)].copy()
        if len(sm) >= 2:
            sm["pt"] = np.hypot(sm.px, sm.py); two = sm.nlargest(2, "pt")
            ms = np.isin(h.particle_id, two.particle_id)
            rs.append(plane_rms(h.loc[ms, ["x", "y", "z"]].values, two[["vx", "vy", "vz"]].values[0]))
    for name, r in (("quirk pair", rq), ("SM pair (two highest-pT tracks)", rs)):
        r = np.array(r)
        print(f"{name:34s} events {len(r):3d}  plane RMS mm: median {np.median(r):.3f}  [16%, 84%] [{np.percentile(r, 16):.3f}, {np.percentile(r, 84):.3f}]")


if __name__ == "__main__":
    main()
