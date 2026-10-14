"""Write ColliderML quirk events in the per-event layout read by burzynski-lab/quirk-tracking
(its hepattn TrackMLDataset): <out>/event<NNNNNNNNN>-hits.parquet and -parts.parquet.

Choices (change them to match the quirk-tracking config you use):
  - quirks only in parts (its --quirks-only mode); events kept if both quirks have >= min_hits hits
    and pass pt > 5 GeV, |eta| < 2.5 (the loader's own selection, applied here so it never sees an
    empty event);
  - volume_id is remapped: pixel volumes -> 8, strip volumes -> 9 (its config uses hit_volume_ids
    [8, 9]); the ODD volume ids are kept in volume_id_odd;
  - the ATLAS cluster-shape columns it lists (charge_frac, cl_charge, cl_size, leta, lphi, lx, ly,
    lz, geta, gphi) do not exist in ColliderML and are written as zeros. Better: drop them from its
    config and set input_size;
  - s_arc is the cumulative 3D path length along the quirk's time-ordered hits; phase is the hit's
    position in that order divided by the number of hits. Both differ from its analytic-trajectory
    definitions;
  - nx, ny, nz is the unit normal to the two quirks' initial momenta (its pair-plane normal).

    python to_quirk_tracking.py <parquet_root> <out_dir> --mass-gev 400 --lambda-ev 1000
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from quirk_loader import QuirkDataset, QUIRK_PDG

CLUSTER_COLS = ("charge_frac", "cl_charge", "cl_size", "leta", "lphi", "lx", "ly", "lz", "geta", "gphi")


def convert_event(ev, mass_gev, lambda_ev, pixel_volumes, min_hits=5):
    hits, parts = ev["hits"], ev["particles"]
    q = parts[parts.is_quirk].copy()
    if len(q) != 2:
        return None
    pt = np.hypot(q.px, q.py); eta = np.arcsinh(q.pz / np.maximum(pt, 1e-9))
    nh = np.array([(hits.particle_id.values == p).sum() for p in q.particle_id])
    if (nh < min_hits).any() or (pt < 5).any() or (np.abs(eta) > 2.5).any():
        return None
    h = hits[hits.is_quirk].copy()
    h["volume_id_odd"] = h.volume_id
    h["volume_id"] = np.where(np.isin(h.volume_id, pixel_volumes), 8, 9)
    h["s_arc"] = 0.0; h["phase"] = 0.0
    for p in q.particle_id:
        m = (h.particle_id == p).values
        o = np.argsort(h.t.values[m], kind="stable")
        xyz = h.loc[m, ["x", "y", "z"]].values[o]
        s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(xyz, axis=0), axis=1))]
        sa = np.empty(m.sum()); sa[o] = s; ph = np.empty(m.sum()); ph[o] = np.arange(m.sum()) / m.sum()
        h.loc[m, "s_arc"] = sa; h.loc[m, "phase"] = ph
    for c in CLUSTER_COLS:
        h[c] = 0.0
    hits_out = h[["x", "y", "z", "t", "volume_id", "volume_id_odd", "layer_id", "particle_id", "s_arc", "phase", *CLUSTER_COLS]]
    p1, p2 = q[["px", "py", "pz"]].values
    n = np.cross(p1, p2); n = n / max(np.linalg.norm(n), 1e-12)
    parts_out = pd.DataFrame(dict(
        particle_id=q.particle_id.values.astype(np.int64), px=q.px.values, py=q.py.values, pz=q.pz.values,
        q=q.charge.values, is_quirk=True, mass_gev=mass_gev, lambda_ev=lambda_ev,
        log10_f_over_m=np.log10(lambda_ev ** 2 / mass_gev),   # quirk-tracking prep_quirks.py: Lambda[eV]^2 / m[GeV]
        nx=n[0], ny=n[1], nz=n[2], vx=q.vx.values, vy=q.vy.values, vz=q.vz.values))
    return hits_out.astype({"particle_id": np.int64}), parts_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("parquet_root"); ap.add_argument("out_dir")
    ap.add_argument("--mass-gev", type=float, required=True); ap.add_argument("--lambda-ev", type=float, required=True)
    ap.add_argument("--quirk-pdg", type=int, default=QUIRK_PDG)
    ap.add_argument("--pixel-volumes", type=int, nargs="+", default=[16, 17, 18],
                    help="ODD pixel volume ids (check against your geometry)")
    ap.add_argument("--min-hits", type=int, default=5)
    a = ap.parse_args()
    ds = QuirkDataset(a.parquet_root, a.quirk_pdg); out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    kept = 0
    for eid in ds.event_ids:
        r = convert_event(ds.event(eid), a.mass_gev, a.lambda_ev, a.pixel_volumes, a.min_hits)
        if r is None:
            continue
        r[0].to_parquet(out / f"event{eid:09d}-hits.parquet", index=False)
        r[1].to_parquet(out / f"event{eid:09d}-parts.parquet", index=False)
        kept += 1
    print(f"wrote {kept} of {len(ds.event_ids)} events to {out}")


if __name__ == "__main__":
    main()
