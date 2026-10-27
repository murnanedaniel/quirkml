"""Loading quirk events from the packaged ColliderML quirk datasets (read only).

For one event: the two quirks (truth/particles, PDG +-10000100, primary), their sim hits (truth/tracker_simhits,
particle_id == quirk id, time in ns) and the reco measurements whose particle_ids contain a quirk
(reco/tracker_hits, with loc0/loc1 variances and the surface key volume/layer/surface).
Units here: mm, GeV, ns as stored.
"""
from __future__ import annotations

import glob
from functools import lru_cache

import numpy as np
import pyarrow.parquet as pq

ROOT = "/global/cfs/cdirs/m4958/data/ColliderML/simulation/quirks"
QUIRK_PDG = 10000100


def dataset_dir(m, lam, mu=0):
    return f"{ROOT}/m{m}_L{lam}_mu{mu}/v1/parquet"


def _file(ds, table, ev):
    lo = (ev // 1000) * 1000
    fs = glob.glob(f"{ds}/{table}/*.events{lo}-{lo + 999}.parquet")
    if len(fs) != 1:
        raise FileNotFoundError(f"{ds}/{table} events {lo}-{lo + 999}: {fs}")
    return fs[0]


@lru_cache(maxsize=32)
def _read(ds, table, lo):
    t = pq.read_table(_file(ds, table, lo))
    ev = np.asarray(t.column("event_id"))
    return t, {int(e): i for i, e in enumerate(ev)}


def _row(ds, table, ev):
    t, idx = _read(ds, table, (ev // 1000) * 1000)
    if ev not in idx:
        return None
    return t.slice(idx[ev], 1).to_pylist()[0]


def surface_key(vol, lay, surf):
    return (int(vol) << 40) | (int(lay) << 24) | int(surf)


def load_event(m, lam, ev, mu=0):
    """dict with quirks (list of 2 dicts: id, q, p[3] GeV, v[3] mm, t0 ns), simhits per quirk, meas per quirk."""
    ds = dataset_dir(m, lam, mu)
    P = _row(ds, "truth/particles", ev)
    if P is None:
        return None
    pdg = np.asarray(P["pdg_id"])
    sel = np.where((np.abs(pdg) == QUIRK_PDG) & np.asarray(P["primary"]))[0]
    if len(sel) != 2:
        return None
    quirks = []
    for i in sel:
        quirks.append(dict(id=int(P["particle_id"][i]), pdg=int(pdg[i]), q=float(P["charge"][i]),
                           p=np.array([P["px"][i], P["py"][i], P["pz"][i]], float),
                           v=np.array([P["vx"][i], P["vy"][i], P["vz"][i]], float),
                           t0=float(P["time"][i] or 0.0), mass=float(P["mass"][i])))
    S = _row(ds, "truth/tracker_simhits", ev)
    R = _row(ds, "reco/tracker_hits", ev)
    spid = np.asarray(S["particle_id"], np.int64)
    sim, meas = [], []
    for qk in quirks:
        k = np.where(spid == qk["id"])[0]
        sim.append(dict(x=np.c_[np.asarray(S["true_x"])[k], np.asarray(S["true_y"])[k], np.asarray(S["true_z"])[k]],
                        t=np.asarray(S["true_time"], float)[k], E=np.asarray(S["tE"])[k], dE=np.asarray(S["dE"])[k],
                        key=np.array([surface_key(a, b, c) for a, b, c in zip(np.asarray(S["volume_id"])[k],
                                      np.asarray(S["layer_id"])[k], np.asarray(S["surface_id"])[k])], np.int64),
                        vol=np.asarray(S["volume_id"])[k]))
        pids = R["particle_ids"]
        k = np.array([i for i, l in enumerate(pids) if qk["id"] in l], int)
        sims = R["simhit_ids"]
        meas.append(dict(x=np.c_[np.asarray(R["x"])[k], np.asarray(R["y"])[k], np.asarray(R["z"])[k]] if len(k) else np.zeros((0, 3)),
                         loc=np.c_[np.asarray(R["loc0"])[k], np.asarray(R["loc1"])[k]] if len(k) else np.zeros((0, 2)),
                         var=np.c_[np.asarray(R["var_loc0"])[k], np.asarray(R["var_loc1"])[k]] if len(k) else np.zeros((0, 2)),
                         sub=np.asarray(R["subspace"])[k] if len(k) else np.zeros(0),
                         key=np.array([surface_key(R["volume_id"][i], R["layer_id"][i], R["surface_id"][i]) for i in k], np.int64),
                         vol=np.asarray(R["volume_id"])[k] if len(k) else np.zeros(0),
                         shared=np.array([len(pids[i]) > 1 for i in k], bool),
                         simhit_ids=[sims[i] for i in k],
                         act=np.asarray(R["sum_activation"])[k] if len(k) else np.zeros(0)))
    return dict(m=m, lam=lam, ev=ev, quirks=quirks, sim=sim, meas=meas, S=S, R=R)


CACHE = "/pscratch/sd/d/danieltm/quirk_tracking/params/event_cache"


def cached_event(m, lam, ev, mu=0):
    """load_event without the full-event tables, through a pickle cache on pscratch (built by build_cache)."""
    import os
    import pickle
    f = f"{CACHE}/m{m}_L{lam}_mu{mu}/{ev}.pkl"
    if os.path.exists(f):
        with open(f, "rb") as fh:
            return pickle.load(fh)
    E = load_event(m, lam, ev, mu)
    if E is not None:
        E = {k: v for k, v in E.items() if k not in ("S", "R")}
    os.makedirs(os.path.dirname(f), exist_ok=True)
    tmp = f"{f}.{os.getpid()}.tmp"
    with open(tmp, "wb") as fh:
        pickle.dump(E, fh)
    os.replace(tmp, f)
    return E


def build_cache(args):
    m, lam, e0, e1 = args
    n = 0
    for ev in range(e0, e1 + 1):
        n += cached_event(m, lam, ev) is not None
    _read.cache_clear()
    return m, lam, n


if __name__ == "__main__":
    import sys
    from multiprocessing import Pool
    pts = [tuple(int(x) for x in p.split(":")) for p in sys.argv[1].split(",")]
    e0, e1 = (int(x) for x in sys.argv[2].split("-"))
    with Pool(min(len(pts), 32)) as p:
        for r in p.imap_unordered(build_cache, [(m, L, e0, e1) for m, L in pts]):
            print("cached", r, flush=True)
