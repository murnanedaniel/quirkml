"""Step 1 (gate): does the toy (straight-string + solenoid, no energy loss) reproduce the Geant4 quirk sim hits,
given the truth initial state? Per hit: residual on the hit's own module plane (u0, u1) and time residual.

    python toy_vs_g4.py --out DIR --events 25 --procs 128
Writes DIR/toy_vs_g4_hits.parquet (one row per sim hit) and DIR/toy_vs_g4_conv.parquet (step-size check).
"""
import argparse
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, "/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/params")
from data import load_event  # noqa: E402
from model import EventModel, theta_truth  # noqa: E402

POINTS = [(m, L) for m in (100, 200, 300) for L in (167, 774, 3594)] + \
         [(200, L) for L in (100, 278, 464, 1292, 2154, 5995, 10000)]


def one(args):
    m, lam, ev, conv = args
    try:
        E = load_event(m, lam, ev)
        if E is None:
            return None, None
        th = theta_truth(E)
        t = time.time()
        M = EventModel(E, hits="sim")
        L = M.local(th)
        cost = time.time() - t
        rows = []
        for qi, ((u0, u1, dt), d) in enumerate(zip(L, M.H)):
            P = np.exp(th[2 + 3 * qi])
            for j in range(len(u0)):
                rows.append(dict(m=m, lam=lam, ev=ev, quirk=qi, t=d["t"][j], vol=d["vol"][j],
                                 r=np.hypot(d["H"][j, 0], d["H"][j, 1]), z=d["H"][j, 2], u0=u0[j], u1=u1[j], dt=dt[j],
                                 p0=P, bg0=P / m, nsteps=M.nsteps, cost=cost, nmiss_geo=d["nmiss"]))
        crow = []
        if conv:
            M2 = EventModel(E, hits="sim", c_sep=0.0025, c_m=0.00125, hcap=0.5)
            L2 = M2.local(th)
            for qi, ((a0, a1, at), (b0, b1, bt)) in enumerate(zip(L, L2)):
                for j in range(len(a0)):
                    crow.append(dict(m=m, lam=lam, ev=ev, quirk=qi, t=M.H[qi]["t"][j], d_pos=np.hypot(a0[j] - b0[j], a1[j] - b1[j]),
                                     d_t=at[j] - bt[j]))
        return rows, crow
    except Exception as e:  # noqa: BLE001
        print("FAIL", m, lam, ev, repr(e), flush=True)
        return None, None


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--events", type=int, default=25)
    ap.add_argument("--procs", type=int, default=64)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    jobs = [(m, L, ev, ev < 5) for (m, L) in POINTS for ev in range(a.events)]
    rows, crow, nfail = [], [], 0
    with Pool(a.procs) as p:
        for r, c in p.imap_unordered(one, jobs):
            if r is None:
                nfail += 1; continue
            rows += r; crow += c
    df = pd.DataFrame(rows)
    df.to_parquet(f"{a.out}/toy_vs_g4_hits.parquet")
    pd.DataFrame(crow).to_parquet(f"{a.out}/toy_vs_g4_conv.parquet")
    print("events", df.groupby(["m", "lam", "ev"]).ngroups, "hits", len(df), "failed/missing", nfail)
    if len(df) == 0:
        sys.exit(1)
