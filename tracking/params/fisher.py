"""Per-event Fisher information (Gauss-Newton J^T W J at the truth) for the quirk parameters.

The Jacobian is taken of the model's own crossings of each measured hit's module plane (noise-free closure at
the truth), so it is the local information content of that event's hits under the toy model. Scenarios differ
only by weights: position errors from the measurement (var_loc0/1, optional floor), hit times with resolution
sigma_t (none / 100 / 30 / 10 ps; t0 free when times are used), vertex free or fixed, field 3 T or 0.

    python fisher.py --out FILE.jsonl --points m:L,... --events 0-29 --procs 120
"""
import argparse
import json
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, "/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/params")
from data import cached_event as load_event  # noqa: E402
from model import EventModel, theta_truth, DEGEN, CMM  # noqa: E402

STEPS = np.r_[1e-4, 1e-4, 1e-4, 1e-5, 1e-5, 1e-4, 1e-5, 1e-5, 1e-3, 1e-3, 1e-3, 1e-4]
TIMES = {"none": None, "100ps": 0.100, "30ps": 0.030, "10ps": 0.010}


def raw(M, th):
    """unweighted residual vector pieces: position rows (mm) with their sigma, time rows (ns)."""
    pos, sp, tim = [], [], []
    L = M.local(th)
    nmiss = sum(int((~np.isfinite(l[0])).sum()) for l in L)
    for (u0, u1, dt), d in zip(L, M.H):
        miss = ~np.isfinite(u0)
        pos.append(np.where(miss, 0, u0)); sp.append(d["s0"])
        two = d["s1"] > 0
        pos.append(np.where(miss, 0, u1)[two]); sp.append(d["s1"][two])
        tim.append(np.where(miss, 0, dt))
    return np.concatenate(pos), np.concatenate(sp), np.concatenate(tim), nmiss


# Basis for the finite differences: the first vector is the scaling direction itself, so that its derivative
# (exactly zero without field) is not the small difference of large per-parameter derivatives with
# truncation errors that do not cancel. theta = th + B phi;  F_theta = B^-T F_phi B^-1.
BASIS = np.eye(12)
BASIS[:, 0] = DEGEN
BINV = np.linalg.inv(BASIS)


def jac(M, th):
    P0, sp, T0, nmiss = raw(M, th)
    JP = np.empty((len(P0), 12)); JT = np.empty((len(T0), 12))
    for k in range(12):
        d = BASIS[:, k] * STEPS[k]
        a = raw(M, th + d); b = raw(M, th - d)
        JP[:, k] = (a[0] - b[0]) / (2 * STEPS[k]); JT[:, k] = (a[2] - b[2]) / (2 * STEPS[k])
    # back to theta coordinates: J_theta = J_phi B^-1
    return JP @ BINV, sp, JT @ BINV, nmiss


def summarize(F, free):
    Fi = F[np.ix_(free, free)]
    out = {}
    ev_ = np.linalg.eigvalsh(Fi)
    if ev_[0] <= 1e-12 * ev_[-1]:
        return dict(singular=True, s_lnm=float("inf"), s_lnL=float("inf"), s_ln_sig_over_m=float("nan"))
    try:
        C = np.linalg.inv(Fi)
    except np.linalg.LinAlgError:
        return dict(singular=True, s_lnm=float("inf"), s_lnL=float("inf"), s_ln_sig_over_m=float("nan"))
    i = {k: free.index(k) for k in (0, 1)}
    s_m = np.sqrt(max(C[i[0], i[0]], 0)); s_L = np.sqrt(max(C[i[1], i[1]], 0))
    rho = C[i[0], i[1]] / (s_m * s_L) if s_m * s_L > 0 else np.nan
    # ln(sigma/m) = 2 lnL - ln m ; ln(m/sqrt(sigma)) = ln m - lnL
    v = np.zeros(len(free)); v[i[0]] = -1; v[i[1]] = 2
    s_sm = np.sqrt(max(v @ C @ v, 0))
    w = np.zeros(len(free)); w[i[0]] = 1; w[i[1]] = -1
    s_mL = np.sqrt(max(w @ C @ w, 0))
    # softest direction of the (scaled) Fisher in theta coordinates
    ev, evec = np.linalg.eigh(Fi)
    soft = evec[:, 0]
    dg = DEGEN[free] / np.linalg.norm(DEGEN[free])
    out.update(s_lnm=s_m, s_lnL=s_L, rho=rho, s_ln_sig_over_m=s_sm, s_ln_m_over_L=s_mL,
               soft_eig=float(ev[0]), soft_cos=float(abs(soft @ dg)),
               soft_vec=[float(x) for x in soft],
               cond=float(ev[-1] / ev[0]),
               s_lnp1=float(np.sqrt(max(C[free.index(2), free.index(2)], 0))),
               cov_mL=[[float(C[i[0], i[0]]), float(C[i[0], i[1]])], [float(C[i[1], i[0]]), float(C[i[1], i[1]])]])
    return out


def one(args):
    m, lam, ev, floor = args
    try:
        E = load_event(m, lam, ev)
        if E is None:
            return None
        th = theta_truth(E)
        rec = dict(m=m, lam=lam, ev=ev, floor=floor, p1=float(np.exp(th[2])), p2=float(np.exp(th[5])))
        t = time.time()
        for bname, bs in (("B3T", 1.0), ("B0", 0.0)):
            M = EventModel(E, hits="meas", floor=floor, bscale=bs, c_m=0.02)
            nmeas = M.nhits
            M.closure_hits(th, np.random.default_rng(0), noise=0.0)  # hits = the model's own crossings at truth
            rec[f"nhits_{bname}"] = M.nhits; rec["nmeas"] = nmeas
            JP, sp, JT, nmiss = jac(M, th)
            rec[f"nmiss_{bname}"] = nmiss
            WP = 1.0 / sp ** 2
            FP = JP.T @ (JP * WP[:, None])
            FT = JT.T @ JT
            Fphi = (JP @ BASIS).T @ ((JP @ BASIS) * WP[:, None])
            rec[f"{bname}_info_degdir"] = float(Fphi[0, 0])  # information along the scaling direction (per unit s)
            for tname, st in TIMES.items():
                F = FP + (FT / st ** 2 if st else 0)
                for vname, free in (("vfree", list(range(11)) + ([11] if st else [])), ("vfix", list(range(8)) + ([11] if st else []))):
                    rec[f"{bname}_{tname}_{vname}"] = summarize(F, free)
        rec["cost_s"] = time.time() - t
        rec["nsteps"] = int(M.nsteps)
        return rec
    except Exception as e:  # noqa: BLE001
        print("FAIL", m, lam, ev, repr(e), flush=True)
        return dict(m=m, lam=lam, ev=ev, floor=floor, error=repr(e))


def parse_points(s):
    return [tuple(int(x) for x in p.split(":")) for p in s.split(",")]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--points", required=True)
    ap.add_argument("--events", default="0-29")
    ap.add_argument("--floors", default="0")
    ap.add_argument("--procs", type=int, default=64)
    a = ap.parse_args()
    e0, e1 = (int(x) for x in a.events.split("-"))
    jobs = [(m, L, ev, float(f)) for (m, L) in parse_points(a.points) for ev in range(e0, e1 + 1) for f in a.floors.split(",")]
    jobs.sort(key=lambda j: -j[1])  # slow (high Lambda) first
    n = 0
    with Pool(a.procs) as p, open(a.out, "a") as fo:
        for r in p.imap_unordered(one, jobs):
            if r is None:
                continue
            fo.write(json.dumps(r) + "\n"); fo.flush(); n += 1
    print("written", n, "of", len(jobs))
