"""Per-event fits of the quirk parameters with truth hit assignment.

For each event and hit set:
  hits = 'closure': the toy's own crossings of the measured modules at the truth, smeared with each measurement's
         error (and times with sigma_t): the statistical ceiling within the toy model;
  hits = 'g4': the Geant4 measurements (reco/tracker_hits), errors = measurement error (+) floor(t), where the
         floor describes the toy-vs-Geant4 difference measured in toy_vs_g4.py.
Fits (scipy least_squares, trf) of all 11 (12 with times) parameters from the truth perturbed by a factor
(1 + f)^(+-1) on m, Lambda and |p| and by f/10 rad on the angles, f in --starts.
Profiles: chi2 minimised over momenta, vertex (and t0) at fixed points along
  s_deg: the scaling direction  ln m + s, ln Lambda + s/2, ln p + s  (start: truth momenta scaled by e^s)
  s_orth: ln(sigma/m) = 2 ln Lambda - ln m shifted by s at fixed ln m (start: truth momenta).

    python fit.py --out FILE.jsonl --points 100:167,... --events 0-29 --hits closure,g4 --times none,30ps
"""
import argparse
import json
import sys
import time
from multiprocessing import Pool

import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, "/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/params")
from data import cached_event as load_event  # noqa: E402
from model import EventModel, theta_truth  # noqa: E402

XS = np.r_[0.01, 0.01, 0.01, 1e-3, 1e-3, 0.01, 1e-3, 1e-3, 0.05, 0.05, 0.5, 0.01]
TIMES = {"none": None, "100ps": 0.100, "30ps": 0.030}
SDEG = [-0.6, -0.3, -0.15, -0.07, -0.03, 0.0, 0.03, 0.07, 0.15, 0.3, 0.6]
SORTH = [-0.2, -0.1, -0.05, -0.02, 0.0, 0.02, 0.05, 0.1, 0.2]


class Floor:
    """error floor (mm) for the toy-vs-Geant4 difference: a + b * t^2 (t = hit time in ns), given as 'a,b'."""

    def __init__(self, spec):
        self.a, self.b = (float(x) for x in spec.split(","))

    def __call__(self, t):
        return self.a + self.b * np.nan_to_num(t, nan=10.0) ** 2


def make_model(E, hits, tname, floor, rng, c_m):
    st = TIMES[tname]
    M = EventModel(E, hits="meas", time_res=st, c_m=c_m)
    th = theta_truth(E)
    if hits == "closure":
        M.closure_hits(th, rng, time_res=st)
    else:
        for d in M.H:
            f = floor(d["t"])
            d["s0"] = np.sqrt(d["s0"] ** 2 + f ** 2)
            d["s1"] = np.where(d["s1"] > 0, np.sqrt(d["s1"] ** 2 + f ** 2), 0.0)
            if st is not None:
                d["st"] = np.sqrt(st ** 2 + (f / 300.0) ** 2) * np.ones(len(d["H"]))  # floor in time: f / c
        # hits without a sim-hit time cannot carry a time residual; NaN rows are skipped in resid
    return M, th


def free_idx(timing, vfix=False):
    idx = list(range(8)) + ([] if vfix else [8, 9, 10])
    return idx + ([11] if timing else [])


def lsq(M, th0, free, nfev):
    base = th0.copy()

    def F(y):
        t = base.copy(); t[free] = y
        r = M.resid(t)
        return np.where(np.isfinite(r), r, 0.0)
    lo = np.full(len(free), -np.inf); hi = np.full(len(free), np.inf)
    for i, k in enumerate(free):  # m in [10, 3000] GeV, Lambda in [30, 30000] eV
        if k == 0: lo[i], hi[i] = np.log(10.0), np.log(3000.0)
        if k == 1: lo[i], hi[i] = np.log(30.0), np.log(30000.0)
    x0 = np.clip(th0[free], lo + 1e-9, hi - 1e-9)
    r = least_squares(F, x0, method="trf", jac="2-point", x_scale=XS[free], max_nfev=nfev, bounds=(lo, hi),
                      xtol=1e-10, ftol=1e-10, gtol=1e-10, diff_step=1e-6)
    t = base.copy(); t[free] = r.x
    return t, 2 * r.cost, r


def perturb(th, f, rng):
    t = th.copy()
    for k in (0, 1, 2, 5):
        t[k] += rng.choice([-1, 1]) * np.log(1 + f)
    for k in (3, 4, 6, 7):
        t[k] += rng.normal() * f / 10
    return t


def cov_from(r, free):
    J = r.jac
    try:
        C = np.linalg.inv(J.T @ J)
    except np.linalg.LinAlgError:
        return None
    return C


def one(args):
    m, lam, ev, hits, tname, floor_spec, starts, prof, c_m, nfev = args
    rng = np.random.default_rng(1000003 * ev + 7919 * m + lam)
    try:
        E = load_event(m, lam, ev)
        if E is None:
            return None
        floor = Floor(floor_spec)
        M, th = make_model(E, hits, tname, floor, rng, c_m)
        timing = TIMES[tname] is not None
        free = free_idx(timing)
        t_task = time.time()
        rec = dict(m=m, lam=lam, ev=ev, hits=hits, times=tname, floor=floor_spec, nhits=M.nhits,
                   p1=float(np.exp(th[2])), p2=float(np.exp(th[5])), truth=th.tolist())
        t = time.time()
        r0 = M.resid(th); rec["chi2_truth"] = float(np.sum(r0 ** 2)); rec["ndof"] = int(len(r0) - len(free))
        rec["eval_s"] = time.time() - t
        fits = []
        for f in starts:
            t0 = perturb(th, f, rng) if f > 0 else th.copy()
            t = time.time()
            tf, chi2, r = lsq(M, t0, free, nfev)
            C = cov_from(r, free)
            fr = dict(f=f, theta=tf.tolist(), chi2=chi2, nfev=int(r.nfev), status=int(r.status), sec=time.time() - t)
            if C is not None:
                fr["s_lnm"] = float(np.sqrt(abs(C[0, 0]))); fr["s_lnL"] = float(np.sqrt(abs(C[1, 1])))
                fr["rho"] = float(C[0, 1] / np.sqrt(abs(C[0, 0] * C[1, 1])))
            fits.append(fr)
        rec["fits"] = fits
        if prof:
            best = min(fits, key=lambda x: x["chi2"])
            tb = np.array(best["theta"])
            pf = [k for k in free if k not in (0, 1)]
            out = {}
            for name, grid in (("deg", SDEG), ("orth", SORTH)):
                vals = []
                for s in grid:
                    if time.time() - t_task > 3000:  # task time budget: leave the rest of the profile empty
                        vals.append(float("nan")); continue
                    tt = th.copy()
                    if name == "deg":
                        tt[0] += s; tt[1] += s / 2; tt[2] += s; tt[5] += s
                    else:
                        tt[1] += s / 2
                    # start: from the best fit's momenta moved the same way, and from the truth-based point
                    tb2 = tb.copy(); tb2[0], tb2[1] = tt[0], tt[1]
                    if name == "deg":
                        tb2[2] = tb[2] + (tt[0] - tb[0]); tb2[5] = tb[5] + (tt[0] - tb[0])
                    vals.append(lsq(M, tb2, pf, max(nfev // 4, 25))[1])
                out[name] = dict(s=grid, chi2=vals)
            rec["profile"] = out
        return rec
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        return dict(m=m, lam=lam, ev=ev, hits=hits, times=tname, error=repr(e))


def parse_points(s):
    return [tuple(int(x) for x in p.split(":")) for p in s.split(",")]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--points", required=True)
    ap.add_argument("--events", default="0-29")
    ap.add_argument("--hits", default="closure,g4")
    ap.add_argument("--times", default="none")
    ap.add_argument("--floor", default="0.02,0.015")
    ap.add_argument("--starts", default="0.03,0.3")
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--c-m", type=float, default=0.02)
    ap.add_argument("--nfev", type=int, default=300)
    ap.add_argument("--procs", type=int, default=64)
    ap.add_argument("--skip-done", default="", help="glob of earlier outputs whose (event, hits, times) are skipped")
    ap.add_argument("--slice", default="0/1", help="i/n: this process runs every n-th job starting at i")
    a = ap.parse_args()
    e0, e1 = (int(x) for x in a.events.split("-"))
    starts = [float(x) for x in a.starts.split(",")]
    jobs = [(m, L, ev, h, tn, a.floor, starts, a.profile, a.c_m, a.nfev) for (m, L) in parse_points(a.points)
            for ev in range(e0, e1 + 1) for h in a.hits.split(",") for tn in a.times.split(",")]
    jobs.sort(key=lambda j: -j[1] * 1000 / j[0])  # slowest first
    done = set()
    import glob as _g
    for fn in _g.glob(a.skip_done) if a.skip_done else []:
        for line in open(fn):
            r = json.loads(line)
            if "error" not in r:
                done.add((r["m"], r["lam"], r["ev"], r["hits"], r["times"]))
    jobs = [j for j in jobs if (j[0], j[1], j[2], j[3], j[4]) not in done]
    print("jobs to run", len(jobs), "skipped", len(done), flush=True)
    si, sn = (int(x) for x in a.slice.split("/"))
    jobs = jobs[si::sn]
    n = 0
    with Pool(a.procs, maxtasksperchild=4) as p, open(a.out, "a") as fo:
        for r in p.imap_unordered(one, jobs):
            if r is None:
                continue
            fo.write(json.dumps(r) + "\n"); fo.flush(); n += 1
    print("written", n, "of", len(jobs))
