"""Per-event posterior of the quirk parameters by random-walk Metropolis (flat priors in theta).

Start: least-squares fit from the truth (fit.lsq) on the chosen hit set; proposal: multivariate normal with the
fit's Gauss-Newton covariance x 2.38^2/d, re-estimated from the chain after each adaptation block.
Records ln m, ln Lambda and ln(sigma/m) quantiles, the acceptance rate and a thinned chain.

    python mcmc.py --out FILE.jsonl --points 100:167,... --events 0-9 --hits closure --times none --steps 6000
"""
import argparse
import json
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, "/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/params")
from data import cached_event as load_event  # noqa: E402
from fit import Floor, make_model, free_idx, lsq, TIMES  # noqa: E402


def one(args):
    m, lam, ev, hits, tname, floor_spec, nsteps, c_m = args
    rng = np.random.default_rng(424242 + 1000003 * ev + 7919 * m + lam)
    try:
        E = load_event(m, lam, ev)
        if E is None:
            return None
        M, th = make_model(E, hits, tname, Floor(floor_spec), rng, c_m)
        free = free_idx(TIMES[tname] is not None)
        t_start = time.time()
        tb, chi2b, r = lsq(M, th.copy(), free, 200)
        J = r.jac
        C = np.linalg.pinv(J.T @ J)
        d = len(free)

        def chi2(y):
            t = tb.copy(); t[free] = y
            v = M.resid(t)
            return float(np.sum(np.where(np.isfinite(v), v, 50.0) ** 2))
        x = tb[free].copy(); c = chi2(x)
        scale = 2.38 ** 2 / d
        L = np.linalg.cholesky(C * scale + 1e-18 * np.eye(d))
        chain = []; acc = 0; block_acc = 0
        for i in range(nsteps):
            y = x + L @ rng.normal(size=d)
            cy = chi2(y)
            if np.log(rng.uniform()) < -0.5 * (cy - c):
                x, c = y, cy; acc += 1; block_acc += 1
            chain.append(np.r_[x, c])
            if (i + 1) % 1000 == 0 and i < nsteps // 2:
                ch = np.array(chain[len(chain) // 2:])[:, :d]
                rate = block_acc / 1000; block_acc = 0
                scale *= np.exp(rate - 0.25)  # steer the acceptance toward ~0.25
                Cc = np.cov(ch.T) if len(ch) > 5 * d else C
                try:
                    L = np.linalg.cholesky(Cc * scale + 1e-18 * np.eye(d))
                except np.linalg.LinAlgError:
                    L = np.linalg.cholesky(C * scale + 1e-18 * np.eye(d))
        ch = np.array(chain[nsteps // 2:])
        lnm = ch[:, 0]; lnL = ch[:, 1]; lsm = 2 * lnL - lnm
        q = lambda a: [float(v) for v in np.percentile(a, [2.5, 16, 50, 84, 97.5])]
        rec = dict(m=m, lam=lam, ev=ev, hits=hits, times=tname, floor=floor_spec, nsteps=nsteps, truth=th.tolist(),
                   best=tb.tolist(), chi2_best=chi2b, acc=acc / nsteps, sec=time.time() - t_start,
                   q_lnm=q(lnm), q_lnL=q(lnL), q_ln_sig_over_m=q(lsm),
                   std_lnm=float(lnm.std()), std_lnL=float(lnL.std()), rho=float(np.corrcoef(lnm, lnL)[0, 1]),
                   gn_s_lnm=float(np.sqrt(C[0, 0])), gn_s_lnL=float(np.sqrt(C[1, 1])),
                   chain_thin=ch[::max(1, len(ch) // 300), :2].tolist())
        return rec
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        return dict(m=m, lam=lam, ev=ev, error=repr(e))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--points", required=True)
    ap.add_argument("--events", default="0-9")
    ap.add_argument("--hits", default="closure")
    ap.add_argument("--times", default="none")
    ap.add_argument("--floor", default="0,0")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--c-m", type=float, default=0.02)
    ap.add_argument("--procs", type=int, default=64)
    a = ap.parse_args()
    e0, e1 = (int(x) for x in a.events.split("-"))
    pts = [tuple(int(x) for x in p.split(":")) for p in a.points.split(",")]
    jobs = [(m, L, ev, h, tn, a.floor, a.steps, a.c_m) for (m, L) in pts for ev in range(e0, e1 + 1)
            for h in a.hits.split(",") for tn in a.times.split(",")]
    jobs.sort(key=lambda j: -j[1] * 1000 / j[0])
    n = 0
    with Pool(a.procs, maxtasksperchild=2) as p, open(a.out, "a") as fo:
        for r in p.imap_unordered(one, jobs):
            if r is None:
                continue
            fo.write(json.dumps(r) + "\n"); fo.flush(); n += 1
    print("written", n, "of", len(jobs))
