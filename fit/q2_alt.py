#!/usr/bin/env python3
"""Force-law comparison with the v2 smooth on-surface residuals, no hit times.

For each event, on (i) quirk_sim closure hits (pilot smear) and (ii) Geant4 hits (pilot smear
+ 0.5 mm floor), no recoil constraint:
  string : (ln m, ln Lambda, ln p, theta, phi x 2)               starts: truth + 1 perturbed (3 %)
  1/r^2  : (ln m, ln k, ln p, theta, phi x 2), F = k/(d^2+eps^2), eps = 1 mm
           starts: k0 x {0.1, 1, 10} with k0 = sigma_true (d_est^2 + eps^2), d_est = 2(g*-1)m/sigma_true,
           each with truth momenta, plus 2 perturbed (10 %) momenta at k0
  helices: per quirk (ln p, theta, phi, d0, z0), no inter-quirk force (10 parameters)
           starts: truth momenta; both quirks along the pair direction (p_i = P/2)
Each model: starts ranked by initial chi2, least squares (2-point Jacobian, <= 15 iterations) from the best 2.
q = chi2_alt - chi2_string (best over starts, plain least squares, chi2 = sum of squared pulls).
"""
import argparse, json, os, sys, time
import numpy as np, pandas as pd
from scipy.optimize import least_squares
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import q2
from q2_kernel import run

EPS = 1.0
COUL_HMIN = float(os.environ.get("COUL_HMIN", "1e-7"))  # mm; raised for the reduced 1/r^2 pass
COUL_FK = tuple(float(x) for x in os.environ.get("COUL_FK", "0.1,1,10").split(","))
XS8 = np.r_[0.01, 0.01, 0.01, 1e-3, 1e-3, 0.01, 1e-3, 1e-3]


def resid(E, s0, mm, strength, mode, hmax, hmin=1e-7):
    H1, H2 = E.H[1], E.H[-1]
    RB1, RE1, RB2, RE2, *_ = run(s0, mm, strength, 1.0, -1.0, 0.01, hmin, hmax, 45000.0, E.env_r, E.env_z,
                                 H1["B"], H1["E"], H2["B"], H2["E"], mode, EPS, 0.005)
    r = np.r_[(RB1 / H1["sB"][:, None]).ravel(), (RE1 / H1["sE"][:, None]).ravel(),
              (RB2 / H2["sB"][:, None]).ravel(), (RE2 / H2["sE"][:, None]).ravel()]
    if getattr(E, "rec", None):  # pair-pT constraint from the measured recoil (any force model)
        pt = (s0[3:5] + s0[9:11]) * 1e-3
        r = np.r_[r, (pt - E.rec_meas) / E.rec_sig]
    return r


def p_of(lp, t, f):
    P = np.exp(lp); return P * np.array([np.sin(t) * np.cos(f), np.sin(t) * np.sin(f), np.cos(t)])


def string_res(E, th):
    m, lam, p1, p2 = q2.unpack(th); sig = q2.sigma_of(lam); mm = m * 1e3
    return resid(E, np.r_[E.v, p1 * 1e3, E.v, p2 * 1e3], mm, sig, 1, float(np.clip(0.005 * mm / sig, 1e-4, 0.5)))


def coul_res(E, th):
    m, k = np.exp(th[0]), np.exp(th[1]); p1 = p_of(*th[2:5]); p2 = p_of(*th[5:8])
    return resid(E, np.r_[E.v, p1 * 1e3, E.v, p2 * 1e3], m * 1e3, k, 3, 0.5, COUL_HMIN)


def helix_res(E, th):
    s0 = []
    for o in (0, 5):
        p = p_of(*th[o:o + 3]); phi = np.arctan2(p[1], p[0])
        r = E.v + np.array([-th[o + 3] * np.sin(phi), th[o + 3] * np.cos(phi), th[o + 4]])
        s0 += [r, p * 1e3]
    return resid(E, np.concatenate(s0), 400e3, 0.0, 4, 0.5)


def best_fit(F, starts, xs, nfev=15, keep=2):
    """rank starts by their initial chi2, run least squares from the best `keep`."""
    c0 = []
    for x0 in starts:
        try:
            c0.append(np.sum(F(x0) ** 2))
        except Exception:
            c0.append(np.inf)
    starts = [starts[i] for i in np.argsort(c0)[:keep]]
    best = (np.inf, None)
    for x0 in starts:
        try:
            r = least_squares(F, x0, method="trf", jac="2-point", x_scale=xs, max_nfev=nfev, xtol=1e-10, ftol=1e-10, gtol=1e-10)
            c = 2 * r.cost
        except Exception:
            continue
        if c < best[0]:
            best = (c, r.x)
    return best


def perturb(th, f, rng, idx):
    t = th.copy()
    for i in idx:
        t[i] = t[i] + np.log(1 + f) * rng.choice([-1, 1]) if i in (0, 1, 2, 5) else t[i] + f * rng.normal()
    return t


def analyse(ev, hits, floor, rng, models=("helix", "coulomb")):
    E = q2.Ev(ev, hits, floor=floor)
    p1t = np.array([ev.qp_px, ev.qp_py, ev.qp_pz]); p2t = np.array([ev.qm_px, ev.qm_py, ev.qm_pz])
    th_t = q2.theta_from(ev.m_GeV, ev.lambda_eV, p1t, p2t)
    string_res(E, th_t)
    t0 = time.time(); string_res(E, th_t); te = time.time() - t0
    if te > 2.0:
        return dict(skipped=True, eval_s=te)
    out = dict(nres=len(string_res(E, th_t)), eval_s=te)
    cs, xs_ = best_fit(lambda y: string_res(E, y), [th_t, perturb(th_t, 0.03, rng, range(8))], XS8)
    out["chi2_string"] = cs
    out["chi2_coulomb"] = np.nan
    if "coulomb" in models:
        out.update(coulomb_part(E, ev, th_t, p1t, p2t, rng))
    out["chi2_helix"] = np.nan
    if "helix" in models:
        out["chi2_helix"] = helix_part(E, p1t, p2t)
    out["q_helix"] = out["chi2_helix"] - cs; out["q_coulomb"] = out["chi2_coulomb"] - cs
    return out


def coulomb_part(E, ev, th_t, p1t, p2t, rng):
    out = {}
    sig_t = q2.sigma_of(ev.lambda_eV); m = ev.m_GeV * 1e3
    Et = ev.qp_E + ev.qm_E; P = p1t + p2t; mcm = np.sqrt(max(Et ** 2 - P @ P, 0)) * 1e3
    dest = max(2 * (mcm / (2 * m) - 1) * m / sig_t, 0.1)
    k0 = sig_t * (dest ** 2 + EPS ** 2)
    base = th_t.copy()
    starts = []
    for fk in COUL_FK:
        b = base.copy(); b[1] = np.log(k0 * fk); starts.append(b)
    for _ in range(2):
        b = perturb(base, 0.10, rng, [0, 2, 3, 4, 5, 6, 7]); b[1] = np.log(k0); starts.append(b)
    cc, _ = best_fit(lambda y: coul_res(E, y), starts, XS8)
    out["chi2_coulomb"] = cc
    return out


def helix_part(E, p1t, p2t):
    P = p1t + p2t
    hp = []
    for p in (p1t, p2t):
        P_ = np.linalg.norm(p); hp.append([np.log(P_), np.arccos(p[2] / P_), np.arctan2(p[1], p[0]), 0.0, 0.0])
    h_truth = np.array(hp[0] + hp[1])
    Ph = P / 2; Pn = np.linalg.norm(Ph)
    pa = [np.log(Pn), np.arccos(Ph[2] / Pn), np.arctan2(Ph[1], Ph[0]), 0.0, 0.0]
    h_pair = np.array(pa + pa)
    hstarts = [h_truth, h_pair]
    ch, _ = best_fit(lambda y: helix_res(E, y), hstarts, np.r_[0.01, 1e-3, 1e-3, 0.01, 0.01, 0.01, 1e-3, 1e-3, 0.01, 0.01])
    return ch


def one(args):
    point, ev, h, kind, models = args
    o = dict(point=point, event=int(ev.event), m=float(ev.m_GeV), lam=float(ev.lambda_eV), kind=kind)
    try:
        rng = np.random.default_rng(777 + int(ev.event))
        if kind == "closure":
            ch, _ = q2.closure_hits(ev, h, np.random.default_rng(4242 + int(ev.event)))
            o.update(analyse(ev, ch, 0.0, rng, models))
        else:
            o.update(analyse(ev, h.assign(x=h.x_s, y=h.y_s, z=h.z_s), 0.5, rng, models))
    except Exception as e:
        o["error"] = repr(e)
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", nargs="+", required=True)
    ap.add_argument("--nev", type=int, default=60)
    ap.add_argument("--procs", type=int, default=120)
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", nargs="+", default=["helix", "coulomb"])
    ap.add_argument("--skip-first", type=int, default=0)
    a = ap.parse_args()
    jobs = []
    for p in a.points:
        E = pd.read_parquet(f"out/pilot/compact/{p}_events.parquet"); H = pd.read_parquet(f"out/pilot/compact/{p}_hits.parquet")
        H = H[np.hypot(H.x, H.y) < 1100]
        cnt = H.groupby(["event", "quirk"]).size().unstack(fill_value=0)
        good = cnt[(cnt.get(1, 0) >= 8) & (cnt.get(-1, 0) >= 8)].index
        for _, ev in E[E.event.isin(good)].head(a.nev).iloc[a.skip_first:].iterrows():
            for kind in ("closure", "geant4"):
                jobs.append((p, ev, H[H.event == ev.event], kind, tuple(a.models)))
    print(f"jobs {len(jobs)}", flush=True)
    from multiprocessing import Pool
    with Pool(a.procs) as pool, open(a.out, "w") as f:
        for i, o in enumerate(pool.imap_unordered(one, jobs)):
            f.write(json.dumps(o, default=float) + "\n"); f.flush()
            print(f"done {i+1}/{len(jobs)}", flush=True)
    print("ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
