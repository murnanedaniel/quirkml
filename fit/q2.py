"""Identifiability study driver (no hit times; smooth surface-crossing residuals).

Parameters theta = (ln m, ln Lambda, ln|p1|, theta1, phi1, ln|p2|, theta2, phi2)  [m in GeV,
Lambda in eV, p in GeV, angles in rad].  The expected near-degenerate direction (scaling of
m, p and sigma = Lambda^2 together; broken only by the magnetic field) is
(1, 1/2, 1, 0, 0, 1, 0, 0) in these coordinates.

Hits: the pilot Geant4 SimTrackerHits of each quirk (truth association), r < 1.1 m, all hit times
(time is not used anywhere).  Barrel hits (pixel / short / long strip barrel) give (R dphi, dz)
residuals on their cylinder; endcap hits give (dx, dy) on their disk plane.
Errors: the pilot smear per subdetector (15 / 40 / 70 um), optionally plus a constant floor.
"""
import numpy as np, pandas as pd
from scipy.optimize import least_squares
from q2_kernel import run, NOHIT

HBARC = 197.3269804e-12
SMEAR = {0: 0.015, 1: 0.015, 2: 0.040, 3: 0.040, 4: 0.070, 5: 0.070}
BARREL = (0, 2, 4)
DEGEN = np.array([1, 0.5, 1, 0, 0, 1, 0, 0], float)


def sigma_of(lam_eV):
    return (lam_eV * 1e-6) ** 2 / HBARC


def theta_from(m, lam, p1, p2):
    out = [np.log(m), np.log(lam)]
    for p in (p1, p2):
        P = np.linalg.norm(p); out += [np.log(P), np.arccos(p[2] / P), np.arctan2(p[1], p[0])]
    return np.array(out)


def unpack(th):
    m, lam = np.exp(th[0]), np.exp(th[1])
    ps = []
    for k in (2, 5):
        P = np.exp(th[k]); t, f = th[k + 1], th[k + 2]
        ps.append(P * np.array([np.sin(t) * np.cos(f), np.sin(t) * np.sin(f), np.cos(t)]))
    return m, lam, ps[0], ps[1]


class Ev:
    """hits: DataFrame with x, y, z (positions to fit), coll, quirk."""

    def __init__(self, ev, hits, c_sep=0.01, c_m=0.005, floor=0.0, rec=None, rec_rng=None, hcap=0.5, time_floor=False, rec_abs=0.0):
        self.ev = ev; self.v = np.array([ev.vx, ev.vy, ev.vz])
        self.c_sep, self.c_m, self.floor, self.hcap = c_sep, c_m, floor, hcap
        self.H = {}
        self.order = []  # hit index (DataFrame label) for each residual pair, in residual order
        for q in (1, -1):
            hq = hits[hits.quirk == q]
            b = hq[hq.coll.isin(BARREL)]; e = hq[~hq.coll.isin(BARREL)]
            b = b.iloc[np.argsort(np.hypot(b.x.values, b.y.values))]; e = e.iloc[np.argsort(e.z.values)]
            self.order += list(b.index) + list(e.index)
            self.H[q] = dict(B=np.ascontiguousarray(b[["x", "y", "z"]].values, float),
                             E=np.ascontiguousarray(e[["x", "y", "z"]].values, float),
                             sB=np.sqrt(np.array([SMEAR[c] for c in b.coll]) ** 2 + (0.05 + 0.004 * b.t.values ** 2 if time_floor else floor) ** 2),
                             sE=np.sqrt(np.array([SMEAR[c] for c in e.coll]) ** 2 + (0.05 + 0.004 * e.t.values ** 2 if time_floor else floor) ** 2),
                             cB=b.coll.values, cE=e.coll.values)
        allh = np.vstack([self.H[q][k] for q in (1, -1) for k in ("B", "E") if len(self.H[q][k])])
        self.env_r = np.hypot(allh[:, 0], allh[:, 1]).max(); self.env_z = np.abs(allh[:, 2]).max()
        self.nhits = len(allh)
        rc = np.array([ev.recoil_px, ev.recoil_py])
        self.rec = rec
        if rec:
            self.rec_sig = np.hypot(rec_abs, rec * np.hypot(*rc))
            self.rec_meas = -rc + (rec_rng.normal(0, self.rec_sig, 2) if rec_rng is not None else 0)

    def raw(self, th, c_sep=None, c_m=None):
        m, lam, p1, p2 = unpack(th)
        sig = sigma_of(lam); mm = m * 1e3
        cs = self.c_sep if c_sep is None else c_sep; cm = self.c_m if c_m is None else c_m
        hmax = float(np.clip(cm * mm / sig, 1e-4, self.hcap))
        s0 = np.r_[self.v, p1 * 1e3, self.v, p2 * 1e3]
        H1, H2 = self.H[1], self.H[-1]
        return run(s0, mm, sig, 1.0, -1.0, cs, 1e-7, hmax, 45000.0, self.env_r, self.env_z,
                   H1["B"], H1["E"], H2["B"], H2["E"])

    def resid(self, th):
        RB1, RE1, RB2, RE2, *_ = self.raw(th)
        H1, H2 = self.H[1], self.H[-1]
        r = np.r_[(RB1 / H1["sB"][:, None]).ravel(), (RE1 / H1["sE"][:, None]).ravel(),
                  (RB2 / H2["sB"][:, None]).ravel(), (RE2 / H2["sE"][:, None]).ravel()]
        if self.rec:
            _, _, p1, p2 = unpack(th)
            r = np.r_[r, ((p1 + p2)[:2] - self.rec_meas) / self.rec_sig]
        return r


def closure_hits(ev, hits, rng, c_sep=0.0025, c_m=0.00125):
    """Hits from the model itself: the model trajectory at truth (fine tolerance) crossing the
    surface of every Geant4 hit; smeared on the surface (barrel: R dphi and z; endcap: x, y)."""
    E = Ev(ev, hits, c_sep, c_m)
    th = theta_from(ev.m_GeV, ev.lambda_eV, np.array([ev.qp_px, ev.qp_py, ev.qp_pz]), np.array([ev.qm_px, ev.qm_py, ev.qm_pz]))
    RB1, RE1, RB2, RE2, PB1, PE1, PB2, PE2, _ = E.raw(th)
    rows = []
    for q, RB, RE, PB, PE in ((1, RB1, RE1, PB1, PE1), (-1, RB2, RE2, PB2, PE2)):
        H = E.H[q]
        for P, R, cl, kind in ((PB, RB, H["cB"], 0), (PE, RE, H["cE"], 1)):
            for i in range(len(P)):
                if abs(R[i, 0]) >= NOHIT and abs(R[i, 1]) >= NOHIT:
                    continue  # surface not crossed by the model: drop the hit
                s = SMEAR[cl[i]]; x, y, z = P[i]
                if kind == 0:
                    Rr = np.hypot(x, y); ph = np.arctan2(y, x) + rng.normal(0, s) / Rr
                    x, y, z = Rr * np.cos(ph), Rr * np.sin(ph), z + rng.normal(0, s)
                else:
                    x, y = x + rng.normal(0, s), y + rng.normal(0, s)
                rows.append(dict(x=x, y=y, z=z, coll=cl[i], quirk=q))
    return pd.DataFrame(rows), th


def fit(E, th0, nfev=200):
    r = least_squares(E.resid, th0, method="trf", jac="3-point", x_scale=np.r_[0.01, 0.01, 0.01, 1e-3, 1e-3, 0.01, 1e-3, 1e-3],
                      max_nfev=nfev, xtol=1e-12, ftol=1e-12, gtol=1e-12)
    return r


def fisher(E, th, steps=None):
    """Gauss-Newton Hessian J^T J from central differences of the smooth residuals."""
    steps = np.r_[1e-4, 1e-4, 1e-4, 1e-5, 1e-5, 1e-4, 1e-5, 1e-5] if steps is None else steps
    r0 = E.resid(th); J = np.empty((len(r0), len(th)))
    for k in range(len(th)):
        d = np.zeros(len(th)); d[k] = steps[k]
        J[:, k] = (E.resid(th + d) - E.resid(th - d)) / (2 * steps[k])
    return J.T @ J, r0


def profile_scan(E, th_best, idx, half, n=11, nfev=60):
    """chi2 profile along parameter idx over +-half, other 7 parameters refit (warm start)."""
    xs = np.linspace(-half, half, n); out = np.full(n, np.nan)
    free = [k for k in range(8) if k != idx]
    order = np.r_[np.arange(n // 2, n), np.arange(n // 2 - 1, -1, -1)]
    cur = th_best.copy()
    for i in order:
        if i == n // 2 - 1:
            cur = th_best.copy()
        fixed = th_best[idx] + xs[i]
        def F(y):
            t = np.empty(8); t[free] = y; t[idx] = fixed
            return E.resid(t)
        r = least_squares(F, cur[free], method="trf", jac="3-point", max_nfev=nfev,
                          x_scale=np.r_[0.01, 0.01, 0.01, 1e-3, 1e-3, 0.01, 1e-3, 1e-3][free], xtol=1e-12, ftol=1e-12, gtol=1e-12)
        out[i] = 2 * r.cost
        cur[free] = r.x; cur[idx] = fixed
    return xs, out
