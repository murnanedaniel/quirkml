"""Event model for the quirk parameter study: parameters, propagation, residuals on ODD module planes.

theta = (ln m [GeV], ln Lambda [eV], ln|p1| [GeV], theta1, phi1, ln|p2|, theta2, phi2, vx, vy, vz [mm], t0 [ns])
Quirk 1 is the +PDG quirk (charge +1), quirk 2 the -PDG one.

Hits: 'meas' = reco/tracker_hits positions (loc0/loc1 errors from var_loc0/var_loc1; 1D long strips have only
loc0), times from their sim hit; 'sim' = truth sim-hit positions (for the toy vs Geant4 check), error set by hand.
Residuals: (X - H).e0 / s0 and (X - H).e1 / s1 on the hit's module plane, X = model crossing of that plane
(closest to H), optionally (T - t_hit) / s_t. Errors can get a floor in quadrature.
"""
import sys

import numpy as np

sys.path.insert(0, "/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/params")
from kernel import run  # noqa: E402

HBARC = 197.3269804e-12  # MeV mm
CMM = 299.792458  # mm / ns
GEO = "/pscratch/sd/d/danieltm/quirk_tracking/params/odd_surfaces.npz"
NOHIT = 50.0  # mm, residual used when the plane is not crossed
THICK = {16: 0.15, 17: 0.15, 18: 0.15, 23: 0.3, 24: 0.3, 25: 0.3, 28: 0.3, 29: 0.3, 30: 0.3}  # mm Si (approx.)
NAMES = ["ln_m", "ln_L", "ln_p1", "th1", "ph1", "ln_p2", "th2", "ph2", "vx", "vy", "vz", "t0"]
DEGEN = np.array([1, 0.5, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0], float)  # m, p, sigma = Lambda^2 scaled together

_G = None


def geo():
    global _G
    if _G is None:
        _G = dict(np.load(GEO))
    return _G


def sigma_of(lam_eV):
    return (lam_eV * 1e-6) ** 2 / HBARC


def theta_truth(E):
    q1, q2 = sorted(E["quirks"], key=lambda q: -q["pdg"])
    out = [np.log(E["m"]), np.log(E["lam"])]
    for q in (q1, q2):
        p = q["p"]; P = np.linalg.norm(p)
        out += [np.log(P), np.arccos(p[2] / P), np.arctan2(p[1], p[0])]
    return np.r_[out, q1["v"], q1["t0"]]


def unpack(th):
    m, lam = np.exp(th[0]), np.exp(th[1])
    ps = []
    for k in (2, 5):
        P = np.exp(th[k]); t, f = th[k + 1], th[k + 2]
        ps.append(P * np.array([np.sin(t) * np.cos(f), np.sin(t) * np.sin(f), np.cos(t)]))
    return m, lam, ps[0], ps[1], th[8:11], th[11]


class EventModel:
    def __init__(self, E, hits="meas", sig_pos=None, time_res=None, floor=0.0, floor_t=0.0, c_sep=0.01, c_m=0.005,
                 hcap=2.0, eloss=0.0, bscale=1.0, drop_shared=False, tmargin_ns=1.0):
        G = geo()
        self.E = E
        order = sorted(range(2), key=lambda i: -E["quirks"][i]["pdg"])
        self.q = [E["quirks"][i]["q"] for i in order]
        self.c_sep, self.c_m, self.hcap, self.eloss, self.bscale = c_sep, c_m, hcap, eloss, bscale
        self.time_res = time_res
        self.H = []
        tmax = 0.0
        for i in order:
            S = E["sim"][i]
            if hits == "meas":
                M = E["meas"][i]
                keep = ~M["shared"] if drop_shared else np.ones(len(M["key"]), bool)
                X = M["x"][keep]; key = M["key"][keep]
                s0 = np.sqrt(M["var"][keep, 0]); s1 = np.sqrt(M["var"][keep, 1])
                # hit time: time of the quirk's sim hit on the same surface (closest in space)
                t = np.empty(len(X))
                for j in range(len(X)):
                    k = np.where(S["key"] == key[j])[0]
                    if len(k) == 0:
                        t[j] = np.nan; continue
                    t[j] = S["t"][k[np.argmin(np.linalg.norm(S["x"][k] - X[j], axis=1))]]
            else:
                X = S["x"]; key = S["key"]; t = S["t"]
                s0 = np.full(len(X), sig_pos if sig_pos else 0.01); s1 = s0.copy()
            idx = np.searchsorted(G["keys"], key)
            idx = np.clip(idx, 0, len(G["keys"]) - 1)
            ok = G["keys"][idx] == key
            vol = (key >> 40).astype(int)
            d = dict(H=np.ascontiguousarray(X[ok], float), C=np.ascontiguousarray(G["c"][idx[ok]]),
                     N=np.ascontiguousarray(G["n"][idx[ok]]), e0=G["e0"][idx[ok]], e1=G["e1"][idx[ok]],
                     s0=np.sqrt(s0[ok] ** 2 + floor ** 2), s1=np.where(s1[ok] > 0, np.sqrt(s1[ok] ** 2 + floor ** 2), 0.0),
                     t=t[ok], vol=vol[ok], key=key[ok], thick=np.array([THICK.get(v, 0.3) for v in vol[ok]]),
                     nmiss=int((~ok).sum()))
            d["st"] = None
            if time_res is not None:
                d["st"] = np.sqrt(time_res ** 2 + floor_t ** 2) * np.ones(len(d["H"]))
            self.H.append(d)
            if len(S["t"]):
                tmax = max(tmax, np.nanmax(S["t"]))
        self.tmax_mm = (tmax + tmargin_ns) * CMM
        self.nhits = sum(len(d["H"]) for d in self.H)

    def predict(self, th):
        m, lam, p1, p2, v, t0 = unpack(th)
        mm = m * 1e3; sig = sigma_of(lam)
        hmax = float(min(self.c_m * mm / sig, self.hcap))
        s0 = np.r_[v, p1 * 1e3, v, p2 * 1e3]
        A, B = self.H
        X1, T1, X2, T2, nst, _ = run(s0, mm, sig, self.q[0], self.q[1], self.bscale, self.c_sep, 1e-7, hmax,
                                     self.tmax_mm - t0 * CMM, A["C"], A["N"], A["H"], B["C"], B["N"], B["H"],
                                     self.eloss, A["thick"], B["thick"])
        self.nsteps = nst
        return (X1, T1 / CMM + t0), (X2, T2 / CMM + t0)

    def local(self, th):
        """per-quirk arrays of (u0, u1, dt): local position residuals (mm) and time residuals (ns)."""
        out = []
        for (X, T), d in zip(self.predict(th), self.H):
            dX = X - d["H"]
            u0 = np.einsum("ij,ij->i", dX, d["e0"]); u1 = np.einsum("ij,ij->i", dX, d["e1"])
            out.append((u0, u1, T - d["t"]))
        return out

    def resid(self, th):
        r = []
        for (u0, u1, dt), d in zip(self.local(th), self.H):
            miss = ~np.isfinite(u0)
            u0 = np.where(miss, NOHIT, u0); u1 = np.where(miss, NOHIT, u1)
            r.append(u0 / d["s0"])
            two = d["s1"] > 0
            r.append(u1[two] / d["s1"][two])
            if d["st"] is not None:
                ok = np.isfinite(d["t"])
                r.append(np.where(miss[ok], NOHIT / CMM, dt[ok]) / d["st"][ok])
        return np.concatenate(r)

    def closure_hits(self, th, rng, time_res=None, noise=1.0):
        """Replace the measured hits by the model's own crossings at th, smeared along e0 / e1 with each hit's
        error (and times with time_res). Hits whose plane the model does not cross are dropped."""
        P = self.predict(th)
        for (X, T), d in zip(P, self.H):
            ok = np.isfinite(X[:, 0])
            for k in ("H", "C", "N", "e0", "e1", "s0", "s1", "t", "vol", "key", "thick"):
                d[k] = np.ascontiguousarray(d[k][ok])
            X = X[ok]; T = T[ok]
            n0 = noise * rng.normal(size=len(X)) * d["s0"]; n1 = noise * rng.normal(size=len(X)) * d["s1"]
            d["H"] = np.ascontiguousarray(X + n0[:, None] * d["e0"] + n1[:, None] * d["e1"])
            d["t"] = T + (noise * rng.normal(size=len(T)) * time_res if time_res else 0.0)
            if d["st"] is not None:
                d["st"] = d["st"][ok]
        self.nhits = sum(len(d["H"]) for d in self.H)
