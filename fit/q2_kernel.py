"""Smooth-residual kernel for identifiability studies (no hit times anywhere).
Force modes: 1 = Nambu-Goto string (default), 3 = softened attractive k/(d^2+eps^2), 4 = none.

Physics: relativistic Nambu-Goto endpoint force (quirk_sim, mode 1) + Lorentz force in the exact
ODD solenoid field (3 T for r < 1160 mm and |z| < 3000 mm, 0.5 T for 1160 <= r < 1200 mm, else 0).
Units mm, MeV, T, e; time in mm/c.

Integrator: RK4 with a variable step  h = clip(c_sep * separation, h_min, h_max)  so the rapid
turn of the string direction when the quirks pass close to each other is resolved.

Residuals: for each hit, the trajectory crossing of the hit's own detector surface is found by
cubic-Hermite interpolation inside the step (barrel hit: cylinder r = R_hit; endcap hit: plane
z = Z_hit).  Barrel residual = (R * dphi, dz); endcap residual = (dx, dy).  Among several
crossings of the same surface the one closest to the hit is kept.  These residuals are smooth
in the parameters (no nearest-sample search).
"""
import numpy as np
from numba import njit

KAPPA = 0.299792458
NOHIT = 50.0  # mm residual if the surface is never crossed


@njit(cache=True)
def _bz(x, y, z):
    if abs(z) >= 3000.0:
        return 0.0
    r = np.sqrt(x * x + y * y)
    if r < 1160.0:
        return 3.0
    if r < 1200.0:
        return 0.5
    return 0.0


@njit(cache=True)
def _deriv(s, m, sig, q1, q2, out, mode, eps):
    E1 = np.sqrt(s[3] * s[3] + s[4] * s[4] + s[5] * s[5] + m * m)
    E2 = np.sqrt(s[9] * s[9] + s[10] * s[10] + s[11] * s[11] + m * m)
    v1x = s[3] / E1; v1y = s[4] / E1; v1z = s[5] / E1
    v2x = s[9] / E2; v2y = s[10] / E2; v2z = s[11] / E2
    sx = s[6] - s[0]; sy = s[7] - s[1]; sz = s[8] - s[2]
    d = np.sqrt(sx * sx + sy * sy + sz * sz)
    if d > 0:
        sx /= d; sy /= d; sz /= d
    # inter-quirk force on 1 (toward 2) and on 2 (toward 1)
    # mode 1: Nambu-Goto string (sig = tension, MeV/mm); mode 3: attractive sig/(d^2+eps^2)
    # (sig = k in MeV mm); mode 4: no force
    f = np.zeros(6)
    if mode == 4:
        out[0] = v1x; out[1] = v1y; out[2] = v1z
        b1 = _bz(s[0], s[1], s[2]); b2 = _bz(s[6], s[7], s[8])
        out[3] = q1 * KAPPA * v1y * b1; out[4] = -q1 * KAPPA * v1x * b1; out[5] = 0.0
        out[6] = v2x; out[7] = v2y; out[8] = v2z
        out[9] = q2 * KAPPA * v2y * b2; out[10] = -q2 * KAPPA * v2x * b2; out[11] = 0.0
        return d
    if mode == 3:
        F = sig / (d * d + eps * eps)
        f[0] = F * sx; f[1] = F * sy; f[2] = F * sz; f[3] = -F * sx; f[4] = -F * sy; f[5] = -F * sz
    for k in range(2):
        if mode == 3:
            break
        if k == 0:
            vx, vy, vz, ux, uy, uz = v1x, v1y, v1z, sx, sy, sz
        else:
            vx, vy, vz, ux, uy, uz = v2x, v2y, v2z, -sx, -sy, -sz
        vpar = vx * ux + vy * uy + vz * uz
        px = vx - vpar * ux; py = vy - vpar * uy; pz = vz - vpar * uz
        arg = 1.0 - (px * px + py * py + pz * pz)
        if arg < 1e-12:
            arg = 1e-12
        root = np.sqrt(arg); c = vpar / root
        f[3 * k] = sig * (root * ux + c * px); f[3 * k + 1] = sig * (root * uy + c * py); f[3 * k + 2] = sig * (root * uz + c * pz)
    b1 = _bz(s[0], s[1], s[2]); b2 = _bz(s[6], s[7], s[8])
    out[0] = v1x; out[1] = v1y; out[2] = v1z
    out[3] = f[0] + q1 * KAPPA * v1y * b1; out[4] = f[1] - q1 * KAPPA * v1x * b1; out[5] = f[2]
    out[6] = v2x; out[7] = v2y; out[8] = v2z
    out[9] = f[3] + q2 * KAPPA * v2y * b2; out[10] = f[4] - q2 * KAPPA * v2x * b2; out[11] = f[5]
    return d


@njit(cache=True)
def _herm(x0, v0, x1, v1, h, s):
    s2 = s * s; s3 = s2 * s
    return ((2 * s3 - 3 * s2 + 1) * x0 + (s3 - 2 * s2 + s) * h * v0 + (-2 * s3 + 3 * s2) * x1 + (s3 - s2) * h * v1)


@njit(cache=True)
def _hermd(x0, v0, x1, v1, h, s):
    s2 = s * s
    return ((6 * s2 - 6 * s) * x0 + (3 * s2 - 4 * s + 1) * h * v0 + (-6 * s2 + 6 * s) * x1 + (3 * s2 - 2 * s) * h * v1)


@njit(cache=True)
def _g(p0, v0, p1, v1, h, kind, target, s):
    x = _herm(p0[0], v0[0], p1[0], v1[0], h, s); y = _herm(p0[1], v0[1], p1[1], v1[1], h, s)
    if kind == 0:
        return np.sqrt(x * x + y * y) - target
    return _herm(p0[2], v0[2], p1[2], v1[2], h, s) - target


@njit(cache=True)
def _cross(p0, v0, p1, v1, h, kind, target):
    """solve r(s) = target (kind 0) or z(s) = target (kind 1) on the Hermite cubic, s in [0,1]
    (regula falsi with bisection safeguard, to |g| < 1e-10 mm)."""
    a = 0.0; b = 1.0
    ga = _g(p0, v0, p1, v1, h, kind, target, a); gb = _g(p0, v0, p1, v1, h, kind, target, b)
    s = 0.5
    for it in range(60):
        s = a - ga * (b - a) / (gb - ga) if gb != ga else 0.5 * (a + b)
        if s <= a or s >= b:
            s = 0.5 * (a + b)
        gs = _g(p0, v0, p1, v1, h, kind, target, s)
        if abs(gs) < 1e-10:
            break
        if (gs > 0) == (ga > 0):
            a = s; ga = gs
        else:
            b = s; gb = gs
    x = _herm(p0[0], v0[0], p1[0], v1[0], h, s); y = _herm(p0[1], v0[1], p1[1], v1[1], h, s)
    z = _herm(p0[2], v0[2], p1[2], v1[2], h, s)
    return x, y, z


@njit(cache=True)
def run(state0, m, sig, q1, q2, c_sep, h_min, h_max, tmax_mm, env_r, env_z,
        HB1, HE1, HB2, HE2, mode=1, eps=1.0, c_m=0.005):
    """HB*: barrel hits (n,3) sorted by r; HE*: endcap hits (n,3) sorted by z.
    Returns residual arrays RB1 (n,2), RE1, RB2, RE2 and crossing points PB1 (n,3), ..., and nsteps."""
    RB1 = np.full((HB1.shape[0], 2), NOHIT); RE1 = np.full((HE1.shape[0], 2), NOHIT)
    RB2 = np.full((HB2.shape[0], 2), NOHIT); RE2 = np.full((HE2.shape[0], 2), NOHIT)
    PB1 = np.zeros((HB1.shape[0], 3)); PE1 = np.zeros((HE1.shape[0], 3))
    PB2 = np.zeros((HB2.shape[0], 3)); PE2 = np.zeros((HE2.shape[0], 3))
    rb1 = np.sqrt(HB1[:, 0] ** 2 + HB1[:, 1] ** 2); rb2 = np.sqrt(HB2[:, 0] ** 2 + HB2[:, 1] ** 2)
    ze1 = HE1[:, 2].copy(); ze2 = HE2[:, 2].copy()
    s = state0.copy(); k1 = np.empty(12); k2 = np.empty(12); k3 = np.empty(12); k4 = np.empty(12); tmp = np.empty(12)
    t = 0.0; nst = 0; maxsep = 0.0
    sep = _deriv(s, m, sig, q1, q2, k1, mode, eps)
    while t < tmax_mm:
        h = c_sep * sep if mode != 4 else h_max
        if mode == 3:
            hF = c_m * m * (sep * sep + eps * eps) / sig
            if h > hF: h = hF
        if h < h_min: h = h_min
        if h > h_max: h = h_max
        old = s.copy(); vold = k1.copy()
        for i in range(12): tmp[i] = s[i] + 0.5 * h * k1[i]
        _deriv(tmp, m, sig, q1, q2, k2, mode, eps)
        for i in range(12): tmp[i] = s[i] + 0.5 * h * k2[i]
        _deriv(tmp, m, sig, q1, q2, k3, mode, eps)
        for i in range(12): tmp[i] = s[i] + h * k3[i]
        _deriv(tmp, m, sig, q1, q2, k4, mode, eps)
        for i in range(12): s[i] += (h / 6.0) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i])
        sep = _deriv(s, m, sig, q1, q2, k1, mode, eps)
        t += h; nst += 1
        for q in range(2):
            o = 6 * q
            p0 = old[o:o + 3]; v0 = vold[o:o + 3]; p1 = s[o:o + 3]; v1 = k1[o:o + 3]
            r0 = np.sqrt(p0[0] ** 2 + p0[1] ** 2); r1 = np.sqrt(p1[0] ** 2 + p1[1] ** 2)
            rb = rb1 if q == 0 else rb2; HB = HB1 if q == 0 else HB2; RB = RB1 if q == 0 else RB2; PB = PB1 if q == 0 else PB2
            lo = min(r0, r1); hi = max(r0, r1)
            j0 = np.searchsorted(rb, lo); j1 = np.searchsorted(rb, hi)
            for j in range(j0, j1):
                x, y, z = _cross(p0, v0, p1, v1, h, 0, rb[j])
                dphi = np.arctan2(y, x) - np.arctan2(HB[j, 1], HB[j, 0])
                if dphi > np.pi: dphi -= 2 * np.pi
                if dphi < -np.pi: dphi += 2 * np.pi
                a = rb[j] * dphi; b = z - HB[j, 2]
                if a * a + b * b < RB[j, 0] ** 2 + RB[j, 1] ** 2:
                    RB[j, 0] = a; RB[j, 1] = b; PB[j, 0] = x; PB[j, 1] = y; PB[j, 2] = z
            ze = ze1 if q == 0 else ze2; HE = HE1 if q == 0 else HE2; RE = RE1 if q == 0 else RE2; PE = PE1 if q == 0 else PE2
            lo = min(p0[2], p1[2]); hi = max(p0[2], p1[2])
            j0 = np.searchsorted(ze, lo); j1 = np.searchsorted(ze, hi)
            for j in range(j0, j1):
                x, y, z = _cross(p0, v0, p1, v1, h, 1, ze[j])
                a = x - HE[j, 0]; b = y - HE[j, 1]
                if a * a + b * b < RE[j, 0] ** 2 + RE[j, 1] ** 2:
                    RE[j, 0] = a; RE[j, 1] = b; PE[j, 0] = x; PE[j, 1] = y; PE[j, 2] = z
        if sep > maxsep: maxsep = sep
        mx = 0.5 * (s[0] + s[6]); my = 0.5 * (s[1] + s[7]); mz = 0.5 * (s[2] + s[8])
        if np.sqrt(mx * mx + my * my) > env_r + maxsep + 50.0 or abs(mz) > env_z + maxsep + 50.0:
            break
    return RB1, RE1, RB2, RE2, PB1, PE1, PB2, PE2, nst
