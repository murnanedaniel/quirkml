"""Quirk-pair propagator with crossings of planar module surfaces (adapted from fit/q2_kernel.py).

Physics as quirk_sim / fit/q2_kernel.py: relativistic Nambu-Goto straight-string endpoint force (both ends at
the same lab time) + Lorentz force in the ODD solenoid (bscale x 3 T for r < 1160 mm, |z| < 3000 mm;
bscale x 0.5 T for 1160 <= r < 1200 mm). No energy loss, no multiple scattering. Units mm, MeV, T, e;
time = c t in mm.

Optional energy loss (eloss > 0): at every crossing of a hit's own plane the quirk loses
eloss * thickness / |cos(incidence)| * bethe(beta gamma) MeV, along its momentum (a simple Bethe-Bloch
shape in silicon, used only to test whether energy loss explains the Geant4 gap).

Integrator: RK4, step h = clip(c_sep * separation, h_min, h_max) in lab time.

For every hit j of quirk k (plane point C, unit normal N), each step tests whether the trajectory changed side
of the plane; the crossing is found on the cubic-Hermite interpolant (regula falsi). Among several crossings
of one plane the one closest (3D) to the hit position H is kept. Returned: crossing points (n,3) and crossing
times (n,) per quirk; points = NaN where the plane was never crossed within RCHK of the hit.
"""
import numpy as np
from numba import njit

KAPPA = 0.299792458  # MeV / (mm T)
RCHK = 400.0  # mm: planes are only tested when the quirk is this close to the hit


@njit(cache=True)
def _bz(x, y, z, bscale):
    if abs(z) >= 3000.0:
        return 0.0
    r2 = x * x + y * y
    if r2 < 1160.0 * 1160.0:
        return 3.0 * bscale
    if r2 < 1200.0 * 1200.0:
        return 0.5 * bscale
    return 0.0


@njit(cache=True)
def _deriv(s, m, sig, q1, q2, bscale, out):
    E1 = np.sqrt(s[3] * s[3] + s[4] * s[4] + s[5] * s[5] + m * m)
    E2 = np.sqrt(s[9] * s[9] + s[10] * s[10] + s[11] * s[11] + m * m)
    v1x = s[3] / E1; v1y = s[4] / E1; v1z = s[5] / E1
    v2x = s[9] / E2; v2y = s[10] / E2; v2z = s[11] / E2
    sx = s[6] - s[0]; sy = s[7] - s[1]; sz = s[8] - s[2]
    d = np.sqrt(sx * sx + sy * sy + sz * sz)
    if d > 0:
        sx /= d; sy /= d; sz /= d
    f0 = 0.0; f1 = 0.0; f2 = 0.0; f3 = 0.0; f4 = 0.0; f5 = 0.0
    for k in range(2):
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
        a = sig * (root * ux + c * px); b = sig * (root * uy + c * py); cc = sig * (root * uz + c * pz)
        if k == 0:
            f0 = a; f1 = b; f2 = cc
        else:
            f3 = a; f4 = b; f5 = cc
    b1 = _bz(s[0], s[1], s[2], bscale); b2 = _bz(s[6], s[7], s[8], bscale)
    out[0] = v1x; out[1] = v1y; out[2] = v1z
    out[3] = f0 + q1 * KAPPA * v1y * b1; out[4] = f1 - q1 * KAPPA * v1x * b1; out[5] = f2
    out[6] = v2x; out[7] = v2y; out[8] = v2z
    out[9] = f3 + q2 * KAPPA * v2y * b2; out[10] = f4 - q2 * KAPPA * v2x * b2; out[11] = f5
    return d


@njit(cache=True)
def _herm(x0, v0, x1, v1, h, s):
    s2 = s * s; s3 = s2 * s
    return ((2 * s3 - 3 * s2 + 1) * x0 + (s3 - 2 * s2 + s) * h * v0 + (-2 * s3 + 3 * s2) * x1 + (s3 - s2) * h * v1)


@njit(cache=True)
def _gpl(p0, v0, p1, v1, h, C, N, s):
    g = 0.0
    for i in range(3):
        g += (_herm(p0[i], v0[i], p1[i], v1[i], h, s) - C[i]) * N[i]
    return g


@njit(cache=True)
def _cross_plane(p0, v0, p1, v1, h, C, N, ga, gb):
    a = 0.0; b = 1.0; s = 0.5
    for it in range(60):
        s = a - ga * (b - a) / (gb - ga) if gb != ga else 0.5 * (a + b)
        if s <= a or s >= b:
            s = 0.5 * (a + b)
        gs = _gpl(p0, v0, p1, v1, h, C, N, s)
        if abs(gs) < 1e-10:
            break
        if (gs > 0) == (ga > 0):
            a = s; ga = gs
        else:
            b = s; gb = gs
    return s


@njit(cache=True)
def _bethe(bg):
    # dE/dx shape in silicon relative to minimum ionising (~1 at bg ~ 3-4), simple form
    b2 = bg * bg / (1.0 + bg * bg)
    if bg < 0.05:
        bg = 0.05
    return (1.0 / b2) * (np.log(2 * 0.511 * bg * bg / 173e-6 * 1.0) - b2) / (np.log(2 * 0.511 * 12.25 / 173e-6) - 0.92)


@njit(cache=True)
def run(state0, m, sig, q1, q2, bscale, c_sep, h_min, h_max, tmax,
        C1, N1, H1, C2, N2, H2, eloss, thick1, thick2, max_steps=15000000):
    """C*, N*, H*: plane points, unit normals, hit positions (n,3) of quirk 1 and 2.
    Returns X1 (n1,3), T1 (n1,), X2, T2, nsteps, final state."""
    n1 = H1.shape[0]; n2 = H2.shape[0]
    X1 = np.full((n1, 3), np.nan); T1 = np.full(n1, np.nan); D1 = np.full(n1, 1e30)
    X2 = np.full((n2, 3), np.nan); T2 = np.full(n2, np.nan); D2 = np.full(n2, 1e30)
    s = state0.copy(); k1 = np.empty(12); k2 = np.empty(12); k3 = np.empty(12); k4 = np.empty(12); tmp = np.empty(12)
    old = np.empty(12); vold = np.empty(12)
    t = 0.0; nst = 0
    R2 = RCHK * RCHK
    sep = _deriv(s, m, sig, q1, q2, bscale, k1)
    while t < tmax and nst < max_steps:
        h = c_sep * sep
        if h < h_min: h = h_min
        if h > h_max: h = h_max
        for i in range(12):
            old[i] = s[i]; vold[i] = k1[i]
        for i in range(12): tmp[i] = s[i] + 0.5 * h * k1[i]
        _deriv(tmp, m, sig, q1, q2, bscale, k2)
        for i in range(12): tmp[i] = s[i] + 0.5 * h * k2[i]
        _deriv(tmp, m, sig, q1, q2, bscale, k3)
        for i in range(12): tmp[i] = s[i] + h * k3[i]
        _deriv(tmp, m, sig, q1, q2, bscale, k4)
        for i in range(12): s[i] += (h / 6.0) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i])
        sep = _deriv(s, m, sig, q1, q2, bscale, k1)
        nst += 1
        lost = False
        for q in range(2):
            o = 6 * q
            p0 = old[o:o + 3]; v0 = vold[o:o + 3]; p1 = s[o:o + 3]; v1 = k1[o:o + 3]
            if q == 0:
                C = C1; N = N1; H = H1; X = X1; T = T1; D = D1; n = n1
            else:
                C = C2; N = N2; H = H2; X = X2; T = T2; D = D2; n = n2
            for j in range(n):
                dx = p0[0] - H[j, 0]; dy = p0[1] - H[j, 1]; dz = p0[2] - H[j, 2]
                if dx * dx + dy * dy + dz * dz > R2:
                    continue
                ga = (p0[0] - C[j, 0]) * N[j, 0] + (p0[1] - C[j, 1]) * N[j, 1] + (p0[2] - C[j, 2]) * N[j, 2]
                gb = (p1[0] - C[j, 0]) * N[j, 0] + (p1[1] - C[j, 1]) * N[j, 1] + (p1[2] - C[j, 2]) * N[j, 2]
                if (ga > 0) == (gb > 0):
                    continue
                u = _cross_plane(p0, v0, p1, v1, h, C[j], N[j], ga, gb)
                x = _herm(p0[0], v0[0], p1[0], v1[0], h, u)
                y = _herm(p0[1], v0[1], p1[1], v1[1], h, u)
                z = _herm(p0[2], v0[2], p1[2], v1[2], h, u)
                dd = (x - H[j, 0]) ** 2 + (y - H[j, 1]) ** 2 + (z - H[j, 2]) ** 2
                if dd < D[j]:
                    D[j] = dd; X[j, 0] = x; X[j, 1] = y; X[j, 2] = z; T[j] = t + u * h
                if eloss > 0 and dd < 25.0:
                    # energy loss on crossing a measured module (within 5 mm of its hit)
                    pp = np.sqrt(s[o + 3] ** 2 + s[o + 4] ** 2 + s[o + 5] ** 2)
                    vv = np.sqrt(v1[0] ** 2 + v1[1] ** 2 + v1[2] ** 2)
                    cosi = abs(v1[0] * N[j, 0] + v1[1] * N[j, 1] + v1[2] * N[j, 2]) / max(vv, 1e-9)
                    if cosi < 0.05: cosi = 0.05
                    th = thick1[j] if q == 0 else thick2[j]
                    dE = eloss * th / cosi * _bethe(pp / m)
                    E = np.sqrt(pp * pp + m * m) - dE
                    if E <= m * 1.0000001:
                        E = m * 1.0000001
                    pn = np.sqrt(E * E - m * m)
                    for i in range(3): s[o + 3 + i] *= pn / pp
                    lost = True
        if lost:
            sep = _deriv(s, m, sig, q1, q2, bscale, k1)
        t += h
    if nst >= max_steps:  # gave up: report no crossings at all (the fit sees NOHIT residuals)
        X1[:, :] = np.nan; X2[:, :] = np.nan
    return X1, T1, X2, T2, nst, s
