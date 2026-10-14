"""Numba-accelerated RK4 kernel for the coupled quirk-pair EOM.

Falls back gracefully: if numba is unavailable, HAVE_NUMBA is False and the
pure-Python integrator in pair.py is used instead.
"""
import numpy as np

try:
    from numba import njit
    HAVE_NUMBA = True
except Exception:  # pragma: no cover
    HAVE_NUMBA = False

    def njit(*a, **k):
        def deco(f):
            return f
        return deco if (a and callable(a[0])) is False else a[0]


@njit(cache=True, fastmath=True)
def _deriv(state, m, sigma, B, kappa, q1, q2, mode, out):
    # state, out : length-12 [r1(3) p1(3) r2(3) p2(3)]
    p1x, p1y, p1z = state[3], state[4], state[5]
    p2x, p2y, p2z = state[9], state[10], state[11]
    E1 = np.sqrt(p1x * p1x + p1y * p1y + p1z * p1z + m * m)
    E2 = np.sqrt(p2x * p2x + p2y * p2y + p2z * p2z + m * m)
    v1x, v1y, v1z = p1x / E1, p1y / E1, p1z / E1
    v2x, v2y, v2z = p2x / E2, p2y / E2, p2z / E2
    # separation unit vector (q1 -> q2)
    sx = state[6] - state[0]
    sy = state[7] - state[1]
    sz = state[8] - state[2]
    dist = np.sqrt(sx * sx + sy * sy + sz * sz)
    if dist > 0:
        sx, sy, sz = sx / dist, sy / dist, sz / dist

    # string force on q1 (shat = +s), on q2 (shat = -s)
    fs1x, fs1y, fs1z = _string(v1x, v1y, v1z, sx, sy, sz, sigma, mode)
    fs2x, fs2y, fs2z = _string(v2x, v2y, v2z, -sx, -sy, -sz, sigma, mode)

    # Lorentz force q*kappa*(v x B)
    fl1x = q1 * kappa * (v1y * B[2] - v1z * B[1])
    fl1y = q1 * kappa * (v1z * B[0] - v1x * B[2])
    fl1z = q1 * kappa * (v1x * B[1] - v1y * B[0])
    fl2x = q2 * kappa * (v2y * B[2] - v2z * B[1])
    fl2y = q2 * kappa * (v2z * B[0] - v2x * B[2])
    fl2z = q2 * kappa * (v2x * B[1] - v2y * B[0])

    out[0], out[1], out[2] = v1x, v1y, v1z
    out[3], out[4], out[5] = fs1x + fl1x, fs1y + fl1y, fs1z + fl1z
    out[6], out[7], out[8] = v2x, v2y, v2z
    out[9], out[10], out[11] = fs2x + fl2x, fs2y + fl2y, fs2z + fl2z


@njit(cache=True, fastmath=True)
def _string(vx, vy, vz, sx, sy, sz, sigma, mode):
    # mode 0 = rigid straight string (constant sigma toward partner)
    # mode 1 = relativistic Nambu-Goto endpoint: sigma[root*s + (vpar/root)*vperp]
    # mode 2 = tension reduction only: sigma*root*s
    if mode == 0:
        return sigma * sx, sigma * sy, sigma * sz
    vpar = vx * sx + vy * sy + vz * sz       # along shat (toward partner)
    pxx = vx - vpar * sx
    pxy = vy - vpar * sy
    pxz = vz - vpar * sz
    vp2 = pxx * pxx + pxy * pxy + pxz * pxz
    arg = 1.0 - vp2
    if arg < 1e-9:
        arg = 1e-9
    root = np.sqrt(arg)
    if mode == 2:
        return sigma * root * sx, sigma * root * sy, sigma * root * sz
    c = vpar / root
    return (sigma * (root * sx + c * pxx),
            sigma * (root * sy + c * pxy),
            sigma * (root * sz + c * pxz))


@njit(cache=True, fastmath=True)
def integrate_kernel(state0, m, sigma, B, kappa, q1, q2, mode,
                     dt, max_steps, r_stop):
    n = max_steps + 1
    T1 = np.empty((n, 3))
    T2 = np.empty((n, 3))
    P1 = np.empty((n, 3))
    P2 = np.empty((n, 3))
    state = state0.copy()
    k1 = np.empty(12); k2 = np.empty(12); k3 = np.empty(12); k4 = np.empty(12)
    tmp = np.empty(12)
    count = 0
    for step in range(n):
        T1[count, 0] = state[0]; T1[count, 1] = state[1]; T1[count, 2] = state[2]
        P1[count, 0] = state[3]; P1[count, 1] = state[4]; P1[count, 2] = state[5]
        T2[count, 0] = state[6]; T2[count, 1] = state[7]; T2[count, 2] = state[8]
        P2[count, 0] = state[9]; P2[count, 1] = state[10]; P2[count, 2] = state[11]
        count += 1
        if step == max_steps:
            break
        _deriv(state, m, sigma, B, kappa, q1, q2, mode, k1)
        for i in range(12):
            tmp[i] = state[i] + 0.5 * dt * k1[i]
        _deriv(tmp, m, sigma, B, kappa, q1, q2, mode, k2)
        for i in range(12):
            tmp[i] = state[i] + 0.5 * dt * k2[i]
        _deriv(tmp, m, sigma, B, kappa, q1, q2, mode, k3)
        for i in range(12):
            tmp[i] = state[i] + dt * k3[i]
        _deriv(tmp, m, sigma, B, kappa, q1, q2, mode, k4)
        for i in range(12):
            state[i] += (dt / 6.0) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i])
        # stop when both quirks are outside r_stop and moving outward
        if r_stop > 0:
            r1 = np.sqrt(state[0] * state[0] + state[1] * state[1])
            r2 = np.sqrt(state[6] * state[6] + state[7] * state[7])
            if r1 > r_stop and r2 > r_stop:
                o1 = state[0] * state[3] + state[1] * state[4]
                o2 = state[6] * state[9] + state[7] * state[10]
                if o1 > 0 and o2 > 0:
                    T1[count, 0] = state[0]; T1[count, 1] = state[1]; T1[count, 2] = state[2]
                    P1[count, 0] = state[3]; P1[count, 1] = state[4]; P1[count, 2] = state[5]
                    T2[count, 0] = state[6]; T2[count, 1] = state[7]; T2[count, 2] = state[8]
                    P2[count, 0] = state[9]; P2[count, 1] = state[10]; P2[count, 2] = state[11]
                    count += 1
                    break
    return T1[:count], P1[:count], T2[:count], P2[:count]
