"""Trajectory sampler for road building: same physics, field and step control as fits/q2_kernel.run
(string mode), returning the positions of both quirks every `every` steps until the pair midpoint
has left the region r < rmax, |z| < zmax by more than the largest separation + 5 cm."""
import os, sys
import numpy as np
from numba import njit
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fits"))
from q2_kernel import _deriv


@njit(cache=True)
def trace(state0, m, sig, c_sep, h_min, h_max, tmax_mm, rmax, zmax, spacing):
    nmax = 2000000
    P1 = np.empty((nmax, 3)); P2 = np.empty((nmax, 3))
    s = state0.copy(); k1 = np.empty(12); k2 = np.empty(12); k3 = np.empty(12); k4 = np.empty(12); tmp = np.empty(12)
    sep = _deriv(s, m, sig, 1.0, -1.0, k1, 1, 1.0)
    t = 0.0; n = 0; last = -1e9; maxsep = 0.0
    P1[0] = s[0:3]; P2[0] = s[6:9]; n = 1
    while t < tmax_mm and n < nmax - 1:
        h = c_sep * sep
        if h < h_min: h = h_min
        if h > h_max: h = h_max
        for i in range(12): tmp[i] = s[i] + 0.5 * h * k1[i]
        _deriv(tmp, m, sig, 1.0, -1.0, k2, 1, 1.0)
        for i in range(12): tmp[i] = s[i] + 0.5 * h * k2[i]
        _deriv(tmp, m, sig, 1.0, -1.0, k3, 1, 1.0)
        for i in range(12): tmp[i] = s[i] + h * k3[i]
        _deriv(tmp, m, sig, 1.0, -1.0, k4, 1, 1.0)
        for i in range(12): s[i] += (h / 6.0) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i])
        sep = _deriv(s, m, sig, 1.0, -1.0, k1, 1, 1.0)
        t += h
        if t - last >= spacing:
            P1[n] = s[0:3]; P2[n] = s[6:9]; n += 1; last = t
        if sep > maxsep: maxsep = sep
        mx = 0.5 * (s[0] + s[6]); my = 0.5 * (s[1] + s[7]); mz = 0.5 * (s[2] + s[8])
        if np.sqrt(mx * mx + my * my) > rmax + maxsep + 50.0 or abs(mz) > zmax + maxsep + 50.0:
            break
    P1[n] = s[0:3]; P2[n] = s[6:9]; n += 1
    return P1[:n], P2[:n]
