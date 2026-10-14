"""Coupled quirk-antiquirk trajectory integration.

Physics model (the macroscopic infracolor-string regime of arXiv:0805.4642,
arXiv:1708.02243, arXiv:2002.07503, and the ATLAS Athena Quirks generator):

A quirk and its antiquirk are joined by an unbreakable infracolor string of
*constant* tension sigma = Lambda^2.  In the taut straight-string limit each
quirk feels a constant-magnitude force sigma directed toward its partner, on
top of the ordinary Lorentz force from the solenoid:

    dp1/dt = +sigma * u + q1 * kappa * (v1 x B)
    dp2/dt = -sigma * u + q2 * kappa * (v2 x B)
    dr_i/dt = v_i = p_i / E_i ,   E_i = sqrt(|p_i|^2 + m^2)

with u = (r2 - r1)/|r2 - r1| the unit vector from quirk 1 toward quirk 2.

This constant (separation-independent) attraction is what makes the pair
oscillate while the centre of mass drifts outward, producing the
characteristic non-helical, multi-hit-per-layer tracks.

Integrated with classical RK4 in lab time (c = 1, so time is measured in mm).
Units are the internal MeV/mm system from units.py.
"""

import numpy as np
from .units import KAPPA
from . import _kernel


class QuirkPair:
    def __init__(self, mass_mev, sigma_mev_per_mm, b_tesla=2.0,
                 b_dir=(0.0, 0.0, 1.0), string_mode=1):
        self.m = float(mass_mev)
        self.sigma = float(sigma_mev_per_mm)
        self.B = float(b_tesla) * np.asarray(b_dir, dtype=float)  # tesla vector
        # string_mode: 0 = rigid straight string (constant sigma toward the
        # partner; the standard leading approximation), 1 = relativistic
        # Nambu-Goto endpoint with the v_perp correction, 2 = tension
        # reduction sqrt(1-v_perp^2) only.
        self.string_mode = int(string_mode)
        self.relativistic_string = self.string_mode != 0

    # --- dynamics ---
    def _energy(self, p):
        return np.sqrt(p @ p + self.m * self.m)

    def _string_force(self, v, shat):
        """Force per quirk from a straight infracolor string of tension sigma.

        shat is the unit vector from this quirk toward its partner (the string
        is attractive, so the leading force is +sigma*shat).  For a relativistic
        Nambu-Goto string with massive endpoints the transverse motion of the
        endpoint reduces the longitudinal tension and adds a term proportional
        to the parallel velocity (arXiv:2002.07503):

            F = sigma [ sqrt(1 - v_perp^2) shat  -  (v_par/sqrt(1-v_perp^2)) v_perp ]

        with v_par = v.shat and v_perp = v - v_par shat.  Setting
        relativistic_string=False recovers the rigid straight string F = sigma*shat.
        """
        if self.string_mode == 0:
            return self.sigma * shat
        v_par = v @ shat
        v_perp = v - v_par * shat
        vp2 = v_perp @ v_perp
        root = np.sqrt(max(1.0 - vp2, 1e-9))
        if self.string_mode == 2:
            return self.sigma * root * shat
        return self.sigma * (root * shat + (v_par / root) * v_perp)

    def _deriv(self, state, q1, q2):
        r1, p1, r2, p2 = state
        E1, E2 = self._energy(p1), self._energy(p2)
        v1, v2 = p1 / E1, p2 / E2
        sep = r2 - r1
        dist = np.linalg.norm(sep)
        u = sep / dist if dist > 0 else np.zeros(3)   # from q1 toward q2
        Fs1 = self._string_force(v1, u)               # q1 pulled toward q2
        Fs2 = self._string_force(v2, -u)              # q2 pulled toward q1
        FL1 = q1 * KAPPA * np.cross(v1, self.B)
        FL2 = q2 * KAPPA * np.cross(v2, self.B)
        return (v1, Fs1 + FL1, v2, Fs2 + FL2)

    def integrate(self, r1, p1, q1, r2, p2, q2,
                  dt=1.0, max_steps=200000, r_stop_mm=None,
                  r_stop_factor=1.4):
        """Integrate the pair, returning per-quirk sampled trajectories.

        r*, p* : initial position [mm] and momentum [MeV/c] 3-vectors.
        q*     : charges in units of e.
        dt     : RK4 time step (mm).  r_stop_mm: stop once both quirks are
                 beyond this radius and moving outward.

        Returns dict with 'traj1', 'traj2' : (N,3) position arrays [mm],
        and 'mom1','mom2' : (N,3) momentum arrays [MeV/c].
        """
        state0 = np.concatenate([
            np.asarray(r1, float), np.asarray(p1, float),
            np.asarray(r2, float), np.asarray(p2, float),
        ])
        r_stop = -1.0 if r_stop_mm is None else float(r_stop_mm)
        if _kernel.HAVE_NUMBA:
            T1, P1, T2, P2 = _kernel.integrate_kernel(
                state0, self.m, self.sigma, self.B, KAPPA,
                float(q1), float(q2), self.string_mode,
                float(dt), int(max_steps), r_stop)
            return dict(traj1=T1, traj2=T2, mom1=P1, mom2=P2)
        # pure-Python fallback
        return self._integrate_py(state0, q1, q2, dt, max_steps, r_stop_mm)

    def _integrate_py(self, state0, q1, q2, dt, max_steps, r_stop_mm):
        s = (state0[0:3].copy(), state0[3:6].copy(),
             state0[6:9].copy(), state0[9:12].copy())
        T1, T2, M1, M2 = [s[0].copy()], [s[2].copy()], [s[1].copy()], [s[3].copy()]

        def add(a, sc, b):
            return tuple(a[i] + sc * b[i] for i in range(4))

        for _ in range(max_steps):
            k1 = self._deriv(s, q1, q2)
            k2 = self._deriv(add(s, dt / 2, k1), q1, q2)
            k3 = self._deriv(add(s, dt / 2, k2), q1, q2)
            k4 = self._deriv(add(s, dt, k3), q1, q2)
            s = tuple(s[i] + (dt / 6.0) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i])
                      for i in range(4))
            T1.append(s[0].copy()); M1.append(s[1].copy())
            T2.append(s[2].copy()); M2.append(s[3].copy())
            if r_stop_mm is not None:
                if np.hypot(*s[0][:2]) > r_stop_mm and np.hypot(*s[2][:2]) > r_stop_mm:
                    if (s[0][:2] @ s[1][:2]) > 0 and (s[2][:2] @ s[3][:2]) > 0:
                        break
        return dict(traj1=np.array(T1), traj2=np.array(T2),
                    mom1=np.array(M1), mom2=np.array(M2))
