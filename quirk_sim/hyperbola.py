"""Faithful Python port of the Athena Quirks string primitives.

These are direct translations of the ATLAS Athena classes
(Simulation/G4Extensions/Quirks/src/{StringVector,InfracolorForce}) used by
the full-simulation HyperbolaStepper.  They are provided so the toy carries the
*exact* string-reflection algebra of the production generator and can be
unit-tested against it.

The production trajectory integrator (HyperbolaStepper) drives these through
Geant4's multi-stack stacking so the quirk and antiquirk exchange string
momentum step by step; that machinery lives in Geant4 (see QuirkDDG4/).  For
the standalone toy the equivalent coupled dynamics are integrated directly in
pair.py.  Here we keep the primitives faithful and verifiable in isolation.

A StringVector stores a 3-vector m_p and an invariant mass m_m; its associated
4-vector is lv = (m_p, t) with t = sqrt(|m_p|^2 + m_m^2).  Minkowski metric is
(+,-,-,-), matching CLHEP's G4LorentzVector.
"""

import numpy as np


def _mdot(a, b):
    """Minkowski 4-dot of two 4-vectors (t,x,y,z) -> a.t b.t - a.vec.b.vec."""
    return a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2]


class StringVector:
    """Verbatim port of StringVector.{h,cxx}."""

    __slots__ = ("p", "m")

    def __init__(self, x=0.0, y=0.0, z=0.0, m=0.0):
        self.p = np.array([x, y, z], dtype=float)
        self.m = float(m)

    @classmethod
    def from_vec(cls, vec, m):
        s = cls(0, 0, 0, m)
        s.p = np.array(vec, dtype=float)
        return s

    def vect(self):
        return self.p

    def mag(self):
        return self.m

    def t(self):
        # StringVector::t() = sqrt(p.mag2() + m*m)
        return np.sqrt(self.p @ self.p + self.m * self.m)

    def lv(self):
        # (px,py,pz, t)
        return np.array([self.p[0], self.p[1], self.p[2], self.t()])

    def reflect(self, axis_lv):
        """StringVector::reflect(axis):
            axis.vect()*(2 * lv()*axis)/axis.mag2() - m_p ,  mass unchanged.
        axis_lv is a 4-vector (px,py,pz,t).  mag2 is the Minkowski square."""
        lv = self.lv()
        num = 2.0 * _mdot(lv, axis_lv)
        axis_mag2 = _mdot(axis_lv, axis_lv)
        new_p = axis_lv[:3] * (num / axis_mag2) - self.p
        return StringVector.from_vec(new_p, self.m)

    def scaled(self, a):
        return StringVector.from_vec(self.p * a, self.m * a)

    def __repr__(self):
        return f"StringVector(p={self.p}, m={self.m}, t={self.t():.4g})"
