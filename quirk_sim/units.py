"""Units, constants, and the infracolor string-tension conversion.

Internal unit system (matches the Athena Quirks code):
    length    : mm
    energy    : MeV
    momentum  : MeV/c
    mass      : MeV/c^2
    time      : mm/c   (i.e. c == 1, so a "time" dt multiplies a velocity
                        v = p/E in [0,1) to give a displacement in mm)
    B field   : tesla
    charge    : units of the positron charge e

The For_Daniel reference data is stored in (cm, GeV).  Use CM, GEV below to
convert into the internal system.
"""

import numpy as np

# --- conversions from the reference-data units into internal units ---
CM = 10.0          # 1 cm  -> mm
GEV = 1000.0       # 1 GeV -> MeV
MM = 1.0
MEV = 1.0

# --- physical constants ---
# Magnetic rigidity factor: a track of transverse momentum pT [MeV/c] in a
# field B [T] has radius of curvature R [mm] = pT / (KAPPA * B * |q|).
#   p_T[GeV] = 0.299792458 * B[T] * R[m] * q   ->   R[mm] = pT[MeV]/(0.2998*B)
KAPPA = 0.299792458          # MeV/(mm * tesla) per unit charge

HBARC_MEV_MM = 197.3269804e-12   # hbar*c = 197.327 MeV*fm = 197.327e-12 MeV*mm


def string_tension_from_lambda_eV(lambda_eV):
    """Infracolor string tension sigma = Lambda^2, expressed in MeV/mm.

    The confinement scale Lambda has units of energy; the string tension is
    sigma = Lambda^2 (energy^2 = energy/length in natural units).  Convert to
    MeV/mm by dividing by hbar*c:

        sigma[MeV/mm] = (Lambda[eV] * 1e-6 MeV/eV)^2 / (hbar*c [MeV*mm])

    e.g. Lambda = 444 eV  ->  ~1000 MeV/mm  (the standalone test value).
    """
    lam_mev = lambda_eV * 1.0e-6
    return lam_mev * lam_mev / HBARC_MEV_MM


def lambda_eV_from_string_tension(sigma_mev_per_mm):
    """Inverse of string_tension_from_lambda_eV: MeV/mm -> Lambda in eV."""
    lam_mev = np.sqrt(sigma_mev_per_mm * HBARC_MEV_MM)
    return lam_mev * 1.0e6
