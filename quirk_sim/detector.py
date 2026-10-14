"""Cylindrical barrel tracker and trajectory -> hit intersection.

The For_Daniel reference detector reconstructed directly from the truth hit
files (mass_*_Lambda_*/truth/.../event*-hits.csv):

    layer_id :  1     2     3     4     5     6    7    ...  31
    r [cm]   : 3.1   5.1   7.1   9.1  11.1  12.0 14.0  ...  62.0

i.e. five "inner" layers at 3.1 + 2*(L-1) cm and 26 "outer" layers at
2*L cm (L = 6..31).  Barrel cylinders; a hit is recorded every time the
trajectory crosses a layer radius, so an oscillating quirk can hit the same
layer several times (the characteristic multi-hit signature).
"""

import numpy as np
from .units import CM

# layer_id -> radius in cm, taken verbatim from the data
INNER_CM = [3.1, 5.1, 7.1, 9.1, 11.1]                 # layers 1..5
OUTER_CM = [float(r) for r in range(12, 64, 2)]       # layers 6..31 -> 12..62
LAYER_R_CM = INNER_CM + OUTER_CM
N_LAYERS = len(LAYER_R_CM)                            # 31


class Barrel:
    """A set of coaxial (z-axis) cylindrical layers."""

    def __init__(self, layer_r_cm=None, z_half_cm=None):
        layer_r_cm = LAYER_R_CM if layer_r_cm is None else layer_r_cm
        # radii in internal units (mm), ascending
        self.radii = np.asarray(layer_r_cm, dtype=float) * CM
        self.layer_ids = np.arange(1, len(self.radii) + 1)
        self.r_max = float(self.radii.max())
        # optional |z| acceptance (None -> unbounded cylinders)
        self.z_half = None if z_half_cm is None else z_half_cm * CM

    def crossings(self, traj_xyz):
        """Find layer crossings along a sampled trajectory.

        Parameters
        ----------
        traj_xyz : (N,3) array of positions [mm], densely sampled.

        Returns a list of hit dicts with keys r [cm], phi, z [cm], layer_id,
        in order of occurrence along the trajectory.
        """
        xyz = np.asarray(traj_xyz, dtype=float)
        r = np.hypot(xyz[:, 0], xyz[:, 1])
        hits = []
        for layer_id, R in zip(self.layer_ids, self.radii):
            d = r - R
            # sign changes between consecutive samples => a crossing
            sign = np.sign(d)
            # treat exact zeros as belonging to the previous sign to avoid
            # double counting
            idx = np.where(sign[:-1] * sign[1:] < 0)[0]
            for i in idx:
                d0, d1 = d[i], d[i + 1]
                t = d0 / (d0 - d1)               # linear interp fraction
                p = xyz[i] + t * (xyz[i + 1] - xyz[i])
                z = p[2]
                if self.z_half is not None and abs(z) > self.z_half:
                    continue
                hits.append(
                    dict(
                        layer_id=int(layer_id),
                        r=R / CM,
                        phi=float(np.arctan2(p[1], p[0])),
                        z=float(z / CM),
                        # arc index for ordering
                        _order=i + t,
                    )
                )
        hits.sort(key=lambda h: h["_order"])
        for h in hits:
            h.pop("_order")
        return hits
