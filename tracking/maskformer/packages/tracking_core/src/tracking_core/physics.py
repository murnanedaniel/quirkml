"""Kinematic helpers used across packages.

All functions are broadcast-safe over `torch.Tensor` and `numpy.ndarray`;
the output type tracks the input. No in-place ops, no device assumptions.

Conventions (ACTS / ColliderML perigee parameters):
- `qop`: charge over momentum, units e/GeV. `|1/qop|` is total momentum in GeV.
- `theta`: polar angle in radians, 0 at +z, π at -z.
- `pt`: transverse momentum in GeV, `pt = |1/qop| * sin(theta)`.
- `eta`: pseudorapidity, `eta = -log(tan(theta/2))`. 0 at θ=π/2.
"""

from __future__ import annotations


def _is_torch(*xs) -> bool:
    try:
        import torch as _t
    except ImportError:
        return False
    return any(isinstance(x, _t.Tensor) for x in xs)


def qop_to_pt(qop, theta):
    """Transverse momentum from charge-over-momentum + polar angle."""
    if _is_torch(qop, theta):
        import torch

        return (1.0 / qop).abs() * torch.sin(theta)
    import numpy as np

    return np.abs(1.0 / qop) * np.sin(theta)


def theta_to_eta(theta):
    """Polar angle → pseudorapidity."""
    if _is_torch(theta):
        import torch

        return -torch.log(torch.tan(theta / 2))
    import numpy as np

    return -np.log(np.tan(theta / 2))


def inv_pt(qop, theta):
    """1 / pt — sometimes cleaner than pt for curvature-driven features."""
    if _is_torch(qop, theta):
        import torch

        return qop.abs() / torch.sin(theta)
    import numpy as np

    return np.abs(qop) / np.sin(theta)


def p_magnitude(qop):
    """Total momentum magnitude. |p| = |1/qop|."""
    if _is_torch(qop):
        return (1.0 / qop).abs()
    import numpy as np

    return np.abs(1.0 / qop)
