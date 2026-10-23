"""Calorimeter-cell clustering for the ColliderML experiments.

``CALO_CLUSTER_FIELDS`` is the canonical model feature order.  Positions and
radial quantities are in metres, energies are raw (uncalibrated) GeV, and
angles are in radians.  Detector IDs 9--11 are ECAL and 12--14 are HCAL.
"""

from __future__ import annotations

from operator import itemgetter

import numpy as np
from sklearn.cluster import DBSCAN

DEFAULT_EPS = 0.08
DEFAULT_MIN_SAMPLES = 3
DEFAULT_MIN_CLUSTER_ENERGY = 0.03

CALO_CLUSTER_FIELDS = (
    "log_energy",
    "log1p_n_cells",
    "x",
    "y",
    "z",
    "r",
    "eta",
    "phi",
    "sigma_eta",
    "sigma_phi",
    "sigma_r",
    "depth_extent",
    "ecal_fraction",
    "mean_abs_z_over_s",
)


def _weighted_mean_and_sigma(values: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    mean = float(np.average(values, weights=weights))
    variance = float(np.average((values - mean) ** 2, weights=weights))
    return mean, np.sqrt(max(variance, 0.0))


def cluster_calo_cells(
    x,
    y,
    z,
    energy,
    detector,
    *,
    eps: float = DEFAULT_EPS,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    min_cluster_energy: float = DEFAULT_MIN_CLUSTER_ENERGY,
) -> dict[str, np.ndarray]:
    """Cluster cells in eta/phi and return energy-weighted aggregate features.

    Phi periodicity is represented continuously by ``(sin(phi), cos(phi))``.
    Thus cells on opposite sides of the -pi/pi boundary remain neighbours.
    Non-finite/non-positive-energy cells, DBSCAN noise, and clusters below the
    raw-energy threshold are discarded. Output clusters are energy-descending.

    Raises:
        ValueError: If array lengths differ or clustering parameters are invalid.
    """
    if eps <= 0 or min_samples <= 0 or min_cluster_energy < 0:
        raise ValueError("eps and min_samples must be positive and min_cluster_energy non-negative")

    arrays = [np.asarray(value).reshape(-1) for value in (x, y, z, energy, detector)]
    if len({len(value) for value in arrays}) != 1:
        raise ValueError("x, y, z, energy, and detector must have equal lengths")
    x_mm, y_mm, z_mm, energy, detector = arrays
    valid = np.isfinite(x_mm) & np.isfinite(y_mm) & np.isfinite(z_mm) & np.isfinite(energy) & (energy > 0)
    x_mm, y_mm, z_mm, energy, detector = (value[valid] for value in arrays)
    if energy.size == 0:
        return {field: np.empty(0, dtype=np.float32) for field in CALO_CLUSTER_FIELDS}

    rho_mm = np.hypot(x_mm, y_mm)
    s_mm = np.hypot(rho_mm, z_mm)
    eta = np.arctanh(np.clip(z_mm / np.maximum(s_mm, 1e-12), -1 + 1e-7, 1 - 1e-7))
    phi = np.arctan2(y_mm, x_mm)
    coordinates = np.column_stack((eta, np.sin(phi), np.cos(phi)))
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(coordinates)

    rows: list[dict[str, float]] = []
    for label in np.unique(labels[labels >= 0]):
        mask = labels == label
        weights = energy[mask].astype(np.float64, copy=False)
        total_energy = float(weights.sum())
        if total_energy < min_cluster_energy:
            continue

        x_m = x_mm[mask] * 1e-3
        y_m = y_mm[mask] * 1e-3
        z_m = z_mm[mask] * 1e-3
        s_m = s_mm[mask] * 1e-3
        centroid_x = float(np.average(x_m, weights=weights))
        centroid_y = float(np.average(y_m, weights=weights))
        centroid_z = float(np.average(z_m, weights=weights))
        centroid_r = float(np.hypot(centroid_x, centroid_y))
        centroid_s = float(np.hypot(centroid_r, centroid_z))
        centroid_eta = float(np.arctanh(np.clip(centroid_z / max(centroid_s, 1e-12), -1 + 1e-7, 1 - 1e-7)))
        centroid_phi = float(np.arctan2(centroid_y, centroid_x))

        _, sigma_eta = _weighted_mean_and_sigma(eta[mask], weights)
        delta_phi = np.arctan2(np.sin(phi[mask] - centroid_phi), np.cos(phi[mask] - centroid_phi))
        sigma_phi = float(np.sqrt(np.average(delta_phi**2, weights=weights)))
        _, sigma_r = _weighted_mean_and_sigma(s_m, weights)

        rows.append(
            {
                "log_energy": np.log(max(total_energy, np.finfo(np.float32).tiny)),
                "log1p_n_cells": np.log1p(np.count_nonzero(mask)),
                "x": centroid_x,
                "y": centroid_y,
                "z": centroid_z,
                "r": centroid_r,
                "eta": centroid_eta,
                "phi": centroid_phi,
                "sigma_eta": sigma_eta,
                "sigma_phi": sigma_phi,
                "sigma_r": sigma_r,
                "depth_extent": float(s_m.max() - s_m.min()),
                "ecal_fraction": float(weights[np.isin(detector[mask], (9, 10, 11))].sum() / total_energy),
                "mean_abs_z_over_s": float(np.average(np.abs(z_m) / np.maximum(s_m, 1e-12), weights=weights)),
            }
        )

    rows.sort(key=itemgetter("log_energy"), reverse=True)
    return {
        field: np.asarray([row[field] for row in rows], dtype=np.float32)
        for field in CALO_CLUSTER_FIELDS
    }
