"""Local calorimeter/ACTS-track accounting features for tracker hits."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ConeAccountingParams:
    grid_size: float = 0.1
    cone_radius: float = 0.4
    seed_energy_threshold: float = 0.01
    max_cones: int = 12
    max_abs_eta: float = 5.0


def _array(values, dtype=None) -> np.ndarray:
    return np.asarray(values, dtype=dtype).reshape(-1)


def _eta_phi(x, y, z) -> tuple[np.ndarray, np.ndarray]:
    x, y, z = _array(x, np.float32), _array(y, np.float32), _array(z, np.float32)
    radius = np.hypot(x, y)
    return np.arcsinh(np.divide(z, radius, out=np.zeros_like(z), where=radius > 0)), np.arctan2(y, x)


def _delta_phi(left, right):
    return (np.asarray(left) - np.asarray(right) + np.pi) % (2 * np.pi) - np.pi


def compute_cone_accounting(
    calo_x,
    calo_y,
    calo_z,
    calo_energy,
    calo_detector,
    track_d0,
    track_z0,
    track_phi,
    track_theta,
    track_qop,
    track_hit_counts,
    params: ConeAccountingParams | None = None,
) -> dict[str, np.ndarray]:
    """Find energy-peak cones and accumulate raw calo and ACTS ingredients.

    ``track_d0``, ``track_z0`` and ``track_hit_counts`` are accepted as part of
    the ACTS perigee record. They deliberately do not enter the accounting.
    """
    del track_d0, track_z0, track_hit_counts
    params = ConeAccountingParams() if params is None else params
    if params.grid_size <= 0 or params.cone_radius <= 0 or params.max_cones < 0:
        raise ValueError("grid_size and cone_radius must be positive and max_cones non-negative")

    energy = _array(calo_energy, np.float32)
    detector = _array(calo_detector, np.int64)
    eta, phi = _eta_phi(calo_x, calo_y, calo_z)
    usable = np.isfinite(eta) & np.isfinite(phi) & np.isfinite(energy) & (energy > 0) & (np.abs(eta) < params.max_abs_eta)
    eta, phi, energy, detector = eta[usable], phi[usable], energy[usable], detector[usable]

    empty = {name: np.zeros(0, dtype=np.float32) for name in ("eta", "phi", "E_ecal_raw", "E_hcal_raw", "n_cells", "sumpt_acts", "N_acts")}
    if energy.size == 0 or params.max_cones == 0:
        return empty

    n_eta = int(np.ceil(2 * params.max_abs_eta / params.grid_size))
    n_phi = int(np.ceil(2 * np.pi / params.grid_size))
    eta_bin = np.clip(((eta + params.max_abs_eta) / (2 * params.max_abs_eta) * n_eta).astype(np.int64), 0, n_eta - 1)
    phi_bin = np.floor((phi + np.pi) / (2 * np.pi) * n_phi).astype(np.int64) % n_phi
    grid = np.zeros((n_eta, n_phi), dtype=np.float64)
    np.add.at(grid, (eta_bin, phi_bin), energy)

    # 3x3 energy sum, periodic only in phi.
    smooth = np.zeros_like(grid)
    for deta in (-1, 0, 1):
        shifted_eta = np.zeros_like(grid)
        if deta < 0:
            shifted_eta[:-1] = grid[1:]
        elif deta > 0:
            shifted_eta[1:] = grid[:-1]
        else:
            shifted_eta = grid
        smooth += np.roll(shifted_eta, -1, axis=1) + shifted_eta + np.roll(shifted_eta, 1, axis=1)

    local_max = np.ones_like(smooth, dtype=bool)
    for deta in (-1, 0, 1):
        for dphi in (-1, 0, 1):
            if deta == dphi == 0:
                continue
            neighbour = np.full_like(smooth, -np.inf)
            src = np.roll(smooth, dphi, axis=1)
            if deta < 0:
                neighbour[:-1] = src[1:]
            elif deta > 0:
                neighbour[1:] = src[:-1]
            else:
                neighbour = src
            local_max &= smooth >= neighbour
    candidate_bins = np.argwhere(local_max & (smooth >= params.seed_energy_threshold))
    order = np.argsort(smooth[tuple(candidate_bins.T)])[::-1]

    centers: list[tuple[float, float]] = []
    for ieta, iphi in candidate_bins[order]:
        seed_cells = (np.abs(eta_bin - ieta) <= 1) & (np.minimum((phi_bin - iphi) % n_phi, (iphi - phi_bin) % n_phi) <= 1)
        seed_weights = energy[seed_cells]
        center_eta = np.average(eta[seed_cells], weights=seed_weights)
        center_phi = np.arctan2(
            np.sum(seed_weights * np.sin(phi[seed_cells])),
            np.sum(seed_weights * np.cos(phi[seed_cells])),
        )
        if any(np.hypot(center_eta - old_eta, _delta_phi(center_phi, old_phi)) < 2 * params.cone_radius for old_eta, old_phi in centers):
            continue
        centers.append((center_eta, center_phi))
        if len(centers) == params.max_cones:
            break

    if not centers:
        return empty
    center_eta = np.asarray([center[0] for center in centers], dtype=np.float32)
    center_phi = np.asarray([center[1] for center in centers], dtype=np.float32)
    num_cones = len(centers)

    # Cones are geometrically non-overlapping, so nearest-centre assignment is
    # equivalent to testing every cone and avoids repeated full-cell scans.
    cell_distances = np.hypot(eta[:, None] - center_eta, _delta_phi(phi[:, None], center_phi))
    cell_cone = cell_distances.argmin(axis=1)
    cell_inside = cell_distances[np.arange(eta.size), cell_cone] < params.cone_radius
    selected_cones = cell_cone[cell_inside]
    selected_energy = energy[cell_inside]
    selected_detector = detector[cell_inside]

    trk_phi = _array(track_phi, np.float32)
    trk_theta = _array(track_theta, np.float32)
    trk_qop = _array(track_qop, np.float32)
    trk_eta = -np.log(np.tan(np.clip(trk_theta, 1e-7, np.pi - 1e-7) / 2))
    trk_pt = np.divide(np.sin(trk_theta), np.abs(trk_qop), out=np.zeros_like(trk_qop), where=trk_qop != 0)
    valid_track = np.isfinite(trk_eta) & np.isfinite(trk_phi) & np.isfinite(trk_pt)
    track_distances = np.hypot(trk_eta[:, None] - center_eta, _delta_phi(trk_phi[:, None], center_phi))
    track_cone = track_distances.argmin(axis=1) if trk_eta.size else np.zeros(0, dtype=np.int64)
    track_inside = valid_track & (track_distances[np.arange(trk_eta.size), track_cone] < params.cone_radius)

    def sums(mask, weights=None):
        return np.bincount(selected_cones[mask], weights=weights, minlength=num_cones)

    out = {
        "eta": center_eta,
        "phi": center_phi,
        "E_ecal_raw": sums(np.isin(selected_detector, (9, 10, 11)), selected_energy[np.isin(selected_detector, (9, 10, 11))]),
        "E_hcal_raw": sums(np.isin(selected_detector, (12, 13, 14)), selected_energy[np.isin(selected_detector, (12, 13, 14))]),
        "n_cells": np.bincount(selected_cones, minlength=num_cones),
        "sumpt_acts": np.bincount(track_cone[track_inside], weights=trk_pt[track_inside], minlength=num_cones),
        "N_acts": np.bincount(track_cone[track_inside], minlength=num_cones),
    }
    return {name: np.asarray(values, dtype=np.float32) for name, values in out.items()}


def assign_cone_features(hit_x, hit_y, hit_z, cones: dict[str, np.ndarray], cone_radius: float = 0.4) -> np.ndarray:
    """Assign the containing cone's six raw-ingredient features to each hit."""
    hit_eta, hit_phi = _eta_phi(hit_x, hit_y, hit_z)
    features = np.zeros((hit_eta.size, 6), dtype=np.float32)
    cone_eta = _array(cones["eta"])
    if cone_eta.size == 0 or hit_eta.size == 0:
        return features
    distances = np.hypot(hit_eta[:, None] - cone_eta[None, :], _delta_phi(hit_phi[:, None], _array(cones["phi"])[None, :]))
    nearest = distances.argmin(axis=1)
    inside = distances[np.arange(hit_eta.size), nearest] < cone_radius
    features[inside, 0] = 1.0
    source_names = ("E_ecal_raw", "E_hcal_raw", "sumpt_acts", "N_acts", "n_cells")
    for column, name in enumerate(source_names, start=1):
        features[inside, column] = np.log1p(np.maximum(_array(cones[name])[nearest[inside]], 0)).astype(np.float32)
    return features
