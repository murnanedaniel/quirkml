from pathlib import Path

import numpy as np
import pytest
import torch

from ml_tracking.experiments.colliderml.data import ColliderMLDataset
from ml_tracking.experiments.colliderml.deficit_features import (
    ConeAccountingParams,
    assign_cone_features,
    compute_cone_accounting,
)

DATASET_ROOT = Path("/global/cfs/cdirs/m4958/data/ColliderML/simulation/hard_scatter/higgs_portal/v1/parquet")


def _xyz(eta, phi, radius=1000.0):
    eta, phi = np.asarray(eta), np.asarray(phi)
    return radius * np.cos(phi), radius * np.sin(phi), radius * np.sinh(eta)


def test_two_peaks_phi_wrap_and_hit_assignment():
    # The cells around +/-pi are one physical peak; the eta=1 peak is separate.
    eta = np.array([0.0, 0.02, 0.0, 1.0, 1.04, 1.0, -2.5])
    phi = np.array([np.pi - 0.03, -np.pi + 0.03, np.pi - 0.08, 0.4, 0.43, 0.36, 1.5])
    x, y, z = _xyz(eta, phi)
    cones = compute_cone_accounting(
        x, y, z, np.array([4, 3, 2, 5, 2, 1, 0.01]), np.array([9, 12, 10, 13, 11, 13, 9]),
        [], [], [np.pi - 0.02, 0.4], [np.pi / 2, 2 * np.arctan(np.exp(-1.0))], [0.5, 0.25], [8, 7],
        ConeAccountingParams(seed_energy_threshold=0.5, max_cones=2),
    )
    assert len(cones["eta"]) == 2
    assert np.isclose((cones["E_ecal_raw"] + cones["E_hcal_raw"]).sum(), 17.0)
    assert sorted(cones["N_acts"].tolist()) == [1.0, 1.0]

    hit_eta = np.array([0.01, 1.02, -2.0])
    hit_phi = np.array([-np.pi + 0.01, 0.4, 1.5])
    hx, hy, hz = _xyz(hit_eta, hit_phi)
    features = assign_cone_features(hx, hy, hz, cones)
    assert np.array_equal(features[:, 0], [1, 1, 0])
    assert np.isfinite(features).all()
    assert (features[:2, 1:].sum(axis=1) > 0).all()
    assert np.array_equal(features[2], np.zeros(6))


@pytest.mark.requiresdata
@pytest.mark.skipif(not DATASET_ROOT.is_dir(), reason="ColliderML higgs_portal v1 is not mounted")
def test_real_event_deficit_features():
    dataset = ColliderMLDataset(
        dataset_root=str(DATASET_ROOT), chunk_range=[0, 0], num_events=2,
        return_deficit_features=True, return_calohits=True, return_tracks=True,
        build_calohit_associations=False,
    )
    inputs, targets = dataset[1]
    names = ("incone", "ecal", "hcal", "trkpt", "ntrk", "ncells")
    for name in names:
        values = inputs[f"sihit_dc_{name}"]
        assert values.shape == inputs["sihit_x"].shape
        assert torch.isfinite(values).all()
    assert inputs["sihit_dc_incone"].sum() > 0

    cones = compute_cone_accounting(
        inputs["calohit_x"], inputs["calohit_y"], inputs["calohit_z"], inputs["calohit_total_energy"], inputs["calohit_detector"],
        targets["track_d0"], targets["track_z0"], targets["track_phi"], targets["track_theta"], targets["track_qop"],
        torch.diff(targets["track_sihit_indptr"]), dataset.deficit_params,
    )
    captured = (cones["E_ecal_raw"] + cones["E_hcal_raw"]).sum()
    assert captured > 0.5 * inputs["calohit_total_energy"].sum().item()
