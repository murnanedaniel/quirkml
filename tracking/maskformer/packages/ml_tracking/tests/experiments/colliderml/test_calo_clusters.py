from pathlib import Path

import numpy as np
import pytest
import torch

from ml_tracking.experiments.colliderml.calo_clusters import CALO_CLUSTER_FIELDS, cluster_calo_cells
from ml_tracking.experiments.colliderml.data import ColliderMLDataset

DATASET_ROOT = Path("/global/cfs/cdirs/m4958/data/ColliderML/simulation/hard_scatter/higgs_portal/v1/parquet")


def _xyz(r_mm, eta, phi):
    return r_mm * np.cos(phi), r_mm * np.sin(phi), r_mm * np.sinh(eta)


def test_two_clusters_have_expected_energy_and_centroids():
    # Two compact, well-separated three-cell blobs at equal detector depth.
    x1, y1, z1 = _xyz(1000.0, np.array([0.0, 0.01, -0.01]), np.array([0.0, 0.01, -0.01]))
    x2, y2, z2 = _xyz(2000.0, np.array([1.0, 1.01, 0.99]), np.array([1.0, 1.01, 0.99]))
    energy = np.array([1.0, 2.0, 1.0, 2.0, 3.0, 5.0])
    result = cluster_calo_cells(
        np.r_[x1, x2], np.r_[y1, y2], np.r_[z1, z2], energy, np.array([10] * 3 + [13] * 3),
        eps=0.05, min_samples=3, min_cluster_energy=0.0,
    )

    assert tuple(result) == CALO_CLUSTER_FIELDS
    assert len(result["x"]) == 2
    np.testing.assert_allclose(np.exp(result["log_energy"]), [10.0, 4.0], rtol=1e-6)
    expected_x2 = np.average(x2 * 1e-3, weights=energy[3:])
    expected_y2 = np.average(y2 * 1e-3, weights=energy[3:])
    np.testing.assert_allclose([result["x"][0], result["y"][0]], [expected_x2, expected_y2], rtol=1e-6)
    np.testing.assert_allclose(result["ecal_fraction"], [0.0, 1.0])


def test_phi_wraparound_cells_form_one_cluster():
    phi = np.array([np.pi - 0.01, -np.pi + 0.01, np.pi - 0.02])
    x, y, z = _xyz(1000.0, np.zeros(3), phi)
    result = cluster_calo_cells(x, y, z, np.ones(3), np.full(3, 10), eps=0.05, min_samples=3, min_cluster_energy=0)
    assert len(result["phi"]) == 1
    assert abs(abs(result["phi"][0]) - np.pi) < 0.02
    assert result["sigma_phi"][0] < 0.02


@pytest.mark.requiresdata
@pytest.mark.skipif(not DATASET_ROOT.is_dir(), reason="ColliderML higgs_portal v1 is not mounted")
def test_real_event_emits_finite_calo_cluster_inputs():
    dataset = ColliderMLDataset(
        dataset_root=str(DATASET_ROOT), chunk_range=[0, 0], num_events=1,
        return_calohits=False, return_calo_clusters=True, return_tracks=False,
    )
    inputs, _ = dataset[0]
    assert inputs["calocls_valid"].dtype == torch.bool
    assert 0 < inputs["calocls_valid"].shape[-1] < 200
    for field in CALO_CLUSTER_FIELDS:
        assert inputs[f"calocls_{field}"].shape == inputs["calocls_valid"].shape
        assert torch.isfinite(inputs[f"calocls_{field}"]).all()
