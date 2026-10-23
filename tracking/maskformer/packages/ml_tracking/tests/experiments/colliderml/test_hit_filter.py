import polars as pl
import torch

from ml_tracking.experiments.colliderml.data import ColliderMLDataset


def test_sihit_target_uses_final_valid_particle_set():
    actual = ColliderMLDataset._build_sihit_on_valid_particle(
        torch.tensor([10, 20, 0, 30, 10]), torch.tensor([10, 30])
    )
    assert torch.equal(actual, torch.tensor([True, False, False, True, True]))


def test_filter_application_preserves_original_id_map(tmp_path):
    path = tmp_path / "filter.parquet"
    pl.DataFrame({
        "event_id": [7, 7, 7, 7],
        "hit_id": [2, 0, 3, 1],
        "keep_prob": [0.8, 0.1, 0.2, 0.9],
    }).write_parquet(path)
    dataset = ColliderMLDataset.__new__(ColliderMLDataset)
    dataset.hit_filter_file = path
    dataset.hit_filter_wp = 0.5
    dataset._hit_filter_by_event = dataset._read_hit_filter(path)
    keep = dataset._hit_filter_mask(7, 4)
    inputs = {
        "sihit_x": torch.tensor([10.0, 11.0, 12.0, 13.0]),
        "sihit_valid": torch.ones(4, dtype=torch.bool),
        "sihit_original_hit_id": torch.arange(4),
    }
    targets = {
        "sihit_valid": torch.ones(4, dtype=torch.bool),
        "sihit_on_valid_particle": torch.tensor([False, True, True, False]),
    }
    dataset._filter_sihits_inplace(inputs, targets, keep)
    assert torch.equal(keep, torch.tensor([False, True, True, False]))
    assert torch.equal(inputs["sihit_x"], torch.tensor([11.0, 12.0]))
    assert torch.equal(inputs["sihit_original_hit_id"], torch.tensor([1, 2]))
    assert torch.equal(targets["sihit_on_valid_particle"], torch.tensor([True, True]))
