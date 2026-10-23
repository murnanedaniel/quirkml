from pathlib import Path

import pytest
import torch

from ml_tracking.experiments.colliderml.data import ColliderMLDataModule, ColliderMLDataset
from ml_tracking.models.matcher import Matcher
from ml_tracking.models.task import ObjectHitMaskTask

DATASET_ROOT = Path("/global/cfs/cdirs/m4958/data/ColliderML/simulation/hard_scatter/higgs_portal/v1/parquet")


def _assert_bit_exact(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert torch.equal(actual.contiguous().reshape(-1).view(torch.uint8), expected.contiguous().reshape(-1).view(torch.uint8))


def _make_dataset_root(tmp_path: Path, chunks: range) -> Path:
    root = tmp_path / "dataset"
    collections = {
        "truth/particles": "truth.particles",
        "reco/tracker_hits": "reco.tracker_hits",
    }
    for directory, label in collections.items():
        path = root / directory
        path.mkdir(parents=True)
        for chunk in chunks:
            start = chunk * 1000
            (path / f"sample.{label}.events{start}-{start + 999}.parquet").touch()
    return root


def test_chunk_range_filters_shards_before_event_truncation(tmp_path, monkeypatch):
    root = _make_dataset_root(tmp_path, range(3, 8))
    monkeypatch.setattr(ColliderMLDataset, "_get_num_events_in_shard", lambda self, path: 2)

    dataset = ColliderMLDataset(
        dataset_root=str(root),
        chunk_range=[4, 6],
        num_events=3,
        return_calohits=False,
        return_tracks=False,
    )

    assert [ColliderMLDataset._event_range_token(path) for path in dataset.particle_shard_paths] == [
        "000000004000-000000004999",
        "000000005000-000000005999",
        "000000006000-000000006999",
    ]
    assert len(dataset.sample_index) == 6
    assert len(dataset) == 3


def test_event_range_token_skips_unrelated_parquet_names():
    assert ColliderMLDataset._event_range_token(Path("events1-2.parquet")) == "000000000001-000000000002"
    assert ColliderMLDataset._event_range_token(Path("sample.events1-2.parquet")) == "000000000001-000000000002"
    assert ColliderMLDataset._event_range_token(Path("metadata.parquet")) is None


def test_datamodule_rejects_overlapping_dataset_root_chunk_ranges(tmp_path):
    with pytest.raises(ValueError, match="train and val overlap"):
        ColliderMLDataModule(
            dataset_root=str(tmp_path),
            train_chunks=[0, 10],
            val_chunks=[10, 11],
            test_chunks=[12, 13],
        )


def test_legacy_datamodule_does_not_require_chunk_ranges(tmp_path):
    datamodule = ColliderMLDataModule(
        train_dir=str(tmp_path / "train"),
        val_dir=str(tmp_path / "val"),
        test_dir=str(tmp_path / "test"),
    )
    assert datamodule.train_chunks is None
    assert datamodule.val_chunks is None
    assert datamodule.test_chunks is None


def test_collate_pads_variable_event_axes_and_preserves_csr_per_event():
    def event(num_hits: int, sample_id: int):
        inputs = {
            "sihit_x": torch.arange(num_hits, dtype=torch.float32),
            "sihit_valid": torch.ones(num_hits, dtype=torch.bool),
        }
        targets = {
            "sihit_valid": inputs["sihit_valid"],
            "particle_valid": torch.tensor([True, False]),
            "particle_pt": torch.tensor([1.0, 0.0]),
            "particle_sihit_valid": torch.tensor([[True] * num_hits, [False] * num_hits]),
            "particle_sihit_indptr": torch.tensor([0, num_hits, num_hits]),
            "particle_sihit_indices": torch.arange(num_hits),
            "particle_sihit_shape": torch.tensor([2, num_hits]),
            "sample_id": torch.tensor(sample_id),
        }
        return inputs, targets

    samples = [event(2, 10), event(4, 11)]
    inputs, targets = ColliderMLDataset.collate_fn(samples)

    assert inputs["sihit_x"].shape == (2, 4)
    assert torch.equal(inputs["sihit_valid"], torch.tensor([[True, True, False, False], [True] * 4]))
    assert targets["particle_sihit_valid"].shape == (2, 2, 4)
    assert not targets["particle_sihit_valid"][0, :, 2:].any()
    assert isinstance(targets["particle_sihit_indices"], list)
    assert len(targets["particle_sihit_indices"]) == 2


def test_single_event_collate_matches_legacy_unsqueeze_bit_exactly():
    inputs = {
        "sihit_x": torch.tensor([1.25, -2.5]),
        "sihit_valid": torch.tensor([True, True]),
    }
    targets = {
        "sihit_valid": inputs["sihit_valid"],
        "particle_valid": torch.tensor([True, False]),
        "particle_sihit_valid": torch.tensor([[True, False], [False, False]]),
        "particle_sihit_indptr": torch.tensor([0, 1, 1]),
        "particle_sihit_indices": torch.tensor([0]),
        "particle_sihit_shape": torch.tensor([2, 2]),
        "sample_id": torch.tensor(7),
    }

    actual_inputs, actual_targets = ColliderMLDataset.collate_fn([(inputs, targets)])
    for key, value in inputs.items():
        _assert_bit_exact(actual_inputs[key], value.unsqueeze(0))
    for key, value in targets.items():
        _assert_bit_exact(actual_targets[key], value.unsqueeze(0))


@pytest.mark.requiresdata
@pytest.mark.skipif(not DATASET_ROOT.is_dir(), reason="ColliderML higgs_portal v1 is not mounted")
def test_higgs_portal_event_dense_targets_and_alignment():
    dataset = ColliderMLDataset(
        dataset_root=str(DATASET_ROOT),
        num_events=1,
        event_max_num_particles=256,
        particle_min_pt=0.5,
        particle_max_abs_eta=4.0,
        particle_hit_cuts={"all": {"min_num_sihit": 3}},
        particle_include_charged=True,
        particle_include_neutral=False,
        particle_require_primary=True,
        return_calohits=False,
        return_tracks=True,
    )

    sample = dataset[0]
    unbatched_inputs, unbatched_targets = sample
    inputs, targets = dataset.collate_fn([sample])
    # Regression guard for the pre-batching __getitem__ implementation.
    for key, value in unbatched_inputs.items():
        _assert_bit_exact(inputs[key], value.unsqueeze(0))
    for key, value in unbatched_targets.items():
        _assert_bit_exact(targets[key], value.unsqueeze(0))
    num_hits = inputs["sihit_particle_id"].shape[-1]
    assert targets["particle_valid"].shape == (1, 256)
    assert targets["particle_sihit_valid"].shape == (1, 256, num_hits)
    assert targets["particle_sihit_valid"].dtype == torch.bool
    assert targets["particle_sihit_indptr"].shape == (1, 257)
    assert targets["particle_sihit_shape"][0, 0] == 256

    valid = targets["particle_valid"][0]
    particle_ids = targets["particle_particle_id"][0]
    hit_particle_ids = inputs["sihit_particle_id"][0]
    expected = particle_ids[:, None] == hit_particle_ids[None, :]
    assert torch.equal(targets["particle_sihit_valid"][0, valid], expected[valid])
    assert not targets["particle_sihit_valid"][0, ~valid].any()
    padded_indptr = targets["particle_sihit_indptr"][0, valid.sum().item() :]
    assert torch.equal(padded_indptr, padded_indptr[-1].expand_as(padded_indptr))
    assert not torch.diff(targets["particle_sihit_indptr"][0])[~valid].any()

    assert targets["particle_primary"][0, valid].all()
    assert (targets["particle_charge"][0, valid] != 0).all()
    assert (targets["particle_pt"][0, valid] >= 0.5).all()
    assert (targets["particle_eta"][0, valid].abs() <= 4.0).all()
    assert (targets["particle_num_sihits"][0, valid] >= 3).all()
    assert torch.allclose(
        targets["particle_r"][0, valid],
        torch.hypot(targets["particle_vx"][0, valid], targets["particle_vy"][0, valid]),
    )

    particle_event_id = targets["particle_event_id"][0, valid].unique()
    hit_event_id = inputs["sihit_event_id"][0].reshape(-1).unique()
    track_event_id = targets["track_event_id"][0].reshape(-1).unique()
    assert particle_event_id.numel() == hit_event_id.numel() == track_event_id.numel() == 1
    assert particle_event_id.item() == hit_event_id.item() == track_event_id.item()


@pytest.mark.requiresdata
@pytest.mark.skipif(not DATASET_ROOT.is_dir(), reason="ColliderML higgs_portal v1 is not mounted")
def test_higgs_portal_batched_mask_costs_equal_individual_events():
    """Padded hits must not alter per-event costs or Hungarian assignments."""
    dataset = ColliderMLDataset(
        dataset_root=str(DATASET_ROOT),
        num_events=4,
        event_max_num_particles=32,
        particle_hit_cuts={"all": {"min_num_sihit": 3}},
        particle_include_neutral=False,
        particle_require_primary=True,
        return_calohits=False,
        return_tracks=False,
    )
    samples = [dataset[index] for index in range(4)]
    _, batched_targets = dataset.collate_fn(samples)
    num_hits = batched_targets["sihit_valid"].shape[1]

    generator = torch.Generator().manual_seed(1234)
    logits = torch.randn((4, 32, num_hits), generator=generator)
    task = ObjectHitMaskTask(
        name="flow_sihit_assignment",
        input_constituent="sihit",
        input_object="query",
        output_object="flow",
        target_object="particle",
        losses={"mask_dice": 2.0, "mask_focal": 100.0},
        costs={"mask_dice": 2.0, "mask_focal": 100.0},
        dim=4,
    )
    matcher = Matcher(default_solver="scipy", adaptive_solver=False)
    batched_cost = sum(task.cost({"flow_sihit_logit": logits}, batched_targets).values())
    batched_matches = matcher(batched_cost, batched_targets["particle_valid"])

    for index, (_, individual_targets) in enumerate(samples):
        n_hits = individual_targets["sihit_valid"].numel()
        individual_targets = {key: value.unsqueeze(0) for key, value in individual_targets.items()}
        individual_cost = sum(task.cost({"flow_sihit_logit": logits[index : index + 1, :, :n_hits]}, individual_targets).values())
        torch.testing.assert_close(batched_cost[index], individual_cost[0], rtol=1e-6, atol=1e-6)
        individual_match = matcher(individual_cost, individual_targets["particle_valid"])
        assert torch.equal(batched_matches[index], individual_match[0])
