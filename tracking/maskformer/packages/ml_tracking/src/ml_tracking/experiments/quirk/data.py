"""Quirk tracker-hit dataset for the MaskFormer track finder.

Reads the ColliderML Release-1 layout (``<root>/truth/particles/*.parquet`` and
``<root>/reco/tracker_hits/*.parquet``, one row per event, per-event list
columns). Derived from the ColliderML LRT data module in this package; trimmed
to tracker hits only so that the quirk target logic is easy to audit.

Target definition (per event):

* ordinary particles: charged, ``primary``, pT > ``particle_min_pt``,
  |eta| < ``particle_max_abs_eta``, >= ``min_num_sihit`` tracker hits;
* quirks (|pdg| in ``quirk_pdg_ids``): kept whenever they leave
  >= ``quirk_min_num_sihit`` hits. No pT, eta, charge or ``primary`` cut is
  applied to them, and they are placed first in the pT-ordered truncation to
  ``event_max_num_particles`` so they can never be dropped (an event with more
  quirks than slots raises).

Splits are by ``event_id`` (inclusive ranges), not by file or row: rows inside a
file are not in event_id order.

``hit_id`` convention: hits are never filtered or reordered here, so the model's
hit axis index equals the row position of the hit in the event's per-hit list
columns (``sihit_original_hit_id`` carries it explicitly for the writer).
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import awkward as ak
import numpy as np
import pyarrow.parquet as pq
import torch
from lightning import LightningDataModule
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

PARTICLE_COLUMNS = ("event_id", "particle_id", "pdg_id", "charge", "px", "py", "pz", "vx", "vy", "vz", "primary")
HIT_POSITION_COLUMNS = ("event_id", "x", "y", "z", "particle_id", "layer_id", "volume_id")
# Cluster-shape columns available in the quirk R1 tracker_hits tables.
HIT_CLUSTER_COLUMNS = ("size_loc0", "size_loc1", "n_channels", "sum_activation", "eta_angle", "phi_angle")


def _normalize_range(name: str, value) -> tuple[int, int] | None:
    if value is None:
        return None
    if len(value) != 2:
        raise ValueError(f"{name} must be an inclusive [start, end] event_id pair")
    start, end = int(value[0]), int(value[1])
    if start < 0 or end < start:
        raise ValueError(f"{name} must satisfy 0 <= start <= end, got {value}")
    return start, end


class QuirkTrackingDataset(Dataset):
    def __init__(
        self,
        dataset_root: str,
        event_range: tuple[int, int],
        num_events: int = -1,
        event_max_num_particles: int = 128,
        particle_min_pt: float = 0.5,
        particle_max_abs_eta: float = 4.0,
        min_num_sihit: int = 3,
        particle_require_primary: bool = True,
        include_quirks: bool = True,
        quirk_pdg_ids: tuple[int, ...] = (10000100,),
        quirk_min_num_sihit: int = 1,
        cluster_features: bool = False,
        row_group_cache_size: int = 64,
    ):
        super().__init__()
        self.root = Path(dataset_root)
        self.event_range = _normalize_range("event_range", event_range)
        self.event_max_num_particles = int(event_max_num_particles)
        self.particle_min_pt = float(particle_min_pt)
        self.particle_max_abs_eta = float(particle_max_abs_eta)
        self.min_num_sihit = int(min_num_sihit)
        self.particle_require_primary = bool(particle_require_primary)
        self.include_quirks = bool(include_quirks)
        self.quirk_pdg_ids = np.asarray([abs(int(p)) for p in quirk_pdg_ids], dtype=np.int64)
        self.quirk_min_num_sihit = int(quirk_min_num_sihit)
        self.cluster_features = bool(cluster_features)
        self.hit_columns = HIT_POSITION_COLUMNS + (HIT_CLUSTER_COLUMNS if cluster_features else ())
        self.row_group_cache_size = int(row_group_cache_size)
        self._cache: OrderedDict[tuple[str, int], ak.Array] = OrderedDict()
        self.num_quirks_dropped = 0  # quirks below quirk_min_num_sihit (cannot be targets: no hits)

        particle_dir = self.root / "truth" / "particles"
        hit_dir = self.root / "reco" / "tracker_hits"
        for d in (particle_dir, hit_dir):
            if not d.is_dir():
                raise ValueError(f"Missing directory {d}")

        # event_id -> (path, row_group, row_in_group) for both collections.
        self.particle_loc = self._index(sorted(particle_dir.glob("*.parquet")))
        self.hit_loc = self._index(sorted(hit_dir.glob("*.parquet")))
        start, end = self.event_range
        ids = sorted(e for e in self.particle_loc if start <= e <= end)
        missing = [e for e in ids if e not in self.hit_loc]
        if missing:
            raise ValueError(f"{len(missing)} events have particles but no tracker hits, e.g. {missing[:5]}")
        if not ids:
            raise ValueError(f"No events with event_id in {self.event_range} under {self.root}")
        if 0 < num_events < len(ids):
            ids = ids[:num_events]
        self.event_ids = ids
        print(f"[QuirkTrackingDataset] {self.root.name}: {len(ids)} events in event_id range {self.event_range}")

    @staticmethod
    def _index(paths: list[Path]) -> dict[int, tuple[str, int, int]]:
        loc: dict[int, tuple[str, int, int]] = {}
        for path in paths:
            pf = pq.ParquetFile(path)
            for rg in range(pf.metadata.num_row_groups):
                event_ids = pf.read_row_group(rg, columns=["event_id"])["event_id"].to_numpy()
                for row, event_id in enumerate(event_ids):
                    event_id = int(event_id)
                    if event_id in loc:
                        raise ValueError(f"Duplicate event_id {event_id} in {path} and {loc[event_id][0]}")
                    loc[event_id] = (str(path), rg, row)
        return loc

    def _read(self, loc: tuple[str, int, int], columns: tuple[str, ...]) -> ak.Record:
        path, rg, row = loc
        key = (path, rg, columns)
        if key not in self._cache:
            table = pq.ParquetFile(path).read_row_group(rg, columns=list(columns))
            self._cache[key] = ak.from_arrow(table)
            if len(self._cache) > self.row_group_cache_size:
                self._cache.popitem(last=False)
        self._cache.move_to_end(key)
        return self._cache[key][row]

    def __len__(self) -> int:
        return len(self.event_ids)

    def is_quirk(self, pdg: np.ndarray) -> np.ndarray:
        return np.isin(np.abs(pdg.astype(np.int64)), self.quirk_pdg_ids)

    def select_targets(self, particles: dict[str, np.ndarray], hit_pid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (target particle row indices in priority order, nhits per particle, is_quirk)."""
        pid = particles["particle_id"].astype(np.int64)
        px, py, pz = particles["px"], particles["py"], particles["pz"]
        pt = np.hypot(px, py)
        eta = np.arcsinh(pz / np.maximum(pt, 1e-12))
        uniq, counts = np.unique(hit_pid, return_counts=True)
        nhits = np.zeros(len(pid), dtype=np.int64)
        pos = np.searchsorted(uniq, pid)
        ok = (pos < len(uniq)) & (uniq[np.minimum(pos, len(uniq) - 1)] == pid)
        nhits[ok] = counts[pos[ok]]

        quirk = self.is_quirk(particles["pdg_id"])
        sel = (particles["charge"] != 0) & (pt > self.particle_min_pt) & (np.abs(eta) < self.particle_max_abs_eta) & (nhits >= self.min_num_sihit)
        if self.particle_require_primary:
            sel &= particles["primary"].astype(bool)
        if self.include_quirks:
            quirk_ok = quirk & (nhits >= self.quirk_min_num_sihit)
            self.num_quirks_dropped += int((quirk & ~quirk_ok).sum())
            sel = (sel & ~quirk) | quirk_ok
        else:
            sel &= ~quirk

        idx = np.nonzero(sel)[0]
        # Quirks first, then descending pT: truncation can only drop soft non-quirks.
        order = np.lexsort((-pt[idx], ~quirk[idx]))
        idx = idx[order]
        n_quirk = int(quirk[idx].sum())
        if n_quirk > self.event_max_num_particles:
            raise ValueError(f"{n_quirk} quirk targets exceed event_max_num_particles={self.event_max_num_particles}")
        return idx[: self.event_max_num_particles], nhits, quirk

    def __getitem__(self, i: int):
        event_id = self.event_ids[int(i)]
        p = self._read(self.particle_loc[event_id], PARTICLE_COLUMNS)
        h = self._read(self.hit_loc[event_id], self.hit_columns)
        if int(p["event_id"]) != event_id or int(h["event_id"]) != event_id:
            raise ValueError(f"event_id mismatch at {event_id}")

        particles = {k: np.asarray(p[k]) for k in PARTICLE_COLUMNS if k != "event_id"}
        hits = {k: np.asarray(h[k]) for k in self.hit_columns if k != "event_id"}
        hit_pid = hits["particle_id"].astype(np.int64)
        n_hits = len(hit_pid)

        idx, nhits, quirk = self.select_targets(particles, hit_pid)
        n_all = int(len(idx))

        # ---- inputs (positions in metres, as in the LRT pipeline)
        x = torch.from_numpy(hits["x"].astype(np.float32)) * 1e-3
        y = torch.from_numpy(hits["y"].astype(np.float32)) * 1e-3
        z = torch.from_numpy(hits["z"].astype(np.float32)) * 1e-3
        r = torch.sqrt(x**2 + y**2)
        s = torch.sqrt(r**2 + z**2).clamp_min(1e-12)
        inputs = {
            "sihit_x": x,
            "sihit_y": y,
            "sihit_z": z,
            "sihit_r": r,
            "sihit_s": s,
            "sihit_eta": torch.arctanh((z / s).clamp(-1 + 1e-7, 1 - 1e-7)),
            "sihit_theta": torch.arccos((z / s).clamp(-1.0, 1.0)),
            "sihit_phi": torch.arctan2(y, x),
            "sihit_layer_id": torch.from_numpy(hits["layer_id"].astype(np.float32)),
            "sihit_volume_id": torch.from_numpy(hits["volume_id"].astype(np.float32)),
            "sihit_valid": torch.ones(n_hits, dtype=torch.bool),
            "sihit_original_hit_id": torch.arange(n_hits, dtype=torch.int64),
        }
        if self.cluster_features:
            # Counts / charge are heavy-tailed: log1p. Angles are passed as stored.
            for col in ("size_loc0", "size_loc1", "n_channels", "sum_activation"):
                inputs[f"sihit_log_{col}"] = torch.from_numpy(np.log1p(hits[col].astype(np.float32)))
            for col in ("eta_angle", "phi_angle"):
                inputs[f"sihit_{col}"] = torch.from_numpy(hits[col].astype(np.float32))

        # ---- targets (fixed size event_max_num_particles)
        P = self.event_max_num_particles
        tgt_pid = particles["particle_id"].astype(np.int64)[idx]
        valid = torch.zeros(P, dtype=torch.bool)
        valid[:n_all] = True
        mask = torch.zeros(P, n_hits, dtype=torch.bool)
        if n_all:
            mask[:n_all] = torch.from_numpy(tgt_pid[:, None] == hit_pid[None, :])
        is_quirk = torch.zeros(P, dtype=torch.bool)
        is_quirk[:n_all] = torch.from_numpy(quirk[idx])
        pt = torch.zeros(P)
        pt[:n_all] = torch.from_numpy(np.hypot(particles["px"], particles["py"])[idx].astype(np.float32))
        targets = {
            "event_id": torch.tensor(event_id, dtype=torch.int64),
            "particle_valid": valid,
            "particle_sihit_valid": mask,
            "particle_is_quirk": is_quirk,
            "particle_pt": pt,
            "particle_particle_id": torch.from_numpy(np.pad(tgt_pid, (0, P - n_all), constant_values=-1)),
            "sihit_valid": inputs["sihit_valid"],
        }
        return inputs, targets

    @staticmethod
    def collate_fn(batch):
        inputs_list, targets_list = zip(*batch, strict=True)
        inputs = {}
        for key in inputs_list[0]:
            values = [b[key] for b in inputs_list]
            inputs[key] = pad_sequence(values, batch_first=True, padding_value=False if values[0].dtype is torch.bool else 0)
        targets = {}
        for key in targets_list[0]:
            values = [b[key] for b in targets_list]
            if key == "particle_sihit_valid":
                targets[key] = pad_sequence([v.transpose(0, 1) for v in values], batch_first=True, padding_value=False).transpose(1, 2)
            elif key == "sihit_valid":
                targets[key] = pad_sequence(values, batch_first=True, padding_value=False)
            else:
                targets[key] = torch.stack(values)
        return inputs, targets


class QuirkDataModule(LightningDataModule):
    def __init__(
        self,
        dataset_root: str,
        train_events: list[int],
        val_events: list[int],
        test_events: list[int],
        num_train: int = -1,
        num_val: int = -1,
        num_test: int = -1,
        allow_split_overlap: bool = False,
        batch_size: int = 1,
        num_workers: int = 4,
        pin_memory: bool = True,
        event_max_num_particles: int = 128,
        particle_min_pt: float = 0.5,
        particle_max_abs_eta: float = 4.0,
        min_num_sihit: int = 3,
        particle_require_primary: bool = True,
        include_quirks: bool = True,
        quirk_pdg_ids: list[int] | None = None,
        quirk_min_num_sihit: int = 1,
        cluster_features: bool = False,
        row_group_cache_size: int = 64,
    ):
        super().__init__()
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.dataset_root = dataset_root
        self.ranges = {
            "train": _normalize_range("train_events", train_events),
            "val": _normalize_range("val_events", val_events),
            "test": _normalize_range("test_events", test_events),
        }
        if not allow_split_overlap:
            names = list(self.ranges)
            for a_i, a in enumerate(names):
                for b in names[a_i + 1 :]:
                    ra, rb = self.ranges[a], self.ranges[b]
                    if max(ra[0], rb[0]) <= min(ra[1], rb[1]):
                        raise ValueError(f"event ranges for {a} {ra} and {b} {rb} overlap")
        self.nums = {"train": num_train, "val": num_val, "test": num_test}
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.kwargs = {
            "event_max_num_particles": event_max_num_particles,
            "particle_min_pt": particle_min_pt,
            "particle_max_abs_eta": particle_max_abs_eta,
            "min_num_sihit": min_num_sihit,
            "particle_require_primary": particle_require_primary,
            "include_quirks": include_quirks,
            "quirk_pdg_ids": tuple(quirk_pdg_ids or [10000100]),
            "quirk_min_num_sihit": quirk_min_num_sihit,
            "cluster_features": cluster_features,
            "row_group_cache_size": row_group_cache_size,
        }

    def _make(self, split: str) -> QuirkTrackingDataset:
        return QuirkTrackingDataset(self.dataset_root, self.ranges[split], num_events=self.nums[split], **self.kwargs)

    def setup(self, stage: str):
        if stage == "fit":
            self.train_dset = self._make("train")
            self.val_dset = self._make("val")
        if stage == "validate":
            self.val_dset = self._make("val")
        if stage in {"test", "predict"}:
            self.test_dset = self._make("test")

    def _loader(self, dataset, shuffle: bool):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            collate_fn=QuirkTrackingDataset.collate_fn,
            num_workers=self.num_workers,
            shuffle=shuffle,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
            **({"prefetch_factor": 4} if self.num_workers > 0 else {}),
        )

    def train_dataloader(self):
        return self._loader(self.train_dset, shuffle=True)

    def val_dataloader(self):
        return self._loader(self.val_dset, shuffle=False)

    def test_dataloader(self):
        return self._loader(self.test_dset, shuffle=False)

    def predict_dataloader(self):
        return self._loader(self.test_dset, shuffle=False)
