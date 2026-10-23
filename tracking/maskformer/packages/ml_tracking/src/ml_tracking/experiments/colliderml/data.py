import re
from collections import OrderedDict
from pathlib import Path
from time import perf_counter
from typing import ClassVar

import awkward as ak
import numpy as np
import pyarrow.parquet as pq
import scipy.sparse as sp
import torch
from lightning import LightningDataModule
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from ml_tracking.experiments.colliderml.calo_clusters import (
    DEFAULT_EPS,
    DEFAULT_MIN_CLUSTER_ENERGY,
    DEFAULT_MIN_SAMPLES,
    cluster_calo_cells,
)
from ml_tracking.experiments.colliderml.deficit_features import (
    ConeAccountingParams,
    assign_cone_features,
    compute_cone_accounting,
)


class ColliderMLDataset(Dataset):
    CALO_SUBSYSTEMS = ("ecb", "ece", "hcb", "hce")
    CALO_SUBSYSTEM_DETECTOR_IDS: ClassVar[dict[str, np.ndarray]] = {
        "ecb": np.array([10], dtype=np.int64),
        "ece": np.array([9, 11], dtype=np.int64),
        "hcb": np.array([13], dtype=np.int64),
        "hce": np.array([12, 14], dtype=np.int64),
    }
    CALO_ECAL_DETECTOR_IDS = np.array([9, 10, 11], dtype=np.int64)
    CALO_HCAL_DETECTOR_IDS = np.array([12, 13, 14], dtype=np.int64)
    CALO_SUBSYSTEM_CALIBRATION: ClassVar[dict[str, float]] = {
        "ecb": 37.5,
        "ece": 38.7,
        "hcb": 45.0,
        "hce": 46.9,
    }
    CALO_GROUP_DETECTOR_IDS: ClassVar[dict[str, np.ndarray]] = {
        "ecalhits": CALO_ECAL_DETECTOR_IDS,
        "hcalhits": CALO_HCAL_DETECTOR_IDS,
    }
    PARTICLE_HIT_CUT_CLASS_TO_MASK: ClassVar[dict[str, str]] = {
        "all": "particle_valid",
        "charged_hadron": "particle_is_charged_hadron",
        "neutral_hadron": "particle_is_neutral_hadron",
        "electron": "particle_is_electron",
        "photon": "particle_is_photon",
        "muon": "particle_is_muon",
        "tau": "particle_is_tau",
        "neutrino": "particle_is_neutrino",
        "other": "particle_is_other",
    }
    PARTICLE_HIT_CUT_KEYS = (
        "min_num_sihit",
        "min_num_ecal",
        "min_num_hcal",
    )
    PARTICLE_HIT_CUT_DEFAULTS: ClassVar[dict[str, int]] = {
        "min_num_sihit": 0,
        "min_num_ecal": 0,
        "min_num_hcal": 0,
    }
    # Sparse associations are diagnostic metadata and are not consumed by the
    # model or its losses.  For multi-event batches they remain one tensor per
    # event, avoiding meaningless padding of CSR index arrays.
    CSR_SUFFIXES = ("_indptr", "_indices", "_shape")

    def __init__(
        self,
        dirpath: str | None = None,
        dataset_root: str | None = None,
        chunk_range: list[int] | tuple[int, int] | None = None,
        num_events: int = -1,
        event_max_num_particles: int = 1024,
        particle_min_pt: float = 0.5,
        particle_max_abs_eta: float = 4.0,
        particle_hit_cuts: dict[str, dict[str, int]] | None = None,
        particle_include_charged: bool = True,
        particle_include_neutral: bool = True,
        particle_require_primary: bool = False,
        return_calohits: bool = True,
        return_calo_clusters: bool = False,
        calo_cluster_eps: float = DEFAULT_EPS,
        calo_cluster_min_samples: int = DEFAULT_MIN_SAMPLES,
        calo_cluster_min_energy: float = DEFAULT_MIN_CLUSTER_ENERGY,
        return_tracks: bool = True,
        return_deficit_features: bool = False,
        deficit_grid_size: float = 0.1,
        deficit_cone_radius: float = 0.4,
        deficit_seed_energy_threshold: float = 0.01,
        deficit_max_cones: int = 12,
        deficit_max_abs_eta: float = 5.0,
        event_type: str = "ttbar",
        build_calohit_associations: bool = True,
        hit_filter_file: str | None = None,
        hit_filter_wp: float = 0.0,
        debug: bool = False,
    ):
        super().__init__()

        if dataset_root is None and dirpath is None:
            raise ValueError("One of dataset_root or dirpath must be provided")
        self.dirpath = Path(dataset_root if dataset_root is not None else dirpath)
        self.chunk_range = self._normalize_chunk_range(chunk_range)
        if dataset_root is not None:
            self.collection_dirs = {
                "particles": self.dirpath / "truth" / "particles",
                "tracker_hits": self.dirpath / "reco" / "tracker_hits",
                "calo_hits": self.dirpath / "reco" / "calo_hits",
                "tracks": self.dirpath / "reco" / "tracks",
            }
            self._pair_shards_by_event_range = True
        else:
            dataset_prefix = f"{event_type}_pu200"
            self.collection_dirs = {
                "particles": self.dirpath / f"{dataset_prefix}_particles",
                "tracker_hits": self.dirpath / f"{dataset_prefix}_tracker_hits",
                "calo_hits": self.dirpath / f"{dataset_prefix}_calo_hits",
                "tracks": self.dirpath / f"{dataset_prefix}_tracks",
            }
            self._pair_shards_by_event_range = False

        self.particle_hit_cuts = self._normalize_particle_hit_cuts(particle_hit_cuts)
        self._requires_calohits_for_hit_cuts = any(cuts["min_num_ecal"] > 0 or cuts["min_num_hcal"] > 0 for cuts in self.particle_hit_cuts.values())

        required_collections = {"particles", "tracker_hits"}
        if return_calohits or return_calo_clusters or return_deficit_features or self._requires_calohits_for_hit_cuts:
            required_collections.add("calo_hits")
        if return_tracks or return_deficit_features:
            required_collections.add("tracks")

        missing_dirs = [name for name in sorted(required_collections) if not self.collection_dirs[name].is_dir()]
        if missing_dirs:
            msg = f"Missing required dataset directories for '{event_type}': {missing_dirs}. Expected these under {self.dirpath}."
            raise ValueError(msg)

        # Use particle shards as the reference and keep only shard names that are
        # available in every required collection.
        self._collection_shards = self._get_shared_shards(required_collections)
        self.particle_shard_paths = [self._collection_shards["particles"][key] for key in sorted(self._collection_shards["particles"])]
        if not self.particle_shard_paths:
            msg = f"No shared parquet shards found for '{event_type}' under {self.dirpath}"
            raise ValueError(msg)

        # Cache parquet row-group metadata and a few decoded row-groups.
        # Must be initialised before counting events in shards.
        self._row_group_starts: dict[str, np.ndarray] = {}
        self._row_group_cache: OrderedDict[tuple[str, int], ak.Array] = OrderedDict()
        self._row_group_cache_size = 8

        # Build a dense sample index: each sample_id maps to (shard index, row/event index in shard).
        self.sample_index = []
        for shard_idx, particle_path in enumerate(self.particle_shard_paths):
            num_events_in_shard = self._get_num_events_in_shard(particle_path)
            self.sample_index.extend((shard_idx, event_idx) for event_idx in range(num_events_in_shard))

        num_events_available = len(self.sample_index)

        if num_events > num_events_available:
            msg = f"Requested {num_events} events, but only {num_events_available} are available in the directory {dirpath}."
            raise ValueError(msg)
        if num_events_available == 0:
            msg = f"No events found in {dirpath}"
            raise ValueError(msg)
        if num_events < 0:
            num_events = num_events_available

        print(f"Found {num_events_available} available events, using {num_events} events.")

        # Sample ID is an integer that can uniquely identify each event/sample, used for picking out events during eval etc
        self.num_events = num_events
        self.sample_ids = list(range(num_events))

        # Particle level cuts
        self.particle_min_pt = particle_min_pt
        self.particle_max_abs_eta = particle_max_abs_eta
        self.particle_include_charged = particle_include_charged
        self.particle_include_neutral = particle_include_neutral
        self.particle_require_primary = particle_require_primary
        if event_max_num_particles <= 0:
            raise ValueError("event_max_num_particles must be positive")
        self.event_max_num_particles = event_max_num_particles
        self.dropped_particle_count = 0
        self._warned_particle_truncation = False

        # Whether to return calo/ACTS track collections
        self.return_calohits = return_calohits
        self.return_calo_clusters = return_calo_clusters
        self.calo_cluster_eps = calo_cluster_eps
        self.calo_cluster_min_samples = calo_cluster_min_samples
        self.calo_cluster_min_energy = calo_cluster_min_energy
        self.return_tracks = return_tracks
        self.return_deficit_features = return_deficit_features
        self.deficit_params = ConeAccountingParams(
            grid_size=deficit_grid_size,
            cone_radius=deficit_cone_radius,
            seed_energy_threshold=deficit_seed_energy_threshold,
            max_cones=deficit_max_cones,
            max_abs_eta=deficit_max_abs_eta,
        )
        self.event_type = event_type
        self.build_calohit_associations = build_calohit_associations
        if not 0.0 <= hit_filter_wp <= 1.0:
            raise ValueError("hit_filter_wp must be in [0, 1]")
        self.hit_filter_file = Path(hit_filter_file) if hit_filter_file else None
        self.hit_filter_wp = hit_filter_wp
        self._hit_filter_by_event = self._read_hit_filter(self.hit_filter_file) if self.hit_filter_file else None
        self.debug = debug

    @staticmethod
    def _read_hit_filter(path: Path) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        if not path.is_file():
            raise ValueError(f"Hit-filter parquet does not exist: {path}")
        table = pq.read_table(path, columns=["event_id", "hit_id", "keep_prob"])
        event_ids = table["event_id"].to_numpy()
        hit_ids = table["hit_id"].to_numpy().astype(np.int64, copy=False)
        probabilities = table["keep_prob"].to_numpy().astype(np.float32, copy=False)
        result = {}
        for event_id in np.unique(event_ids):
            selected = event_ids == event_id
            ids = hit_ids[selected]
            probs = probabilities[selected]
            if len(np.unique(ids)) != len(ids):
                raise ValueError(f"Duplicate hit_id values in hit filter for event {int(event_id)}")
            result[int(event_id)] = (ids, probs)
        return result

    def _hit_filter_mask(self, event_id: int, num_hits: int) -> torch.Tensor:
        if self._hit_filter_by_event is None:
            return torch.ones(num_hits, dtype=torch.bool)
        if event_id not in self._hit_filter_by_event:
            raise KeyError(f"Event {event_id} is missing from hit-filter parquet {self.hit_filter_file}")
        hit_ids, probabilities = self._hit_filter_by_event[event_id]
        if len(hit_ids) != num_hits or not np.array_equal(np.sort(hit_ids), np.arange(num_hits)):
            raise ValueError(
                f"Hit-filter rows for event {event_id} must contain each original positional hit_id "
                f"exactly once (expected {num_hits}, found {len(hit_ids)})"
            )
        ordered = np.empty(num_hits, dtype=np.float32)
        ordered[hit_ids] = probabilities
        return torch.from_numpy(ordered >= self.hit_filter_wp)

    @staticmethod
    def _filter_sihits_inplace(inputs: dict[str, torch.Tensor], targets: dict[str, torch.Tensor], keep: torch.Tensor) -> None:
        for mapping in (inputs, targets):
            for key, value in list(mapping.items()):
                if key.startswith("sihit_") and isinstance(value, torch.Tensor) and value.ndim and value.shape[0] == keep.numel():
                    mapping[key] = value[keep]

    @staticmethod
    def _build_sihit_on_valid_particle(
        sihit_particle_ids: torch.Tensor, valid_particle_ids: torch.Tensor
    ) -> torch.Tensor:
        """Label hits using the final, cut-passing particle set."""
        return torch.isin(sihit_particle_ids, valid_particle_ids)

    @staticmethod
    def _coerce_non_negative_int(name: str, value) -> int:
        if isinstance(value, bool):
            msg = f"Cut '{name}' must be numeric, got bool."
            raise TypeError(msg)
        as_float = float(value)
        if as_float < 0:
            msg = f"Cut '{name}' must be non-negative, got {value}."
            raise ValueError(msg)
        as_int = int(as_float)
        if as_int != as_float:
            msg = f"Cut '{name}' must be an integer-like value, got {value}."
            raise ValueError(msg)
        return as_int

    @classmethod
    def _normalize_particle_hit_cuts(cls, particle_hit_cuts: dict[str, dict[str, int]] | None) -> dict[str, dict[str, int]]:
        if particle_hit_cuts is None:
            return {}
        if not isinstance(particle_hit_cuts, dict):
            msg = "particle_hit_cuts must be a dict like {'electron': {'min_num_sihit': 6, ...}}."
            raise TypeError(msg)

        normalized: dict[str, dict[str, int]] = {}
        for particle_type, cut_cfg in particle_hit_cuts.items():
            class_key = str(particle_type).strip().lower()
            if class_key not in cls.PARTICLE_HIT_CUT_CLASS_TO_MASK:
                valid = ", ".join(sorted(set(cls.PARTICLE_HIT_CUT_CLASS_TO_MASK)))
                msg = f"Unknown particle_hit_cuts class '{particle_type}'. Valid classes: {valid}"
                raise ValueError(msg)
            if not isinstance(cut_cfg, dict):
                msg = f"particle_hit_cuts['{particle_type}'] must be a dict."
                raise TypeError(msg)

            mask_key = cls.PARTICLE_HIT_CUT_CLASS_TO_MASK[class_key]
            if mask_key not in normalized:
                normalized[mask_key] = cls.PARTICLE_HIT_CUT_DEFAULTS.copy()

            for cut_name, cut_value in cut_cfg.items():
                cut_key = str(cut_name).strip().lower()
                if cut_key not in cls.PARTICLE_HIT_CUT_KEYS:
                    valid = ", ".join(sorted(cls.PARTICLE_HIT_CUT_KEYS))
                    msg = f"Unknown cut key '{cut_name}' for '{particle_type}'. Valid keys: {valid}"
                    raise ValueError(msg)
                cut_threshold = cls._coerce_non_negative_int(cut_name, cut_value)
                normalized[mask_key][cut_key] = max(normalized[mask_key][cut_key], cut_threshold)

        return {mask_key: cfg for mask_key, cfg in normalized.items() if any(cfg[cut_key] > 0 for cut_key in cls.PARTICLE_HIT_CUT_KEYS)}

    def __len__(self):
        return int(self.num_events)

    @staticmethod
    def _normalize_chunk_range(chunk_range: list[int] | tuple[int, int] | None) -> tuple[int, int] | None:
        if chunk_range is None:
            return None
        if len(chunk_range) != 2:
            raise ValueError("chunk_range must be an inclusive [start, end] pair")
        start, end = chunk_range
        if isinstance(start, bool) or isinstance(end, bool) or not isinstance(start, int) or not isinstance(end, int):
            raise TypeError("chunk_range values must be integers")
        if start < 0 or end < start:
            raise ValueError("chunk_range must satisfy 0 <= start <= end")
        return start, end

    @staticmethod
    def _event_range_token(path: Path) -> str | None:
        match = re.search(r"(?:^|\.)events(\d+-\d+)\.parquet$", path.name)
        if match is None:
            return None
        start, end = (int(value) for value in match.group(1).split("-"))
        return f"{start:012d}-{end:012d}"

    def _get_shared_shards(self, required_collections: set[str]) -> dict[str, dict[str, Path]]:
        shards: dict[str, dict[str, Path]] = {}
        for collection in required_collections:
            paths = sorted(self.collection_dirs[collection].glob("*.parquet"))
            keyed = {}
            for path in paths:
                key = self._event_range_token(path) if self._pair_shards_by_event_range else path.name
                if key is not None:
                    keyed[key] = path
            shards[collection] = keyed

        if self._pair_shards_by_event_range and self.chunk_range is not None:
            chunk_start, chunk_end = self.chunk_range
            unfiltered = shards
            shards = {
                collection: {
                    key: path for key, path in mapping.items()
                    if chunk_start <= int(key.split("-", maxsplit=1)[0]) // 1000 <= chunk_end
                }
                for collection, mapping in shards.items()
            }
            ignored_missing = {
                collection: sorted(set(unfiltered["particles"]) - set(mapping) - set(shards["particles"]))
                for collection, mapping in unfiltered.items()
            }
            ignored_missing = {collection: keys for collection, keys in ignored_missing.items() if keys}
            if ignored_missing:
                print(f"Ignoring unpaired shards outside chunk_range {self.chunk_range}: {ignored_missing}")

        shared_keys = set(shards["particles"])
        for collection in required_collections - {"particles"}:
            shared_keys &= set(shards[collection])
        if not self._pair_shards_by_event_range:
            return {collection: {key: mapping[key] for key in shared_keys} for collection, mapping in shards.items()}
        missing = {collection: sorted(set(shards["particles"]) - set(mapping)) for collection, mapping in shards.items()}
        missing = {collection: keys for collection, keys in missing.items() if keys}
        if missing:
            raise ValueError(f"Collections do not contain matching event-range shards: {missing}")
        return {collection: {key: mapping[key] for key in shared_keys} for collection, mapping in shards.items()}

    def _get_row_group_starts(self, path: Path) -> np.ndarray:
        path_key = str(path)
        if path_key in self._row_group_starts:
            return self._row_group_starts[path_key]

        parquet_file = pq.ParquetFile(path)
        num_row_groups = parquet_file.metadata.num_row_groups
        row_group_sizes = np.fromiter(
            (parquet_file.metadata.row_group(i).num_rows for i in range(num_row_groups)),
            dtype=np.int64,
            count=num_row_groups,
        )

        row_group_starts = np.zeros(num_row_groups + 1, dtype=np.int64)
        row_group_starts[1:] = np.cumsum(row_group_sizes)
        self._row_group_starts[path_key] = row_group_starts
        return row_group_starts

    def _get_num_events_in_shard(self, path: Path) -> int:
        row_group_starts = self._get_row_group_starts(path)
        return int(row_group_starts[-1])

    def _get_collection_path(self, collection: str, particle_path: Path) -> Path:
        key = self._event_range_token(particle_path) if self._pair_shards_by_event_range else particle_path.name
        return self._collection_shards[collection][key]

    @staticmethod
    def _event_id(record: ak.Record) -> int:
        return int(np.asarray(ak.to_numpy(record["event_id"])).reshape(-1)[0])

    def _assert_event_alignment(self, reference: ak.Record, collection: ak.Record, name: str) -> None:
        reference_id = self._event_id(reference)
        collection_id = self._event_id(collection)
        if reference_id != collection_id:
            raise ValueError(f"event_id mismatch for {name}: particles={reference_id}, {name}={collection_id}")

    def _get_row_group_lookup(self, path: Path, event_idx: int) -> tuple[int, int]:
        row_group_starts = self._get_row_group_starts(path)
        num_rows = int(row_group_starts[-1])
        if event_idx < 0 or event_idx >= num_rows:
            msg = f"Event index {event_idx} is out of range [0, {num_rows - 1}] for {path}"
            raise ValueError(msg)

        row_group_idx = int(np.searchsorted(row_group_starts, event_idx, side="right") - 1)
        row_in_group_idx = int(event_idx - row_group_starts[row_group_idx])
        return row_group_idx, row_in_group_idx

    def _get_row_group_array(self, path: Path, row_group_idx: int) -> ak.Array:
        key = (str(path), row_group_idx, None)
        if key in self._row_group_cache:
            self._row_group_cache.move_to_end(key)
            return self._row_group_cache[key]

        row_group_array = ak.from_parquet(path, row_groups=[row_group_idx])
        self._row_group_cache[key] = row_group_array
        self._row_group_cache.move_to_end(key)

        if len(self._row_group_cache) > self._row_group_cache_size:
            self._row_group_cache.popitem(last=False)

        return row_group_array

    def _read_event_columns(self, path: Path, event_idx: int, columns: tuple[str, ...]) -> ak.Record:
        """Read only selected parquet columns (notably excluding contrib_*)."""
        row_group_idx, row_in_group_idx = self._get_row_group_lookup(path, event_idx)
        key = (str(path), row_group_idx, columns)
        if key not in self._row_group_cache:
            table = pq.ParquetFile(path).read_row_group(row_group_idx, columns=list(columns))
            self._row_group_cache[key] = ak.from_arrow(table)
            self._row_group_cache.move_to_end(key)
            if len(self._row_group_cache) > self._row_group_cache_size:
                self._row_group_cache.popitem(last=False)
        else:
            self._row_group_cache.move_to_end(key)
        return self._row_group_cache[key][row_in_group_idx]

    def _read_event_from_file(self, path: Path, event_idx: int) -> ak.Record:
        row_group_idx, row_in_group_idx = self._get_row_group_lookup(path, event_idx)
        row_group_array = self._get_row_group_array(path, row_group_idx)
        return row_group_array[row_in_group_idx]

    def _debug(self, message: str) -> None:
        if self.debug:
            print(f"[ColliderMLDataset] {message}", flush=True)

    @staticmethod
    def _record_to_tensors(
        record: ak.Record,
        prefix: str,
        *,
        int_fields: set[str] | None = None,
        skip_fields: set[str] | None = None,
        default_dtype: torch.dtype | None = torch.float32,
    ) -> dict[str, torch.Tensor]:
        int_fields = set() if int_fields is None else set(int_fields)
        skip_fields = set() if skip_fields is None else set(skip_fields)
        tensors: dict[str, torch.Tensor] = {}

        for field in record.fields:
            if field in skip_fields:
                continue

            tensor = ak.to_torch(record[field])
            if field in int_fields:
                tensor = tensor.to(torch.int64)
            elif default_dtype is not None:
                tensor = tensor.to(default_dtype)
            tensors[f"{prefix}_{field}"] = tensor

        return tensors

    @staticmethod
    def _scale_xyz_inplace(tensors: dict[str, torch.Tensor], prefix: str, scale: float) -> None:
        for axis in ("x", "y", "z"):
            key = f"{prefix}_{axis}"
            if key in tensors:
                tensors[key] = tensors[key] * scale

    @staticmethod
    def _add_cylindrical_coords_inplace(tensors: dict[str, torch.Tensor], prefix: str) -> None:
        x = tensors[f"{prefix}_x"]
        y = tensors[f"{prefix}_y"]
        z = tensors[f"{prefix}_z"]
        r = torch.sqrt(x**2 + y**2)
        s = torch.sqrt(r**2 + z**2).clamp_min(1e-12)
        cos_theta = (z / s).clamp(-1.0, 1.0)

        tensors[f"{prefix}_r"] = r
        tensors[f"{prefix}_s"] = s
        tensors[f"{prefix}_eta"] = torch.arctanh(z / s)
        tensors[f"{prefix}_theta"] = torch.arccos(cos_theta)
        tensors[f"{prefix}_phi"] = torch.arctan2(y, x)

    @staticmethod
    def _valid_mask_like(values: torch.Tensor) -> torch.Tensor:
        return torch.full_like(values, True, dtype=torch.bool)

    @staticmethod
    def _apply_mask_to_prefixed_keys(
        tensors: dict[str, torch.Tensor],
        prefix: str,
        mask: torch.Tensor,
        *,
        skip_prefixes: tuple[str, ...] = (),
    ) -> None:
        for key in list(tensors.keys()):
            if not key.startswith(prefix):
                continue
            if any(key.startswith(skip) for skip in skip_prefixes):
                continue
            tensors[key] = tensors[key][mask]

    @staticmethod
    def _lookup_row_indices(reference_ids: np.ndarray, query_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if reference_ids.size == 0 or query_ids.size == 0:
            return np.zeros(0, dtype=np.int64), np.zeros(query_ids.size, dtype=bool)

        sort_idx = np.argsort(reference_ids)
        sorted_reference_ids = reference_ids[sort_idx]

        search_idx = np.searchsorted(sorted_reference_ids, query_ids)
        in_bounds = search_idx < sorted_reference_ids.size
        matched = np.zeros(query_ids.shape[0], dtype=bool)
        matched[in_bounds] = sorted_reference_ids[search_idx[in_bounds]] == query_ids[in_bounds]

        if not np.any(matched):
            return np.zeros(0, dtype=np.int64), matched

        row_indices = sort_idx[search_idx[matched]]
        return row_indices, matched

    @staticmethod
    def _build_csr_components(
        num_rows: int,
        num_cols: int,
        row_indices: np.ndarray,
        col_indices: np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row_indices = np.asarray(row_indices, dtype=np.int64)
        col_indices = np.asarray(col_indices, dtype=np.int64)

        if row_indices.size == 0:
            indptr = np.zeros(num_rows + 1, dtype=np.int64)
            indices = np.zeros(0, dtype=np.int64)
            shape = np.array([num_rows, num_cols], dtype=np.int64)
            return torch.from_numpy(indptr), torch.from_numpy(indices), torch.from_numpy(shape)

        # Build canonical CSR with optimized sparse backend:
        # deduplicates duplicate entries and stores row-sorted indices.
        values = np.ones(row_indices.shape[0], dtype=np.uint8)
        csr = sp.coo_matrix((values, (row_indices, col_indices)), shape=(num_rows, num_cols), dtype=np.uint8).tocsr()
        csr.sum_duplicates()
        indptr = np.array(csr.indptr, dtype=np.int64, copy=True)
        indices = np.array(csr.indices, dtype=np.int64, copy=True)
        shape = np.array([num_rows, num_cols], dtype=np.int64)
        return torch.from_numpy(indptr), torch.from_numpy(indices), torch.from_numpy(shape)

    @staticmethod
    def _empty_csr_components(num_rows: int, num_cols: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        empty = np.zeros(0, dtype=np.int64)
        return ColliderMLDataset._build_csr_components(num_rows, num_cols, empty, empty)

    @staticmethod
    def _float32_bincount(
        indices: np.ndarray,
        *,
        minlength: int,
        weights: np.ndarray | None = None,
    ) -> np.ndarray:
        counts = np.bincount(indices, weights=weights, minlength=minlength)
        return counts.astype(np.float32, copy=False)

    @staticmethod
    def _get_calohit_detector_ids(calohits: ak.Record) -> np.ndarray:
        return np.asarray(ak.to_numpy(calohits["detector"]), dtype=np.int64).reshape(-1)

    def _build_particle_sihit_csr(
        self,
        particle_ids: torch.Tensor,
        sihit_particle_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_particles = int(particle_ids.numel())
        num_sihits = int(sihit_particle_ids.numel())
        if num_particles == 0 or num_sihits == 0:
            return self._empty_csr_components(num_particles, num_sihits)

        particle_ids_np = particle_ids.cpu().numpy()
        sihit_particle_ids_np = sihit_particle_ids.cpu().numpy()

        row_indices, matched = self._lookup_row_indices(particle_ids_np, sihit_particle_ids_np)
        if row_indices.size == 0:
            return self._empty_csr_components(num_particles, num_sihits)

        col_indices = np.nonzero(matched)[0]
        return self._build_csr_components(num_particles, num_sihits, row_indices, col_indices)

    def _build_particle_calohit_csr(self, particle_ids: torch.Tensor, calohits) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_particles = int(particle_ids.numel())
        num_calohits = len(calohits["contrib_particle_ids"])
        if num_particles == 0 or num_calohits == 0:
            return self._empty_csr_components(num_particles, num_calohits)

        contrib_counts = ak.to_numpy(ak.num(calohits["contrib_particle_ids"], axis=1))
        if contrib_counts.sum() == 0:
            return self._empty_csr_components(num_particles, num_calohits)

        flat_contrib_particle_ids = ak.to_numpy(ak.flatten(calohits["contrib_particle_ids"], axis=1))
        calohit_indices = np.repeat(np.arange(num_calohits, dtype=np.int64), contrib_counts)

        particle_ids_np = particle_ids.cpu().numpy()
        row_indices, matched = self._lookup_row_indices(particle_ids_np, flat_contrib_particle_ids)
        if row_indices.size == 0:
            return self._empty_csr_components(num_particles, num_calohits)

        col_indices = calohit_indices[matched]
        return self._build_csr_components(num_particles, num_calohits, row_indices, col_indices)

    def _build_track_sihit_csr(self, track_hit_ids, num_sihits: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_tracks = len(track_hit_ids)
        if num_tracks == 0 or num_sihits == 0:
            return self._empty_csr_components(num_tracks, num_sihits)

        row_counts = ak.to_numpy(ak.num(track_hit_ids, axis=1))
        if row_counts.sum() == 0:
            return self._empty_csr_components(num_tracks, num_sihits)

        row_indices = np.repeat(np.arange(num_tracks, dtype=np.int64), row_counts)
        col_indices = ak.to_numpy(ak.flatten(track_hit_ids, axis=1)).astype(np.int64, copy=False)

        valid = (col_indices >= 0) & (col_indices < num_sihits)
        row_indices = row_indices[valid]
        col_indices = col_indices[valid]
        return self._build_csr_components(num_tracks, num_sihits, row_indices, col_indices)

    def _build_particle_targets(self, particles: ak.Record) -> dict[str, torch.Tensor]:
        targets = self._record_to_tensors(
            particles,
            "particle",
            int_fields={"particle_id", "pdg_id", "primary", "vertex_primary"},
            skip_fields={"parent_id"},
            default_dtype=torch.float32,
        )
        targets["particle_event_id"] = targets["particle_event_id"].expand_as(targets["particle_particle_id"])

        pt = torch.sqrt(targets["particle_px"] ** 2 + targets["particle_py"] ** 2)
        p = torch.sqrt(targets["particle_px"] ** 2 + targets["particle_py"] ** 2 + targets["particle_pz"] ** 2).clamp_min(1e-12)

        targets["particle_pt"] = pt
        targets["particle_p"] = p
        targets["particle_qopt"] = targets["particle_charge"] / pt.clamp_min(1e-6)
        targets["particle_eta"] = torch.arctanh(targets["particle_pz"] / p)
        targets["particle_theta"] = torch.arccos((targets["particle_pz"] / p).clamp(-1.0, 1.0))
        targets["particle_phi"] = torch.arctan2(targets["particle_py"], targets["particle_px"])
        targets["particle_d0"] = (-targets["particle_vx"] * targets["particle_py"] + targets["particle_vy"] * targets["particle_px"]) / pt.clamp_min(
            1e-6
        )
        targets["particle_z0"] = targets["particle_vz"]
        targets["particle_r"] = torch.hypot(targets["particle_vx"], targets["particle_vy"])
        targets["particle_primary"] = targets["particle_primary"].to(torch.bool)
        targets["particle_charged"] = targets["particle_charge"] != 0
        targets["particle_neutral"] = ~targets["particle_charged"]
        self._add_particle_classes_inplace(targets)
        return targets

    @staticmethod
    def _add_particle_classes_inplace(targets: dict[str, torch.Tensor]) -> None:
        pdg_key = "particle_pdg_id" if "particle_pdg_id" in targets else "particle_pdgid"
        pdg_id = targets[pdg_key].to(torch.int64)
        abs_pdg_id = torch.abs(pdg_id)

        class_masks: dict[str, torch.Tensor] = {
            "is_photon": abs_pdg_id == 22,
            "is_electron": abs_pdg_id == 11,
            "is_muon": abs_pdg_id == 13,
            "is_tau": abs_pdg_id == 15,
            "is_neutrino": (abs_pdg_id == 12) | (abs_pdg_id == 14) | (abs_pdg_id == 16) | (abs_pdg_id == 18),
        }

        is_known_nonhadron = torch.zeros_like(abs_pdg_id, dtype=torch.bool)
        for mask in class_masks.values():
            is_known_nonhadron |= mask

        class_masks["is_charged_hadron"] = (~is_known_nonhadron) & targets["particle_charged"]
        class_masks["is_neutral_hadron"] = (~is_known_nonhadron) & targets["particle_neutral"]

        class_assignment = {
            "is_neutral_hadron": 0,
            "is_charged_hadron": 1,
            "is_photon": 3,
            "is_electron": 4,
            "is_muon": 5,
            "is_tau": 6,
            "is_neutrino": 7,
        }
        class_id = torch.full_like(abs_pdg_id, -1, dtype=torch.int64)
        for class_name, class_value in class_assignment.items():
            class_id[class_masks[class_name]] = class_value

        is_other = torch.ones_like(abs_pdg_id, dtype=torch.bool)
        for class_name in class_assignment:
            is_other &= ~class_masks[class_name]
        class_masks["is_other"] = is_other

        targets["particle_class"] = class_id
        for class_name, class_mask in class_masks.items():
            targets[f"particle_{class_name}"] = class_mask

    def _build_particle_kinematic_mask(self, targets: dict[str, torch.Tensor]) -> torch.Tensor:
        particle_valid = targets["particle_pt"] > self.particle_min_pt
        particle_valid = particle_valid & (torch.abs(targets["particle_eta"]) < self.particle_max_abs_eta)

        if not self.particle_include_neutral:
            particle_valid = particle_valid & (~targets["particle_neutral"])
        if not self.particle_include_charged:
            particle_valid = particle_valid & (~targets["particle_charged"])
        if self.particle_require_primary:
            particle_valid = particle_valid & targets["particle_primary"]
        return particle_valid

    def _build_particle_hit_cut_mask(
        self,
        targets: dict[str, torch.Tensor],
        particle_num_sihits: torch.Tensor,
        particle_num_calo_hit_fields: dict[str, torch.Tensor] | None,
    ) -> torch.Tensor:
        particle_valid = torch.ones_like(targets["particle_valid"], dtype=torch.bool)
        num_ecalhits = (
            particle_num_calo_hit_fields["particle_num_ecalhits"]
            if particle_num_calo_hit_fields is not None
            else torch.zeros_like(particle_num_sihits)
        )
        num_hcalhits = (
            particle_num_calo_hit_fields["particle_num_hcalhits"]
            if particle_num_calo_hit_fields is not None
            else torch.zeros_like(particle_num_sihits)
        )

        for class_mask_key, cuts in self.particle_hit_cuts.items():
            class_mask = targets[class_mask_key].to(torch.bool)
            class_valid = torch.ones_like(class_mask)
            for cut_key, counts in (
                ("min_num_sihit", particle_num_sihits),
                ("min_num_ecal", num_ecalhits),
                ("min_num_hcal", num_hcalhits),
            ):
                threshold = cuts[cut_key]
                if threshold > 0:
                    class_valid &= counts >= threshold

            particle_valid &= (~class_mask) | class_valid

        return particle_valid

    def _build_particle_sihit_fields(
        self,
        particle_ids: torch.Tensor,
        sihit_particle_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        indptr, indices, shape = self._build_particle_sihit_csr(
            particle_ids,
            sihit_particle_ids,
        )
        num_sihits = torch.diff(indptr).to(torch.float32)
        return indptr, indices, shape, num_sihits

    def _build_sihit_inputs(self, sihits: ak.Record) -> dict[str, torch.Tensor]:
        inputs = self._record_to_tensors(
            sihits,
            "sihit",
            int_fields={"particle_id"},
            default_dtype=torch.float32,
        )
        self._scale_xyz_inplace(inputs, "sihit", scale=1e-3)
        self._add_cylindrical_coords_inplace(inputs, "sihit")
        inputs["sihit_valid"] = self._valid_mask_like(inputs["sihit_x"])
        return inputs

    def _read_calohits(self, particle_path: Path, event_idx: int) -> ak.Record:
        t_calo_read = perf_counter()
        calohits_path = self._get_collection_path("calo_hits", particle_path)
        calohits = self._read_event_from_file(calohits_path, event_idx)
        self._debug(f"read calohits in {perf_counter() - t_calo_read:.3f}s")
        return calohits

    def _add_calo_clusters(self, inputs: dict[str, torch.Tensor], particle_path: Path, event_idx: int) -> None:
        calohits_path = self._get_collection_path("calo_hits", particle_path)
        calohits = self._read_event_columns(
            calohits_path,
            event_idx,
            ("event_id", "detector", "total_energy", "x", "y", "z"),
        )
        clusters = cluster_calo_cells(
            calohits["x"],
            calohits["y"],
            calohits["z"],
            calohits["total_energy"],
            calohits["detector"],
            eps=self.calo_cluster_eps,
            min_samples=self.calo_cluster_min_samples,
            min_cluster_energy=self.calo_cluster_min_energy,
        )
        inputs.update({f"calocls_{field}": torch.from_numpy(values) for field, values in clusters.items()})
        inputs["calocls_valid"] = torch.ones(len(clusters["log_energy"]), dtype=torch.bool)

    def _build_particle_num_calo_hit_fields(self, particle_ids: torch.Tensor, calohits: ak.Record) -> dict[str, torch.Tensor]:
        particle_calohit_indptr, particle_calohit_indices, _ = self._build_particle_calohit_csr(particle_ids, calohits)
        num_calohits = torch.diff(particle_calohit_indptr).to(torch.float32)
        num_particles = int(num_calohits.numel())
        if num_particles == 0 or particle_calohit_indices.numel() == 0:
            empty_counts = {
                f"particle_num_{group_name}": torch.zeros(num_particles, dtype=torch.float32) for group_name in self.CALO_GROUP_DETECTOR_IDS
            }
            return {
                "particle_num_calohits": num_calohits,
                **empty_counts,
            }

        indptr_np = particle_calohit_indptr.cpu().numpy().astype(np.int64, copy=False)
        indices_np = particle_calohit_indices.cpu().numpy().astype(np.int64, copy=False)
        row_indices = np.repeat(np.arange(num_particles, dtype=np.int64), np.diff(indptr_np))

        linked_detector_ids = self._get_calohit_detector_ids(calohits)[indices_np]
        grouped_counts = {
            f"particle_num_{group_name}": torch.from_numpy(
                self._float32_bincount(
                    row_indices[np.isin(linked_detector_ids, detector_ids)],
                    minlength=num_particles,
                )
            )
            for group_name, detector_ids in self.CALO_GROUP_DETECTOR_IDS.items()
        }

        return {
            "particle_num_calohits": num_calohits,
            **grouped_counts,
        }

    def _build_particle_num_calo_hit_fields_if_available(
        self,
        particle_ids: torch.Tensor,
        calohits: ak.Record | None,
    ) -> tuple[dict[str, torch.Tensor] | None, torch.Tensor | None]:
        if calohits is None:
            return None, None

        fields = self._build_particle_num_calo_hit_fields(particle_ids, calohits)
        return fields, fields["particle_num_calohits"]

    def _build_particle_calo_energy_fields(self, particle_ids: torch.Tensor, calohits: ak.Record) -> dict[str, torch.Tensor]:
        num_particles = int(particle_ids.numel())
        particle_energy_raw_np = {key: np.zeros(num_particles, dtype=np.float32) for key in (*self.CALO_SUBSYSTEMS, "calo_sum")}

        contrib_counts = ak.to_numpy(ak.num(calohits["contrib_particle_ids"], axis=1))
        if num_particles > 0 and contrib_counts.sum() > 0:
            flat_contrib_particle_ids = ak.to_numpy(ak.flatten(calohits["contrib_particle_ids"], axis=1))
            flat_contrib_energies = ak.to_numpy(ak.flatten(calohits["contrib_energies"], axis=1)).astype(np.float32, copy=False)
            flat_contrib_detector = np.repeat(self._get_calohit_detector_ids(calohits), contrib_counts)

            particle_ids_np = particle_ids.cpu().numpy()
            row_indices, matched = self._lookup_row_indices(particle_ids_np, flat_contrib_particle_ids)
            if row_indices.size > 0:
                matched_energies = flat_contrib_energies[matched]
                matched_detectors = flat_contrib_detector[matched]
                particle_energy_raw_np["calo_sum"] = self._float32_bincount(
                    row_indices,
                    minlength=num_particles,
                    weights=matched_energies,
                )
                for subsystem, detector_ids in self.CALO_SUBSYSTEM_DETECTOR_IDS.items():
                    subsystem_mask = np.isin(matched_detectors, detector_ids)
                    if not np.any(subsystem_mask):
                        continue

                    particle_energy_raw_np[subsystem] = self._float32_bincount(
                        row_indices[subsystem_mask],
                        minlength=num_particles,
                        weights=matched_energies[subsystem_mask],
                    )

        particle_energy_raw = {key: torch.from_numpy(values) for key, values in particle_energy_raw_np.items()}
        particle_energy_raw["ecal"] = particle_energy_raw["ecb"] + particle_energy_raw["ece"]
        particle_energy_raw["hcal"] = particle_energy_raw["hcb"] + particle_energy_raw["hce"]

        particle_energy_calib = {
            f"{subsystem}_calib": particle_energy_raw[subsystem] * scale for subsystem, scale in self.CALO_SUBSYSTEM_CALIBRATION.items()
        }
        particle_energy_calib["ecal_calib"] = particle_energy_calib["ecb_calib"] + particle_energy_calib["ece_calib"]
        particle_energy_calib["hcal_calib"] = particle_energy_calib["hcb_calib"] + particle_energy_calib["hce_calib"]
        particle_energy_calib["calo_calib"] = particle_energy_calib["ecal_calib"] + particle_energy_calib["hcal_calib"]

        out = {f"particle_energy_{subsystem}": particle_energy_raw[subsystem] for subsystem in (*self.CALO_SUBSYSTEMS, "ecal", "hcal")}
        out["particle_energy_calo_sum"] = particle_energy_raw["calo_sum"]
        out.update({f"particle_energy_{name}": values for name, values in particle_energy_calib.items()})
        return out

    @staticmethod
    def _build_calohit_contrib_energy_sum(calohits: ak.Record) -> torch.Tensor:
        contrib_energy_sum = ak.to_numpy(ak.sum(calohits["contrib_energies"], axis=1, mask_identity=False)).astype(np.float32, copy=False)
        return torch.from_numpy(contrib_energy_sum)

    def _add_calohits(
        self,
        inputs: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        particle_path: Path,
        event_idx: int,
        calohits: ak.Record | None = None,
    ) -> None:
        if calohits is None:
            calohits = self._read_calohits(particle_path, event_idx)
        if self._event_id(calohits) != int(targets["event_id"]):
            raise ValueError(f"event_id mismatch for calo_hits: particles={int(targets['event_id'])}, calo_hits={self._event_id(calohits)}")

        inputs.update(
            self._record_to_tensors(
                calohits,
                "calohit",
                skip_fields={"detector", "contrib_particle_ids", "contrib_energies", "contrib_times"},
                default_dtype=torch.float32,
            )
        )
        inputs["calohit_detector"] = ak.to_torch(calohits["detector"]).to(torch.int64)
        inputs["calohit_contrib_energy_sum"] = self._build_calohit_contrib_energy_sum(calohits)
        self._scale_xyz_inplace(inputs, "calohit", scale=1e-3)
        inputs["calohit_valid"] = self._valid_mask_like(inputs["calohit_x"])
        targets["calohit_valid"] = inputs["calohit_valid"]
        targets.update(self._build_particle_num_calo_hit_fields(targets["particle_particle_id"], calohits))
        targets.update(self._build_particle_calo_energy_fields(targets["particle_particle_id"], calohits))

        if not self.build_calohit_associations:
            self._debug("skipping calo association build (build_calohit_associations=False)")
            return

        self._debug(f"building calo associations: n_particles={targets['particle_particle_id'].size(0)} n_calohits={inputs['calohit_x'].size(0)}")
        t_calo_assoc = perf_counter()
        particle_calohit_indptr, particle_calohit_indices, particle_calohit_shape = self._build_particle_calohit_csr(
            targets["particle_particle_id"],
            calohits,
        )
        self._debug(
            f"built calo CSR in {perf_counter() - t_calo_assoc:.3f}s "
            f"shape=({int(particle_calohit_shape[0])}, {int(particle_calohit_shape[1])}) nnz={particle_calohit_indices.numel()}"
        )

        targets["particle_calohit_indptr"] = particle_calohit_indptr
        targets["particle_calohit_indices"] = particle_calohit_indices
        targets["particle_calohit_shape"] = particle_calohit_shape

    def _add_tracks(self, inputs: dict[str, torch.Tensor], targets: dict[str, torch.Tensor], particle_path: Path, event_idx: int) -> None:
        t_track_read = perf_counter()
        tracks_path = self._get_collection_path("tracks", particle_path)
        tracks = self._read_event_from_file(tracks_path, event_idx)
        if self._event_id(tracks) != int(targets["event_id"]):
            raise ValueError(f"event_id mismatch for tracks: particles={int(targets['event_id'])}, tracks={self._event_id(tracks)}")
        self._debug(f"read tracks in {perf_counter() - t_track_read:.3f}s")

        targets.update(
            self._record_to_tensors(
                tracks,
                "track",
                skip_fields={"hit_ids"},
                default_dtype=None,
            )
        )
        targets["track_valid"] = self._valid_mask_like(targets["track_phi"])

        t_track_assoc = perf_counter()
        original_ids = inputs["sihit_original_hit_id"].numpy()
        inverse = {int(original): dense for dense, original in enumerate(original_ids)}
        filtered_track_hit_ids = ak.Array([[inverse[int(hit)] for hit in row if int(hit) in inverse] for row in ak.to_list(tracks["hit_ids"])])
        track_sihit_indptr, track_sihit_indices, track_sihit_shape = self._build_track_sihit_csr(
            filtered_track_hit_ids,
            num_sihits=len(inputs["sihit_valid"]),
        )
        self._debug(
            f"built track_sihit CSR in {perf_counter() - t_track_assoc:.3f}s "
            f"shape=({int(track_sihit_shape[0])}, {int(track_sihit_shape[1])}) nnz={track_sihit_indices.numel()}"
        )
        targets["track_sihit_indptr"] = track_sihit_indptr
        targets["track_sihit_indices"] = track_sihit_indices
        targets["track_sihit_shape"] = track_sihit_shape

    def _add_deficit_features(
        self,
        inputs: dict[str, torch.Tensor],
        particle_path: Path,
        event_idx: int,
        calohits: ak.Record | None,
    ) -> None:
        if calohits is None:
            calohits = self._read_event_columns(
                self._get_collection_path("calo_hits", particle_path),
                event_idx,
                ("event_id", "detector", "total_energy", "x", "y", "z"),
            )
        tracks = self._read_event_columns(
            self._get_collection_path("tracks", particle_path),
            event_idx,
            ("event_id", "phi", "theta", "qop"),
        )
        cones = compute_cone_accounting(
            calohits["x"], calohits["y"], calohits["z"], calohits["total_energy"], calohits["detector"],
            (), (), tracks["phi"], tracks["theta"], tracks["qop"], (), self.deficit_params,
        )
        features = assign_cone_features(
            inputs["sihit_x"].numpy(), inputs["sihit_y"].numpy(), inputs["sihit_z"].numpy(),
            cones, self.deficit_params.cone_radius,
        )
        for column, name in enumerate(("incone", "ecal", "hcal", "trkpt", "ntrk", "ncells")):
            inputs[f"sihit_dc_{name}"] = torch.from_numpy(features[:, column])

    def _pad_particle_targets_inplace(self, targets: dict[str, torch.Tensor]) -> None:
        particle_keys = [
            k for k in targets if k.startswith("particle_") and not k.startswith("particle_sihit_") and not k.startswith("particle_calohit_")
        ]
        if not particle_keys:
            return

        pad_size = self.event_max_num_particles
        for key in particle_keys:
            values = targets[key]
            if values.size(0) > pad_size:
                values = values[:pad_size]
                targets[key] = values
            extra = pad_size - int(values.size(0))
            if extra <= 0:
                continue

            pad_shape = (extra, *values.shape[1:])
            pad_value = False if values.dtype is torch.bool else 0
            targets[key] = torch.cat([values, values.new_full(pad_shape, pad_value)], dim=0)

    def load_event(self, sample_id):
        t0 = perf_counter()
        if sample_id < 0 or sample_id >= len(self.sample_index):
            msg = f"sample_id {sample_id} is out of range [0, {len(self.sample_index) - 1}]"
            raise ValueError(msg)

        shard_idx, event_idx = self.sample_index[sample_id]
        particle_path = self.particle_shard_paths[shard_idx]
        sihit_path = self._get_collection_path("tracker_hits", particle_path)
        self._debug(f"sample_id={sample_id} shard_idx={shard_idx} event_idx={event_idx} file={particle_path.name}")

        # Read in the data
        t_read = perf_counter()
        particles = self._read_event_from_file(particle_path, event_idx)
        sihits = self._read_event_from_file(sihit_path, event_idx)
        self._assert_event_alignment(particles, sihits, "tracker_hits")
        self._debug(f"read particles+sihits in {perf_counter() - t_read:.3f}s")

        targets = self._build_particle_targets(particles)
        targets["event_id"] = torch.tensor(self._event_id(particles), dtype=torch.int64)
        particle_valid = self._build_particle_kinematic_mask(targets)
        self._apply_mask_to_prefixed_keys(targets, "particle_", particle_valid)
        self._debug(f"after kinematic cuts: n_particles={targets['particle_particle_id'].size(0)}")

        targets["particle_valid"] = self._valid_mask_like(targets["particle_pt"])
        inputs = self._build_sihit_inputs(sihits)
        # Stable map from the model's dense hit axis back to the unfiltered event.
        inputs["sihit_original_hit_id"] = torch.arange(len(inputs["sihit_valid"]), dtype=torch.int64)
        targets["sihit_valid"] = inputs["sihit_valid"]

        t_assoc = perf_counter()
        particle_sihit_indptr, particle_sihit_indices, particle_sihit_shape, particle_num_sihits = self._build_particle_sihit_fields(
            targets["particle_particle_id"],
            inputs["sihit_particle_id"],
        )

        calohits = self._read_calohits(particle_path, event_idx) if self.return_calohits or self._requires_calohits_for_hit_cuts else None
        if calohits is not None:
            self._assert_event_alignment(particles, calohits, "calo_hits")
        particle_num_calo_hit_fields, particle_num_calohits = self._build_particle_num_calo_hit_fields_if_available(
            targets["particle_particle_id"],
            calohits,
        )

        # Apply constituent based particle cuts
        particle_valid = self._build_particle_hit_cut_mask(
            targets,
            particle_num_sihits,
            particle_num_calo_hit_fields,
        )

        self._apply_mask_to_prefixed_keys(
            targets,
            "particle_",
            particle_valid,
            skip_prefixes=("particle_sihit_", "particle_calohit_"),
        )
        if targets["particle_particle_id"].numel() > self.event_max_num_particles:
            n_kept = targets["particle_particle_id"].numel()
            dropped = n_kept - self.event_max_num_particles
            self.dropped_particle_count += dropped
            if not self._warned_particle_truncation:
                print(
                    f"Warning: event {int(targets['event_id'])} has {n_kept} cut-passing particles; "
                    f"keeping {self.event_max_num_particles} and dropping {dropped} softest particles "
                    "(further truncation warnings suppressed)"
                )
                self._warned_particle_truncation = True
            # Particle targets form an order-invariant set, so one shared pT ordering
            # is safe and is applied to every particle_* field before associations build.
            keep = torch.argsort(targets["particle_pt"], descending=True)[: self.event_max_num_particles]
            self._apply_mask_to_prefixed_keys(
                targets,
                "particle_",
                keep,
                skip_prefixes=("particle_sihit_", "particle_calohit_"),
            )
        particle_sihit_indptr, particle_sihit_indices, particle_sihit_shape, particle_num_sihits = self._build_particle_sihit_fields(
            targets["particle_particle_id"],
            inputs["sihit_particle_id"],
        )
        valid_particle_ids = targets["particle_particle_id"]
        targets["sihit_on_valid_particle"] = self._build_sihit_on_valid_particle(
            inputs["sihit_particle_id"], valid_particle_ids
        )

        # Apply learned filtering only after the reconstructibility target has
        # been built, but before any optional hit-derived branches or collation.
        hit_keep = self._hit_filter_mask(int(targets["event_id"]), len(inputs["sihit_valid"]))
        self._filter_sihits_inplace(inputs, targets, hit_keep)
        particle_sihit_indptr, particle_sihit_indices, particle_sihit_shape, particle_num_sihits = self._build_particle_sihit_fields(
            targets["particle_particle_id"], inputs["sihit_particle_id"]
        )
        particle_num_calo_hit_fields, particle_num_calohits = self._build_particle_num_calo_hit_fields_if_available(
            targets["particle_particle_id"],
            calohits,
        )
        if particle_num_calo_hit_fields is not None:
            targets.update(particle_num_calo_hit_fields)

        constituent_debug = (
            f"after constituent cuts: n_particles={targets['particle_particle_id'].size(0)} n_sihits={inputs['sihit_particle_id'].size(0)}"
        )
        if particle_num_calohits is not None:
            constituent_debug += f" n_calohits={int(particle_num_calohits.sum().item())}"
            if particle_num_calo_hit_fields is not None:
                constituent_debug += (
                    f" n_ecalhits={int(particle_num_calo_hit_fields['particle_num_ecalhits'].sum().item())}"
                    f" n_hcalhits={int(particle_num_calo_hit_fields['particle_num_hcalhits'].sum().item())}"
                )
        self._debug(constituent_debug)

        self._debug(
            f"built particle_sihit CSR in {perf_counter() - t_assoc:.3f}s "
            f"shape=({int(particle_sihit_shape[0])}, {int(particle_sihit_shape[1])}) nnz={particle_sihit_indices.numel()}"
        )
        targets["particle_sihit_indptr"] = particle_sihit_indptr
        targets["particle_sihit_indices"] = particle_sihit_indices
        targets["particle_sihit_shape"] = particle_sihit_shape
        targets["particle_num_sihits"] = particle_num_sihits
        targets["particle_sihit_valid"] = (
            targets["particle_particle_id"].unsqueeze(-1) == inputs["sihit_particle_id"].unsqueeze(0)
        )

        if self.return_deficit_features:
            self._add_deficit_features(inputs, particle_path, event_idx, calohits)

        # Return the calorimeter hit info if requested
        if self.return_calohits:
            self._add_calohits(inputs, targets, particle_path, event_idx, calohits=calohits)

        if self.return_calo_clusters:
            self._add_calo_clusters(inputs, particle_path, event_idx)

        # Return ACTS track info if requested
        if self.return_tracks:
            self._add_tracks(inputs, targets, particle_path, event_idx)

        # Add metadata
        targets["sample_id"] = torch.tensor(sample_id, dtype=torch.int64)
        self._debug(f"load_event finished in {perf_counter() - t0:.3f}s")

        return inputs, targets

    def __getitem__(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = int(idx.item())

        if idx < 0:
            idx += len(self.sample_ids)
        if idx < 0 or idx >= len(self.sample_ids):
            msg = f"Dataset index {idx} out of range for length {len(self.sample_ids)}"
            raise IndexError(msg)

        sample_id = self.sample_ids[idx]
        inputs, targets = self.load_event(sample_id)

        self._pad_particle_targets_inplace(targets)

        indptr = targets["particle_sihit_indptr"]
        expected_indptr_size = self.event_max_num_particles + 1
        if indptr.numel() < expected_indptr_size:
            targets["particle_sihit_indptr"] = torch.cat(
                [indptr, indptr[-1:].repeat(expected_indptr_size - indptr.numel())]
            )
            targets["particle_sihit_shape"] = targets["particle_sihit_shape"].clone()
            targets["particle_sihit_shape"][0] = self.event_max_num_particles

        dense = targets["particle_sihit_valid"]
        if dense.size(0) > self.event_max_num_particles:
            dense = dense[: self.event_max_num_particles]
        elif dense.size(0) < self.event_max_num_particles:
            dense = torch.cat(
                [dense, torch.zeros((self.event_max_num_particles - dense.size(0), dense.size(1)), dtype=torch.bool)],
                dim=0,
            )
        targets["particle_sihit_valid"] = dense

        return inputs, targets

    @classmethod
    def collate_fn(
        cls,
        batch: list[tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]],
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor | list[torch.Tensor]]]:
        """Batch events, padding each variable-length collection to its batch maximum.

        Input collections follow the hepattn convention of a boolean
        ``{name}_valid`` mask. Particle targets are already fixed-size; the hit
        axis of ``particle_sihit_valid`` is padded with the corresponding input.
        CSR diagnostic fields stay as per-event lists for batch sizes above one.

        Raises:
            ValueError: If ``batch`` is empty.
        """
        if not batch:
            raise ValueError("Cannot collate an empty batch")

        input_samples, target_samples = zip(*batch, strict=True)
        input_lengths = {
            key.removesuffix("_valid"): [int(sample[key].shape[0]) for sample in input_samples]
            for key in input_samples[0]
            if key.endswith("_valid")
        }

        def pad_first(values: list[torch.Tensor], padding_value: float | bool = 0) -> torch.Tensor:
            return pad_sequence(values, batch_first=True, padding_value=padding_value)

        batched_inputs: dict[str, torch.Tensor] = {}
        for key in input_samples[0]:
            values = [sample[key] for sample in input_samples]
            collection = next(
                (name for name in input_lengths if key == f"{name}_valid" or key.startswith(f"{name}_")),
                None,
            )
            if collection is None:
                batched_inputs[key] = torch.stack(values)
            else:
                batched_inputs[key] = pad_first(values, False if key.endswith("_valid") else 0)

        batched_targets: dict[str, torch.Tensor | list[torch.Tensor]] = {}
        for key in target_samples[0]:
            values = [sample[key] for sample in target_samples]

            if key.endswith(cls.CSR_SUFFIXES):
                # Preserve the historical tensor shape for the default batch_size=1.
                batched_targets[key] = torch.stack(values) if len(values) == 1 else values
                continue

            if key == "particle_sihit_valid":
                # pad_sequence operates on the leading axis, so temporarily put
                # the variable hit axis first: [particle, hit] -> [hit, particle].
                batched_targets[key] = pad_first([value.transpose(0, 1) for value in values], False).transpose(1, 2)
                continue

            if key.startswith("sihit_") and any(value.shape != values[0].shape for value in values):
                batched_targets[key] = pad_first(values, False if values[0].dtype is torch.bool else 0)
                continue

            collection = next(
                (
                    name
                    for name in input_lengths
                    if key == f"{name}_valid" or (name != "sihit" and key.startswith(f"{name}_"))
                ),
                None,
            )
            if collection is not None and any(value.shape != values[0].shape for value in values):
                batched_targets[key] = pad_first(values, False if key.endswith("_valid") else 0)
            else:
                batched_targets[key] = torch.stack(values)

        return batched_inputs, batched_targets


class ColliderMLDataModule(LightningDataModule):
    def __init__(
        self,
        train_dir: str | None = None,
        val_dir: str | None = None,
        num_workers: int = 0,
        num_train: int = -1,
        num_val: int = 100,
        num_test: int = 1000,
        batch_size: int = 1,
        test_dir: str | None = None,
        dataset_root: str | None = None,
        train_chunks: list[int] | tuple[int, int] | None = None,
        val_chunks: list[int] | tuple[int, int] | None = None,
        test_chunks: list[int] | tuple[int, int] | None = None,
        event_max_num_particles: int = 1024,
        pin_memory: bool = True,
        return_calo_clusters: bool = False,
        calo_cluster_eps: float = DEFAULT_EPS,
        calo_cluster_min_samples: int = DEFAULT_MIN_SAMPLES,
        calo_cluster_min_energy: float = DEFAULT_MIN_CLUSTER_ENERGY,
        particle_min_pt: float = 0.5,
        particle_max_abs_eta: float = 4.0,
        particle_hit_cuts: dict[str, dict[str, int]] | None = None,
        particle_include_charged: bool = True,
        particle_include_neutral: bool = True,
        particle_require_primary: bool = False,
        return_calohits: bool = True,
        return_tracks: bool = True,
        return_deficit_features: bool = False,
        deficit_grid_size: float = 0.1,
        deficit_cone_radius: float = 0.4,
        deficit_seed_energy_threshold: float = 0.01,
        deficit_max_cones: int = 12,
        deficit_max_abs_eta: float = 5.0,
        event_type: str = "ttbar",
        build_calohit_associations: bool = True,
        hit_filter_file: str | None = None,
        hit_filter_wp: float = 0.0,
        debug: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.num_workers = num_workers
        self.num_train = num_train
        self.num_val = num_val
        self.num_test = num_test
        if batch_size < 1:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.dataset_root = dataset_root
        self.train_chunks = ColliderMLDataset._normalize_chunk_range(train_chunks)
        self.val_chunks = ColliderMLDataset._normalize_chunk_range(val_chunks)
        self.test_chunks = ColliderMLDataset._normalize_chunk_range(test_chunks)
        if dataset_root is not None:
            split_ranges = {
                "train": self.train_chunks,
                "val": self.val_chunks,
                "test": self.test_chunks,
            }
            missing_ranges = [name for name, chunk_range in split_ranges.items() if chunk_range is None]
            if missing_ranges:
                raise ValueError(f"dataset_root requires chunk ranges for all splits; missing: {missing_ranges}")
            for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
                left_range = split_ranges[left]
                right_range = split_ranges[right]
                if max(left_range[0], right_range[0]) <= min(left_range[1], right_range[1]):
                    raise ValueError(f"ColliderML chunk ranges for {left} and {right} overlap: {left_range} and {right_range}")
        self.event_max_num_particles = event_max_num_particles
        self.kwargs = {
            **kwargs,
            "return_calo_clusters": return_calo_clusters,
            "calo_cluster_eps": calo_cluster_eps,
            "calo_cluster_min_samples": calo_cluster_min_samples,
            "calo_cluster_min_energy": calo_cluster_min_energy,
            "particle_min_pt": particle_min_pt,
            "particle_max_abs_eta": particle_max_abs_eta,
            "particle_hit_cuts": particle_hit_cuts,
            "particle_include_charged": particle_include_charged,
            "particle_include_neutral": particle_include_neutral,
            "particle_require_primary": particle_require_primary,
            "return_calohits": return_calohits,
            "return_tracks": return_tracks,
            "return_deficit_features": return_deficit_features,
            "deficit_grid_size": deficit_grid_size,
            "deficit_cone_radius": deficit_cone_radius,
            "deficit_seed_energy_threshold": deficit_seed_energy_threshold,
            "deficit_max_cones": deficit_max_cones,
            "deficit_max_abs_eta": deficit_max_abs_eta,
            "event_type": event_type,
            "build_calohit_associations": build_calohit_associations,
            "hit_filter_file": hit_filter_file,
            "hit_filter_wp": hit_filter_wp,
            "debug": debug,
        }

    def setup(self, stage: str):
        if stage in {"fit", "test"}:
            self.train_dset = ColliderMLDataset(
                dirpath=self.train_dir, dataset_root=self.dataset_root, num_events=self.num_train,
                chunk_range=self.train_chunks,
                event_max_num_particles=self.event_max_num_particles, **self.kwargs
            )

        if stage in {"fit", "validate"}:
            self.val_dset = ColliderMLDataset(
                dirpath=self.val_dir, dataset_root=self.dataset_root, num_events=self.num_val,
                chunk_range=self.val_chunks,
                event_max_num_particles=self.event_max_num_particles, **self.kwargs
            )

        # Only print train/val dataset details when actually training
        if stage == "fit" and (getattr(self, "_trainer", None) is None or self.trainer.is_global_zero):
            print(f"Created training dataset with {len(self.train_dset):,} events")
            print(f"Created validation dataset with {len(self.val_dset):,} events")

        if stage in {"test", "predict"}:
            if self.test_dir is None and self.dataset_root is None:
                raise ValueError("No test data specified; set --data.test_dir or --data.dataset_root")
            self.test_dset = ColliderMLDataset(
                dirpath=self.test_dir, dataset_root=self.dataset_root, num_events=self.num_test,
                chunk_range=self.test_chunks,
                event_max_num_particles=self.event_max_num_particles, **self.kwargs
            )
            print(f"Created test dataset with {len(self.test_dset):,} events")

    def get_dataloader(self, stage: str, dataset: ColliderMLDataset, shuffle: bool):
        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            collate_fn=dataset.collate_fn,
            sampler=None,
            num_workers=self.num_workers,
            shuffle=shuffle,
            pin_memory=self.pin_memory,
            **({"prefetch_factor": 4} if self.num_workers > 0 else {}),
        )

    def train_dataloader(self):
        return self.get_dataloader(dataset=self.train_dset, stage="fit", shuffle=True)

    def val_dataloader(self):
        return self.get_dataloader(dataset=self.val_dset, stage="test", shuffle=False)

    def test_dataloader(self):
        return self.get_dataloader(dataset=self.test_dset, stage="test", shuffle=False)

    def predict_dataloader(self):
        return self.get_dataloader(dataset=self.test_dset, stage="predict", shuffle=False)
