"""ColliderML-backed IReader implementations for pyacts.

Reads ColliderML parquet files and feeds truth particles and measurements
into the ACTS sequencer.

Workflow: truth particles + tracker_hits → ColliderMLMeasurementReader
  (bypasses DigitizationAlgorithm; projects positions onto surfaces directly).

Surface mapping: ColliderML uses gen1 ODD geometry identifiers
(volume_id, layer_id, surface_id). `config/surface_map.csv.gz`
(produced by scripts/build_colliderml_surface_map.py against the gen3 ODD
TrackingGeometry) maps each ColliderML tuple to the correct gen3
ACTS GeometryIdentifier via KD-tree + normal-distance matching.
"""

from __future__ import annotations

import math
from pathlib import Path

import acts
import acts.examples
import numpy as np
import polars as pl


def _read_surface_map(path: Path | str) -> pl.DataFrame:
    import gzip
    p = Path(path)
    if p.suffixes[-2:] == [".csv", ".gz"]:
        with gzip.open(p, "rt") as f:
            return pl.read_csv(f.read().encode())
    return pl.read_csv(p)


def _load_parquet_with_event_map(
    path: Path | str,
) -> tuple[pl.DataFrame, dict[int, int], int, int]:
    """Load a ColliderML parquet and build an event-number → row-index map."""
    df = pl.read_parquet(str(path))
    event_ids = df["event_id"].to_list()
    entry_map = {int(eid): i for i, eid in enumerate(event_ids)}
    return df, entry_map, min(entry_map), max(entry_map)


def load_tracking_geometry_from_json(jsonPath: Path | str) -> acts.TrackingGeometry:
    """Load the ODD gen3 TrackingGeometry from its JSON export (.json or .json.gz)."""
    import gzip

    import acts.ActsPluginsPythonBindingsJson as pj

    p = Path(jsonPath)
    if p.suffix == ".gz":
        with gzip.open(p, "rt") as f:
            encoded = f.read()
    else:
        encoded = p.read_text()
    gctx = acts.GeometryContext.dangerouslyDefaultConstruct()
    conv = pj.TrackingGeometryJsonConverter(pj.TrackingGeometryJsonConverter.Config())
    return conv.fromJson(gctx, encoded)


class ColliderMLParticleReader(acts.examples.IReader):
    """Reads truth particles from a ColliderML parquet file into ACTS.

    ColliderML stores one row per event, with list-valued columns.
    Momentum and mass are in GeV; vertex positions in mm; charge in
    elementary-charge units.
    """

    def __init__(
        self,
        particlePath: Path | str,
        outputParticles: str = "particles",
        level: acts.logging.Level = acts.logging.INFO,
    ):
        acts.examples.IReader.__init__(self, "ColliderMLParticleReader", level)

        self._outputParticles = outputParticles
        self._handle = acts.examples.WriteDataHandle(
            self, acts.examples.SimParticleContainer, "OutputParticles"
        )
        self._handle.initialize(self._outputParticles)

        self._df, self._entry_map, self._min_event, self._max_event = (
            _load_parquet_with_event_map(particlePath)
        )

        # Populated in read() so SimHitReader can look up barcodes by particle_id.
        self.last_barcode_map: dict[int, acts.examples.SimBarcode] = {}
        # Populated in read() so SimHitReader can look up per-particle momentum.
        self.last_momentum_map: dict[int, tuple[float, float, float, float]] = {}

    def availableEvents(self):
        return (self._min_event, self._max_event + 1)

    def read(self, context):
        u = acts.UnitConstants
        event_number = context.eventNumber
        particles = acts.examples.SimParticleContainer()

        self.last_barcode_map = {}
        self.last_momentum_map = {}

        if event_number in self._entry_map:
            idx = self._entry_map[event_number]
            row = self._df.row(idx, named=True)

            particle_ids = row["particle_id"]
            pdg_ids = row["pdg_id"]
            masses = row["mass"]
            energies = row["energy"]
            charges = row["charge"]
            vxs = row["vx"]
            vys = row["vy"]
            vzs = row["vz"]
            times = row["time"]
            pxs = row["px"]
            pys = row["py"]
            pzs = row["pz"]
            vertex_primaries = row["vertex_primary"]

            for i in range(len(particle_ids)):
                px_val = float(pxs[i])
                py_val = float(pys[i])
                pz_val = float(pzs[i])
                p_mag = math.sqrt(px_val**2 + py_val**2 + pz_val**2)

                barcode = acts.examples.SimBarcode()
                barcode.vertexPrimary = int(vertex_primaries[i])
                barcode.particle = i + 1  # 1-indexed so no zero barcode

                p = acts.examples.SimParticle(
                    barcode,
                    acts.PdgParticle(int(pdg_ids[i])),
                    float(charges[i]) * u.e,
                    float(masses[i]) * u.GeV,
                )
                p.fourPosition = acts.Vector4(
                    float(vxs[i]) * u.mm,
                    float(vys[i]) * u.mm,
                    float(vzs[i]) * u.mm,
                    float(times[i]) * u.mm,
                )
                if p_mag > 0:
                    p.direction = acts.Vector3(px_val / p_mag, py_val / p_mag, pz_val / p_mag)
                    p.absoluteMomentum = p_mag * u.GeV
                else:
                    p.direction = acts.Vector3(0.0, 0.0, 1.0)
                    p.absoluteMomentum = 0.0

                particles.insert(p)

                pid = int(particle_ids[i])
                self.last_barcode_map[pid] = barcode
                self.last_momentum_map[pid] = (px_val, py_val, pz_val, float(energies[i]))

        self._handle(context, particles)
        return acts.examples.ProcessCode.SUCCESS


class ColliderMLSimHitReader(acts.examples.IReader):
    """Reads tracker hits from a ColliderML parquet file into ACTS SimHits.

    Uses truth positions (true_x/y/z) as the simhit coordinates. Per-hit
    momentum is approximated from the parent particle's total momentum (no
    per-hit energy-loss information is available in ColliderML). The
    `config/surface_map.csv.gz` (built by scripts/build_colliderml_surface_map.py
    against the gen3 ODD geometry) maps each ColliderML (volume_id, layer_id,
    surface_id) tuple to the correct gen3 GeometryIdentifier. Hits without a
    mapping entry are dropped.

    The particle reader must appear before the hit reader in the Sequencer
    so its barcode/momentum maps are populated first.
    """

    def __init__(
        self,
        hitPath: Path | str,
        particleReader: ColliderMLParticleReader,
        surfaceMapPath: Path | str,
        outputSimHits: str = "simhits",
        level: acts.logging.Level = acts.logging.INFO,
    ):
        acts.examples.IReader.__init__(self, "ColliderMLSimHitReader", level)

        self._outputSimHits = outputSimHits
        self._particle_reader = particleReader
        self._handle = acts.examples.WriteDataHandle(
            self, acts.examples.SimHitContainer, "OutputSimHits"
        )
        self._handle.initialize(self._outputSimHits)

        self._df, self._entry_map, self._min_event, self._max_event = (
            _load_parquet_with_event_map(hitPath)
        )

        # Load pre-built (volume_id, layer_id, surface_id) → gen3 geoid table.
        smap = _read_surface_map(surfaceMapPath)
        self._geoid_map: dict[tuple[int, int, int], acts.GeometryIdentifier] = {
            (int(r["volume_id"]), int(r["layer_id"]), int(r["surface_id"])): acts.GeometryIdentifier(int(r["geoid"]))
            for r in smap.iter_rows(named=True)
            if int(r["geoid"]) != 0
        }

    def availableEvents(self):
        return (self._min_event, self._max_event + 1)

    def read(self, context):
        u = acts.UnitConstants
        event_number = context.eventNumber
        hits = acts.examples.SimHitContainer()

        if event_number not in self._entry_map:
            self._handle(context, hits)
            return acts.examples.ProcessCode.SUCCESS

        idx = self._entry_map[event_number]
        row = self._df.row(idx, named=True)

        true_xs = row["true_x"]
        true_ys = row["true_y"]
        true_zs = row["true_z"]
        times = row["time"]
        particle_ids = row["particle_id"]
        volume_ids = row["volume_id"]
        layer_ids = row["layer_id"]
        surface_ids = row["surface_id"]

        barcode_map = self._particle_reader.last_barcode_map
        momentum_map = self._particle_reader.last_momentum_map

        for i in range(len(true_xs)):
            key = (int(volume_ids[i]), int(layer_ids[i]), int(surface_ids[i]))
            geoid = self._geoid_map.get(key)
            if geoid is None:
                continue

            pid = int(particle_ids[i])
            barcode = barcode_map.get(pid, acts.examples.SimBarcode())

            pos4 = acts.Vector4(
                float(true_xs[i]) * u.mm,
                float(true_ys[i]) * u.mm,
                float(true_zs[i]) * u.mm,
                float(times[i]) * u.mm,
            )

            if pid in momentum_map:
                px, py, pz, e = momentum_map[pid]
            else:
                px, py, pz, e = 0.0, 0.0, 1.0, 1.0

            mom4 = acts.Vector4(px * u.GeV, py * u.GeV, pz * u.GeV, e * u.GeV)

            # before == after: no per-hit energy loss in ColliderML
            hit = acts.examples.SimHit(geoid, barcode, pos4, mom4, mom4, i)
            hits.insert(hit)

        self._handle(context, hits)
        return acts.examples.ProcessCode.SUCCESS


class ColliderMLMeasurementReader(acts.examples.IReader):
    """Reads ColliderML hits as ACTS measurements, bypassing DigitizationAlgorithm.

    For each hit:
    1. Maps (volume_id, layer_id, surface_id) → gen3 GeometryIdentifier via surface_map.parquet.
    2. Projects the truth position onto the matched surface plane (removes the
       ~0.1 mm normal-distance error from gen1 vs gen3 surface placement).
    3. Converts the projected global position to local (loc0, loc1) coordinates
       via dot products with the surface local-frame axes derived from localToGlobal.
    4. Emplaces a VariableBoundMeasurement into the MeasurementContainer.

    The surface local-frame axes are precomputed at startup (one localToGlobal
    call per unique surface); the read() loop is purely numpy dot products.

    """

    def __init__(
        self,
        hitPath: Path | str,
        particleReader: ColliderMLParticleReader,
        surfaceMapPath: Path | str,
        trackingGeometry: acts.TrackingGeometry,
        sigmaLoc0: float = 0.05,
        sigmaLoc1: float = 0.05,
        outputMeasurements: str = "measurements",
        outputMeasParticlesMap: str = "measurement_particles_map",
        outputPartMeasMap: str = "particle_measurements_map",
        level: acts.logging.Level = acts.logging.INFO,
    ):
        acts.examples.IReader.__init__(self, "ColliderMLMeasurementReader", level)

        self._p_reader = particleReader
        self._var0 = sigmaLoc0 ** 2
        self._var1 = sigmaLoc1 ** 2
        self._output_meas = outputMeasurements
        self._output_meas_part = outputMeasParticlesMap
        self._output_part_meas = outputPartMeasMap
        self._gctx = acts.GeometryContext.dangerouslyDefaultConstruct()

        self.last_measurement_count: int = 0
        self.last_mapped_particle_count: int = 0
        self.last_hit_indices: list[int] = []

        self._meas_handle = acts.examples.WriteDataHandle(
            self, acts.examples.MeasurementContainer, "OutputMeasurements"
        )
        self._meas_handle.initialize(self._output_meas)

        self._meas_part_handle = acts.examples.WriteDataHandle(
            self, acts.examples.MeasurementParticlesMap, "OutputMeasParticlesMap"
        )
        self._meas_part_handle.initialize(self._output_meas_part)

        self._part_meas_handle = acts.examples.WriteDataHandle(
            self, acts.examples.ParticleMeasurementsMap, "OutputPartMeasMap"
        )
        self._part_meas_handle.initialize(self._output_part_meas)

        self._df, self._entry_map, self._min_event, self._max_event = (
            _load_parquet_with_event_map(hitPath)
        )

        smap = _read_surface_map(surfaceMapPath)
        geoid_lookup: dict[tuple[int, int, int], int] = {
            (int(r["volume_id"]), int(r["layer_id"]), int(r["surface_id"])): int(r["geoid"])
            for r in smap.iter_rows(named=True)
            if int(r["geoid"]) != 0
        }

        surface_map_raw: dict[int, object] = {}
        trackingGeometry.visitSurfaces(
            lambda s: surface_map_raw.update({s.geometryId.value: s})
            if s.geometryId.sensitive != 0 else None
        )

        # Precompute (center, local-x axis, local-y axis, normal, geoid) per surface key.
        # Local axes are derived from localToGlobal at unit displacements so that
        # globalToLocal = dot(pos - center, axis), which is not exposed in Python bindings.
        gctx = self._gctx
        dx_loc = acts.Vector2(1.0, 0.0)
        dy_loc = acts.Vector2(0.0, 1.0)

        seen_geoids: dict[int, tuple] = {}
        self._surf_cache: dict[tuple[int, int, int], tuple] = {}

        for key, geoid_val in geoid_lookup.items():
            if geoid_val in seen_geoids:
                self._surf_cache[key] = seen_geoids[geoid_val]
                continue
            surf = surface_map_raw.get(geoid_val)
            if surf is None:
                continue
            c_v3 = surf.center(gctx)
            c_np = np.array([float(c_v3[0]), float(c_v3[1]), float(c_v3[2])])
            n_v3 = surf.normal(gctx, c_v3, c_v3)
            n_np = np.array([float(n_v3[0]), float(n_v3[1]), float(n_v3[2])])

            gdx = surf.localToGlobal(gctx, dx_loc, n_v3)
            gdy = surf.localToGlobal(gctx, dy_loc, n_v3)
            xax = np.array([float(gdx[0]) - c_np[0], float(gdx[1]) - c_np[1], float(gdx[2]) - c_np[2]])
            yax = np.array([float(gdy[0]) - c_np[0], float(gdy[1]) - c_np[1], float(gdy[2]) - c_np[2]])

            entry = (c_np, xax, yax, n_np, acts.GeometryIdentifier(geoid_val))
            seen_geoids[geoid_val] = entry
            self._surf_cache[key] = entry

    def availableEvents(self):
        return (self._min_event, self._max_event + 1)

    def read(self, context):
        event_number = context.eventNumber
        container = acts.examples.MeasurementContainer()
        meas_part_map = acts.examples.MeasurementParticlesMap()
        part_meas_map = acts.examples.ParticleMeasurementsMap()
        hit_indices: list[int] = []
        mapped_particle_count = 0

        if event_number not in self._entry_map:
            self.last_measurement_count = 0
            self.last_mapped_particle_count = 0
            self.last_hit_indices = []
            self._meas_handle(context, container)
            self._meas_part_handle(context, meas_part_map)
            self._part_meas_handle(context, part_meas_map)
            return acts.examples.ProcessCode.SUCCESS

        idx = self._entry_map[event_number]
        row = self._df.row(idx, named=True)

        meas_xs = row["x"]
        meas_ys = row["y"]
        meas_zs = row["z"]
        particle_ids = row["particle_id"]
        volume_ids = row["volume_id"]
        layer_ids = row["layer_id"]
        surface_ids = row["surface_id"]

        barcode_map = self._p_reader.last_barcode_map

        for i in range(len(meas_xs)):
            key = (int(volume_ids[i]), int(layer_ids[i]), int(surface_ids[i]))
            entry = self._surf_cache.get(key)
            if entry is None:
                continue

            c_np, xax, yax, n_np, geoid = entry

            pos = np.array([float(meas_xs[i]), float(meas_ys[i]), float(meas_zs[i])])
            delta = pos - c_np
            pos_proj = pos - np.dot(delta, n_np) * n_np
            d_proj = pos_proj - c_np
            loc0 = float(np.dot(d_proj, xax))
            loc1 = float(np.dot(d_proj, yax))

            meas = container.emplaceMeasurement(
                geometryId=geoid,
                indices=[0, 1],
                parameters=[loc0, loc1],
                covariance=[self._var0, self._var1],
            )

            hit_indices.append(i)
            pid = int(particle_ids[i])
            barcode = barcode_map.get(pid)
            if barcode is not None:
                meas_part_map.insert(meas.index, barcode)
                part_meas_map.insert(barcode, meas.index)
                mapped_particle_count += 1

        self.last_measurement_count = len(container)
        self.last_mapped_particle_count = mapped_particle_count
        self.last_hit_indices = hit_indices
        self._meas_handle(context, container)
        self._meas_part_handle(context, meas_part_map)
        self._part_meas_handle(context, part_meas_map)
        return acts.examples.ProcessCode.SUCCESS


class MeasurementSubsetWriter(acts.examples.IAlgorithm):
    """Writes a MeasurementSubset covering all entries in a MeasurementContainer.

    Must follow ColliderMLMeasurementReader in the Sequencer. Exists as a
    separate algorithm because MeasurementSubset holds a const C++ reference
    to its MeasurementContainer; pybind11 disowns the Python container object
    on WriteDataHandle write, so the stable whiteboard reference is only
    available via a ReadDataHandle in a subsequent algorithm step.
    """

    def __init__(
        self,
        inputMeasurements: str = "measurements",
        outputMeasurementSubset: str = "measurement_subset",
        level: acts.logging.Level = acts.logging.INFO,
    ):
        acts.examples.IAlgorithm.__init__(self, "MeasurementSubsetWriter", level)

        self._read_handle = acts.examples.ReadDataHandle(
            self, acts.examples.MeasurementContainer, "InputMeasurements"
        )
        self._read_handle.initialize(inputMeasurements)

        self._write_handle = acts.examples.WriteDataHandle(
            self, acts.examples.MeasurementSubset, "OutputMeasurementSubset"
        )
        self._write_handle.initialize(outputMeasurementSubset)

    def execute(self, context):
        container = self._read_handle(context.eventStore)
        subset = acts.examples.MeasurementSubset(container, list(range(len(container))))
        self._write_handle(context, subset)
        return acts.examples.ProcessCode.SUCCESS


class PredictionReader(acts.examples.IAlgorithm):
    """Converts a predictions DataFrame into a ProtoTrackContainer per event.

    Maps hit_id (ColliderML index) → measurement index via
    measurement_reader.last_hit_indices, then groups by predicted_track_id.
    Rows with predicted_track_id == -1 are ignored.
    """

    def __init__(
        self,
        predictions: pl.DataFrame,
        measurement_reader: ColliderMLMeasurementReader,
        outputProtoTracks: str = "prototracks",
        level: acts.logging.Level = acts.logging.INFO,
    ):
        acts.examples.IAlgorithm.__init__(self, "PredictionReader", level)
        self._preds = predictions
        self._m_reader = measurement_reader
        self._handle = acts.examples.WriteDataHandle(
            self, acts.examples.ProtoTrackContainer, "OutputProtoTracks"
        )
        self._handle.initialize(outputProtoTracks)

    def execute(self, context):
        hit_to_meas = {h: m for m, h in enumerate(self._m_reader.last_hit_indices)}
        tracks: dict[int, list[int]] = {}
        for row in self._preds.filter(
            (pl.col("event_id") == context.eventNumber)
            & (pl.col("predicted_track_id") >= 0)
        ).iter_rows(named=True):
            meas_idx = hit_to_meas.get(int(row["hit_id"]))
            if meas_idx is not None:
                tracks.setdefault(int(row["predicted_track_id"]), []).append(meas_idx)
        container = acts.examples.ProtoTrackContainer()
        for meas_indices in tracks.values():
            t = acts.examples.ProtoTrack()
            for idx in meas_indices:
                t.append(idx)
            container.append(t)
        self._handle(context, container)
        return acts.examples.ProcessCode.SUCCESS
