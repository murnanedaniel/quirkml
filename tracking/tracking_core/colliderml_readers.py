"""ColliderML <-> pyacts glue for the eval harness.

The actual ColliderML -> ACTS translation (particles, tracker hits,
measurements, geometry-id remapping) is done by
`acts.examples.dataset.addColliderML`, backed by
`acts.examples.arrow.ColliderMLRelease1InputConverter`. That converter
guarantees measurement index `i` in its `measurements`/`measurement_subset`
output corresponds exactly to ColliderML hit row `i` in the input hits
table for that event -- it does not silently drop or reindex a hit it
can't place (an unmappable geometry id or an out-of-bounds projected
position aborts the whole run instead), specifically so this identity
holds for every measurement it does emit. That's what lets
`PredictionReader` below turn `predictions.parquet`'s `hit_id` directly
into a measurement index with no bridging table at all.

This module holds what layers on top of that:

- `load_tracking_geometry_from_json`: load the gen3 ODD TrackingGeometry
  addColliderML needs (its `trackingGeometry` argument), optionally
  decorated with a material map for the KF fit's multiple-scattering/
  energy-loss corrections.
- `PredictionReader`: turn a predictions.parquet DataFrame (see
  `tracking_core.schema`) into a `ProtoTrackContainer` for scoring.
"""

from __future__ import annotations

import contextlib
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import acts
    import polars as pl


@contextlib.contextmanager
def _plain_path(path: Path | str, suffix: str = ""):
    """Yield a filesystem path to decompressed content for `path`.

    ACTS's file readers (`TrackingGeometryJsonConverter.fromFile`,
    `JsonMaterialDecorator`, `loadGeoIdMapFromCsv`) take a real filesystem
    path and read it themselves -- they don't decompress a `.gz`/`.zst`
    path, so a compressed input needs decompressing to a plain temp file
    first. `suffix` picks the temp file's extension (e.g. ".json", ".csv")
    for readers that care about it. Yields the original path unchanged if
    it isn't `.gz`/`.zst`.
    """
    import gzip
    import tempfile

    p = Path(path)
    if p.suffix == ".gz":
        with gzip.open(p, "rb") as f, tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
            tmp.write(f.read())
            tmp.flush()
            yield tmp.name
    elif p.suffix == ".zst":
        import zstandard  # eval extra

        with zstandard.open(p, "rb") as f, tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
            tmp.write(f.read())
            tmp.flush()
            yield tmp.name
    else:
        yield str(p)


def _apply_material_map(trackingGeometry: acts.TrackingGeometry, material_json: Path | str) -> None:
    """Decorate `trackingGeometry`'s portals with material from a JSON material map.

    `TrackingGeometryJsonConverter.fromFile` only reconstructs topology
    (geometry ids, surfaces, volumes) -- no material; gen3's Blueprint-built
    geometry doesn't accept a material decorator at construction time the
    way gen1's `getOpenDataDetector` does. Material has to be attached
    afterwards via the generic `TrackingGeometryMutableVisitor` pattern
    instead (mirrors the ACTS Examples reference pipeline
    `colliderml_full_chain.py`).

    Decorates `visitPortal`'s surfaces, not `visitSurface`'s -- gen3's
    sensitive module surfaces don't carry material themselves; the ODD's
    material (beam pipe, support structures, services) lives on portal
    surfaces. Confirmed empirically: decorating sensitive surfaces mutates
    the object handed to the visitor but the mutation never persists into
    the geometry's real navigation structures (`geoIdSurfaceMap()` reads
    back `None` for all of them regardless); decorating portals does
    persist (into `geoIdSurfaceMap()`) and measurably changes a downstream
    KF fit's covariance.
    """
    import acts
    import acts.ActsPluginsPythonBindingsJson as pj

    class _PortalMaterialDecoratorVisitor(acts.TrackingGeometryMutableVisitor):
        def __init__(self, decorator):
            super().__init__()
            self.decorator = decorator

        def visitPortal(self, portal):
            self.decorator.decorate(portal.surface)

    with _plain_path(material_json, suffix=".json") as plain_path:
        decorator = pj.JsonMaterialDecorator(
            rConfig=pj.MaterialMapJsonConverter.Config(),
            jFileName=plain_path,
            level=acts.logging.WARNING,
        )
        trackingGeometry.apply(_PortalMaterialDecoratorVisitor(decorator))


def detector_subregion_volumes(odd_json: Path | str) -> dict[str, set[int]]:
    """Group the ODD's sensitive-layer volume ids by detector subregion.

    Reads the raw JSON export directly (the pyacts `TrackingVolume`
    binding doesn't expose volume names, only `load_tracking_geometry_from_json`'s
    input does), grouping by substring match on each volume's ACTS name
    (e.g. `"OpenDataTracker|Pixels|PixelBarrel|PixelLayer0"`). Only the
    three subdetectors' actual sensitive-layer volumes are matched, not
    their `::GapN` navigation-gap siblings, which carry no measurements.

    Feeds `PythonPatternRecognitionPerformanceWriter.Config.subDetectorTrackSummaryVolumes`
    for the per-region track-quality report pages.
    """
    import json

    _MARKERS = {
        "OpenDataTracker|Pixels|": "pixel",
        "OpenDataTracker|ShortStrips|": "shortstrip",
        "OpenDataTracker|LongStrips|": "longstrip",
    }
    groups: dict[str, set[int]] = {region: set() for region in _MARKERS.values()}
    with _plain_path(odd_json, suffix=".json") as plain_path:
        data = json.loads(Path(plain_path).read_text())
    for volume in data["volumes"]:
        for marker, region in _MARKERS.items():
            if marker in volume["name"]:
                groups[region].add(volume["volume_id"])
    return groups


def load_tracking_geometry_from_json(
    jsonPath: Path | str,
    material_json: Path | str | None = None,
) -> acts.TrackingGeometry:
    """Load the ODD gen3 TrackingGeometry from its JSON export (.json/.json.gz/.json.zst).

    If `material_json` is given, the geometry's portals are decorated with
    material from that JSON material map (.json/.json.gz/.json.zst -- see
    `_apply_material_map`), needed for the KF fit's multiple-scattering and
    energy-loss corrections to be physically meaningful rather than a
    no-op on a material-less geometry.
    """
    import acts
    import acts.ActsPluginsPythonBindingsJson as pj

    gctx = acts.GeometryContext.dangerouslyDefaultConstruct()
    conv = pj.TrackingGeometryJsonConverter(pj.TrackingGeometryJsonConverter.Config())

    with _plain_path(jsonPath, suffix=".json") as plain_path:
        trackingGeometry = conv.fromFile(gctx, plain_path)

    if material_json is not None:
        _apply_material_map(trackingGeometry, material_json)

    return trackingGeometry


def make_prediction_reader(
    predictions: pl.DataFrame,
    level: acts.logging.Level,
    event_id_remap: Mapping[int, int] | None = None,
) -> acts.examples.IAlgorithm:
    """Build the `PredictionReader` algorithm (needs `acts` importable; see module docstring).

    `predictions`'s `event_id` column is matched against ACTS's Sequencer
    `context.eventNumber` (a 0-based position within the hits/particles
    directory being read) -- true by construction for any predictions
    written against a contiguous event slice starting at 0, where position
    and the underlying dataset's own event id are numerically identical.
    `event_id_remap` (position -> the id predictions.parquet actually
    carries) is needed only when that identity doesn't hold, e.g. a
    non-contiguous/reindexed event selection.

    Raises ValueError for an event without predictions, or with a hit_id
    beyond the event's measurements: both mean the predictions don't
    belong to the events being read.
    """
    import acts
    import acts.examples
    import polars as pl

    # Quirk pilot: split once by event instead of filtering the whole frame for every event.
    by_event = {int(key[0]): frame for key, frame in predictions.partition_by("event_id", as_dict=True).items()}

    class PredictionReader(acts.examples.IAlgorithm):
        """Converts a predictions DataFrame into a ProtoTrackContainer per event.

        `hit_id` is used directly as the measurement index -- see this
        module's docstring for why that's safe. Rows with
        `predicted_track_id == -1` are ignored.
        """

        def __init__(self):
            acts.examples.IAlgorithm.__init__(self, "PredictionReader", level)
            self._handle = acts.examples.WriteDataHandle(
                self, acts.examples.ProtoTrackContainer, "OutputProtoTracks"
            )
            self._handle.initialize("prototracks")
            self._measurements = acts.examples.ReadDataHandle(
                self, acts.examples.MeasurementContainer, "InputMeasurements"
            )
            self._measurements.initialize("measurements")

        def execute(self, context):
            position = context.eventNumber
            event_id = position if event_id_remap is None else event_id_remap[position]
            event = by_event.get(event_id)
            if event is None or event.is_empty():
                raise ValueError(
                    f"no predictions for event_id {event_id} (Sequencer event {position}); "
                    "a non-contiguous event selection needs event_id_remap"
                )
            max_hit_id = event["hit_id"].max()
            n_measurements = len(self._measurements(context.eventStore))
            if max_hit_id >= n_measurements:
                raise ValueError(
                    f"event_id {event_id} (Sequencer event {position}): hit_id {max_hit_id} "
                    f"but the event has {n_measurements} measurements"
                )
            tracks: dict[int, list[int]] = {}
            for row in event.filter(pl.col("predicted_track_id") >= 0).iter_rows(named=True):
                tracks.setdefault(int(row["predicted_track_id"]), []).append(int(row["hit_id"]))
            container = acts.examples.ProtoTrackContainer()
            for meas_indices in tracks.values():
                t = acts.examples.ProtoTrack()
                for idx in meas_indices:
                    t.append(idx)
                container.append(t)
            self._handle(context, container)
            return acts.examples.ProcessCode.SUCCESS

    return PredictionReader()
