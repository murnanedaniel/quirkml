"""ACTS evaluation chain: the only evaluator of track candidates in this repository.

:func:`dump_eval_histograms` reads a predictions.parquet written by any
baseline or model, runs the ODD chain (prototracks, seeding, Kalman fit,
TrackTruthMatcher, the ACTS performance writers) against ColliderML truth
and pickles every ACTS histogram. :mod:`tracking_core.summary` turns that
bundle into summary.json and reads it back; :mod:`tracking_core.report`
renders the optional PDF. No matching or metric is computed outside ACTS:
the summary only sums ACTS histogram entries.

Particle selections (efficiency denominators, :data:`SELECTIONS`), each its
own ACTS ParticleSelector, TrackTruthMatchers and performance writers in the
same Sequencer pass:

- ``physics`` (the headline): pT > 1 GeV, |eta| < 3, charged
  (``removeNeutral``), primary (``removeSecondaries``), prompt (production
  vertex ``rhoMax`` 2 mm, ``absZMax`` 200 mm), no measurement requirement
  (``measurementsMin`` 0); hard scatter and pileup;
- ``physics_hs``: physics, hard-scatter vertex only (``minPrimaryVertexId`` 1,
  ``maxPrimaryVertexId`` 2: ColliderML ``vertex_primary`` 1);
- ``technical``: the same charged-primary, prompt selection as ``physics``,
  additionally requiring at least 7 measurements (configurable).

Before 2026-10-07, technical selected any particle passing the kinematic and
measurement cuts. Stored selection dictionaries distinguish those older results.

The default "proto"/"kf" levels match at the very_loose working point
(matchingRatio 0.5, double matching) and the other working points add
"proto@<name>"/"kf@<name>" levels from the same tracks and KF pass. Level names
carry the selection, "proto[physics]@tight"; technical levels keep the bare
names ("proto@tight"), see :mod:`tracking_core.summary`.

    Particles enter the ACTS particle collection only when ColliderML records at
least one tracker hit for them (ColliderMLRelease1InputConverter
``keepParticlesWithoutHits`` false, its default). ColliderML marks generator
intermediate states (rho, K*, Delta, D and B hadrons, decayed before reaching
the tracker) as primary; this keeps them out of the physics denominator. On
the checked PU50 event no stable charged particle (pion, kaon, proton, lepton)
passing the physics cuts was left without hits.

The built-in truth tracks use the technical selection. They are not an
unrestricted physics oracle. Selection-specific matching changes efficiency
denominators, not the track population of the KF track-summary histograms.

Quirk pilot additions (quirkml/tracking, 2026-10-16): the ``quirk`` and ``all`` selections
(:func:`selection_cuts`), a choice of samples, a wide eta binning for selections without an eta cut, and
per-event efficiency counts read from the ACTS writers during the run (``per_event_selections``).

Install the optional pyacts dependency::

    uv pip install -e "packages/tracking_core[eval]"
"""

from __future__ import annotations

import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from .cli import DEFAULT_MIN_MEASUREMENTS
from .histogram_counts import track_classification
from .schema import RunMeta, read_predictions, read_predictions_meta
from .summary import DEFAULT_SELECTION, LEGACY_SELECTION, SELECTIONS, level_key, split_level

if TYPE_CHECKING:
    import acts.examples

#: Samples always scored; "reference" is added only when tracks_dir is given.
_ALWAYS_ON_SAMPLES = ("truth", "input")

#: Matching working points, as TrackTruthMatcher (matchingRatio, doubleMatching).
#: very_loose: purity and completeness >= 50%; loose: purity >= 75% (the CMS quickTrackAssociatorByHits
#: default cut, purity only); tight: both >= 90%; perfect: both 100%.
WORKING_POINTS = {
    "very_loose": (0.5, True),
    "loose": (0.75, False),
    "tight": (0.9, True),
    "perfect": (1.0, True),
}

#: The working point of the default "proto"/"kf" levels. Every other working point adds levels
#: "proto@<name>" and "kf@<name>" from the same tracks and the same KF pass; the default one is not
#: matched a second time.
DEFAULT_WORKING_POINT = "very_loose"

#: TrackTruthMatcher's hit-majority threshold of the default levels (double matching). Kept as a name so
#: report.py can quote it.
TRUTH_MATCHING_RATIO = WORKING_POINTS[DEFAULT_WORKING_POINT][0]

#: Selected-particle defaults (efficiency denominator) and the reco-track measurement cut.
DEFAULT_PT_MIN_GEV = 1.0
DEFAULT_ETA_MAX = 3.0

#: Prompt production-vertex window of the physics selections, an analysis choice (Daniel, 2026-10-05): the
#: particle's production vertex within rho < 2 mm and |z| < 200 mm of the beamline origin (ParticleSelector
#: rhoMax, absZMax).
PROMPT_RHO_MAX_MM = 2.0
PROMPT_ABS_Z_MAX_MM = 200.0

#: ColliderML vertex_primary of the hard scatter (docs/dataset.md; the synthesised pileup levels keep it as vertex 1,
#: vertex_subset.select_collision_particles). physics_hs keeps minPrimaryVertexId <= vertex < maxPrimaryVertexId.
HARD_SCATTER_VERTEX_ID = 1

#: ParticleSelector.Config fields in mm or GeV, converted with acts.UnitConstants when the chain is built.
_SELECTOR_UNITS = {"ptMin": "GeV", "rhoMax": "mm", "absZMax": "mm", "mMin": "GeV"}

#: Selections used unless the caller names others (the original three of the ML-Tracking-Review evaluator).
DEFAULT_SELECTIONS = ("physics", "physics_hs", "technical")

#: Quirk selection: |pdg| 10000100 has mass 200-400 GeV in the pilot; at mu = 0 no other particle with hits is
#: heavier than this mass cut. Quirks are charged primaries (ColliderML primary = True, parent_id -1). No pT, eta or
#: prompt cut: quirk tracks are not helices and may start anywhere the pair is produced.
QUIRK_MASS_MIN_GEV = 10.0
QUIRK_MIN_MEASUREMENTS = 3
QUIRK_PDG = 10000100

#: Selections without an eta cut get this eta axis on their efficiency / fake / duplicate histograms, so that no
#: selected particle falls outside the axis (ACTS drops out-of-range entries from some histograms).
WIDE_ETA_MAX = 10.0
_NO_ETA_CUT_SELECTIONS = ("quirk", "all")  # also no pT cut: EffPlotTool minTruthPt set to 0 for them


def selection_cuts(name: str, pt_min_gev: float = DEFAULT_PT_MIN_GEV, eta_max: float = DEFAULT_ETA_MAX,
                   min_measurements: int = DEFAULT_MIN_MEASUREMENTS) -> dict:
    """The ACTS ParticleSelector.Config fields of particle selection `name` (lengths in mm, momenta in GeV)."""
    kinematics = {"ptMin": float(pt_min_gev), "absEtaMax": float(eta_max)}
    physics = {**kinematics, "removeNeutral": True, "removeSecondaries": True, "rhoMax": PROMPT_RHO_MAX_MM,
               "absZMax": PROMPT_ABS_Z_MAX_MM, "measurementsMin": 0}
    if name == "technical":
        return {**physics, "measurementsMin": int(min_measurements)}
    if name == "physics":
        return physics
    if name == "physics_hs":
        return {**physics, "minPrimaryVertexId": HARD_SCATTER_VERTEX_ID, "maxPrimaryVertexId": HARD_SCATTER_VERTEX_ID + 1}
    if name == "quirk":
        # Charged, primary, heavy, at least QUIRK_MIN_MEASUREMENTS measurements; no pT / eta / prompt cut.
        return {"mMin": QUIRK_MASS_MIN_GEV, "removeNeutral": True, "removeSecondaries": True,
                "measurementsMin": QUIRK_MIN_MEASUREMENTS}
    if name == "technical_sm":
        # Technical without quirks: the Standard Model particles of a quirk event.
        return {**physics, "measurementsMin": int(min_measurements), "excludeAbsPdgs": [QUIRK_PDG]}
    if name == "all":
        # Every particle the converter keeps (at least one hit): every track with a truth hit is then matched,
        # duplicate or fake, so the fake rate covers all tracks with truth hits.
        return {}
    raise ValueError(f"Unknown particle selection {name!r}; known: {SELECTIONS}.")


def add_particle_selections(sequencer, selections: Mapping[str, Mapping]) -> None:
    """Add one ACTS ParticleSelector per selection ({name: selection_cuts}) reading the addColliderML collections
    "particles" and "particle_measurements_map" and writing `selected_particles_key(name)`. Every cut is a native
    ParticleSelector.Config option; nothing is selected outside ACTS."""
    import acts
    import acts.examples

    u = acts.UnitConstants
    for name, cuts in selections.items():
        config = acts.examples.ParticleSelector.Config()
        config.inputParticles = "particles"
        config.inputParticleMeasurementsMap = "particle_measurements_map"
        config.outputParticles = selected_particles_key(name)
        for option, value in cuts.items():
            unit = _SELECTOR_UNITS.get(option)
            setattr(config, option, value * getattr(u, unit) if unit else value)
        sequencer.addAlgorithm(acts.examples.ParticleSelector(config, acts.logging.WARNING))


def selected_particles_key(name: str) -> str:
    """Whiteboard name of selection `name`'s particles (technical keeps the historical "selected_particles")."""
    return "selected_particles" if name == LEGACY_SELECTION else f"selected_particles_{name}"


def extra_working_points(working_points: Mapping[str, tuple[float, bool]] | None) -> dict[str, tuple[float, bool]]:
    """The working points that need their own matcher: those not equal to the default levels' matching."""
    default = (TRUTH_MATCHING_RATIO, True)
    return {name: (float(value[0]), bool(value[1])) for name, value in (working_points or {}).items()
            if (float(value[0]), bool(value[1])) != default}


def level_working_points(levels) -> dict[str, str]:
    """{level: working point name} for the proto/kf levels of a bundle ("proto" -> very_loose,
    "proto[physics]@tight" -> tight)."""
    out = {}
    for level in levels:
        base, _, point = split_level(level)
        if base in ("proto", "kf"):
            out[level] = point or DEFAULT_WORKING_POINT
    return out

#: Gen3 ODD volume ids for pixel + short-strip (confirmed against
#: odd.json.zst). Long-strip excluded: its 1D measurements need stereo
#: pairing across layers to form space points, not implemented.
_PIXEL_SHORTSTRIP_VOLUMES = (
    9, 10, 11, 12, 21, 22, 23, 24, 25, 26, 27, 36, 37, 38, 39, 40, 41, 42,  # pixel
    48, 49, 50, 51, 59, 60, 61, 62, 63, 64, 72, 73, 74, 75, 76, 77,  # shortstrip
)

#: Reco track pT cut (GeV), below pt_min_gev to give headroom against
#: fit-resolution smearing at the particle-selection boundary.
RECO_PT_CUT = 0.5


def _run_perf_writers(
    predictions_dir: Path | str,
    hits_dir: Path | str,
    particles_dir: Path | str,
    geoid_map: Path | str,
    odd_json: Path | str,
    material_map: Path | str | None,
    num_events: int,
    pt_min_gev: float,
    eta_max: float,
    min_measurements: int,
    bfield_tesla: float,
    tracks_dir: Path | str | None = None,
    event_id_remap: Mapping[int, int] | None = None,
    num_threads: int = 1,
    skip_events: int = 0,
    working_points: Mapping[str, tuple[float, bool]] | None = None,
    selections: Mapping[str, Mapping] | None = None,
    samples: tuple[str, ...] = _ALWAYS_ON_SAMPLES,
    per_event_selections: tuple[str, ...] = (),
    predictions_df: pl.DataFrame | None = None,
):
    """Drive the ODD KF chain once and return the raw performance writers.

    Called by :func:`dump_eval_histograms`, so the (expensive) Sequencer
    run happens once per evaluation. `working_points` here are the extra
    matchers only (see `extra_working_points`). `selections` is {name:
    ParticleSelector cuts} (default: every :data:`SELECTIONS` entry from
    :func:`selection_cuts`); it must hold the technical selection, which
    TruthTrackFinder, the reco-track count and the fit writer use.

    Scores up to three hit-assignment samples through the identical
    downstream chain: "truth" (TruthTrackFinder), "input"
    (predictions.parquet, see `colliderml_readers.make_prediction_reader`),
    and "reference" (ColliderML's published tracks, only if `tracks_dir`
    is given).

    `event_id_remap`: see `make_prediction_reader`'s docstring (needed
    only for a non-contiguous/reindexed event selection). `num_threads`
    is `Sequencer.numThreads` (wall time only, results unchanged).
    `skip_events` is `Sequencer.skip` (`context.eventNumber` stays
    absolute) -- lets a bad event be excluded by running the slices
    around it separately and summing their histograms.

    `samples` (quirk pilot): the samples to score, a subset of
    ("truth", "input", "reference"). `per_event_selections`: for the
    "input" proto levels of these selections, record per Sequencer event
    the ACTS trackeff_vs_eta accepted/total increments (needs
    num_threads 1, see `_PerEventProbe`). `predictions_df`: an already
    read predictions frame (otherwise read from `predictions_dir`).

    Returns a fifth element, `per_event` (quirk pilot): {"positions",
    "n_measurements", "n_selected": {selection: [...]},
    "quirk_particles": [[(particle, pdg, charge, mass_gev), ...]],
    "levels": {level: {"accepted": [...], "total": [...]}}}, lists in
    Sequencer-event order.

    Returns ``(perf_writers, track_stats, meta, prediction_event_ids)``:
    ``perf_writers`` is
    ``{sample: {level: PythonPatternRecognitionPerformanceWriter,
    "fit": PythonTrackFitterPerformanceWriter}}`` with levels named by
    :func:`tracking_core.summary.level_key`; ``track_stats`` is
    ``{sample: {"prototracks": float, "prototracks_per_event": float,
    "selected_fitted_tracks_per_event": float}}`` (per-event values averaged over
    `num_events`); ``meta`` and the sorted ``prediction_event_ids`` describe
    the predictions, so callers needn't read them again.
    """
    try:
        import acts
        import acts.examples
        import acts.examples.scipy
        from acts.examples.dataset import addColliderML
    except ImportError as exc:
        raise RuntimeError(
            "pyacts is required. Install with: "
            "`uv pip install -e \"packages/tracking_core[eval]\"`"
        ) from exc

    from .colliderml_readers import (
        _plain_path,
        detector_subregion_volumes,
        load_tracking_geometry_from_json,
        make_prediction_reader,
    )

    u = acts.UnitConstants
    tg = load_tracking_geometry_from_json(odd_json, material_json=material_map)
    subregion_volumes = detector_subregion_volumes(odd_json)
    field = acts.ConstantBField(acts.Vector3(0, 0, bfield_tesla * u.T))
    if predictions_df is None:
        predictions_df, meta = read_predictions(predictions_dir)
    else:
        meta = read_predictions_meta(predictions_dir)
    if per_event_selections and num_threads != 1:
        raise ValueError("per-event efficiency counts need num_threads 1 (events in order).")
    prediction_event_ids = sorted(predictions_df["event_id"].unique().to_list())

    have_reference = tracks_dir is not None
    samples = [sample for sample in samples if sample != "reference"] + (["reference"] if have_reference else [])
    if "input" not in samples:
        raise ValueError("The input sample is required.")

    with tempfile.TemporaryDirectory() as tmpdir:
        s = acts.examples.Sequencer(
            events=num_events,
            skip=skip_events,
            numThreads=num_threads,
            outputDir=tmpdir,
            logLevel=acts.logging.WARNING,
            # SpacePointMaker/ProtoTracksToParameters below hit a handful of
            # FPEs on the outer strip layers' 1D measurements (see the
            # SpacePointMaker TODO); non-fatal, so don't abort the run.
            failOnUnmaskedFpe=False,
        )
        with _plain_path(geoid_map, suffix=".csv") as plain_geoid_map:
            addColliderML(
                s,
                tg,
                particlesDir=str(particles_dir),
                hitsDir=str(hits_dir),
                tracksDir=str(tracks_dir) if have_reference else None,
                readTracks=have_reference,
                outputTracks="colliderml_tracks",
                geoIdMapPath=Path(plain_geoid_map),
                logLevel=acts.logging.WARNING,
            )
        # ColliderMLRelease1InputConverter guarantees measurement index i ==
        # ColliderML hit_id i for that event (see colliderml_readers'
        # docstring), so predictions.parquet's hit_id can be used as a
        # measurement index directly -- no bridging table needed.
        s.addAlgorithm(make_prediction_reader(predictions_df, acts.logging.WARNING, event_id_remap))

        # Long-strip volumes excluded -- see _PIXEL_SHORTSTRIP_VOLUMES.
        s.addAlgorithm(acts.examples.SpacePointMaker(
            level=acts.logging.WARNING,
            trackingGeometry=tg,
            inputMeasurements="measurement_subset",
            outputSpacePoints="spacepoints",
            geometrySelection=[
                acts.GeometryIdentifier(volume=v) for v in _PIXEL_SHORTSTRIP_VOLUMES
            ],
        ))

        if selections is None:
            selections = {name: selection_cuts(name, pt_min_gev, eta_max, min_measurements) for name in SELECTIONS}
        if LEGACY_SELECTION not in selections:
            raise ValueError("The technical selection is required (truth tracks, reco-track count, fit writer).")
        add_particle_selections(s, selections)

        if "truth" in samples:
            s.addAlgorithm(acts.examples.TruthTrackFinder(
                level=acts.logging.WARNING,
                inputParticles="selected_particles",
                inputParticleMeasurementsMap="particle_measurements_map",
                inputMeasurements="measurements",
                inputSimHits="simhits",
                inputMeasurementSimHitsMap="measurement_simhits_map",
                outputProtoTracks="truth_prototracks",
            ))

        if have_reference:
            s.addAlgorithm(acts.examples.TracksToTrajectories(
                level=acts.logging.WARNING,
                inputTracks="colliderml_tracks",
                outputTrajectories="reference_trajectories",
            ))
            s.addAlgorithm(acts.examples.TrajectoriesToProtoTracks(
                level=acts.logging.WARNING,
                inputTrajectories="reference_trajectories",
                outputProtoTracks="reference_prototracks",
            ))

        # "input" reuses predictions.parquet's prototracks (written above
        # by make_prediction_reader under the fixed name "prototracks").
        prototrack_collections = {
            "truth": "truth_prototracks",
            "input": "prototracks",
            "reference": "reference_prototracks",
        }

        fit_fn = acts.examples.makeKalmanFitterFunction(
            tg, field,
            multipleScattering=True,
            energyLoss=True,
            reverseFilteringMomThreshold=0 * u.GeV,
            reverseFilteringCovarianceScaling=100.0,
            freeToBoundCorrection=acts.examples.FreeToBoundCorrection(False),
            # inf means the KF never marks a bad measurement as an outlier,
            # so nOutliers_vs_eta is trivially zero -- a finite cut is
            # needed for that plot to mean anything. 50.0 is a first pass
            # (100.0 was tried and visibly distorted ColliderML resolution);
            # revisit if outlier/resolution plots look off.
            chi2Cut=50.0,
            useJosephFormulation=False,
            level=acts.logging.WARNING,
        )
        calibrator = acts.examples.makePassThroughCalibrator()

        # No writer exposes prototrack container size directly, so read it
        # off the whiteboard and count per event.
        class _ProtoTrackCounter(acts.examples.IAlgorithm):
            def __init__(self, input_key: str, level: acts.logging.Level) -> None:
                super().__init__(f"ProtoTrackCounter_{input_key}", level)
                self._handle = acts.examples.ReadDataHandle(
                    self, acts.examples.ProtoTrackContainer, "inputProtoTracks"
                )
                self._handle.initialize(input_key)
                self.per_event: list[int] = []
                # Sequencer event number -> (prototracks, hits on them), for the consistency check after s.run().
                self.by_event: dict[int, tuple[int, int]] = {}

            def execute(self, ctx: acts.examples.AlgorithmContext) -> acts.examples.ProcessCode:
                tracks = self._handle(ctx.eventStore)
                self.per_event.append(len(tracks))
                self.by_event[ctx.eventNumber] = (len(tracks), sum(len(track) for track in tracks))
                return acts.examples.ProcessCode.SUCCESS

        proto_counters: dict[str, _ProtoTrackCounter] = {}

        perf_writers: dict[str, dict[str, acts.examples.IWriter]] = {}
        for sample in samples:
            prototracks_key = prototrack_collections[sample]

            proto_counter = _ProtoTrackCounter(prototracks_key, acts.logging.WARNING)
            s.addAlgorithm(proto_counter)
            proto_counters[sample] = proto_counter

            cfg_p2p = acts.examples.ProtoTracksToParameters.Config()
            cfg_p2p.inputProtoTracks = prototracks_key
            cfg_p2p.inputSpacePoints = "spacepoints"
            cfg_p2p.outputParameters = f"{sample}_estimatedparameters"
            cfg_p2p.outputProtoTracks = f"{sample}_remaining_prototracks"
            cfg_p2p.outputSeeds = f"{sample}_seeds"
            cfg_p2p.geometry = tg
            cfg_p2p.magneticField = field
            s.addAlgorithm(acts.examples.ProtoTracksToParameters(cfg_p2p, acts.logging.WARNING))

            s.addAlgorithm(acts.examples.TrackFittingAlgorithm(
                level=acts.logging.WARNING,
                inputMeasurements="measurements",
                inputProtoTracks=f"{sample}_remaining_prototracks",
                inputInitialTrackParameters=f"{sample}_estimatedparameters",
                inputClusters="",
                outputTracks=f"{sample}_kf_tracks",
                pickTrack=-1,
                fit=fit_fn,
                calibrator=calibrator,
            ))

            # Reco track selection (numerator-side): no proto-level
            # equivalent, unfitted prototracks have no fitted pT to cut on.
            track_sel_cfg = acts.TrackSelector.Config()
            track_sel_cfg.ptMin = RECO_PT_CUT * u.GeV
            track_sel_cfg.minMeasurements = min_measurements
            trk_sel_algo_cfg = acts.examples.TrackSelectorAlgorithm.Config()
            trk_sel_algo_cfg.inputTracks = f"{sample}_kf_tracks"
            trk_sel_algo_cfg.outputTracks = f"{sample}_selected_kf_tracks"
            trk_sel_algo_cfg.selectorConfig = track_sel_cfg
            s.addAlgorithm(acts.examples.TrackSelectorAlgorithm(trk_sel_algo_cfg, acts.logging.WARNING))

            p2t_cfg = acts.examples.ProtoTracksToTracks.Config()
            p2t_cfg.inputProtoTracks = prototracks_key
            p2t_cfg.inputMeasurements = "measurement_subset"
            p2t_cfg.outputTracks = f"{sample}_proto_tracks_container"
            s.addAlgorithm(acts.examples.ProtoTracksToTracks(p2t_cfg, acts.logging.WARNING))

            writers: dict[str, acts.examples.IWriter] = {}
            for level, tracks_key in [
                ("proto", f"{sample}_proto_tracks_container"),
                ("kf", f"{sample}_selected_kf_tracks"),
            ]:
                for selection in selections:
                    particles_key = selected_particles_key(selection)
                    # Matching at the default levels' working point, then at every extra one; the same tracks.
                    for point, (ratio, double) in [(None, (TRUTH_MATCHING_RATIO, True)),
                                                   *(working_points or {}).items()]:
                        name = level_key(level, selection, point)
                        key = f"{sample}_{name}"
                        s.addAlgorithm(acts.examples.TrackTruthMatcher(
                            level=acts.logging.WARNING,
                            inputTracks=tracks_key,
                            inputParticles=particles_key,
                            inputMeasurementParticlesMap="measurement_particles_map",
                            outputTrackParticleMatching=f"{key}_track_particle_matching",
                            outputParticleTrackMatching=f"{key}_particle_track_matching",
                            matchingRatio=ratio,
                            doubleMatching=double,
                        ))
                        perf_cfg = acts.examples.PythonPatternRecognitionPerformanceWriter.Config()
                        perf_cfg.inputTracks = tracks_key
                        perf_cfg.inputParticles = particles_key
                        perf_cfg.inputTrackParticleMatching = f"{key}_track_particle_matching"
                        perf_cfg.inputParticleTrackMatching = f"{key}_particle_track_matching"
                        perf_cfg.inputParticleMeasurementsMap = "particle_measurements_map"
                        if selection in _NO_ETA_CUT_SELECTIONS:
                            wide_eta = acts.Axis.regular(round(2 * WIDE_ETA_MAX / 0.2), -WIDE_ETA_MAX, WIDE_ETA_MAX,
                                                         "#eta")
                            for tool in ("effPlotToolConfig", "fakePlotToolConfig", "duplicationPlotToolConfig"):
                                tool_cfg = getattr(perf_cfg, tool)
                                binning = dict(tool_cfg.varBinning)
                                binning["Eta"] = wide_eta
                                tool_cfg.varBinning = binning
                            # EffPlotTool fills its vs-eta / vs-phi histograms only for truth pT above minTruthPt (default
                            # 1 GeV); these selections have no pT cut, so fill them for every selected particle.
                            perf_cfg.effPlotToolConfig.minTruthPt = 0.0
                        if level == "kf" and point is None:
                            # Detector-subregion breakdown -- only meaningful for
                            # fitted tracks (nHoles/nOutliers come from the KF pass).
                            perf_cfg.subDetectorTrackSummaryVolumes = subregion_volumes
                            # Default eta binning is |eta| < 4, wider than the
                            # eta_max particle-selection cut -- narrow it to match
                            # so track-quality plots don't carry empty bins.
                            quality_binning = dict(perf_cfg.trackSummaryPlotToolConfig.varBinning)
                            quality_binning["Eta"] = acts.Axis.regular(
                                round(2 * eta_max / 0.2), -eta_max, eta_max, "#eta"
                            )
                            perf_cfg.trackSummaryPlotToolConfig.varBinning = quality_binning
                        writer = acts.examples.PythonPatternRecognitionPerformanceWriter(perf_cfg, acts.logging.WARNING)
                        s.addWriter(writer)
                        writers[name] = writer

            fit_perf_cfg = acts.examples.PythonTrackFitterPerformanceWriter.Config()
            fit_perf_cfg.inputTracks = f"{sample}_selected_kf_tracks"
            fit_perf_cfg.inputParticles = selected_particles_key(LEGACY_SELECTION)
            fit_perf_cfg.inputTrackParticleMatching = f"{sample}_kf_track_particle_matching"
            fit_perf_cfg.fitFunction = acts.examples.scipy.makeScipyHistogramFitFunction()
            # Same eta-binning narrowing as trackSummaryPlotToolConfig above.
            res_binning = dict(fit_perf_cfg.resPlotToolConfig.varBinning)
            res_binning["Eta"] = acts.Axis.regular(round(2 * eta_max / 0.2), -eta_max, eta_max, "#eta")
            # Upstream Residual_* defaults are too narrow for ColliderML/ODD
            # and silently drop out-of-range entries rather than clip, so
            # the per-eta-bin Gaussian fit starves. Ranges below were sized
            # empirically to comfortably contain each eta bin's true width.
            for key, (nbins, lo, hi) in {
                "Residual_d0": (400, -2.0, 2.0),
                "Residual_z0": (400, -10.0, 10.0),
                "Residual_phi": (400, -0.05, 0.05),
                "Residual_theta": (400, -0.02, 0.02),
                "Residual_qop": (400, -0.05, 0.05),
                "Residual_qopt": (400, -1.5, 1.5),
                "Residual_qopt_rel": (400, -0.3, 0.3),
            }.items():
                res_binning[key] = acts.Axis.regular(nbins, lo, hi, res_binning[key].label)
            fit_perf_cfg.resPlotToolConfig.varBinning = res_binning
            fit_writer = acts.examples.PythonTrackFitterPerformanceWriter(fit_perf_cfg, acts.logging.WARNING)
            s.addWriter(fit_writer)
            writers["fit"] = fit_writer

            perf_writers[sample] = writers

        probe = _PerEventProbe(
            {name: writer for name, writer in perf_writers["input"].items()
             if split_level(name)[0] == "proto" and split_level(name)[1] in per_event_selections},
            [name for name in selections], acts.logging.WARNING)
        s.addAlgorithm(probe)

        s.run()
        per_event = probe.finish()

        # The "input" prototracks ACTS scored must be exactly the predictions on disk. Under heavy node load
        # (tens of concurrent evaluations) some runs scored fewer or altered prototracks without any error
        # (reports/acts48/diagnose, 2026-10-01); fail instead of returning those numbers.
        assigned = predictions_df.filter(pl.col("predicted_track_id") >= 0)
        expected = {
            row["event_id"]: (row["tracks"], row["hits"])
            for row in assigned.group_by("event_id").agg(
                pl.col("predicted_track_id").n_unique().alias("tracks"), pl.len().alias("hits")
            ).iter_rows(named=True)
        }
        for position, seen in sorted(proto_counters["input"].by_event.items()):
            event_id = position if event_id_remap is None else event_id_remap[position]
            if seen != expected.get(event_id, (0, 0)):
                raise RuntimeError(
                    f"ACTS read {seen[0]} prototracks / {seen[1]} hits for event_id {event_id} (Sequencer event "
                    f"{position}) but the predictions hold {expected.get(event_id, (0, 0))}; result discarded."
                )

        track_stats: dict[str, dict[str, float]] = {}
        for sample in samples:
            n_proto = sum(proto_counters[sample].per_event)
            # nMeasurements_vs_eta is filled once per selected KF-fitted
            # track with its eta -- summing per-bin counts gives the
            # container size with no extra writer needed.
            n_measurements_hist = perf_writers[sample]["kf"].histograms()["nMeasurements_vs_eta"].histogram
            n_fitted = float(np.asarray(n_measurements_hist.counts()).sum())
            track_stats[sample] = {
                "prototracks": float(n_proto),
                "prototracks_per_event": n_proto / num_events,
                "selected_fitted_tracks_per_event": n_fitted / num_events,
            }

    return perf_writers, track_stats, meta, prediction_event_ids, per_event


def _efficiency_counts(writer) -> tuple[float, float]:
    """(accepted, total) of a pattern-recognition writer's trackeff_vs_eta, flow bins included."""
    from .histogram_counts import histogram_counts

    counts = histogram_counts(_to_bh(writer.histograms()["trackeff_vs_eta"]))
    return counts["accepted"], counts["total"]


def _make_probe_class():
    import acts
    import acts.examples

    class PerEventProbe(acts.examples.IAlgorithm):
        """Per-event counts read from the whiteboard and from the ACTS performance writers (quirk pilot).

        The pyacts 48 Sequencer runs readers, algorithms and writers of an event in the order they were added; this
        probe is added after every writer, so at event k the writers hold the sums over events up to and including k
        (checked at the end: the writers do not change after the last probe call). With one thread, events run in
        order, so the increment between consecutive snapshots is the ACTS count of one event. No matching is done here:
        accepted / total are TrackTruthMatcher results as filled into trackeff_vs_eta by
        PatternRecognitionPerformanceCollector.

        Also records, per event: the number of measurements (to check against the hits table), the number of
        selected particles of each selection (to check that no selected particle left the histogram axes) and the
        particles of the quirk selection (particle number, pdg, charge, mass in GeV).
        """

        def __init__(self, writers, selections, level):
            super().__init__("PerEventProbe", level)
            self._writers = dict(writers)
            self._measurements = acts.examples.ReadDataHandle(self, acts.examples.MeasurementContainer, "M")
            self._measurements.initialize("measurements")
            self._selected = {}
            for name in selections:
                handle = acts.examples.ReadDataHandle(self, acts.examples.SimParticleContainer, f"P_{name}")
                handle.initialize(selected_particles_key(name))
                self._selected[name] = handle
            self.positions, self.n_measurements, self.quirk_particles = [], [], []
            self.n_selected = {name: [] for name in selections}
            self._snapshots = []
            self._last = None

        def _snapshot(self):
            return {name: _efficiency_counts(writer) for name, writer in self._writers.items()}

        def execute(self, ctx):
            if self._last is not None and ctx.eventNumber != self._last + 1:
                raise RuntimeError(f"PerEventProbe: event {ctx.eventNumber} after {self._last}; events out of order.")
            self._last = ctx.eventNumber
            self.positions.append(ctx.eventNumber)
            self.n_measurements.append(len(self._measurements(ctx.eventStore)))
            for name, handle in self._selected.items():
                particles = handle(ctx.eventStore)
                self.n_selected[name].append(len(particles))
                if name == "quirk":
                    self.quirk_particles.append([
                        (int(p.particleId.particle), int(p.pdg), float(p.charge),
                         float(p.mass / acts.UnitConstants.GeV)) for p in particles])
            if self._writers:
                self._snapshots.append(self._snapshot())
            return acts.examples.ProcessCode.SUCCESS

        def finish(self):
            levels = {}
            if self._writers:
                final = self._snapshot()
                if self._snapshots and final != self._snapshots[-1]:
                    raise RuntimeError("PerEventProbe: writers changed after the last event's probe; the probe did "
                                       "not run after the writers, per-event counts would be shifted.")
                zero = {name: (0.0, 0.0) for name in self._writers}
                previous = [zero] + self._snapshots[:-1]
                for name in self._writers:
                    levels[name] = {
                        "accepted": [now[name][0] - before[name][0] for before, now in zip(previous, self._snapshots)],
                        "total": [now[name][1] - before[name][1] for before, now in zip(previous, self._snapshots)]}
            return {"positions": self.positions, "n_measurements": self.n_measurements,
                    "n_selected": self.n_selected, "quirk_particles": self.quirk_particles, "levels": levels}

    return PerEventProbe


def _PerEventProbe(writers, selections, level):
    return _make_probe_class()(writers, selections, level)


def _to_bh(h):
    """Convert one ACTS performance-writer histogram wrapper to a picklable
    `boost_histogram.Histogram` (or, for `Efficiency1`, a 2-key dict of
    them) -- so `tracking_core.report.render_report` can plot from a
    pickle without importing pyacts. `bh.Histogram(acts_obj)` uses the
    `_to_boost_histogram_` protocol ACTS patches onto `Histogram1/2/3`
    (Weight storage) and `ProfileHistogram1` (Mean storage); axis labels
    ride along as bh axis `metadata`. `Efficiency1` is the one type ACTS
    doesn't patch, but its `.accepted`/`.total` are themselves
    `BoostHistogram`, so it converts as a dict of two.
    """
    import boost_histogram as bh

    if hasattr(h, "accepted"):  # Efficiency1
        return {"accepted": bh.Histogram(h.accepted), "total": bh.Histogram(h.total)}
    # ProfileHistogram1's sampleAxisTitle (its y-axis label) has no bh
    # equivalent -- stash it in the histogram's own metadata dict, which
    # pickles with it.
    return bh.Histogram(h, metadata={"sample_axis_title": getattr(h, "sampleAxisTitle", None)})


def dump_eval_histograms(
    predictions_dir: Path | str,
    hits_dir: Path | str,
    particles_dir: Path | str,
    geoid_map: Path | str,
    odd_json: Path | str,
    output_pickle: Path | str,
    material_map: Path | str | None = None,
    *,
    num_events: int,
    pt_min_gev: float = DEFAULT_PT_MIN_GEV,
    eta_max: float = DEFAULT_ETA_MAX,
    min_measurements: int = DEFAULT_MIN_MEASUREMENTS,
    bfield_tesla: float = 3.0,
    tracks_dir: Path | str | None = None,
    event_id_remap: Mapping[int, int] | None = None,
    num_threads: int = 1,
    skip_events: int = 0,
    working_points: Mapping[str, tuple[float, bool]] | None = WORKING_POINTS,
    selections=DEFAULT_SELECTIONS,
    samples: tuple[str, ...] = _ALWAYS_ON_SAMPLES,
    per_event_selections: tuple[str, ...] = (),
    predictions_df: pl.DataFrame | None = None,
) -> Path:
    """Run the ODD chain once over `num_events` events and pickle every ACTS histogram to `output_pickle`.

    Summaries (:func:`tracking_core.summary.summarize`) and the PDF report
    (:func:`tracking_core.report.render_report`) read the pickle without pyacts.

    Denominators (particle selections, one `ParticleSelector` each, identical
    for every sample): `selections` names entries of :data:`SELECTIONS`
    (default all; see :func:`selection_cuts`). `pt_min_gev` and `eta_max` apply
    to every selection, `min_measurements` to the technical one (and to
    reco tracks). Numerator: a track counts for a particle when
    TrackTruthMatcher matches them (default levels: matchingRatio 0.5,
    double matching, the very_loose working point). KF-fitted tracks must
    first pass reco selection (fitted pT > RECO_PT_CUT, at least
    `min_measurements` measurements); prototracks have no fitted pT.

    `hits_dir`/`particles_dir`/`tracks_dir` are ColliderML object
    directories; `tracks_dir` adds a "reference" sample from the published
    tracks. `geoid_map` is the gen1->gen3 geometry-id CSV; `geoid_map` and
    `material_map` may be `.gz`/`.zst`. `num_threads` is
    `Sequencer.numThreads` (wall time only). `working_points` (default
    WORKING_POINTS; `{}` for the default levels only) adds "proto@<name>" /
    "kf@<name>" levels for every point whose matching differs from the
    default levels'. `event_id_remap`/`skip_events`: see `_run_perf_writers`.

    The bundle also holds `prefit_classification`: matched / duplicate /
    fake prototrack totals per proto level, read from ACTS's particle-side
    histograms (`histogram_counts.track_classification`).
    """
    import pickle

    if num_events < 1 or skip_events < 0:
        raise ValueError("num_events must be positive and skip_events nonnegative.")
    positions = range(skip_events, skip_events + num_events)
    evaluated_event_ids = [event_id_remap[position] if event_id_remap is not None else position
                           for position in positions]
    names = list(selections)
    if LEGACY_SELECTION not in names:
        names.append(LEGACY_SELECTION)
    cuts = {name: selection_cuts(name, pt_min_gev, eta_max, min_measurements) for name in names}
    perf_writers, track_stats, meta, prediction_event_ids, per_event = _run_perf_writers(
        predictions_dir, hits_dir, particles_dir, geoid_map, odd_json, material_map,
        num_events, pt_min_gev, eta_max, min_measurements, bfield_tesla, tracks_dir,
        event_id_remap, num_threads, skip_events, extra_working_points(working_points), cuts,
        tuple(samples), tuple(per_event_selections), predictions_df,
    )

    hists = {
        sample: {
            level: {key: _to_bh(h) for key, h in writer.histograms().items()}
            for level, writer in writers.items()
        }
        for sample, writers in perf_writers.items()
    }
    # Pre-fit (prototrack) classification. ACTS fills fakeRatio/duplicationRatio and purity/completeness
    # only for tracks with a reference surface (none at proto level), so read the per-particle totals its
    # particle loop fills; see histogram_counts.track_classification.
    prefit_classification = {
        sample: {
            level: track_classification(level_hists, track_stats[sample]["prototracks"])
            for level, level_hists in sample_hists.items() if split_level(level)[0] == "proto"
        }
        for sample, sample_hists in hists.items()
    }

    bundle = {
        "meta": {"implementation": meta.implementation, "run_id": meta.run_id,
                 "wrote_at": meta.wrote_at, "config": meta.config},
        "eval_kwargs": dict(
            num_events=num_events, pt_min_gev=pt_min_gev, eta_max=eta_max, min_measurements=min_measurements,
        ),
        "samples": list(perf_writers),
        "selections": cuts,
        "default_selection": DEFAULT_SELECTION if DEFAULT_SELECTION in cuts else LEGACY_SELECTION,
        "provenance": {
            "version": 1,
            **{key: str(Path(value).resolve()) if value is not None else None for key, value in {
                "hits_dir": hits_dir, "particles_dir": particles_dir, "tracks_dir": tracks_dir,
                "geoid_map": geoid_map, "odd_json": odd_json, "material_map": material_map,
            }.items()},
            "bfield_tesla": bfield_tesla,
            "skip_events": skip_events,
            "event_id_remap": sorted((event_id_remap or {}).items()),
            "prediction_event_ids": prediction_event_ids,
            "reco_pt_cut": RECO_PT_CUT,
            "truth_matching_ratio": TRUTH_MATCHING_RATIO,
            "working_points": {name: list(value) for name, value in (working_points or {}).items()},
            "default_working_point": DEFAULT_WORKING_POINT,
            "prompt_vertex_mm": {"rho_max": PROMPT_RHO_MAX_MM, "abs_z_max": PROMPT_ABS_Z_MAX_MM},
            "hard_scatter_vertex_id": HARD_SCATTER_VERTEX_ID,
            "keep_particles_without_hits": False,
            "evaluated_event_ids": evaluated_event_ids,
        },
        "levels": level_working_points(hists["input"]),
        "track_stats": track_stats,
        "prefit_classification": prefit_classification,
        "hists": hists,
        "per_event": per_event,
    }

    output_pickle = Path(output_pickle)
    output_pickle.parent.mkdir(parents=True, exist_ok=True)
    with open(output_pickle, "wb") as f:
        pickle.dump(bundle, f)
    return output_pickle
