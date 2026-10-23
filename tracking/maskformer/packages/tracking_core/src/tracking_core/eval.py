"""Eval harness — wraps pyacts to compute tracking metrics.

:func:`score` reads a predictions.parquet written by any baseline or lego
pipeline, runs a full ODD Kalman-filter chain against a ColliderML shard,
and returns aggregated efficiency metrics.

Install the optional pyacts dependency::

    uv pip install -e "packages/tracking_core[eval]"
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from .schema import read_predictions


def score(
    predictions_dir: Path | str,
    hit_shard: Path | str,
    particle_shard: Path | str,
    surface_map: Path | str,
    odd_json: Path | str,
    num_events: int = 1,
    pt_min_gev: float = 1.0,
    eta_max: float = 3.0,
    min_measurements: int = 5,
    bfield_tesla: float = 2.0,
) -> dict[str, float]:
    """Score predictions against ColliderML truth via the ODD KF chain.

    Drives a pyacts Sequencer over num_events events and returns efficiency
    for both KF-fitted and unfitted proto-tracks. Particle selection
    (pT > pt_min_gev GeV, |eta| < eta_max, measurements >= min_measurements)
    is applied to the denominator.

    Returns dict with keys "kf_efficiency" and "proto_efficiency".
    """
    try:
        import acts
        import acts.examples
    except ImportError as exc:
        raise RuntimeError(
            "pyacts is required. Install with: "
            "`uv pip install -e \"packages/tracking_core[eval]\"`"
        ) from exc

    from .colliderml_readers import (
        ColliderMLMeasurementReader,
        ColliderMLParticleReader,
        MeasurementSubsetWriter,
        PredictionReader,
        load_tracking_geometry_from_json,
    )

    u = acts.UnitConstants
    tg = load_tracking_geometry_from_json(odd_json)
    field = acts.ConstantBField(acts.Vector3(0, 0, bfield_tesla * u.T))
    predictions_df, _ = read_predictions(predictions_dir)

    p_reader = ColliderMLParticleReader(str(particle_shard), level=acts.logging.WARNING)
    m_reader = ColliderMLMeasurementReader(
        str(hit_shard),
        p_reader,
        surfaceMapPath=str(surface_map),
        trackingGeometry=tg,
        level=acts.logging.WARNING,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        s = acts.examples.Sequencer(
            events=num_events,
            numThreads=1,
            outputDir=tmpdir,
            logLevel=acts.logging.WARNING,
        )
        s.addReader(p_reader)
        s.addReader(m_reader)
        s.addAlgorithm(MeasurementSubsetWriter(level=acts.logging.WARNING))
        s.addAlgorithm(PredictionReader(predictions_df, m_reader, level=acts.logging.WARNING))

        # TODO: outer strip layers have 1D measurements; meaningful space points there
        # require combining stereo strip pairs from adjacent layers.
        s.addAlgorithm(acts.examples.SpacePointMaker(
            level=acts.logging.WARNING,
            trackingGeometry=tg,
            inputMeasurements="measurement_subset",
            outputSpacePoints="spacepoints",
            geometrySelection=[acts.GeometryIdentifier()],  # volume=0 → all sensitive surfaces
        ))

        cfg_p2p = acts.examples.ProtoTracksToParameters.Config()
        cfg_p2p.inputProtoTracks = "prototracks"
        cfg_p2p.inputSpacePoints = "spacepoints"
        cfg_p2p.outputParameters = "estimatedparameters"
        cfg_p2p.outputProtoTracks = "remaining-protoTracks"
        cfg_p2p.outputSeeds = "seeds-from-protoTracks"
        cfg_p2p.geometry = tg
        cfg_p2p.magneticField = field
        s.addAlgorithm(acts.examples.ProtoTracksToParameters(cfg_p2p, acts.logging.WARNING))

        s.addAlgorithm(acts.examples.TrackFittingAlgorithm(
            level=acts.logging.WARNING,
            inputMeasurements="measurements",
            inputProtoTracks="remaining-protoTracks",
            inputInitialTrackParameters="estimatedparameters",
            inputClusters="",
            outputTracks="kf_tracks",
            pickTrack=-1,
            fit=acts.examples.makeKalmanFitterFunction(
                tg, field,
                multipleScattering=True,
                energyLoss=True,
                reverseFilteringMomThreshold=0 * u.GeV,
                reverseFilteringCovarianceScaling=100.0,
                freeToBoundCorrection=acts.examples.FreeToBoundCorrection(False),
                chi2Cut=float("inf"),
                useJosephFormulation=False,
                level=acts.logging.WARNING,
            ),
            calibrator=acts.examples.makePassThroughCalibrator(),
        ))

        sel_cfg = acts.examples.ParticleSelector.Config()
        sel_cfg.inputParticles = "particles"
        sel_cfg.inputParticleMeasurementsMap = "particle_measurements_map"
        sel_cfg.outputParticles = "selected_particles"
        sel_cfg.ptMin = pt_min_gev * u.GeV
        sel_cfg.absEtaMax = eta_max
        sel_cfg.measurementsMin = min_measurements
        s.addAlgorithm(acts.examples.ParticleSelector(sel_cfg, acts.logging.WARNING))

        perf_writers = {}
        for tracks_key, prefix in [("kf_tracks", "kf"), ("proto_tracks_container", "proto")]:
            if tracks_key == "proto_tracks_container":
                p2t_cfg = acts.examples.ProtoTracksToTracks.Config()
                p2t_cfg.inputProtoTracks = "prototracks"
                p2t_cfg.inputMeasurements = "measurement_subset"
                p2t_cfg.outputTracks = "proto_tracks_container"
                s.addAlgorithm(acts.examples.ProtoTracksToTracks(p2t_cfg, acts.logging.WARNING))

            s.addAlgorithm(acts.examples.TrackTruthMatcher(
                level=acts.logging.WARNING,
                inputTracks=tracks_key,
                inputParticles="selected_particles",
                inputMeasurementParticlesMap="measurement_particles_map",
                outputTrackParticleMatching=f"{prefix}_track_particle_matching",
                outputParticleTrackMatching=f"{prefix}_particle_track_matching",
                doubleMatching=True,
            ))
            perf_cfg = acts.examples.PythonTrackFinderPerformanceWriter.Config()
            perf_cfg.inputTracks = tracks_key
            perf_cfg.inputParticles = "selected_particles"
            perf_cfg.inputTrackParticleMatching = f"{prefix}_track_particle_matching"
            perf_cfg.inputParticleTrackMatching = f"{prefix}_particle_track_matching"
            perf_cfg.inputParticleMeasurementsMap = "particle_measurements_map"
            writer = acts.examples.PythonTrackFinderPerformanceWriter(perf_cfg, acts.logging.WARNING)
            s.addWriter(writer)
            perf_writers[prefix] = writer

        s.run()

    result = {}
    for prefix, writer in perf_writers.items():
        eff = writer.histograms()["trackeff_vs_pT"]
        total = sum(eff.total.values())
        result[f"{prefix}_efficiency"] = float(sum(eff.accepted.values()) / total) if total > 0 else 0.0
    return result
