"""Score track candidates on a quirk dataset with the ACTS evaluator (tracking_core, pyacts 48).

    python tracking/evaluate.py --predictions-dir P --dataset R1_ROOT --events A B --out O [--jobs N]

P holds predictions.parquet + meta.json (tracking_core/schema.py: event_id, hit_id, predicted_track_id, -1 =
unassigned). Events A <= event_id < B are scored; P may hold more events, but must hold every event of the range.
Writes O/summary.json (the numbers below), O/acts_summary.json (every ACTS efficiency-type histogram sum,
tracking_core.summary layout), O/histograms.pkl (all ACTS histograms) and O/per_event.parquet.

Metrics, at the proto-track level (the predicted hit sets as they are, no fit), for the working points very_loose
(TrackTruthMatcher matchingRatio 0.5, double matching) and tight (0.9, double matching):
  quirk_track_efficiency   matched quirks / quirks in the quirk selection (charged, primary, mass > 10 GeV, i.e.
                           |pdg| 10000100, at least 3 measurements; no pT / eta / prompt cut). Primary figure of merit.
  quirk_event_efficiency   events in which both quirks are matched / events with both quirks in the quirk selection.
                           Per-event counts are read from the ACTS writers during the run (eval._PerEventProbe).
  technical_efficiency     matched / selected particles of the technical selection (pT > 1 GeV, |eta| < 3, charged,
                           primary, production vertex rho < 2 mm and |z| < 200 mm, at least 7 measurements); this
                           includes quirks.
  technical_sm_efficiency  the same without quirks (|pdg| 10000100 excluded): the other particles of the event.
  fake_rate, duplicate_rate  ACTS track classification against all particles with hits ("all" selection): fake and
                           duplicate tracks / all prototracks.
  prototracks_per_event
The Kalman-fit ("kf") levels are kept in acts_summary.json and summary.json["kf_not_meaningful_for_quirks"]: the fit
assumes a helix, which quirk tracks are not.

A Python re-implementation of the TrackTruthMatcher rule cross-checks the per-quirk and per-event ACTS numbers
(summary.json["checks"]); the run fails if they differ.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import polars as pl

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from r1_data import QUIRK_PDG, read_events, stage_events  # noqa: E402

CONFIG = HERE / "config"
WORKING_POINTS = {"very_loose": (0.5, True), "tight": (0.9, True)}
SELECTIONS = ("quirk", "technical", "technical_sm", "all")
DEFAULT_STAGING = Path(os.environ.get("PSCRATCH", "/tmp")) / "quirk_tracking" / "staging"


def _level(selection: str, point: str) -> str:
    from tracking_core.summary import level_key

    return level_key("proto", selection, None if point == "very_loose" else point)


# Section: ACTS run, in chunks of events


def _run_chunk(job: dict) -> str:
    """One ACTS Sequencer pass over staged positions [skip, skip + n); returns the chunk's pickle path."""
    from tracking_core.eval import dump_eval_histograms

    first = job["first"]
    preds = (pl.scan_parquet(Path(job["predictions_dir"]) / "predictions.parquet")
             .filter((pl.col("event_id") >= first + job["skip"]) & (pl.col("event_id") < first + job["skip"] + job["n"]))
             .collect())
    remap = {position: first + position for position in range(job["skip"], job["skip"] + job["n"])}
    dump_eval_histograms(
        predictions_dir=job["predictions_dir"], hits_dir=job["staged"] + "/reco/tracker_hits",
        particles_dir=job["staged"] + "/truth/particles", geoid_map=CONFIG / "geoid_map.csv.zst",
        odd_json=CONFIG / "odd.json.zst", material_map=CONFIG / "material_map.json.zst",
        output_pickle=job["output"], num_events=job["n"], skip_events=job["skip"], event_id_remap=remap,
        num_threads=1, working_points=WORKING_POINTS, selections=SELECTIONS, samples=("input",),
        per_event_selections=("quirk",), predictions_df=preds,
    )
    return job["output"]


def _add(a, b):
    if isinstance(a, dict):
        return {key: _add(a[key], b[key]) for key in a}
    return a + b


def merge_bundles(bundles: list[dict]) -> dict:
    """Sum the histograms of chunk bundles (consecutive event ranges) and recount the derived numbers."""
    from tracking_core.histogram_counts import track_classification
    from tracking_core.summary import split_level

    out = dict(bundles[0])
    out["hists"] = {sample: {level: {key: h for key, h in hists.items()} for level, hists in levels.items()}
                    for sample, levels in bundles[0]["hists"].items()}
    for bundle in bundles[1:]:
        for sample, levels in bundle["hists"].items():
            for level, hists in levels.items():
                for key, h in hists.items():
                    out["hists"][sample][level][key] = _add(out["hists"][sample][level][key], h)
    n_events = sum(b["eval_kwargs"]["num_events"] for b in bundles)
    out["eval_kwargs"] = {**bundles[0]["eval_kwargs"], "num_events": n_events}
    stats = {}
    for sample in bundles[0]["track_stats"]:
        proto = sum(b["track_stats"][sample]["prototracks"] for b in bundles)
        fitted = sum(b["track_stats"][sample]["selected_fitted_tracks_per_event"] * b["eval_kwargs"]["num_events"]
                     for b in bundles)
        stats[sample] = {"prototracks": proto, "prototracks_per_event": proto / n_events,
                         "selected_fitted_tracks_per_event": fitted / n_events}
    out["track_stats"] = stats
    out["prefit_classification"] = {
        sample: {level: track_classification(hists, stats[sample]["prototracks"])
                 for level, hists in levels.items() if split_level(level)[0] == "proto"}
        for sample, levels in out["hists"].items()}
    provenance = dict(bundles[0]["provenance"])
    provenance["evaluated_event_ids"] = [e for b in bundles for e in b["provenance"]["evaluated_event_ids"]]
    provenance["prediction_event_ids"] = sorted({e for b in bundles for e in b["provenance"]["prediction_event_ids"]})
    provenance["skip_events"] = 0
    provenance["event_id_remap"] = []
    out["provenance"] = provenance
    per_event = {"positions": [], "n_measurements": [], "quirk_particles": [],
                 "n_selected": {k: [] for k in bundles[0]["per_event"]["n_selected"]},
                 "levels": {k: {"accepted": [], "total": []} for k in bundles[0]["per_event"]["levels"]}}
    for b in bundles:
        pe = b["per_event"]
        for key in ("positions", "n_measurements", "quirk_particles"):
            per_event[key] += pe[key]
        for k in per_event["n_selected"]:
            per_event["n_selected"][k] += pe["n_selected"][k]
        for k in per_event["levels"]:
            for c in ("accepted", "total"):
                per_event["levels"][k][c] += pe["levels"][k][c]
    out["per_event"] = per_event
    return out


# Section: the same matching rule outside ACTS (cross-check only)


def python_matching(hits: pl.DataFrame, preds: pl.DataFrame, ratio: float, double: bool) -> pl.DataFrame:
    """TrackTruthMatcher's rule on (event_id, hit_id, particle_id) hits and predictions, rows in prototrack order.

    Per track, the majority particle; the track matches it if purity (majority hits / track hits) >= ratio and, with
    double matching, completeness (majority hits / the particle's hits) >= ratio. Ties in the majority count follow
    ACTS (Validation/TrackClassification): the contributing particles are collected in the order of
    trackStatesReversed(), i.e. from the last hit of the prototrack (= last predictions row of the track) backwards,
    then sorted by hit count with std::sort. For up to 16 distinct particles std::sort is an insertion sort, which
    keeps that order, so the tied particle whose last hit comes latest wins. Above 16 the order of ties is not
    defined; such ties are counted in `tied_unordered`.

    Returns (event_id, particle_id, matched_tracks, tied, tied_unordered) for every particle that is the majority
    particle of a track."""
    particle_hits = hits.group_by("event_id", "particle_id").agg(pl.len().alias("particle_hits"))
    assigned = (preds.with_row_index("row").filter(pl.col("predicted_track_id") >= 0)
                .join(hits, on=["event_id", "hit_id"], how="left"))
    if assigned["particle_id"].null_count():
        raise ValueError("predictions hold hit_ids outside the events' hits")
    counts = assigned.group_by("event_id", "predicted_track_id", "particle_id").agg(
        pl.len().alias("n"), pl.col("row").max().alias("last_row"))
    per_track = counts.sort(["n", "last_row"], descending=[True, True]).group_by(
        "event_id", "predicted_track_id", maintain_order=True).agg(
        pl.col("n").sum().alias("track_hits"), pl.col("n").first().alias("n_major"),
        (pl.col("n") == pl.col("n").max()).sum().alias("n_at_max"), pl.len().alias("n_particles"),
        pl.col("particle_id").first().alias("particle_id"))
    per_track = per_track.join(particle_hits, on=["event_id", "particle_id"], how="left").with_columns(
        matched=(pl.col("n_major") / pl.col("track_hits") >= ratio)
        & (~pl.lit(double) | (pl.col("n_major") / pl.col("particle_hits") >= ratio)))
    return per_track.group_by("event_id", "particle_id").agg(
        pl.col("matched").sum().alias("matched_tracks"), (pl.col("n_at_max") > 1).any().alias("tied"),
        ((pl.col("n_at_max") > 1) & (pl.col("n_particles") > 16)).any().alias("tied_unordered"))


# Section: summary


def _ratio(accepted, total):
    if not total:
        return {"eff": None, "err": None, "accepted": accepted, "total": total}
    eff = accepted / total
    return {"eff": eff, "err": (eff * (1 - eff) / total) ** 0.5, "accepted": accepted, "total": total}


def build_summary(bundle, acts_summary, args, predictions_meta, per_event_df, checks) -> dict:
    from tracking_core import summary as S

    out = {
        "dataset": str(Path(args.dataset).resolve()), "events": [args.events[0], args.events[1]],
        "num_events": args.events[1] - args.events[0],
        "predictions_dir": str(Path(args.predictions_dir).resolve()), "predictions_meta": predictions_meta,
        "level": "proto (predicted hit sets, no fit)",
        "working_points": {k: {"matchingRatio": v[0], "doubleMatching": v[1]} for k, v in WORKING_POINTS.items()},
        "selections": acts_summary["selections"],
        "quirk_track_efficiency": {}, "quirk_event_efficiency": {}, "technical_efficiency": {},
        "technical_sm_efficiency": {},
        "fake_rate": {}, "duplicate_rate": {},
        "prototracks_per_event": acts_summary["track_stats"]["input"]["prototracks_per_event"],
        "track_classification_all_particles": {},
    }
    for point in WORKING_POINTS:
        out["quirk_track_efficiency"][point] = S.efficiency(acts_summary, _level("quirk", point), selection="quirk")
        out["technical_efficiency"][point] = S.efficiency(acts_summary, _level("technical", point),
                                                          selection="technical")
        out["technical_sm_efficiency"][point] = S.efficiency(acts_summary, _level("technical_sm", point),
                                                             selection="technical_sm")
        two = per_event_df.filter(pl.col(f"quirks_selected_{point}") == 2)
        out["quirk_event_efficiency"][point] = {
            **_ratio(int((two[f"quirks_matched_{point}"] == 2).sum()), two.height),
            "events_by_selected_quirks": {str(k): v for k, v in
                                          per_event_df[f"quirks_selected_{point}"].value_counts().iter_rows()}}
        cls = S.classification(acts_summary, _level("all", point), selection="all")
        out["track_classification_all_particles"][point] = cls
        out["fake_rate"][point] = cls["fake_rate"]
        out["duplicate_rate"][point] = cls["duplicate_rate"]
    out["kf_not_meaningful_for_quirks"] = {
        "note": "Kalman-fit levels assume helices; quirk tracks are not helices. Kept for reference only.",
        "selected_fitted_tracks_per_event": acts_summary["track_stats"]["input"]["selected_fitted_tracks_per_event"],
        **{f"{sel}_{point}": S.efficiency(acts_summary, ("kf" if point == "very_loose" else f"kf@{point}"),
                                          selection=sel)
           for sel in ("quirk", "technical") for point in WORKING_POINTS},
    }
    out["checks"] = checks
    out["definitions"] = {
        "quirk_track_efficiency": "matched quirks / quirks in the quirk selection (ACTS trackeff_vs_eta, proto level)",
        "quirk_event_efficiency": "events with both quirks matched / events with both quirks in the quirk selection",
        "technical_efficiency": "matched / selected particles, technical selection (quirks included), proto level",
        "technical_sm_efficiency": "matched / selected particles, technical selection without quirks, proto level",
        "fake_rate": "ACTS fake prototracks / all prototracks, against all particles with hits",
        "duplicate_rate": "ACTS duplicate prototracks / all prototracks, against all particles with hits",
    }
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions-dir", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path, help="R1 dataset root (truth/, reco/)")
    parser.add_argument("--events", required=True, type=int, nargs=2, metavar=("A", "B"),
                        help="score events A <= event_id < B")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--jobs", type=int, default=min(32, os.cpu_count() or 1),
                        help="parallel ACTS passes over consecutive event chunks (each single-threaded, about 5 GB "
                             "of memory each: at most about 60 on one 512 GB CPU node)")
    parser.add_argument("--staging-dir", type=Path, default=DEFAULT_STAGING,
                        help="where the sorted event-range copy of the dataset is written (reused when unchanged)")
    parser.add_argument("--keep-chunks", action="store_true", help="keep the per-chunk histogram pickles")
    args = parser.parse_args(argv)
    first, stop = args.events
    if stop <= first:
        parser.error("--events A B needs A < B")
    t0 = time.time()
    args.out.mkdir(parents=True, exist_ok=True)

    preds = (pl.scan_parquet(args.predictions_dir / "predictions.parquet")
             .filter((pl.col("event_id") >= first) & (pl.col("event_id") < stop))
             .select(pl.col("event_id").cast(pl.Int64), pl.col("hit_id").cast(pl.Int64),
                     pl.col("predicted_track_id").cast(pl.Int64)).collect())
    missing = sorted(set(range(first, stop)) - set(preds["event_id"].unique().to_list()))
    if missing:
        raise SystemExit(f"predictions lack {len(missing)} events of [{first}, {stop}), e.g. {missing[:10]}; "
                         "write one row per hit (predicted_track_id -1 when unassigned) for every event")
    if preds.select(pl.struct("event_id", "hit_id").is_duplicated().any()).item():
        raise SystemExit("predictions hold a (event_id, hit_id) pair more than once")
    predictions_meta = json.loads((args.predictions_dir / "meta.json").read_text())

    staged = stage_events(args.dataset, first, stop, args.staging_dir)
    print(f"staged events [{first}, {stop}) at {staged} ({time.time() - t0:.0f} s)", flush=True)

    n = stop - first
    jobs_n = max(1, min(args.jobs, n))
    edges = np.linspace(0, n, jobs_n + 1).round().astype(int)
    chunk_dir = args.out / "chunks"
    chunk_dir.mkdir(exist_ok=True)
    for stale in chunk_dir.glob("chunk_*.pkl"):  # from an earlier, interrupted run
        stale.unlink()
    jobs = [{"predictions_dir": str(args.predictions_dir.resolve()), "staged": str(staged), "first": first,
             "skip": int(lo), "n": int(hi - lo), "output": str(chunk_dir / f"chunk_{i:04d}.pkl")}
            for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])) if hi > lo]
    if len(jobs) == 1:
        paths = [_run_chunk(jobs[0])]
    else:
        # ProcessPoolExecutor raises BrokenProcessPool if a worker dies (e.g. out of memory); multiprocessing.Pool
        # would wait forever for the lost chunk.
        with ProcessPoolExecutor(len(jobs), mp_context=mp.get_context("spawn")) as pool:
            paths = list(pool.map(_run_chunk, jobs))
    print(f"ACTS done: {len(jobs)} chunks ({time.time() - t0:.0f} s)", flush=True)
    bundles = []
    for path in paths:
        with open(path, "rb") as f:
            bundles.append(pickle.load(f))
    bundle = merge_bundles(bundles) if len(bundles) > 1 else bundles[0]
    with open(args.out / "histograms.pkl", "wb") as f:
        pickle.dump(bundle, f)
    if not args.keep_chunks:
        for path in paths:
            Path(path).unlink()
        chunk_dir.rmdir()

    from tracking_core.summary import summarize

    acts_summary = summarize(bundle, scope="quirk pilot: quirk, technical and all-particle selections")
    (args.out / "acts_summary.json").write_text(json.dumps(acts_summary, indent=1) + "\n")

    # Per-event table from the ACTS run.
    pe = bundle["per_event"]
    if pe["positions"] != list(range(n)):
        raise RuntimeError("per-event ACTS records are not the staged positions 0..n-1 in order")
    per_event = {"event_id": np.arange(first, stop), "n_measurements": pe["n_measurements"],
                 "quirks_selected": pe["n_selected"]["quirk"],
                 "quirk_particles": [[q[0] for q in qs] for qs in pe["quirk_particles"]],
                 "quirk_pdgs": [[q[1] for q in qs] for qs in pe["quirk_particles"]]}
    for point in WORKING_POINTS:
        level = pe["levels"][_level("quirk", point)]
        per_event[f"quirks_matched_{point}"] = np.rint(level["accepted"]).astype(int)
        per_event[f"quirks_selected_{point}"] = np.rint(level["total"]).astype(int)
    per_event_df = pl.DataFrame(per_event)

    # Checks: event alignment, selection content, ACTS per-event sums, Python matching.
    hits_tab = read_events(staged, "tracker_hits", first, stop, columns=["event_id", "particle_id"])
    parts_tab = read_events(staged, "particles", first, stop,
                            columns=["event_id", "particle_id", "pdg_id", "charge", "primary", "mass"])
    n_hits = [len(v) for v in hits_tab["particle_id"].to_pylist()]
    hits = pl.DataFrame({
        "event_id": np.repeat(np.arange(first, stop), n_hits),
        "hit_id": np.concatenate([np.arange(k) for k in n_hits]),
        "particle_id": np.concatenate([np.asarray(v, dtype=np.int64) for v in hits_tab["particle_id"].to_pylist()]),
    })
    parts = pl.DataFrame({c: np.concatenate([np.asarray(v) for v in parts_tab[c].to_pylist()])
                          for c in ["particle_id", "pdg_id", "charge", "primary", "mass"]}).with_columns(
        event_id=pl.Series(np.repeat(np.arange(first, stop), [len(v) for v in parts_tab["particle_id"].to_pylist()])))
    hit_counts = hits.group_by("event_id", "particle_id").agg(pl.len().alias("particle_hits"))
    quirks = (parts.filter(pl.col("pdg_id").abs() == QUIRK_PDG)
              .join(hit_counts, on=["event_id", "particle_id"], how="left").fill_null(0))
    quirks_sel = quirks.filter(pl.col("particle_hits") >= 3)
    acts_quirk_sets = {e: sorted(qs) for e, qs in zip(per_event_df["event_id"].to_list(),
                                                       per_event_df["quirk_particles"].to_list())}
    py_quirk_sets = {e: [] for e in range(first, stop)}
    for e, p in quirks_sel.select("event_id", "particle_id").iter_rows():
        py_quirk_sets[e].append(int(p))
    checks = {
        "n_measurements_equal_hits_per_event": per_event_df["n_measurements"].to_list() == n_hits,
        "quirk_selection_all_pdg_10000100": all(abs(p) == QUIRK_PDG for ps in per_event["quirk_pdgs"] for p in ps),
        "quirk_selection_equals_python_selection": all(acts_quirk_sets[e] == sorted(py_quirk_sets[e])
                                                       for e in py_quirk_sets),
        "quirk_truth_table": {
            "quirks": quirks.height, "charge_values": sorted(set(quirks["charge"].to_list())),
            "primary_values": sorted(set(quirks["primary"].to_list())),
            "mass_gev": sorted(set(np.round(quirks["mass"].to_numpy(), 3).tolist())),
            "with_at_least_3_hits": quirks_sel.height,
            "hit_count_quantiles": {str(q): float(quirks["particle_hits"].quantile(q))
                                    for q in (0.0, 0.1, 0.5, 0.9, 1.0)}},
    }
    for point, (ratio, double) in WORKING_POINTS.items():
        level = _level("quirk", point)
        inclusive = acts_summary["inclusive_metrics"]["input"][level]["trackeff_vs_eta"]
        matched_py = python_matching(hits, preds, ratio, double).filter(pl.col("matched_tracks") > 0)
        q_matched = quirks_sel.join(matched_py, on=["event_id", "particle_id"], how="inner")
        py_per_event = dict(q_matched.group_by("event_id").agg(pl.len()).iter_rows())
        py_counts = [py_per_event.get(e, 0) for e in range(first, stop)]
        acts_counts = per_event_df[f"quirks_matched_{point}"].to_list()
        per_event_df = per_event_df.with_columns(pl.Series(f"python_quirks_matched_{point}", py_counts))
        checks[point] = {
            "acts_quirk_accepted_total": [inclusive["accepted"], inclusive["total"]],
            "per_event_sums_equal_inclusive": [sum(acts_counts), sum(per_event_df[f"quirks_selected_{point}"])]
            == [inclusive["accepted"], inclusive["total"]],
            "per_event_selected_equal_selection_size": per_event_df[f"quirks_selected_{point}"].to_list()
            == per_event_df["quirks_selected"].to_list(),
            "python_quirks_matched": sum(py_counts),
            "python_equals_acts_per_event": py_counts == acts_counts,
            "events_differing": [e for e, a, b in zip(range(first, stop), acts_counts, py_counts) if a != b][:20],
            "python_majority_ties_on_quirk_tracks": int(q_matched["tied"].sum()),
            "python_ties_with_undefined_order": int(q_matched["tied_unordered"].sum()),
        }
        for sel in ("technical", "technical_sm", "all"):
            lv = _level(sel, point)
            total = acts_summary["inclusive_metrics"]["input"][lv]["trackeff_vs_eta"]["total"]
            checks[point][f"{sel}_histogram_total_equals_selected"] = total == sum(pe["n_selected"][sel])
    per_event_df.write_parquet(args.out / "per_event.parquet")
    summary = build_summary(bundle, acts_summary, args, predictions_meta, per_event_df, checks)
    summary["wall_seconds"] = round(time.time() - t0, 1)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")

    failed = [name for name, ok in _flatten_checks(checks) if ok is False]
    for point in WORKING_POINTS:
        q = summary["quirk_track_efficiency"][point]
        ev = summary["quirk_event_efficiency"][point]
        t = summary["technical_efficiency"][point]
        sm = summary["technical_sm_efficiency"][point]
        print(f"{point:10s} quirk track eff {_fmt(q)}  quirk event eff {_fmt(ev)}  technical eff {_fmt(t)}  "
              f"technical_sm eff {_fmt(sm)}  "
              f"fake {_num(summary['fake_rate'][point])}  dup {_num(summary['duplicate_rate'][point])}")
    print(f"prototracks/event {summary['prototracks_per_event']:.1f}; wrote {args.out / 'summary.json'} "
          f"({summary['wall_seconds']:.0f} s)")
    if failed:
        raise SystemExit(f"checks failed: {failed}")


def _num(x):
    return "n/a" if x is None else f"{x:.4f}"


def _fmt(r):
    return "n/a" if r is None or r["eff"] is None else f"{r['eff']:.4f} ({r['accepted']:.0f}/{r['total']:.0f})"


def _flatten_checks(checks, prefix=""):
    for key, value in checks.items():
        if isinstance(value, dict):
            yield from _flatten_checks(value, f"{prefix}{key}.")
        elif isinstance(value, bool):
            yield f"{prefix}{key}", value


if __name__ == "__main__":
    main()
