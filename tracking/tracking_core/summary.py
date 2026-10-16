"""summary.json: the scalar record of one ACTS evaluation, and the readers every consumer uses.

:func:`summarize` turns the histogram bundle of :func:`tracking_core.eval.dump_eval_histograms` into a JSON-able
dict. Every number in it is a sum of ACTS histogram entries (or the ACTS particle-side track classification);
nothing is matched or counted here.

Particle selections (efficiency denominators, ``tracking_core.eval.SELECTIONS``), all scored in one ACTS pass:

    physics      the headline: pT > 1 GeV, |eta| < 3, charged, primary, prompt (production vertex rho < 2 mm,
                 |z| < 200 mm), no measurement requirement; all vertices (hard scatter and pileup)
    physics_hs   physics, hard-scatter vertex only (ColliderML vertex_primary 1)
    technical    physics plus at least `min_measurements` (7) measurements

Technical results produced before 2026-10-07 used only the kinematic and measurement
cuts, without the charged-primary/prompt requirements. Consult the stored selections.

Level names carry the selection: "proto[physics]@tight", "kf[physics_hs]" (the default working point is left
out, as before). Technical levels keep the bare names of earlier summaries ("proto", "kf@tight"), so a summary
written before the physics selections (``schema_version`` 1 or 2) reads as technical, and asking it for a
physics level gives None.

Layout (``schema_version`` 3)::

    schema_version, scope, events, min_measurements, pt_min_gev, eta_max, num_events,
    default_selection                "physics"
    selections                       {name: ParticleSelector cuts (mm, GeV)}
    default_working_point            "very_loose": the matching of the default "proto"/"kf" levels
    working_points                   {name: [matchingRatio, doubleMatching]}, all points scored
    levels                           {level: working point name}, e.g. {"proto[physics]@tight": "tight"}
    metrics[sample][level][hist]     {accepted, total, fraction} of every ACTS efficiency-type histogram (with flow)
    inclusive_metrics[sample][level] {trackeff_vs_eta, fakeRatio_vs_phi, duplicationRatio_vs_phi}: {accepted, total}
    prefit_classification[sample][level]   proto levels: tracks, matched, duplicate, fake, unassociated and rates
    track_stats[sample]              prototracks (per event) and selected fitted tracks per event (technical)

``schema_version`` 2 has the technical levels only; ``schema_version`` 1 (no version field) may also hold a
"proto@very_loose" level equal to "proto". The readers accept every version: level "proto@very_loose" resolves to
"proto" (and back), so callers can name every working point the same way.

Readers: :func:`efficiency` (inclusive pre-fit or fitted efficiency with a binomial error), :func:`rates`
(efficiency, fake and duplicate rate of one level; proto fake/duplicate from the ACTS particle-side classification,
since ACTS fills its track-side fake/duplicate histograms only for fitted tracks), :func:`binned` (per-bin efficiency
from histograms.pkl, needs boost_histogram) and :func:`load` / :func:`load_bundle`. Each takes `selection`
(default "physics").
"""

from __future__ import annotations

import json
import pickle
from collections.abc import Mapping
from pathlib import Path

SCHEMA_VERSION = 3
SUMMARY_NAME = "summary.json"
BUNDLE_NAME = "histograms.pkl"
DEFAULT_WORKING_POINT = "very_loose"  # tracking_core.eval.DEFAULT_WORKING_POINT, without importing pyacts helpers
INCLUSIVE_METRICS = ("trackeff_vs_eta", "fakeRatio_vs_phi", "duplicationRatio_vs_phi")

#: Particle selections, in report order (tracking_core.eval.selection_cuts defines them). "quirk" and "all" were
#: added for the quirk tracking pilot (quirkml/tracking): see tracking_core.eval.selection_cuts.
SELECTIONS = ("physics", "physics_hs", "technical", "quirk", "all", "technical_sm")
#: The headline selection every reader defaults to.
DEFAULT_SELECTION = "physics"
#: The selection of summaries written before the physics selections; its levels keep the bare names.
LEGACY_SELECTION = "technical"
TRACK_LEVELS = ("proto", "kf")


# Section: level names


def level_key(base: str, selection: str = DEFAULT_SELECTION, point: str | None = None) -> str:
    """Level name of track level `base` ("proto"/"kf"), `selection` and working point `point` (None: the default
    levels' matching, left out of the name): ("proto", "physics", "tight") -> "proto[physics]@tight";
    ("kf", "technical", None) -> "kf"."""
    if selection not in SELECTIONS:
        raise ValueError(f"Unknown particle selection {selection!r}; known: {SELECTIONS}.")
    name = base if selection == LEGACY_SELECTION else f"{base}[{selection}]"
    return f"{name}@{point}" if point else name


def split_level(level: str) -> tuple[str, str, str]:
    """(base, selection, point) of a level name; point is "" for the default levels: "proto[physics]@tight" ->
    ("proto", "physics", "tight"); "kf" -> ("kf", "technical", ""); "fit" -> ("fit", "technical", "")."""
    name, _, point = level.partition("@")
    base, bracket, rest = name.partition("[")
    selection = rest.rstrip("]") if bracket else LEGACY_SELECTION
    return base, selection, point


def _resolve(levels, level: str, selection: str | None) -> str | None:
    """The key holding `level` of `selection` in `levels`. A level with a bracket names its own selection."""
    if "[" not in level:
        base, _, point = level.partition("@")
        level = level_key(base, selection or DEFAULT_SELECTION, point or None)
    return resolve_level(levels, level)


# Section: writing


def _histogram_counts(histogram):
    from .histogram_counts import histogram_counts

    return histogram_counts(histogram)


def summarize(bundle, *, scope: str | None = None, events=None) -> dict:
    """summary.json content (``schema_version`` 3) from a histogram bundle (dict or path to histograms.pkl).

    `events` defaults to the evaluated original event ids in Sequencer order. Older bundles derive them from
    the evaluated position range and event remapping. A bundle without selections is recorded
    with its technical selection only."""
    if not isinstance(bundle, Mapping):
        bundle = load_bundle(bundle)
    provenance = bundle.get("provenance", {})
    if events is None:
        events = provenance.get("evaluated_event_ids")
        if events is None:
            start = provenance.get("skip_events", 0)
            positions = range(start, start + bundle["eval_kwargs"]["num_events"])
            remap = dict(provenance.get("event_id_remap") or [])
            events = [remap[position] if remap else position for position in positions]
    kwargs = bundle["eval_kwargs"]
    selections = bundle.get("selections") or {LEGACY_SELECTION: {
        "ptMin": kwargs["pt_min_gev"], "absEtaMax": kwargs["eta_max"], "measurementsMin": kwargs["min_measurements"]}}
    summary = {
        "schema_version": SCHEMA_VERSION,
        "scope": scope or "common ACTS metrics",
        "events": list(events),
        "num_events": kwargs["num_events"],
        "min_measurements": kwargs["min_measurements"],
        "pt_min_gev": kwargs["pt_min_gev"],
        "eta_max": kwargs["eta_max"],
        "default_selection": bundle.get("default_selection", DEFAULT_SELECTION if DEFAULT_SELECTION in selections
                                        else LEGACY_SELECTION),
        "selections": selections,
        "default_working_point": provenance.get("default_working_point", DEFAULT_WORKING_POINT),
        "working_points": provenance.get("working_points", {}),
        "levels": bundle.get("levels", {}),
        "metrics": {},
        "inclusive_metrics": {},
        "prefit_classification": bundle.get("prefit_classification", {}),
        "track_stats": bundle.get("track_stats", {}),
    }
    for sample, levels in bundle["hists"].items():
        summary["metrics"][sample] = {}
        summary["inclusive_metrics"][sample] = {}
        for level, hists in levels.items():
            if split_level(level)[0] in TRACK_LEVELS:
                summary["inclusive_metrics"][sample][level] = {name: _histogram_counts(hists[name])
                                                               for name in INCLUSIVE_METRICS}
            summary["metrics"][sample][level] = {}
            for name, histogram in hists.items():
                if isinstance(histogram, dict) and {"accepted", "total"} <= set(histogram):
                    accepted = float(histogram["accepted"].values(flow=True).sum())
                    total = float(histogram["total"].values(flow=True).sum())
                    summary["metrics"][sample][level][name] = {
                        "accepted": accepted, "total": total, "fraction": accepted / total if total else None}
    return summary


def write_summary(bundle, out: Path | str, **kwargs) -> dict:
    """Write :func:`summarize` of `bundle` to `out` (a file, or a directory receiving summary.json)."""
    summary = summarize(bundle, **kwargs)
    out = Path(out)
    if out.is_dir():
        out = out / SUMMARY_NAME
    out.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


# Section: reading


def load(summary_or_dir) -> dict:
    """A summary dict from a dict, a summary.json path or an evaluation directory."""
    if isinstance(summary_or_dir, Mapping):
        return dict(summary_or_dir)
    path = Path(summary_or_dir)
    if path.is_dir():
        path = path / SUMMARY_NAME
    return json.loads(path.read_text())


def load_bundle(bundle_or_dir):
    """The histogram bundle from a dict, a histograms.pkl path or an evaluation directory (needs boost_histogram)."""
    if isinstance(bundle_or_dir, Mapping):
        return bundle_or_dir
    path = Path(bundle_or_dir)
    if path.is_dir():
        path = path / BUNDLE_NAME
    with path.open("rb") as stream:
        return pickle.load(stream)


def schema_version(summary) -> int:
    return int(load(summary).get("schema_version", 1))


def selections(summary_or_dir) -> dict:
    """{selection: ParticleSelector cuts} scored in a summary; summaries before ``schema_version`` 3 hold the
    technical selection only."""
    summary = load(summary_or_dir)
    return summary.get("selections") or {LEGACY_SELECTION: {
        "ptMin": summary.get("pt_min_gev"), "absEtaMax": summary.get("eta_max"),
        "measurementsMin": summary.get("min_measurements")}}


def resolve_level(levels, level: str) -> str | None:
    """The key holding `level` in `levels` (a mapping or set of level names): "proto@very_loose" and "proto" name
    the same matching (likewise for kf and for every selection), whichever the summary recorded. None when
    absent."""
    if level in levels:
        return level
    name, _, point = level.partition("@")
    if point == DEFAULT_WORKING_POINT and name in levels:
        return name
    if not point and f"{name}@{DEFAULT_WORKING_POINT}" in levels:
        return f"{name}@{DEFAULT_WORKING_POINT}"
    return None


def _ratio(accepted, total):
    accepted, total = float(accepted), float(total)
    if not total:
        return None
    value = accepted / total
    return {"eff": value, "err": (value * (1 - value) / total) ** 0.5, "accepted": accepted, "total": total}


def efficiency(summary_or_dir, level: str = "proto", sample: str = "input",
               selection: str = DEFAULT_SELECTION) -> dict | None:
    """Inclusive efficiency of `level` (default: pre-fit, very loose) of particle selection `selection` (default
    physics) from trackeff_vs_eta: {eff, err (binomial), accepted, total}. None when the level is absent (a
    physics level of a summary written before the physics selections) or empty."""
    levels = load(summary_or_dir)["inclusive_metrics"].get(sample, {})
    key = _resolve(levels, level, selection)
    if key is None:
        return None
    entry = levels[key].get("trackeff_vs_eta")
    return _ratio(entry["accepted"], entry["total"]) if entry else None


def classification(summary_or_dir, level: str = "proto", sample: str = "input",
                   selection: str = DEFAULT_SELECTION) -> dict | None:
    """ACTS pre-fit track classification of a proto level (tracks, matched, duplicate, fake, unassociated, rates),
    or None (fitted levels, or a summary written before the classification existed). Tracks whose majority
    particle is outside `selection` are unassociated."""
    levels = load(summary_or_dir).get("prefit_classification", {}).get(sample, {})
    key = _resolve(levels, level, selection)
    return None if key is None else levels[key]


def rates(summary_or_dir, level: str = "proto", sample: str = "input",
          selection: str = DEFAULT_SELECTION) -> dict | None:
    """{efficiency, fake_rate, duplicate_rate}: {accepted, total} of one level, None when the level is absent.

    Fake and duplicate rates are per associated track (the definition of the ACTS fakeRatio/duplicationRatio
    histograms). For fitted levels they are those histograms; for proto levels, where ACTS leaves them empty, they
    come from the particle-side classification (accepted = fake or duplicate tracks, total = matched + duplicate +
    fake tracks), and are absent from summaries without it."""
    summary = load(summary_or_dir)
    levels = summary["inclusive_metrics"].get(sample, {})
    key = _resolve(levels, level, selection)
    if key is None:
        return None
    counts = levels[key]
    out = {"efficiency": dict(counts["trackeff_vs_eta"]),
           "fake_rate": dict(counts["fakeRatio_vs_phi"]),
           "duplicate_rate": dict(counts["duplicationRatio_vs_phi"])}
    prefit = (classification(summary, key, sample, split_level(key)[1]) if split_level(key)[0] == "proto"
              else None)
    if prefit is not None:
        associated = float(prefit["matched"] + prefit["duplicate"] + prefit["fake"])
        out["fake_rate"] = {"accepted": float(prefit["fake"]), "total": associated}
        out["duplicate_rate"] = {"accepted": float(prefit["duplicate"]), "total": associated}
    return out


def binned(bundle_or_dir, level: str, name: str, sample: str = "input",
           selection: str = DEFAULT_SELECTION) -> dict | None:
    """Per-bin efficiency of one ACTS efficiency histogram from histograms.pkl (bins with total > 0): numpy arrays
    lo, hi (bin edges), eff, err (binomial), accepted, total. None when the level or histogram is absent."""
    import numpy as np

    bundle = load_bundle(bundle_or_dir)
    levels = bundle["hists"].get(sample, {})
    key = _resolve(levels, level, selection)
    entry = None if key is None else levels[key].get(name)
    if entry is None:
        return None
    accepted = np.asarray(entry["accepted"].values(), dtype=float)
    total = np.asarray(entry["total"].values(), dtype=float)
    edges = np.asarray(entry["total"].axes[0].edges, dtype=float)
    keep = total > 0
    eff = np.divide(accepted, total, out=np.zeros_like(total), where=keep)
    err = np.sqrt(np.divide(eff * (1 - eff), total, out=np.zeros_like(total), where=keep))
    return {"lo": edges[:-1][keep], "hi": edges[1:][keep], "eff": eff[keep], "err": err[keep],
            "accepted": accepted[keep], "total": total[keep]}
