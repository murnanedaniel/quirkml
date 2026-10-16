"""Inclusive counters from ACTS histograms that cover the selected phase space.

The default pT axes stop at 100 GeV and ACTS can discard overflow, so summing
them is not an inclusive efficiency/fake-rate estimate. Truth eta covers the
configured ParticleSelector eta interval; reconstructed phi covers all finite
track directions, unlike reconstructed eta (which has no selection cut).
"""

INCLUSIVE_METRICS = ("trackeff_vs_eta", "fakeRatio_vs_phi", "duplicationRatio_vs_phi")


def histogram_counts(histogram):
    result = {}
    for kind in ("accepted", "total"):
        component = histogram[kind] if isinstance(histogram, dict) else getattr(histogram, kind)
        try:
            values = component.values(flow=True)
        except TypeError:
            # Native ACTS histogram bindings have no flow keyword.
            values = component.values()
        result[kind] = float(sum(values))
    return result


def inclusive_counts(histograms):
    return {metric: histogram_counts(histograms[metric]) for metric in INCLUSIVE_METRICS}


def _entries_times_value(histograms, key):
    """Sum of entries times the "Num" (second) axis bin centre of an ACTS FakePlotTool 2D histogram."""
    histogram = histograms[key]
    if histogram.values(flow=True).sum() != histogram.values().sum():
        raise ValueError(f"{key}: entries outside the histogram range; the per-particle count is lost.")
    return float((histogram.values().sum(axis=0) * histogram.axes[1].centers).sum())


def track_classification(histograms, n_tracks):
    """Matched / duplicate / fake track totals from the particle-side histograms ACTS fills.

    PatternRecognitionPerformanceCollector fills its track-side histograms (fakeRatio_vs_*,
    duplicationRatio_vs_*, purity/completeness) only for tracks with a reference surface, which unfitted
    prototracks do not have. Its particle loop needs no track parameters: for every selected particle
    it fills nTruthMatchedTracks (the TrackTruthMatcher-matched track plus its duplicates), nFakeTracks
    (fake tracks whose majority particle it is) and nDuplicated (matched - 1). TrackTruthMatcher keys every
    matched, duplicate and fake track to exactly one selected particle, so the sums over particles are
    the collector's own nTotalMatchedTracks (matched + duplicate) / nTotalDuplicateTracks /
    nTotalFakeTracks. Returned "matched", "duplicate" and "fake" are TrackTruthMatcher's
    TrackMatchClassification. Tracks whose majority particle is outside the particle selection, or which
    hold no truth hit, get no TrackTruthMatcher entry and are in none of the three ("unassociated" =
    n_tracks - matched - duplicate - fake).

    `n_tracks` is the size of the scored track container. Rates follow ACTS's logSummary (per track,
    all tracks in the denominator) and its fakeRatio/duplicationRatio histograms (per associated track).
    """
    matched = _entries_times_value(histograms, "nTruthMatchedTracks_vs_eta")
    fake = _entries_times_value(histograms, "nFakeTracks_vs_eta")
    profile = histograms["nDuplicated_vs_eta"].view(flow=True)
    duplicate = float((profile["count"] * profile["value"]).sum())
    counts = {"matched": matched, "duplicate": duplicate, "fake": fake}
    for key, value in counts.items():
        if abs(value - round(value)) > 1e-6:
            raise ValueError(f"{key} track total {value} from ACTS histograms is not an integer.")
        counts[key] = int(round(value))
    particles = histograms["trackeff_vs_eta"]["total"].values(flow=True).sum()
    if profile["count"].sum() != particles:
        raise ValueError("nDuplicated_vs_eta does not hold one entry per selected particle.")
    associated = counts["matched"] + counts["fake"]
    # TrackMatchClassification partition: Matched (one per reconstructed particle), Duplicate, Fake.
    counts["matched"] -= counts["duplicate"]
    if associated > n_tracks:
        raise ValueError(f"{associated} classified tracks exceed the {n_tracks} scored tracks.")
    return {
        "tracks": int(n_tracks), **counts, "unassociated": int(n_tracks) - associated,
        "fake_rate": counts["fake"] / n_tracks if n_tracks else None,
        "duplicate_rate": counts["duplicate"] / n_tracks if n_tracks else None,
        "fake_rate_associated": counts["fake"] / associated if associated else None,
        "duplicate_rate_associated": counts["duplicate"] / associated if associated else None,
    }
