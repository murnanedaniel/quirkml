"""Efficiency vs fake-rate curves for a per-track score cut (reusable; B2).

Input: track tables (common.track_table) of the same tracks on the normal truth and on the pair-truth copy, each with
a 'score' column (higher = keep), and the particle tables of both truths. A track is kept if score >= cut. Strict
matching (> 50%). Per cut:
  fake_rate (normal truth) and fake_rate_pair (pair truth): kept fakes / kept tracks
  quirk_eff: quirks (>= 3 hits) with a kept matched track / quirks; quirk_event_eff: both quirks of the event
  coupled_eff: pair objects (pair truth, >= 3 hits) with a kept matched track / pairs
  technical_sm_eff: technical_sm particles with a kept matched track / technical_sm particles
  tracks_per_event
    python curves.py TRACKS_NORMAL.parquet TRACKS_PAIR.parquet PARTS_NORMAL.parquet PARTS_PAIR.parquet OUT.csv
"""
from __future__ import annotations

import sys

import numpy as np
import polars as pl


def _found_cut(tracks, parts, sel):
    """Per selected particle: the largest score among its matched tracks (-inf if none)."""
    best = (tracks.filter(pl.col("matched")).group_by("event_id", "maj_pid").agg(pl.col("score").max().alias("best"))
            .rename({"maj_pid": "particle_id"}))
    p = parts.filter(sel).select("event_id", "particle_id").join(best, on=["event_id", "particle_id"], how="left")
    return p.with_columns(pl.col("best").fill_null(-np.inf))


def _eff(best, cuts):
    b = np.sort(best)
    return 1 - np.searchsorted(b, cuts, side="left") / max(len(b), 1)


def _rate(scores, flag, cuts):
    o = np.argsort(scores)
    s, f = scores[o], flag[o].astype(np.int64)
    tail_f = np.concatenate([np.cumsum(f[::-1])[::-1], [0]])
    i = np.searchsorted(s, cuts, side="left")
    kept = len(s) - i
    return np.where(kept > 0, tail_f[i] / np.maximum(kept, 1), np.nan), kept


def curve(tn: pl.DataFrame, tp: pl.DataFrame, pn: pl.DataFrame, pp: pl.DataFrame, cuts=None) -> pl.DataFrame:
    n_events = pn["event_id"].n_unique()
    if cuts is None:
        cuts = np.unique(np.concatenate([[-np.inf], np.quantile(tn["score"].to_numpy(), np.linspace(0, 0.995, 400))]))
    cuts = np.maximum(np.asarray(cuts, float), -1e300)  # -inf (no cut) -> keep every track, unmatched stay -inf
    fr, kept = _rate(tn["score"].to_numpy(), (tn["cls"] == "fake").to_numpy(), cuts)
    frp, _ = _rate(tp["score"].to_numpy(), (tp["cls"] == "fake").to_numpy(), cuts)
    q = _found_cut(tn, pn, pl.col("is_quirk") & (pl.col("n_hits") >= 3))
    c = _found_cut(tp, pp, pl.col("is_quirk") & (pl.col("n_hits") >= 3))
    t = _found_cut(tn, pn, pl.col("technical_sm"))
    # quirk events: both quirks found; an event's cut is the smaller of its two best scores
    ev = q.group_by("event_id").agg(pl.len().alias("k"), pl.col("best").min()).filter(pl.col("k") == 2)
    return pl.DataFrame({"cut": cuts, "tracks_per_event": kept / n_events, "fake_rate": fr, "fake_rate_pair": frp,
                         "quirk_eff": _eff(q["best"].to_numpy(), cuts), "quirk_event_eff": _eff(ev["best"].to_numpy(), cuts),
                         "coupled_eff": _eff(c["best"].to_numpy(), cuts), "technical_sm_eff": _eff(t["best"].to_numpy(), cuts)})


def operating_point(val_curve: pl.DataFrame, test_curve: pl.DataFrame, fake_col: str, target: float) -> dict:
    """Smallest cut whose val fake rate (fake_col) is <= target; the test row at that cut (test_curve on the same cuts)."""
    ok = val_curve.with_row_index("i").filter(pl.col(fake_col) <= target)
    if ok.height == 0:
        return {"target": target, "cut": None}
    i = ok["i"][0]
    return {"target": target, "cut": float(val_curve["cut"][i]), "val": val_curve.row(i, named=True),
            "test": test_curve.row(i, named=True)}


if __name__ == "__main__":
    tn, tp, pn, pp = (pl.read_parquet(x) for x in sys.argv[1:5])
    curve(tn, tp, pn, pp).write_csv(sys.argv[5])
