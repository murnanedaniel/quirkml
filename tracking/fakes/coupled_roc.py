"""Coupled efficiency vs background rejection for one model (reusable).

Knob: a per-track quality score, the same for every model: gradient-boosted trees (quality.py features) trained on
val 16000-16499 with label "the track is not a fake on the pair-truth copy" and quirk-majority good tracks weighted
x30 (the quirk-protected B2 score, here always with pair-truth labels, also for the per-quirk GNN reference). The
test tracks 17000-19999 are kept if score >= cut; the cut is swept.

Per cut, on the pair-truth copy (pair = one particle), strict matching:
  coupled_eff       pairs (>= 3 hits) with a kept track matched at > 50% / pairs
  fake              kept tracks that are fakes at > 50% / kept tracks (fake: majority particle in the particle
                    table, not matched); rejection = 1 / fake
  coupled_eff_tight, fake_tight: the same at > 90%
  technical_sm_eff  (normal truth), tracks_per_event
    python coupled_roc.py MODEL OUT_ROOT     -> OUT_ROOT/MODEL/roc_test.csv, roc.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
import quality as QL  # noqa: E402

VAL_A_END = QL.VAL_A_END


def _best(tracks, parts, flag, sel):
    best = (tracks.filter(pl.col(flag)).group_by("event_id", "maj_pid").agg(pl.col("score").max().alias("best"))
            .rename({"maj_pid": "particle_id"}))
    p = parts.filter(sel).select("event_id", "particle_id").join(best, on=["event_id", "particle_id"], how="left")
    return np.sort(p["best"].fill_null(-np.inf).to_numpy())


def roc(t: pl.DataFrame, pn: pl.DataFrame, pp: pl.DataFrame, score: np.ndarray, n_cuts: int = 800) -> pl.DataFrame:
    t = t.with_columns(score=pl.Series(score))
    cuts = np.concatenate([[-1e300], np.quantile(score, np.linspace(0, 0.9995, n_cuts))])
    s = np.sort(score)
    o = np.argsort(score)
    i = np.searchsorted(s, cuts, side="left")
    kept = len(s) - i

    def tail(flag):
        f = flag[o].astype(np.int64)
        c = np.concatenate([np.cumsum(f[::-1])[::-1], [0]])
        return c[i]
    assoc = t["associated_pair"].to_numpy()
    fake = tail(assoc & ~t["matched_pair"].to_numpy())
    fake_t = tail(assoc & ~t["matched_tight_pair"].to_numpy())
    tp = t.select("event_id", "score", pl.col("maj_pid_pair").alias("maj_pid"), pl.col("matched_pair").alias("m"),
                  pl.col("matched_tight_pair").alias("mt"))
    tn = t.select("event_id", "score", "maj_pid", pl.col("matched").alias("m"))
    sel_pair = pl.col("is_quirk") & (pl.col("n_hits") >= 3)
    b, bt = _best(tp, pp, "m", sel_pair), _best(tp, pp, "mt", sel_pair)
    bs = _best(tn, pn, "m", pl.col("technical_sm"))
    eff = lambda arr: 1 - np.searchsorted(arr, cuts, side="left") / len(arr)  # noqa: E731
    n_ev = pn["event_id"].n_unique()
    return pl.DataFrame({"cut": cuts, "tracks_per_event": kept / n_ev, "coupled_eff": eff(b),
                         "fake": fake / np.maximum(kept, 1), "coupled_eff_tight": eff(bt),
                         "fake_tight": fake_t / np.maximum(kept, 1), "technical_sm_eff": eff(bs),
                         "n_fake": fake, "n_kept": kept})


def main():
    model, root = sys.argv[1], Path(sys.argv[2])
    out = root / model
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    tv, _, _ = QL.build_split(model, "val", out)
    tt, pnt, ppt = QL.build_split(model, "test", out)
    cols = QL.feature_cols(tv)
    A = tv.filter(pl.col("event_id") < VAL_A_END)
    y = (A["cls_pair"] != "fake").to_numpy()
    w = np.where(A["maj_is_quirk_pair"].to_numpy() & y, 30.0, 1.0)
    clf = HistGradientBoostingClassifier(max_iter=500, learning_rate=0.08, max_leaf_nodes=63, early_stopping=True,
                                         validation_fraction=0.15, random_state=0)
    clf.fit(A.select(cols).to_numpy().astype(np.float64), y, sample_weight=w)
    score = clf.predict_proba(tt.select(cols).to_numpy().astype(np.float64))[:, 1]
    r = roc(tt, pnt, ppt, score)
    r.write_csv(out / "roc_test.csv")
    tt.select("event_id", "track_id").with_columns(score=pl.Series(score)).write_parquet(out / "roc_score_test.parquet")
    info = {"model": model, "knob": "quality score, pair-truth label, quirk-majority good tracks x30, trained on val "
            "16000-16499", "n_train": A.height, "no_cut": r.row(0, named=True), "seconds": round(time.time() - t0)}
    (out / "roc.json").write_text(json.dumps(info, indent=1, default=float) + "\n")
    print(model, {k: round(v, 4) for k, v in r.row(0, named=True).items()}, f"({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
