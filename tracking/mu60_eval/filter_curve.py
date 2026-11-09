"""Hit-filter trade-off curve: quirk-hit recall and target-hit recall vs hits kept per event, over a threshold sweep.

Targets as in the filter training: hits of charged primary particles with pT > PT_MIN GeV, |eta| < 4, >= 3 hits; quirk
hits = hits of |pdg| 10000100 particles. Recall is pooled over hits; also the fraction of events keeping >= 99% of
their quirk hits, and the per-event pair-hit completeness ceilings (> 50%, > 90% of the quirk hits kept).

    python filter_curve.py FILTER_PREDICTIONS.parquet DATASET A B_EXCL OUT_JSON [PT_MIN]
"""
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.compute as pc

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from r1_data import QUIRK_PDG, read_events  # noqa: E402

THRESHOLDS = [0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 0.9]


def main():
    pred, dataset, first, stop, out = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), Path(sys.argv[5])
    pt_min = float(sys.argv[6]) if len(sys.argv) > 6 else 1.0
    h = read_events(dataset, "tracker_hits", first, stop, columns=["event_id", "particle_id"])
    n = np.asarray(pc.list_value_length(h["particle_id"]))
    hits = pl.DataFrame({"event_id": np.repeat(h["event_id"].to_numpy().astype(np.int64), n),
                         "hit_id": np.concatenate([np.arange(k) for k in n]),
                         "particle_id": np.asarray(pc.list_flatten(h["particle_id"])).astype(np.int64)})
    cols = ["event_id", "particle_id", "pdg_id", "charge", "px", "py", "pz", "primary"]
    p = read_events(dataset, "particles", first, stop, columns=cols)
    m = np.asarray(pc.list_value_length(p["particle_id"]))
    d = {"event_id": np.repeat(p["event_id"].to_numpy().astype(np.int64), m)}
    for c in cols[1:]:
        d[c] = np.asarray(pc.list_flatten(p[c]))
    parts = pl.DataFrame(d).with_columns(pl.col("particle_id").cast(pl.Int64))
    nh = hits.group_by("event_id", "particle_id").agg(pl.len().alias("nh"))
    pt = (pl.col("px") ** 2 + pl.col("py") ** 2).sqrt()
    eta = (pl.col("pz") / pt).arcsinh()
    parts = parts.join(nh, on=["event_id", "particle_id"], how="left").with_columns(
        quirk=pl.col("pdg_id").abs() == QUIRK_PDG,
        target=(pl.col("charge") != 0) & pl.col("primary") & (pt > pt_min) & (eta.abs() < 4.0) & (pl.col("nh").fill_null(0) >= 3))
    hits = hits.join(parts.select("event_id", "particle_id", "quirk", "target"), on=["event_id", "particle_id"], how="left").with_columns(
        pl.col("quirk").fill_null(False), pl.col("target").fill_null(False))
    pr = pl.read_parquet(pred).with_columns(pl.col("event_id").cast(pl.Int64), pl.col("hit_id").cast(pl.Int64)).filter(
        (pl.col("event_id") >= first) & (pl.col("event_id") < stop))
    hits = hits.join(pr, on=["event_id", "hit_id"], how="left")
    if hits["keep_prob"].null_count():
        raise ValueError(f"{hits['keep_prob'].null_count()} hits without a filter probability")
    n_ev = hits["event_id"].n_unique()
    rows = []
    for t in THRESHOLDS:
        k = hits.with_columns(keep=pl.col("keep_prob") >= t)
        per_ev = k.filter(pl.col("quirk")).group_by("event_id").agg(pl.col("keep").mean().alias("f"))
        rows.append({"threshold": t, "kept_per_event": k["keep"].sum() / n_ev,
                     "quirk_hit_recall": float(k.filter(pl.col("quirk"))["keep"].mean()),
                     "target_hit_recall": float(k.filter(pl.col("target"))["keep"].mean()),
                     "events_quirk_recall_ge_0.99": float((per_ev["f"] >= 0.99).mean()),
                     # completeness ceiling of the coupled (pair) efficiency: pair keeps > 50% / > 90% of its hits
                     "pair_ceiling_loose_strict": float((per_ev["f"] > 0.5).mean()),
                     "pair_ceiling_tight_strict": float((per_ev["f"] > 0.9).mean())})
    res = {"predictions": str(pred), "dataset": str(dataset), "events": [first, stop], "n_events": n_ev,
           "hits_per_event": hits.height / n_ev, "quirk_hits_per_event": hits["quirk"].sum() / n_ev,
           "target_hits_per_event": hits["target"].sum() / n_ev, "pt_min": pt_min, "curve": rows}
    out.write_text(json.dumps(res, indent=1))
    print(f"events {n_ev}, hits/event {res['hits_per_event']:.0f}, quirk hits/event {res['quirk_hits_per_event']:.1f}, "
          f"target hits/event {res['target_hits_per_event']:.0f}")
    print("threshold  kept/event  quirk_recall  target_recall  events_quirk>=0.99  pair>50%  pair>90%")
    for r in rows:
        print(f"{r['threshold']:9.3f}  {r['kept_per_event']:10.0f}  {r['quirk_hit_recall']:12.4f}  {r['target_hit_recall']:13.4f}"
              f"  {r['events_quirk_recall_ge_0.99']:18.3f}  {r['pair_ceiling_loose_strict']:8.3f}  {r['pair_ceiling_tight_strict']:8.3f}")


if __name__ == "__main__":
    main()
