"""Python scoring with ACTS's rules (bulk default at mu 60; ACTS for headline numbers, cross-checked per event).

Matching (as evaluate.python_matching / TrackTruthMatcher): per predicted track the majority particle (ties: the tied
particle whose last hit comes latest in the track's prediction rows wins); double matching: the track matches its
majority particle if purity n_major / track_hits and completeness n_major / particle_hits pass the ratio.
Working points: very_loose (>= 0.5), tight (>= 0.9), very_loose_strict (> 0.5), tight_strict (> 0.9) (integer tests).
Track classes (ACTS TrackFinderPerformanceWriter): matched (first matching track of a particle), duplicate (further
matching tracks), fake (no match). Selections as evaluate.py: quirk (|pdg| 10000100, charged, primary, >= 3 hits),
technical (pT > 1 GeV, |eta| < 3, charged, primary, vertex rho < 2 mm, |z| < 200 mm, >= 7 hits), technical_sm
(technical without quirks). On a pair-truth copy the quirk efficiency is the coupled efficiency.

    python py_score.py PRED_DIR DATASET A B_EXCL OUT_JSON [PER_EVENT_PARQUET]
"""
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.compute as pc

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from r1_data import QUIRK_PDG, read_events  # noqa: E402

POINTS = {"very_loose": (1, 2, False), "tight": (9, 10, False), "very_loose_strict": (1, 2, True), "tight_strict": (9, 10, True)}


def load(dataset, first, stop):
    h = read_events(dataset, "tracker_hits", first, stop, columns=["event_id", "particle_id"])
    n = np.asarray(pc.list_value_length(h["particle_id"]))
    ev = np.repeat(h["event_id"].to_numpy().astype(np.int64), n)
    hits = pl.DataFrame({"event_id": ev, "hit_id": np.concatenate([np.arange(k) for k in n]),
                         "particle_id": np.asarray(pc.list_flatten(h["particle_id"])).astype(np.int64)})
    cols = ["event_id", "particle_id", "pdg_id", "charge", "px", "py", "pz", "vx", "vy", "vz", "primary"]
    p = read_events(dataset, "particles", first, stop, columns=cols)
    m = np.asarray(pc.list_value_length(p["particle_id"]))
    d = {"event_id": np.repeat(p["event_id"].to_numpy().astype(np.int64), m)}
    for c in cols[1:]:
        d[c] = np.asarray(pc.list_flatten(p[c]))
    parts = pl.DataFrame(d).with_columns(pl.col("particle_id").cast(pl.Int64))
    counts = hits.group_by("event_id", "particle_id").agg(pl.len().alias("particle_hits"))
    parts = parts.join(counts, on=["event_id", "particle_id"], how="left").with_columns(pl.col("particle_hits").fill_null(0))
    pt = (pl.col("px") ** 2 + pl.col("py") ** 2).sqrt()
    eta = (pl.col("pz") / pt).arcsinh()
    rho = (pl.col("vx") ** 2 + pl.col("vy") ** 2).sqrt()
    base = (pl.col("charge") != 0) & pl.col("primary")
    tech = base & (pt > 1.0) & (eta.abs() < 3.0) & (rho < 2.0) & (pl.col("vz").abs() < 200.0) & (pl.col("particle_hits") >= 7)
    parts = parts.with_columns(quirk=base & (pl.col("pdg_id").abs() == QUIRK_PDG) & (pl.col("particle_hits") >= 3),
                               technical=tech, technical_sm=tech & (pl.col("pdg_id").abs() != QUIRK_PDG))
    return hits, counts, parts


def per_track(hits, counts, preds):
    a = (preds.with_row_index("row").filter(pl.col("predicted_track_id") >= 0)
         .join(hits, on=["event_id", "hit_id"], how="left"))
    if a["particle_id"].null_count():
        raise ValueError("predictions hold hit_ids outside the events' hits")
    c = a.group_by("event_id", "predicted_track_id", "particle_id").agg(pl.len().alias("n"), pl.col("row").max().alias("last_row"))
    t = c.sort(["n", "last_row"], descending=[True, True]).group_by("event_id", "predicted_track_id", maintain_order=True).agg(
        pl.col("n").sum().alias("track_hits"), pl.col("n").first().alias("n_major"), pl.col("particle_id").first().alias("particle_id"))
    return t.join(counts, on=["event_id", "particle_id"], how="left")


def main():
    pred_dir, dataset, first, stop, out = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), Path(sys.argv[5])
    per_event_out = Path(sys.argv[6]) if len(sys.argv) > 6 else None
    hits, counts, parts = load(dataset, first, stop)
    preds = pl.read_parquet(pred_dir / "predictions.parquet").filter(
        (pl.col("event_id") >= first) & (pl.col("event_id") < stop)).with_columns(
        pl.col("event_id").cast(pl.Int64), pl.col("hit_id").cast(pl.Int64))
    t = per_track(hits, counts, preds)
    n, m, p = t["n_major"].to_numpy(), t["track_hits"].to_numpy(), t["particle_hits"].to_numpy()
    res = {"events": [first, stop], "dataset": str(dataset), "predictions": str(pred_dir), "tracks": t.height}
    pe_frames = []
    for name, (a, b, strict) in POINTS.items():
        ok = ((b * n > a * m) & (b * n > a * p)) if strict else ((b * n >= a * m) & (b * n >= a * p))
        tt = t.with_columns(ok=pl.Series(ok))
        per_p = tt.group_by("event_id", "particle_id").agg(pl.col("ok").sum().alias("k"))
        matched_particles = per_p.filter(pl.col("k") > 0)
        n_matched = matched_particles.height
        n_dup = int((matched_particles["k"] - 1).sum()) if n_matched else 0
        n_fake = int((~tt["ok"]).sum())
        q = parts.join(matched_particles, on=["event_id", "particle_id"], how="left").with_columns(found=pl.col("k").fill_null(0) > 0)
        r = {}
        for sel in ("quirk", "technical", "technical_sm"):
            s = q.filter(pl.col(sel))
            r[sel] = {"accepted": int(s["found"].sum()), "total": s.height, "eff": float(s["found"].mean()) if s.height else None}
        ev = q.filter(pl.col("quirk")).group_by("event_id").agg(pl.len().alias("nsel"), pl.col("found").sum().alias("nf"))
        e2 = ev.filter(pl.col("nsel") == 2)
        r["quirk_event"] = {"accepted": int((e2["nf"] == 2).sum()), "total": e2.height,
                            "eff": float((e2["nf"] == 2).mean()) if e2.height else None}
        assoc = n_matched + n_dup + n_fake
        r["tracks"] = {"tracks": t.height, "matched": n_matched, "duplicate": n_dup, "fake": n_fake,
                       "fake_rate": n_fake / assoc if assoc else None, "duplicate_rate": n_dup / assoc if assoc else None}
        res[name] = r
        if per_event_out is not None:
            pe_frames.append(ev.with_columns(point=pl.lit(name)))
    out.write_text(json.dumps(res, indent=1))
    if per_event_out is not None and pe_frames:
        pl.concat(pe_frames).write_parquet(per_event_out)
    for name in POINTS:
        r = res[name]
        print(f"{name}: quirk {r['quirk']['eff']:.4f} ({r['quirk']['accepted']}/{r['quirk']['total']}) event {r['quirk_event']['eff']}"
              f" tech_sm {r['technical_sm']['eff']:.4f} fake {r['tracks']['fake_rate']:.4f} dup {r['tracks']['duplicate_rate']:.4f}")


if __name__ == "__main__":
    main()
