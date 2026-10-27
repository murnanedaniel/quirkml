"""Shared tables for the fake-rate studies (work package B): hits and particles of an R1 event range, and one row per
built track with its truth composition and the strict TrackTruthMatcher classification.

Strict matching (the standard from 2026-10-26): a track matches its majority particle (ties as
evaluate.python_matching) if purity > 50% and completeness > 50% (tight: > 90%). As in the ACTS evaluator ("all"
selection), a track whose majority particle is in the particle table and that is not matched is a fake; matched
tracks beyond the first per particle are duplicates.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

TRK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRK))
from r1_data import QUIRK_PDG, read_events  # noqa: E402

SRC = "/global/cfs/cdirs/m4958/data/ColliderML/simulation/quirks/m200_L774_mu0/v1/parquet_r1"
PAIR_TRUTH = "/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirk_tracking_data/pair_truth/m200_L774_mu0"
HIT_COLS = ["event_id", "particle_id", "x", "y", "z", "volume_id", "layer_id"]
PART_COLS = ["event_id", "particle_id", "pdg_id", "parent_id", "primary", "charge", "mass", "px", "py", "pz",
             "vx", "vy", "vz"]


def _explode(tab, cols):
    n = [len(v) for v in tab[cols[1]].to_pylist()]
    ev = np.repeat(tab["event_id"].to_numpy(), n)
    out = {"event_id": ev.astype(np.int64)}
    for c in cols[1:]:
        out[c] = tab[c].combine_chunks().flatten().to_numpy(zero_copy_only=False)
    return pl.DataFrame(out), n


def load_truth(root: str, first: int, stop: int) -> tuple[pl.DataFrame, pl.DataFrame]:
    """hits (event_id, hit_id, particle_id, x, y, z, r, eta, layer_key) and particles (event_id, particle_id, pdg_id,
    parent_id, primary, charge, pt, eta, rho, vz, n_hits, is_quirk, is_quirk_desc, technical, technical_sm)."""
    h, n = _explode(read_events(root, "tracker_hits", first, stop, columns=HIT_COLS), HIT_COLS)
    h = h.with_columns(hit_id=pl.Series(np.concatenate([np.arange(k) for k in n])),
                       particle_id=pl.col("particle_id").cast(pl.Int64))
    h = h.with_columns(r=(pl.col("x") ** 2 + pl.col("y") ** 2).sqrt(),
                       layer_key=pl.col("volume_id").cast(pl.Int32) * 10000 + pl.col("layer_id").cast(pl.Int32))
    h = h.with_columns(eta=-((pl.arctan2(pl.col("r"), pl.col("z")) / 2).tan().log()))
    p, _ = _explode(read_events(root, "particles", first, stop, columns=PART_COLS), PART_COLS)
    p = p.with_columns(particle_id=pl.col("particle_id").cast(pl.Int64), parent_id=pl.col("parent_id").cast(pl.Int64))
    nh = h.group_by("event_id", "particle_id").agg(pl.len().alias("n_hits"))
    pt = (pl.col("px") ** 2 + pl.col("py") ** 2).sqrt()
    p = p.join(nh, on=["event_id", "particle_id"], how="left").with_columns(
        pl.col("n_hits").fill_null(0), pt=pt, rho=(pl.col("vx") ** 2 + pl.col("vy") ** 2).sqrt(),
        eta=pl.arctan2(pt, pl.col("pz")).truediv(2).tan().log().neg(),
        is_quirk=pl.col("pdg_id").abs() == QUIRK_PDG)
    # quirk descendants: iterate parent links
    desc = p.filter(pl.col("is_quirk")).select("event_id", "particle_id")
    allq = desc
    while True:
        new = (p.join(allq.rename({"particle_id": "parent_id"}), on=["event_id", "parent_id"], how="semi")
               .select("event_id", "particle_id").join(allq, on=["event_id", "particle_id"], how="anti"))
        if new.height == 0:
            break
        allq = pl.concat([allq, new])
    qd = allq.join(desc, on=["event_id", "particle_id"], how="anti").with_columns(is_quirk_desc=pl.lit(True))
    p = p.join(qd, on=["event_id", "particle_id"], how="left").with_columns(pl.col("is_quirk_desc").fill_null(False))
    tech = ((pl.col("pt") > 1.0) & (pl.col("eta").abs() < 3.0) & (pl.col("charge") != 0) & pl.col("primary")
            & (pl.col("rho") < 2.0) & (pl.col("vz").abs() < 200.0) & (pl.col("n_hits") >= 7))
    p = p.with_columns(technical=tech, technical_sm=tech & ~pl.col("is_quirk"))
    return h, p.drop("px", "py", "pz", "vx", "vy")


def read_preds(pred_dir, first, stop) -> pl.DataFrame:
    return (pl.scan_parquet(Path(pred_dir) / "predictions.parquet")
            .filter((pl.col("event_id") >= first) & (pl.col("event_id") < stop) & (pl.col("predicted_track_id") >= 0))
            .select(pl.col("event_id").cast(pl.Int64), pl.col("hit_id").cast(pl.Int64),
                    pl.col("predicted_track_id").cast(pl.Int64).alias("track_id"),
                    *([pl.col("score").cast(pl.Float64)] if "score" in pl.read_parquet_schema(
                        Path(pred_dir) / "predictions.parquet") else []))
            .collect())


def track_table(hits: pl.DataFrame, parts: pl.DataFrame, preds: pl.DataFrame) -> pl.DataFrame:
    """One row per built track: composition, majority particle (evaluate.python_matching tie rule), strict / >= match
    flags, ACTS-style class (matched / duplicate / fake / unassociated) at > 50%, and geometry summaries."""
    pinfo = parts.select("event_id", "particle_id", "is_quirk", "is_quirk_desc", pl.col("n_hits").alias("p_hits"))
    pmaj = parts.select("event_id", pl.col("particle_id").alias("maj_pid"), pl.col("pt").alias("maj_pt"),
                        pl.col("primary").alias("maj_primary"), pl.col("technical").alias("maj_technical"))
    a = (preds.with_row_index("row").join(hits.select("event_id", "hit_id", "particle_id", "r", "z", "eta", "layer_key"),
                                          on=["event_id", "hit_id"], how="left")
         .join(pinfo, on=["event_id", "particle_id"], how="left")
         .with_columns(pl.col("is_quirk").fill_null(False), pl.col("is_quirk_desc").fill_null(False),
                       in_table=pl.col("p_hits").is_not_null()))
    c = a.group_by("event_id", "track_id", "particle_id").agg(pl.len().alias("n"), pl.col("row").max().alias("last_row"))
    maj = (c.sort(["n", "last_row"], descending=[True, True]).group_by("event_id", "track_id", maintain_order=True)
           .agg(pl.col("particle_id").first().alias("maj_pid"), pl.col("n").first().alias("n_major"),
                pl.len().alias("n_particles")))
    g = a.group_by("event_id", "track_id").agg(
        pl.len().alias("n_hits"), pl.col("layer_key").n_unique().alias("n_layers"),
        pl.col("r").min().alias("r_min"), pl.col("r").max().alias("r_max"),
        pl.col("z").sort_by("r").first().alias("z_first"), pl.col("eta").sort_by("r").first().alias("eta_first"),
        pl.col("eta").mean().alias("eta_mean"),
        pl.col("is_quirk").sum().alias("n_quirk_hits"), pl.col("is_quirk_desc").sum().alias("n_qdesc_hits"),
        (pl.col("particle_id") == 0).sum().alias("n_pid0_hits"))
    t = g.join(maj, on=["event_id", "track_id"]).join(
        pinfo.rename({"particle_id": "maj_pid"}), on=["event_id", "maj_pid"], how="left").with_columns(
        pl.col("is_quirk").fill_null(False).alias("maj_is_quirk"), pl.col("is_quirk_desc").fill_null(False).alias("maj_is_qdesc"),
        pl.col("p_hits").is_not_null().alias("associated")).drop("is_quirk", "is_quirk_desc").join(
        pmaj, on=["event_id", "maj_pid"], how="left")
    n, m, P = pl.col("n_major"), pl.col("n_hits"), pl.col("p_hits")
    t = t.with_columns(purity=n / m, completeness=n / P,
                       matched=(2 * n > m) & (2 * n > P), matched_ge=(2 * n >= m) & (2 * n >= P),
                       matched_tight=(10 * n > 9 * m) & (10 * n > 9 * P),
                       n_q_major=pl.when(pl.col("maj_is_quirk")).then(n).otherwise(0),
                       hits_per_layer=m / pl.col("n_layers"))
    t = t.with_columns(n_q_other=pl.col("n_quirk_hits") - pl.col("n_q_major"))
    # ACTS classification: first matched track per particle = matched (ACTS keeps the one with more hits), others dup
    t = t.with_columns(rank=pl.col("n_hits").rank("ordinal", descending=True).over("event_id", "maj_pid", "matched"))
    cls = (pl.when(~pl.col("associated")).then(pl.lit("unassociated"))
           .when(pl.col("matched") & (pl.col("rank") == 1)).then(pl.lit("matched"))
           .when(pl.col("matched")).then(pl.lit("duplicate")).otherwise(pl.lit("fake")))
    return t.with_columns(cls=cls).drop("rank")


FAKE_CATEGORIES = {
    "a": "piece of a quirk (majority quirk, other quirk < 25% of the track)",
    "b": "mixes the two quirks (majority quirk, other quirk >= 25% of the track)",
    "c": "quirk secondaries (majority particle a quirk descendant, or > 50% of hits)",
    "f": "mostly noise / unassigned truth (majority particle_id 0)",
    "d": "merged ordinary tracks (majority ordinary particle, purity <= 50%)",
    "e": "piece of an ordinary track (purity > 50%, completeness <= 50%)",
}


def fake_category(t: pl.DataFrame) -> pl.DataFrame:
    m = pl.col("n_hits")
    cat = (pl.when(pl.col("cls") != "fake").then(None)
           .when(pl.col("maj_is_quirk") & (4 * pl.col("n_q_other") >= m)).then(pl.lit("b"))
           .when(pl.col("maj_is_quirk")).then(pl.lit("a"))
           .when(pl.col("maj_is_qdesc") | (2 * pl.col("n_qdesc_hits") > m)).then(pl.lit("c"))
           .when(pl.col("maj_pid") == 0).then(pl.lit("f"))
           .when(2 * pl.col("n_major") <= m).then(pl.lit("d"))
           .otherwise(pl.lit("e")))
    return t.with_columns(fake_cat=cat)
