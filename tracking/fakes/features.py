"""Per-track features for the track quality score (B2) and the quirk tagger (B3). Model-agnostic geometry and
composition-free features, plus GNN edge-score features (scored edge files) or MaskFormer per-hit mask
probabilities (predictions 'score') when available. No truth is used here.

Features: n_hits, n_layers, hits_per_layer, n_dup_layer_hits, r_min, r_max, r_span, z_first (signed z of the
innermost hit), abs_eta_first, eta_mean, eta_span, phi_span, step mean / max (3D distance between r-ordered hits),
turning angle mean / max / p90 (between consecutive r-ordered segments), rz-line residual rms, xy-circle (Kasa) radius
and residual rms, n_shared_hits (hits on > 1 track); GNN: n_int_edges, int_score_mean, hit_best_int min / mean,
frac_hits_best_int_lt_cut, ext_score_max, n_ext_gt_cut; MaskFormer: mask_mean, mask_min, mask_std.
"""
from __future__ import annotations

import glob

import numpy as np
import polars as pl


def _circle(df):
    """Kasa circle fit per track from group moments; returns (track keys, R, residual rms)."""
    w = pl.col("x") ** 2 + pl.col("y") ** 2
    g = df.group_by("event_id", "track_id").agg(
        n=pl.len(), sx=pl.col("x").sum(), sy=pl.col("y").sum(), sxx=(pl.col("x") ** 2).sum(), syy=(pl.col("y") ** 2).sum(),
        sxy=(pl.col("x") * pl.col("y")).sum(), sw=w.sum(), sxw=(pl.col("x") * w).sum(), syw=(pl.col("y") * w).sum())
    A = np.stack([np.stack([g["sxx"], g["sxy"], g["sx"]], -1), np.stack([g["sxy"], g["syy"], g["sy"]], -1),
                  np.stack([g["sx"], g["sy"], g["n"].cast(pl.Float64)], -1)], 1).astype(np.float64)
    b = np.stack([g["sxw"], g["syw"], g["sw"]], -1).astype(np.float64)
    A = A + np.eye(3)[None] * 1e-6 * (np.abs(A).max(axis=(1, 2), keepdims=True) + 1)
    sol = np.linalg.solve(A, b[..., None])[..., 0]
    xc, yc = sol[:, 0] / 2, sol[:, 1] / 2
    R = np.sqrt(np.maximum(sol[:, 2] + xc ** 2 + yc ** 2, 0))
    c = g.select("event_id", "track_id").with_columns(xc=pl.Series(xc), yc=pl.Series(yc), circ_R=pl.Series(R))
    res = (df.join(c, on=["event_id", "track_id"])
           .with_columns(d=((pl.col("x") - pl.col("xc")) ** 2 + (pl.col("y") - pl.col("yc")) ** 2).sqrt() - pl.col("circ_R"))
           .group_by("event_id", "track_id").agg(circ_rms=(pl.col("d") ** 2).mean().sqrt()))
    return c.select("event_id", "track_id", "circ_R").join(res, on=["event_id", "track_id"])


def geometry_features(hits: pl.DataFrame, preds: pl.DataFrame) -> pl.DataFrame:
    a = preds.join(hits.select("event_id", "hit_id", "x", "y", "z", "r", "eta", "layer_key"), on=["event_id", "hit_id"])
    shared = preds.group_by("event_id", "hit_id").agg(pl.len().alias("ntr"))
    a = a.join(shared, on=["event_id", "hit_id"]).with_columns(phi=pl.arctan2("y", "x"))
    a = a.sort(["event_id", "track_id", "r"])
    k = ["event_id", "track_id"]
    a = a.with_columns(
        dx=pl.col("x").shift(-1).over(k) - pl.col("x"), dy=pl.col("y").shift(-1).over(k) - pl.col("y"),
        dz=pl.col("z").shift(-1).over(k) - pl.col("z"))
    a = a.with_columns(step=(pl.col("dx") ** 2 + pl.col("dy") ** 2 + pl.col("dz") ** 2).sqrt())
    a = a.with_columns(
        cosang=((pl.col("dx") * pl.col("dx").shift(1).over(k) + pl.col("dy") * pl.col("dy").shift(1).over(k)
                 + pl.col("dz") * pl.col("dz").shift(1).over(k)) / (pl.col("step") * pl.col("step").shift(1).over(k))))
    a = a.with_columns(ang=pl.col("cosang").clip(-1, 1).arccos())
    rz = pl.col("r"), pl.col("z")
    g = a.group_by(k).agg(
        n_hits=pl.len(), n_layers=pl.col("layer_key").n_unique(),
        r_min=pl.col("r").min(), r_max=pl.col("r").max(), z_first=pl.col("z").first(),
        abs_eta_first=pl.col("eta").first().abs(), eta_mean=pl.col("eta").mean(),
        eta_span=pl.col("eta").max() - pl.col("eta").min(),
        phi_span=(pl.col("phi") - pl.col("phi").first() + np.pi).mod(2 * np.pi).max()
        - (pl.col("phi") - pl.col("phi").first() + np.pi).mod(2 * np.pi).min(),
        step_mean=pl.col("step").mean(), step_max=pl.col("step").max(),
        ang_mean=pl.col("ang").mean(), ang_max=pl.col("ang").max(), ang_p90=pl.col("ang").quantile(0.9),
        n_shared_hits=(pl.col("ntr") > 1).sum(),
        rz_cov=((rz[0] - rz[0].mean()) * (rz[1] - rz[1].mean())).mean(), r_var=((rz[0] - rz[0].mean()) ** 2).mean(),
        z_var=((rz[1] - rz[1].mean()) ** 2).mean())
    g = g.with_columns(hits_per_layer=pl.col("n_hits") / pl.col("n_layers"),
                       n_dup_layer_hits=pl.col("n_hits") - pl.col("n_layers"), r_span=pl.col("r_max") - pl.col("r_min"),
                       rz_res_rms=(pl.col("z_var") - pl.col("rz_cov") ** 2 / (pl.col("r_var") + 1e-9)).clip(0).sqrt()
                       ).drop("rz_cov", "r_var", "z_var")
    return g.join(_circle(a.select("event_id", "track_id", "x", "y")), on=k)


def mask_features(preds: pl.DataFrame) -> pl.DataFrame:
    return preds.group_by("event_id", "track_id").agg(mask_mean=pl.col("score").mean(), mask_min=pl.col("score").min(),
                                                      mask_std=pl.col("score").std().fill_null(0))


def edge_features(preds: pl.DataFrame, scored_glob: str, first: int, stop: int, cut: float = 0.5,
                  chunk: int = 100) -> pl.DataFrame:
    """Edge-score features from the GNN scored files (event_id, src, dst, score lists; src/dst = hit_id), read once
    per file and processed in chunks of `chunk` events."""
    outs = []
    for f in sorted(glob.glob(scored_glob)):
        t = (pl.scan_parquet(f).filter((pl.col("event_id") >= first) & (pl.col("event_id") < stop))
             .select(pl.col("event_id").cast(pl.Int64), "src", "dst", "score").collect())
        evs = t["event_id"].to_numpy()
        for lo in range(0, len(evs), chunk):
            sub = t.slice(lo, chunk)
            e0, e1 = int(sub["event_id"].min()), int(sub["event_id"].max())
            pr = preds.filter((pl.col("event_id") >= e0) & (pl.col("event_id") <= e1))
            outs.append(_edge_chunk(pr, sub.explode("src", "dst", "score"), cut))
    return pl.concat(outs)


def _edge_chunk(preds, e, cut):
    e = e.with_columns(pl.col("src").cast(pl.Int64), pl.col("dst").cast(pl.Int64))
    h2t = preds.select("event_id", "hit_id", "track_id")
    both = pl.concat([e.select("event_id", pl.col("src").alias("a"), pl.col("dst").alias("b"), "score"),
                      e.select("event_id", pl.col("dst").alias("a"), pl.col("src").alias("b"), "score")])
    j = (both.join(h2t.rename({"hit_id": "a"}), on=["event_id", "a"], how="inner")
         .join(h2t.rename({"hit_id": "b", "track_id": "tb"}), on=["event_id", "b"], how="left")
         .with_columns(internal=pl.col("tb") == pl.col("track_id")).with_columns(pl.col("internal").fill_null(False)))
    per_hit = j.group_by("event_id", "track_id", "a").agg(
        best_int=pl.when(pl.col("internal")).then(pl.col("score")).max(),
        best_ext=pl.when(~pl.col("internal")).then(pl.col("score")).max())
    trk_int = j.filter(pl.col("internal")).group_by("event_id", "track_id").agg(
        n_int_edges=(pl.len() / 2), int_score_mean=pl.col("score").mean(), int_score_min=pl.col("score").min())
    trk_ext = j.filter(~pl.col("internal")).group_by("event_id", "track_id").agg(
        ext_score_max=pl.col("score").max(), n_ext_gt_cut=(pl.col("score") > cut).sum())
    hp = per_hit.group_by("event_id", "track_id").agg(
        hit_best_int_min=pl.col("best_int").fill_null(0).min(), hit_best_int_mean=pl.col("best_int").fill_null(0).mean(),
        frac_hits_best_int_lt_cut=(pl.col("best_int").fill_null(0) < cut).mean(),
        hit_best_ext_mean=pl.col("best_ext").fill_null(0).mean())
    out = preds.select("event_id", "track_id").unique()
    for x in (trk_int, trk_ext, hp):
        out = out.join(x, on=["event_id", "track_id"], how="left")
    return out.fill_null(0)
