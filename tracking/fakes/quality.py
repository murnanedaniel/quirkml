"""B2 track quality score and B3 quirk tagger for one model (gradient-boosted trees, CPU).

Protocol: features (features.py) and strict truth labels for every built track on val 16000-16999 and test
17000-19999. Train on val 16000-16499, choose cuts on val 16500-16999, report test once.
  B2 label: the track is not a fake in the model's own truth (per-quirk models: normal truth; pair models: pair
        truth, so pair tracks are good). Curves (curves.py): efficiency vs fake rate for the score, for a
        hit-count-only cut (baseline), operating points at fake 0.05 and 0.02 (cut chosen on val B).
  Variants: quality_qw (quirk-majority good tracks weighted x30 in training), quality_or_tag (keep if quality >= cut
        OR quirk tag above its 90% cut).
  B3 label: the track's majority particle is a quirk (normal truth). Cut at 90% efficiency for quirk tracks on val B
        (quirk tracks = quirk-majority tracks that are not fakes in the model's own truth); report ordinary
        (non-quirk-majority) tracks tagged per event on test.

    python quality.py MODEL OUT_ROOT
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
import common as C  # noqa: E402
import curves as CV  # noqa: E402
import features as F  # noqa: E402

Q = "/pscratch/sd/d/danieltm/quirk_tracking"
MODELS = {
    "gnn_v3_s0_walk": dict(truth="normal", val=Q + "/fakes/val_preds/g2_s0_walk",
                           test=Q + "/gnn/main/g2_q50_dir_s0/test/pred_stitch0.7_mh4",
                           scored=Q + "/gnn/main/g2_q50_dir_s0/gnn/scored/events{}*.parquet"),
    "pair_gnn_s0_cc": dict(truth="pair", val=Q + "/fakes/val_preds/e2_s0_cc",
                           test=Q + "/gnn/main/e2_pair_dir_s0/test/pred_cc0.5_mh4",
                           scored=Q + "/gnn/main/e2_pair_dir_s0/gnn/scored/events{}*.parquet"),
    "mf_long_ot0.1": dict(truth="pair", val=Q + "/maskformer/predictions/t2e_pairdir_long_e049_val_ot0.1",
                          test=Q + "/maskformer/predictions/t2e_pairdir_long_e049_test_ot0.1", scored=None),
    "mf_long_ot0.2": dict(truth="pair", val=Q + "/maskformer/predictions/t2e_pairdir_long_e049_val_ot0.2",
                          test=Q + "/maskformer/predictions/t2e_pairdir_long_e049_test_ot0.2", scored=None),
}
# coupled-efficiency study (coupled_roc.py): more pair models, same split rule
P = Q + "/maskformer/predictions/"
for _s in (1, 2):
    MODELS[f"pair_gnn_s{_s}_cc"] = dict(truth="pair", val=Q + f"/fakes/val_preds/e2_s{_s}_cc",
                                        test=Q + f"/gnn/main/e2_pair_dir_s{_s}/test/pred_cc0.5_mh4",
                                        scored=Q + f"/gnn/main/e2_pair_dir_s{_s}/gnn/scored/events{{}}*.parquet")
MODELS["pair_gnn_s0_walk"] = dict(truth="pair", val=Q + "/gnn/main/e2_pair_dir_s0/val/pred_stitch0.7_mh4",
                                  test=Q + "/gnn/main/e2_pair_dir_s0/test/pred_stitch0.7_mh4",
                                  scored=Q + "/gnn/main/e2_pair_dir_s0/gnn/scored/events{}*.parquet")
MODELS["pair_gnn_v3_p3_cc07"] = dict(truth="pair", val=Q + "/fakes/val_preds/p3_cc07",
                                     test=Q + "/gnn/main/p3_pairv3_s0/test/pred_cc0.7_mh4",
                                     scored=Q + "/gnn/main/p3_pairv3_s0/gnn/scored/events{}*.parquet")
for _n, _k in [("mf_pair_s0", "t2d_pair_lr5e-5"), ("mf_pair_s1", "t2e_pair_lr5e-5_s1"), ("mf_pair_s2", "t2e_pair_lr5e-5_s2"),
               ("mf_pairdir_s0", "t2e_pairdir_lr5e-5"), ("mf_pairdir_s1", "t2e_pairdir_lr5e-5_s1"),
               ("mf_pairdir_s2", "t2e_pairdir_lr5e-5_s2")]:
    MODELS[f"{_n}_ot0.1"] = dict(truth="pair", val=P + f"{_k}_val_ot0.1", test=P + f"{_k}_test_ot0.1", scored=None)
SPLITS = {"val": (16000, 17000), "test": (17000, 20000)}
VAL_A_END = 16500
TRUTH_COLS = ["cls", "matched", "matched_tight", "associated", "maj_pid", "maj_is_quirk", "maj_is_qdesc", "n_major",
              "purity", "completeness"]
DROP = {"event_id", "track_id"}


def build_split(model, split, out):
    path = out / f"tracks_{split}.parquet"
    if path.exists():
        return pl.read_parquet(path), pl.read_parquet(out / f"parts_{split}_normal.parquet"), \
            pl.read_parquet(out / f"parts_{split}_pair.parquet")
    cfg = MODELS[model]
    first, stop = SPLITS[split]
    hn, pn = C.load_truth(C.SRC, first, stop)
    hp, pp = C.load_truth(C.PAIR_TRUTH, first, stop)
    preds = C.read_preds(cfg[split], first, stop)
    feats = F.geometry_features(hn, preds)
    if cfg["scored"]:
        tag = "16" if split == "val" else "1[7-9]"
        feats = feats.join(F.edge_features(preds, cfg["scored"].format(tag), first, stop), on=["event_id", "track_id"],
                           how="left").fill_null(0)
    elif "score" in preds.columns:
        feats = feats.join(F.mask_features(preds), on=["event_id", "track_id"], how="left")
    tn = C.track_table(hn, pn, preds).select("event_id", "track_id", *TRUTH_COLS)
    tp = C.track_table(hp, pp, preds).select("event_id", "track_id", *TRUTH_COLS)
    t = (feats.join(tn, on=["event_id", "track_id"])
         .join(tp.rename({c: c + "_pair" for c in TRUTH_COLS}), on=["event_id", "track_id"]))
    t.write_parquet(path)
    keep = ["event_id", "particle_id", "is_quirk", "n_hits", "technical_sm"]
    pn.select(keep).write_parquet(out / f"parts_{split}_normal.parquet")
    pp.select(keep).write_parquet(out / f"parts_{split}_pair.parquet")
    return t, pn.select(keep), pp.select(keep)


def feature_cols(t):
    truthish = set(TRUTH_COLS) | {c + "_pair" for c in TRUTH_COLS}
    return [c for c in t.columns if c not in DROP and c not in truthish and not c.startswith("score")]


def as_tables(t, score):
    """track tables for curves.py (normal / pair truth) with a 'score' column."""
    t = t.with_columns(score=pl.Series(score))
    tn = t.select("event_id", "track_id", "score", "cls", "matched", "maj_pid")
    tp = t.select("event_id", "track_id", "score", pl.col("cls_pair").alias("cls"), pl.col("matched_pair").alias("matched"),
                  pl.col("maj_pid_pair").alias("maj_pid"))
    return tn, tp


def main():
    model, out_root = sys.argv[1], Path(sys.argv[2])
    out = out_root / model
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    tv, pnv, ppv = build_split(model, "val", out)
    tt, pnt, ppt = build_split(model, "test", out)
    print(f"tables: val {tv.height} test {tt.height} tracks ({time.time() - t0:.0f} s)", flush=True)
    own = MODELS[model]["truth"]
    cls_col = "cls" if own == "normal" else "cls_pair"
    fake_col = "fake_rate" if own == "normal" else "fake_rate_pair"
    cols = feature_cols(tv)
    A = tv.filter(pl.col("event_id") < VAL_A_END)
    B = tv.filter(pl.col("event_id") >= VAL_A_END)
    pnB, ppB = pnv.filter(pl.col("event_id") >= VAL_A_END), ppv.filter(pl.col("event_id") >= VAL_A_END)
    X = lambda d: d.select(cols).to_numpy().astype(np.float64)  # noqa: E731
    res = {"model": model, "own_truth": own, "features": cols, "protocol": "train val 16000-16499, cuts on val "
           "16500-16999, test 17000-19999", "n_train_tracks": A.height}
    # B3 quirk tagger (first: its 90% cut is reused by the quality_or_tag variant)
    yq = A["maj_is_quirk"].to_numpy()
    tag = HistGradientBoostingClassifier(max_iter=500, learning_rate=0.08, max_leaf_nodes=63, early_stopping=True,
                                         validation_fraction=0.15, random_state=0)
    tag.fit(X(A), yq)
    qB, qT = tag.predict_proba(X(B))[:, 1], tag.predict_proba(X(tt))[:, 1]
    b3 = {"n_quirk_majority_train": int(yq.sum())}
    for defn, sel in {"quirk_tracks_good_own_truth": lambda d: d["maj_is_quirk"].to_numpy() & (d[cls_col] != "fake").to_numpy(),
                      "quirk_majority_tracks": lambda d: d["maj_is_quirk"].to_numpy()}.items():
        mB, mT = sel(B), sel(tt)
        cut = float(np.quantile(qB[mB], 0.10))
        ordinary_T = ~tt["maj_is_quirk"].to_numpy()
        n_ev = tt["event_id"].n_unique()
        evs = tt.filter(pl.Series(ordinary_T & (qT >= cut)))["event_id"]
        b3[defn] = {"cut_from_valB": cut, "quirk_track_eff_valB": float((qB[mB] >= cut).mean()),
                    "quirk_track_eff_test": float((qT[mT] >= cut).mean()), "n_quirk_tracks_test": int(mT.sum()),
                    "ordinary_tagged_per_event_test": float((ordinary_T & (qT >= cut)).sum() / n_ev),
                    "ordinary_tracks_per_event_test": float(ordinary_T.sum() / n_ev),
                    "events_with_ordinary_tagged_test": float(evs.n_unique() / n_ev),
                    "ordinary_tagged_nonqdesc_per_event_test": float(
                        (ordinary_T & (qT >= cut) & ~tt["maj_is_qdesc"].to_numpy()).sum() / n_ev)}
    res["b3"] = b3
    tcut = b3["quirk_majority_tracks"]["cut_from_valB"]
    # B2 quality score: plain; with quirk-majority good tracks weighted x30 (quality_qw); plain OR quirk tag
    y = (A[cls_col] != "fake").to_numpy()
    res["train_fake_fraction"] = float(1 - y.mean())
    clf = HistGradientBoostingClassifier(max_iter=500, learning_rate=0.08, max_leaf_nodes=63, early_stopping=True,
                                         validation_fraction=0.15, random_state=0)
    clf.fit(X(A), y)
    sB, sT = clf.predict_proba(X(B))[:, 1], clf.predict_proba(X(tt))[:, 1]
    w = np.where(yq & y, 30.0, 1.0)
    clf_w = HistGradientBoostingClassifier(max_iter=500, learning_rate=0.08, max_leaf_nodes=63, early_stopping=True,
                                           validation_fraction=0.15, random_state=0)
    clf_w.fit(X(A), y, sample_weight=w)
    wB, wT = clf_w.predict_proba(X(B))[:, 1], clf_w.predict_proba(X(tt))[:, 1]
    scores = {"quality": (sB, sT), "quality_qw": (wB, wT),
              "quality_or_tag": (sB + (qB >= tcut), sT + (qT >= tcut)),
              "n_hits": (B["n_hits"].to_numpy().astype(float), tt["n_hits"].to_numpy().astype(float))}
    curves = {}
    for name, (scB, scT) in scores.items():
        cuts = np.unique(np.concatenate([[-np.inf], np.quantile(scB, np.linspace(0, 0.995, 600))]))
        if name == "quality_or_tag":
            cuts = np.unique(np.concatenate([[-np.inf], np.quantile(sB, np.linspace(0, 0.995, 600))]))
        if name == "n_hits":
            cuts = np.concatenate([[-np.inf], np.arange(2, 31)])
        cvB = CV.curve(*as_tables(B, scB), pnB, ppB, cuts)
        cvT = CV.curve(*as_tables(tt, scT), pnt, ppt, cuts)
        cvB.write_csv(out / f"curve_{name}_valB.csv")
        cvT.write_csv(out / f"curve_{name}_test.csv")
        curves[name] = {"no_cut_test": cvT.row(0, named=True),
                        "op": [CV.operating_point(cvB, cvT, fake_col, f) for f in (0.05, 0.02)]}
        if name == "n_hits":
            i4 = int(np.where(cuts == 4)[0][0])
            curves[name]["min_hits_4_test"] = cvT.row(i4, named=True)
            curves[name]["min_hits_4_valB"] = cvB.row(i4, named=True)
    res["b2"] = curves
    tt.select("event_id", "track_id").with_columns(quality=pl.Series(sT), quality_qw=pl.Series(wT),
                                                   quirk_tag=pl.Series(qT)).write_parquet(out / "scores_test.parquet")
    (out / "result.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
    print(json.dumps({"b2": {k: {"no_cut": v["no_cut_test"], "op": v["op"]} for k, v in curves.items()}, "b3": b3},
                     indent=1, default=float)[:6000])
    print(f"done ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
