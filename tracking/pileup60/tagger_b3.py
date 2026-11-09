"""B3 quirk tagger, saved to disk (the 2026-10-26 B2/B3 models were refitted per run and never saved).

fit:   refit the B3 tagger exactly as fakes/quality.py (pair_gnn_s0_cc: train on val 16000-16499 tracks, cut at 90%
       efficiency for quirk tracks good in pair truth on val 16500-16999), from the saved mu 0 track table; check the
       cut and the test scores against the 2026-10-26 outputs; save model + cut.
apply: features (fakes/features.py: geometry + GNN edge-score features) for built tracks of any set, then the tag.

  python tagger_b3.py fit OUT_DIR
  python tagger_b3.py apply OUT_DIR PRED_DIR SCORED_GLOB R1_ROOT FIRST STOP OUT.parquet
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fakes"))
import common as C  # noqa: E402
import features as F  # noqa: E402
import quality as QY  # noqa: E402

B2 = Path("/pscratch/sd/d/danieltm/quirk_tracking/fakes/b2/pair_gnn_s0_cc")


def fit(out):
    out.mkdir(parents=True, exist_ok=True)
    tv = pl.read_parquet(B2 / "tracks_val.parquet")
    tt = pl.read_parquet(B2 / "tracks_test.parquet")
    cols = QY.feature_cols(tv)
    A = tv.filter(pl.col("event_id") < QY.VAL_A_END)
    B = tv.filter(pl.col("event_id") >= QY.VAL_A_END)
    X = lambda d: d.select(cols).to_numpy().astype(np.float64)  # noqa: E731
    tag = HistGradientBoostingClassifier(max_iter=500, learning_rate=0.08, max_leaf_nodes=63, early_stopping=True,
                                         validation_fraction=0.15, random_state=0)
    tag.fit(X(A), A["maj_is_quirk"].to_numpy())
    qB, qT = tag.predict_proba(X(B))[:, 1], tag.predict_proba(X(tt))[:, 1]
    mB = B["maj_is_quirk"].to_numpy() & (B["cls_pair"] != "fake").to_numpy()
    cut = float(np.quantile(qB[mB], 0.10))
    mBm = B["maj_is_quirk"].to_numpy()
    cut_maj = float(np.quantile(qB[mBm], 0.10))
    old = json.load(open(B2 / "result.json"))["b3"]
    ref = pl.read_parquet(B2 / "scores_test.parquet").join(
        tt.select("event_id", "track_id").with_columns(new=pl.Series(qT)), on=["event_id", "track_id"])
    check = dict(cut=cut, cut_old=old["quirk_tracks_good_own_truth"]["cut_from_valB"], cut_majority=cut_maj,
                 cut_majority_old=old["quirk_majority_tracks"]["cut_from_valB"],
                 max_abs_diff_test_scores=float((ref["new"] - ref["quirk_tag"]).abs().max()), n_test=ref.height)
    print(json.dumps(check), flush=True)
    pickle.dump(tag, open(out / "tagger.pkl", "wb"))
    json.dump(dict(features=cols, cut_pair_good=cut, cut_quirk_majority=cut_maj, source="pair_gnn_s0_cc mu 0, fakes/quality.py protocol",
                   check=check), open(out / "tagger.json", "w"), indent=1)


def apply(out, pred_dir, scored_glob, r1, first, stop, dest):
    tag = pickle.load(open(out / "tagger.pkl", "rb"))
    cfg = json.load(open(out / "tagger.json"))
    preds = C.read_preds(pred_dir, first, stop)
    hn, _ = C.load_truth(r1, first, stop)  # positions only are used by the features
    feats = F.geometry_features(hn, preds)
    feats = feats.join(F.edge_features(preds, scored_glob, first, stop), on=["event_id", "track_id"], how="left").fill_null(0)
    q = tag.predict_proba(feats.select(cfg["features"]).to_numpy().astype(np.float64))[:, 1]
    feats.select("event_id", "track_id").with_columns(quirk_tag=pl.Series(q)).write_parquet(dest)
    print("tracks", feats.height, "tagged at pair-good cut", int((q >= cfg["cut_pair_good"]).sum()), flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "fit":
        fit(Path(sys.argv[2]))
    else:
        apply(Path(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5], int(sys.argv[6]), int(sys.argv[7]), sys.argv[8])
