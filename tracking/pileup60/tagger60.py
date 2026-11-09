"""B3 quirk tagger retrained at mu 60 (fakes/quality.py recipe) on a pair GNN's val tracks: label = majority particle is
a quirk (normal truth); trained on val 2000-2499; cut on val 2500-2999 at 90% efficiency for quirk tracks good in the
pair truth (the gate used at mu 0); test 0-1999 scored once. Model saved (tagger.pkl, tagger.json) to MODEL_DIR.
  python tagger60.py NAME VAL_PRED TEST_PRED SCORED_DIR OUT_ROOT MODEL_DIR
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
import quality as QL  # noqa: E402

Q = "/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirk_tracking_data"
C.SRC = Q + "/parquet_r1_mu60/m200_L774_mu60"
C.PAIR_TRUTH = Q + "/pair_truth_mu60/m200_L774_mu60"
QL.SPLITS = {"val": (2000, 3000), "test": (0, 2000)}
VAL_A_END = 2500

name, vpred, tpred, scored, root, mdir = sys.argv[1:7]
QL.MODELS[name] = dict(truth="pair", val=vpred, test=tpred, scored=scored + "/events*.parquet")
out = Path(root) / name
out.mkdir(parents=True, exist_ok=True)
mdir = Path(mdir); mdir.mkdir(parents=True, exist_ok=True)
tv, _, _ = QL.build_split(name, "val", out)
tt, _, _ = QL.build_split(name, "test", out)
cols = QL.feature_cols(tv)
A = tv.filter(pl.col("event_id") < VAL_A_END)
B = tv.filter(pl.col("event_id") >= VAL_A_END)
X = lambda d: d.select(cols).to_numpy().astype(np.float64)  # noqa: E731
# mu 60: ~2e-4 of the training tracks are quirk-majority (514 of 2.4M); unweighted, the mu 0 recipe gives a constant
# score. Positives are weighted to balance the classes; no early stopping (its 15% split holds ~80 positives).
yA = A["maj_is_quirk"].to_numpy()
tag = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, max_leaf_nodes=63, early_stopping=False, random_state=0)
tag.fit(X(A), yA, sample_weight=np.where(yA, (~yA).sum() / max(yA.sum(), 1), 1.0))
qB, qT = tag.predict_proba(X(B))[:, 1], tag.predict_proba(X(tt))[:, 1]
res = {"model": name, "features": cols, "n_train": A.height, "n_quirk_majority_train": int(A["maj_is_quirk"].sum())}
for defn, sel in {"pair_good": lambda d: d["maj_is_quirk"].to_numpy() & (d["cls_pair"] != "fake").to_numpy(),
                  "quirk_majority": lambda d: d["maj_is_quirk"].to_numpy()}.items():
    mB, mT = sel(B), sel(tt)
    cut = float(np.quantile(qB[mB], 0.10))
    ordT = ~tt["maj_is_quirk"].to_numpy()
    res[defn] = {"cut_from_valB": cut, "quirk_track_eff_valB": float((qB[mB] >= cut).mean()),
                 "quirk_track_eff_test": float((qT[mT] >= cut).mean()), "n_quirk_tracks_test": int(mT.sum()),
                 "tagged_per_event_test": float((qT >= cut).sum() / 2000),
                 "ordinary_tagged_per_event_test": float((ordT & (qT >= cut)).sum() / 2000)}
pickle.dump(tag, open(mdir / "tagger.pkl", "wb"))
json.dump(res, open(mdir / "tagger.json", "w"), indent=1)
tt.select("event_id", "track_id").with_columns(quirk_tag=pl.Series(qT)).write_parquet(out / "b3_scores_test.parquet")
print(json.dumps(res, indent=1), flush=True)
