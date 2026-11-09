"""Coupled efficiency vs background rejection at mu 60 (fakes/coupled_roc.py recipe: B2 quality score with pair-truth
labels, quirk-majority good tracks x30, HistGradientBoosting), for the pair GNN trained at mu 60. Train on val
2000-2499, report the curve on test 0-1999 (cuts swept; an operating point is read off at the val-B (2500-2999) cut
for a given coupled efficiency). The model is saved (quality.pkl).
  python roc60.py NAME VAL_PRED TEST_PRED SCORED_DIR OUT_ROOT
"""
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fakes"))
import common as C  # noqa: E402
import quality as QL  # noqa: E402
import coupled_roc as CR  # noqa: E402

Q = "/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirk_tracking_data"
C.SRC = Q + "/parquet_r1_mu60/m200_L774_mu60"
C.PAIR_TRUTH = Q + "/pair_truth_mu60/m200_L774_mu60"
QL.SPLITS = {"val": (2000, 3000), "test": (0, 2000)}
VAL_A_END = 2500

name, vpred, tpred, scored, root = sys.argv[1:6]
QL.MODELS[name] = dict(truth="pair", val=vpred, test=tpred, scored=scored + "/events*.parquet")
out = Path(root) / name
out.mkdir(parents=True, exist_ok=True)
t0 = time.time()
tv, pnv, ppv = QL.build_split(name, "val", out)
tt, pnt, ppt = QL.build_split(name, "test", out)
cols = QL.feature_cols(tv)
A = tv.filter(pl.col("event_id") < VAL_A_END)
B = tv.filter(pl.col("event_id") >= VAL_A_END)
y = (A["cls_pair"] != "fake").to_numpy()
w = np.where(A["maj_is_quirk_pair"].to_numpy() & y, 30.0, 1.0)
clf = HistGradientBoostingClassifier(max_iter=500, learning_rate=0.08, max_leaf_nodes=63, early_stopping=True,
                                     validation_fraction=0.15, random_state=0)
clf.fit(A.select(cols).to_numpy().astype(np.float64), y, sample_weight=w)
pickle.dump(clf, open(out / "quality.pkl", "wb"))
X = lambda d: d.select(cols).to_numpy().astype(np.float64)  # noqa: E731
sB, sT = clf.predict_proba(X(B))[:, 1], clf.predict_proba(X(tt))[:, 1]
rB = CR.roc(B, pnv.filter(pl.col("event_id") >= VAL_A_END), ppv.filter(pl.col("event_id") >= VAL_A_END), sB)
rT = CR.roc(tt, pnt, ppt, sT)
rB.write_csv(out / "roc_valB.csv"); rT.write_csv(out / "roc_test.csv")
# operating points: cut chosen on val B where the coupled efficiency first drops below a fraction of its no-cut value
ops = {}
for frac in (0.99, 0.95, 0.90):
    target = frac * rB["coupled_eff"][0]
    okB = rB.filter(pl.col("coupled_eff") >= target)
    cut = float(okB["cut"].max())
    rowT = rT.filter(pl.col("cut") >= cut).row(0, named=True) if rT.filter(pl.col("cut") >= cut).height else rT.row(-1, named=True)
    ops[f"keep_{frac}"] = {"cut_valB": cut, "valB": okB.sort("cut").row(-1, named=True), "test": rowT}
# fixed-fake working points: the loosest cut whose val-B fake rate is <= target, applied to test once
for target in (0.07, 0.02):
    okB = rB.filter(pl.col("fake") <= target)
    if okB.height:
        cut = float(okB["cut"].min())
        sel = rT.filter(pl.col("cut") >= cut)
        ops[f"fake_{target}"] = {"cut_valB": cut, "valB": okB.sort("cut").row(0, named=True),
                                 "test": sel.row(0, named=True) if sel.height else rT.row(-1, named=True)}
info = {"model": name, "features": cols, "n_train": A.height, "no_cut_test": rT.row(0, named=True), "operating_points": ops,
        "seconds": round(time.time() - t0)}
(out / "roc.json").write_text(json.dumps(info, indent=1, default=float) + "\n")
print(json.dumps({"no_cut": {k: round(v, 4) for k, v in rT.row(0, named=True).items()},
                  **{k: {kk: round(vv, 4) for kk, vv in v["test"].items()} for k, v in ops.items()}}, indent=1), flush=True)
