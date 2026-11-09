"""mu60, other string scales: apply the Lambda = 774 tagger and quality model of a pair GNN (no retraining) to the test
tracks (0-1999) of another set, at the cuts chosen on L774 val B.
  python lam_apply60.py DS NAME TEST_PRED SCORED_DIR OUT_ROOT TAGGER_DIR QUALITY_DIR
Writes OUT_ROOT/NAME/{b3_scores_test.parquet, roc_test.csv, apply.json}.
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fakes"))
import common as C  # noqa: E402
import quality as QL  # noqa: E402
import coupled_roc as CR  # noqa: E402

ds, name, tpred, scored, root, tagdir, qdir = sys.argv[1:8]
Q = "/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirk_tracking_data"
C.SRC = f"{Q}/parquet_r1_mu60/{ds}"
C.PAIR_TRUTH = f"{Q}/pair_truth_mu60/{ds}"
QL.SPLITS = {"test": (0, 2000)}
QL.MODELS[name] = dict(truth="pair", val=None, test=tpred, scored=scored + "/events*.parquet")
out = Path(root) / name
out.mkdir(parents=True, exist_ok=True)
tt, pnt, ppt = QL.build_split(name, "test", out)

tag = pickle.load(open(Path(tagdir) / "tagger.pkl", "rb"))
tj = json.load(open(Path(tagdir) / "tagger.json"))
qual = pickle.load(open(Path(qdir) / "quality.pkl", "rb"))
qj = json.load(open(Path(qdir) / "roc.json"))
assert tj["features"] == qj["features"] == QL.feature_cols(tt), "feature columns differ from the L774 models"
X = tt.select(tj["features"]).to_numpy().astype(np.float64)

qT = tag.predict_proba(X)[:, 1]
cut = tj["pair_good"]["cut_from_valB"]
good = tt["maj_is_quirk"].to_numpy() & (tt["cls_pair"] != "fake").to_numpy()
ordT = ~tt["maj_is_quirk"].to_numpy()
res = {"dataset": ds, "model": name, "tagger": {
    "cut_L774_valB": cut, "pair_good_eff_test": float((qT[good] >= cut).mean()) if good.any() else None,
    "n_pair_good_test": int(good.sum()), "tagged_per_event_test": float((qT >= cut).sum() / 2000),
    "ordinary_tagged_per_event_test": float((ordT & (qT >= cut)).sum() / 2000)}}
tt.select("event_id", "track_id").with_columns(quirk_tag=pl.Series(qT)).write_parquet(out / "b3_scores_test.parquet")

sT = qual.predict_proba(X)[:, 1]
rT = CR.roc(tt, pnt, ppt, sT)
rT.write_csv(out / "roc_test.csv")
res["quality"] = {"no_cut_test": rT.row(0, named=True)}
for wp in ("fake_0.07", "fake_0.02"):
    c = qj["operating_points"][wp]["cut_valB"]
    res["quality"][wp] = {"cut_L774_valB": c, "test": CR.roc(tt, pnt, ppt, sT, cuts=np.array([c])).row(0, named=True)}
(out / "apply.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
print(json.dumps(res, indent=1, default=float), flush=True)
