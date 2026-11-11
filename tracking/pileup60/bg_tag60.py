"""mu60 background tag rate: the L774 pair-GNN chain's tagger (and quality cut) on quirk-free events (noquirk60.py).
Counts tagged tracks per event and the fraction of events with >= 1 tagged track, without and with the quality cut
at the L774 val-B working points.
  python bg_tag60.py DS FIRST STOP NAME PRED SCORED_DIR OUT_ROOT TAGGER_DIR QUALITY_DIR
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

ds, first, stop, name, pred, scored, root, tagdir, qdir = sys.argv[1:10]
first, stop = int(first), int(stop)
Q = "/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirk_tracking_data"
C.SRC = C.PAIR_TRUTH = f"{Q}/parquet_r1_mu60/{ds}"  # no quirks: pair truth = normal truth
QL.SPLITS = {"test": (first, stop)}
QL.MODELS[name] = dict(truth="pair", val=None, test=pred, scored=scored + "/events*.parquet")
out = Path(root) / name
out.mkdir(parents=True, exist_ok=True)
tt, _, _ = QL.build_split(name, "test", out)
tj = json.load(open(Path(tagdir) / "tagger.json"))
qj = json.load(open(Path(qdir) / "roc.json"))
X = tt.select(tj["features"]).to_numpy().astype(np.float64)
qT = pickle.load(open(Path(tagdir) / "tagger.pkl", "rb")).predict_proba(X)[:, 1]
sT = pickle.load(open(Path(qdir) / "quality.pkl", "rb")).predict_proba(X)[:, 1]
n_ev = stop - first
assert not tt["maj_is_quirk"].any(), "quirk-majority track in a quirk-free sample"
ev = tt["event_id"].to_numpy()
tag_cut = tj["pair_good"]["cut_from_valB"]
res = {"dataset": ds, "events": [first, stop], "n_events": n_ev, "tracks_per_event": tt.height / n_ev, "tagger_cut": tag_cut}
for label, sel in [("tagger", np.ones(len(qT), bool))] + [
        (f"tagger+quality_{wp}", sT >= qj["operating_points"][wp]["cut_valB"]) for wp in ("fake_0.07", "fake_0.02")]:
    t = (qT >= tag_cut) & sel
    per_ev = np.bincount(ev[t] - first, minlength=n_ev)
    res[label] = {"tagged_per_event": float(t.sum() / n_ev), "frac_events_ge1": float((per_ev >= 1).mean()),
                  "frac_events_ge2": float((per_ev >= 2).mean()), "n_tagged": int(t.sum())}
# tag-score tail: fraction of events whose highest tag score exceeds each cut (for a tighter working point)
mx = np.full(n_ev, -1.0)
np.maximum.at(mx, ev - first, qT)
res["event_max_tag_quantiles"] = {str(q): float(np.quantile(mx, q)) for q in (0.5, 0.9, 0.99, 0.999)}
tt.select("event_id", "track_id").with_columns(quirk_tag=pl.Series(qT), quality=pl.Series(sT)).write_parquet(out / "bg_scores.parquet")
(out / "bg_tag.json").write_text(json.dumps(res, indent=1) + "\n")
print(json.dumps(res, indent=1), flush=True)
