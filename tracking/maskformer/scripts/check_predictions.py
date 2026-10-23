"""Quick truth check of a predictions.parquet against the R1 tracker_hits (not the official evaluator).

Joins predictions on (event_id, hit_id = row position in the event's hit lists) and
reports, per quirk and for all particles with pT > 1 GeV and >= 7 hits, the fraction
matched by a predicted track with >= 50% hit efficiency and >= 50% purity, plus the
fraction of predicted tracks (>= 3 hits) that are not >= 50% pure (fake proxy).

Usage: python check_predictions.py PRED_DIR DATASET_ROOT
"""

import glob
import sys

import numpy as np
import pyarrow.parquet as pq

pred = pq.read_table(f"{sys.argv[1]}/predictions.parquet").to_pandas()
root = sys.argv[2]
want = set(pred.event_id.unique().tolist())
QUIRK = 10000100

hits = {}
parts = {}
for f in sorted(glob.glob(f"{root}/reco/tracker_hits/*.parquet")):
    t = pq.read_table(f, columns=["event_id", "particle_id"]).to_pydict()
    for e, pid in zip(t["event_id"], t["particle_id"], strict=True):
        if e in want:
            hits[e] = np.asarray(pid, dtype=np.int64)
for f in sorted(glob.glob(f"{root}/truth/particles/*.parquet")):
    t = pq.read_table(f, columns=["event_id", "particle_id", "pdg_id", "px", "py", "charge"]).to_pydict()
    for i, e in enumerate(t["event_id"]):
        if e in want:
            parts[e] = {k: np.asarray(t[k][i]) for k in ("particle_id", "pdg_id", "px", "py", "charge")}

n_q = n_q_ok = n_t = n_t_ok = n_tracks = n_fake = 0
n_hits_checked = 0
for e, g in pred.groupby("event_id"):
    pid = hits[e]
    assert g.hit_id.max() < len(pid) and len(g) == len(pid), f"event {e}: hit count mismatch {len(g)} vs {len(pid)}"
    tid = np.full(len(pid), -1, dtype=np.int64)
    tid[g.hit_id.to_numpy()] = g.predicted_track_id.to_numpy()
    n_hits_checked += len(pid)
    # best (track -> majority particle) table
    track_major = {}
    for t in np.unique(tid[tid >= 0]):
        m = tid == t
        ids, c = np.unique(pid[m], return_counts=True)
        k = np.argmax(c)
        track_major[t] = (ids[k], c[k], m.sum())
        if m.sum() >= 3:
            n_tracks += 1
            n_fake += c[k] / m.sum() < 0.5
    p = parts[e]
    pt = np.hypot(p["px"], p["py"])
    nh = {i: (pid == i).sum() for i in p["particle_id"]}
    for j, i in enumerate(p["particle_id"]):
        ok = any(maj == i and cnt / tot >= 0.5 and cnt / max(nh[i], 1) >= 0.5 for maj, cnt, tot in track_major.values())
        if abs(p["pdg_id"][j]) == QUIRK:
            n_q += 1
            n_q_ok += ok
        elif p["charge"][j] != 0 and pt[j] > 1.0 and nh[i] >= 7:
            n_t += 1
            n_t_ok += ok
print(f"events {pred.event_id.nunique()}, hits checked {n_hits_checked}, assigned frac {np.mean(pred.predicted_track_id >= 0):.3f}")
print(f"quirk eff (all quirks incl. 0-hit) {n_q_ok}/{n_q} = {n_q_ok / max(n_q, 1):.3f}")
print(f"technical eff (pT>1, >=7 hits, non-quirk) {n_t_ok}/{n_t} = {n_t_ok / max(n_t, 1):.3f}")
print(f"fake proxy (tracks >=3 hits, <50% pure) {n_fake}/{n_tracks} = {n_fake / max(n_tracks, 1):.3f}")
