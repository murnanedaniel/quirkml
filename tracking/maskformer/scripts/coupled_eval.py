"""Coupled quirk efficiency of a predictions.parquet (Python, not the ACTS evaluator).

Events in the selection: both quirks (|pdg| 10000100) leave >= 3 hits. U = union of both quirks' hits.
Coupled match: some predicted track T with |T & U| >= 0.5 |U| and |T & U| >= 0.5 |T| (double 50% matching of the
pair as one object). Also prints the per-quirk double-majority rate with the same rule (quirk vs its own hits), and
the fraction of selected events whose best coupled track holds both quirks with >= 3 hits each.

Usage: python coupled_eval.py PRED_DIR DATASET_ROOT FIRST LAST_EXCL [OUT_JSON]
"""

import glob
import json
import sys

import numpy as np
import pyarrow.parquet as pq

pred_dir, root, first, stop = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
out_json = sys.argv[5] if len(sys.argv) > 5 else None
pred = pq.read_table(f"{pred_dir}/predictions.parquet").to_pandas()
pred = pred[(pred.event_id >= first) & (pred.event_id < stop)]
want = set(pred.event_id.unique().tolist())
hits, quirks = {}, {}
for f in sorted(glob.glob(f"{root}/reco/tracker_hits/*.parquet")):
    t = pq.read_table(f, columns=["event_id", "particle_id"]).to_pydict()
    for e, pid in zip(t["event_id"], t["particle_id"], strict=True):
        if e in want:
            hits[e] = np.asarray(pid, dtype=np.int64)
for f in sorted(glob.glob(f"{root}/truth/particles/*.parquet")):
    t = pq.read_table(f, columns=["event_id", "particle_id", "pdg_id"]).to_pydict()
    for i, e in enumerate(t["event_id"]):
        if e in want:
            ids, pdg = np.asarray(t["particle_id"][i]), np.asarray(t["pdg_id"][i])
            quirks[e] = ids[np.abs(pdg) == 10000100]


def matched(tid, sel, strict=False):
    """Return (any double-50% match, best track id) for the hit set sel; strict: > 50% instead of >= 50%."""
    tr, c = np.unique(tid[sel & (tid >= 0)], return_counts=True)
    best, ok = -1, False
    for t, n in zip(tr, c, strict=True):
        size = (tid == t).sum()
        passes = (n > 0.5 * sel.sum() and n > 0.5 * size) if strict else (n >= 0.5 * sel.sum() and n >= 0.5 * size)
        if passes:
            ok, best = True, t
    if best < 0 and len(tr):
        best = tr[np.argmax(c)]
    return ok, best


n_ev = n_coupled = n_both_in = n_q = n_q_ok = n_coupled_strict = 0
for e, g in pred.groupby("event_id"):
    pid = hits[e]
    tid = np.full(len(pid), -1)
    tid[g.hit_id.to_numpy()] = g.predicted_track_id.to_numpy()
    qs = [q for q in quirks[e] if (pid == q).sum() >= 3]
    for q in qs:
        n_q += 1
        n_q_ok += matched(tid, pid == q)[0]
    if len(qs) != 2:
        continue
    n_ev += 1
    u = np.isin(pid, qs)
    ok, best = matched(tid, u)
    n_coupled += ok
    n_coupled_strict += matched(tid, u, strict=True)[0]
    if ok:
        in_t = tid == best
        n_both_in += all(((pid == q) & in_t).sum() >= 3 for q in qs)
res = {"events": n_ev, "coupled_eff": n_coupled / max(n_ev, 1), "coupled_eff_strict": n_coupled_strict / max(n_ev, 1), "coupled_with_both_quirks": n_both_in / max(n_ev, 1),
       "quirks": n_q, "per_quirk_double_majority": n_q_ok / max(n_q, 1)}
print(json.dumps(res))
if out_json:
    with open(out_json, "w") as fh:
        json.dump(res, fh, indent=1)
