"""Per-quirk failure breakdown of a predictions.parquet (not the official evaluator).

For each quirk with >= 3 hits: best track = predicted track holding most of its hits. Reports medians of
hit efficiency / purity of that track, the share of its wrong hits that belong to the other quirk, the number
of tracks the quirk's hits are spread over, and the fraction of quirk hits left unassigned.
Usage: python quirk_autopsy.py PRED_DIR DATASET_ROOT
"""
import glob
import sys

import numpy as np
import pyarrow.parquet as pq

pred = pq.read_table(f"{sys.argv[1]}/predictions.parquet").to_pandas()
root = sys.argv[2]
want = set(pred.event_id.unique().tolist())
hits, parts = {}, {}
for f in sorted(glob.glob(f"{root}/reco/tracker_hits/*.parquet")):
    t = pq.read_table(f, columns=["event_id", "particle_id"]).to_pydict()
    for e, pid in zip(t["event_id"], t["particle_id"], strict=True):
        if e in want:
            hits[e] = np.asarray(pid, dtype=np.int64)
for f in sorted(glob.glob(f"{root}/truth/particles/*.parquet")):
    t = pq.read_table(f, columns=["event_id", "particle_id", "pdg_id"]).to_pydict()
    for i, e in enumerate(t["event_id"]):
        if e in want:
            parts[e] = (np.asarray(t["particle_id"][i]), np.asarray(t["pdg_id"][i]))
rows = []
for e, g in pred.groupby("event_id"):
    pid = hits[e]
    tid = np.full(len(pid), -1)
    tid[g.hit_id.to_numpy()] = g.predicted_track_id.to_numpy()
    ids, pdg = parts[e]
    quirks = ids[np.abs(pdg) == 10000100]
    for q in quirks:
        m = pid == q
        n = m.sum()
        if n < 3:
            continue
        other = [o for o in quirks if o != q]
        t_q = tid[m]
        unassigned = (t_q < 0).mean()
        tr, c = np.unique(t_q[t_q >= 0], return_counts=True)
        if len(tr) == 0:
            rows.append((n, 0, 0, np.nan, 0, unassigned))
            continue
        best = tr[np.argmax(c)]
        in_best = tid == best
        eff = c.max() / n
        pur = c.max() / in_best.sum()
        wrong = in_best & ~m
        frac_other = np.isin(pid[wrong], other).mean() if wrong.any() else np.nan
        rows.append((n, eff, pur, frac_other, len(tr), unassigned))
a = np.array(rows, dtype=float)
print(f"quirks {len(a)}; hits median {np.median(a[:, 0]):.0f}")
print(f"best-track hit eff median {np.median(a[:, 1]):.2f}; purity median {np.nanmedian(a[:, 2]):.2f}")
print(f"eff>=0.5 {np.mean(a[:, 1] >= 0.5):.3f}; pur>=0.5 {np.mean(a[:, 2] >= 0.5):.3f}; both {np.mean((a[:, 1] >= 0.5) & (a[:, 2] >= 0.5)):.3f}")
print(f"wrong hits in best track from the other quirk (median over quirks) {np.nanmedian(a[:, 3]):.2f}")
print(f"tracks per quirk median {np.median(a[:, 4]):.0f}; unassigned quirk-hit fraction median {np.median(a[:, 5]):.2f}")
