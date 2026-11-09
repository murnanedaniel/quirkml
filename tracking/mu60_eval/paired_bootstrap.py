"""Paired event bootstrap of the coupled (pair-truth) efficiency difference between two runs scored by py_score.py.

Inputs: the per-event parquets py_score.py writes on a pair-truth copy (columns event_id, nsel, nf, point); on pair
truth every event has one selected pair object (nsel = 1) and nf = 1 if it was found. Events are resampled jointly
(same event ids for both runs), N_BOOT times; prints the efficiency of each run, the difference B - A and its 68% /
95% intervals per working point.

    python paired_bootstrap.py A_EVENTS.parquet B_EVENTS.parquet [N_BOOT]
"""
import sys

import numpy as np
import polars as pl

a_path, b_path = sys.argv[1], sys.argv[2]
n_boot = int(sys.argv[3]) if len(sys.argv) > 3 else 10000
a, b = pl.read_parquet(a_path), pl.read_parquet(b_path)
rng = np.random.default_rng(0)
for point in a["point"].unique(maintain_order=True):
    x = a.filter(pl.col("point") == point).select("event_id", (pl.col("nf") >= pl.col("nsel")).alias("fa"))
    y = b.filter(pl.col("point") == point).select("event_id", (pl.col("nf") >= pl.col("nsel")).alias("fb"))
    j = x.join(y, on="event_id", how="inner")
    if j.height != x.height or j.height != y.height:
        print(f"{point}: warning, events differ ({x.height} / {y.height} / common {j.height})")
    fa, fb = j["fa"].to_numpy().astype(float), j["fb"].to_numpy().astype(float)
    idx = rng.integers(0, len(fa), size=(n_boot, len(fa)))
    d = fb[idx].mean(1) - fa[idx].mean(1)
    q = np.quantile(d, [0.025, 0.16, 0.84, 0.975])
    print(f"{point}: A {fa.mean():.4f}  B {fb.mean():.4f}  B-A {fb.mean() - fa.mean():+.4f}  68% [{q[1]:+.4f}, {q[2]:+.4f}]"
          f"  95% [{q[0]:+.4f}, {q[3]:+.4f}]  (n = {len(fa)}, P(B-A <= 0) = {(d <= 0).mean():.3f})")
