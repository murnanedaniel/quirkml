"""Paired event bootstrap between two runs' val builder results (builders60.py json with per_event), one builder spec.
  python compare_runs.py A.json B.json SPEC [SPEC_B]"""
import json, sys
import numpy as np
A = json.load(open(sys.argv[1]))["results"][sys.argv[3]]["per_event"]
B = json.load(open(sys.argv[2]))["results"][sys.argv[4] if len(sys.argv) > 4 else sys.argv[3]]["per_event"]
assert A["event_id"] == B["event_id"]
n = np.array(A["pairs"], float); rng = np.random.default_rng(0); I = rng.integers(0, len(n), (2000, len(n)))
for k in ("loose", "tight"):
    a, b = np.array(A[k], float), np.array(B[k], float)
    d = (b[I].sum(1) - a[I].sum(1)) / n[I].sum(1)
    print(k, "A %.3f B %.3f  B-A %+.3f [%+.3f, %+.3f]" % (a.sum() / n.sum(), b.sum() / n.sum(), d.mean(), *np.percentile(d, [2.5, 97.5])))
