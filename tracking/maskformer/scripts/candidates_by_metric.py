"""Print the checkpoint paths of the K validation points with the highest val/coupled_strict of run TAG (ckpt files
are named by val/loss, matched to the metrics.csv row of the same validation). Usage: candidates_by_metric.py TAG [K]"""
import glob
import sys

import pandas as pd

tag, k = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 3
R = f"/pscratch/sd/d/danieltm/quirk_tracking/maskformer/runs/{tag}"
d = pd.concat([pd.read_csv(f) for f in glob.glob(f"{R}/*/*/version_0/metrics.csv")])
v = d.groupby("step")[["val/coupled_strict", "val/loss"]].last().dropna().sort_values("val/coupled_strict", ascending=False)
ck = {round(float(p.split("val_loss=")[1].removesuffix(".ckpt")), 5): p for p in glob.glob(f"{R}/*/*/version_0/ckpts/*.ckpt")}
out = []
for step, r in v.iterrows():
    p = ck.get(round(float(r["val/loss"]), 5))
    if p:
        out.append(p)
        print(f"# step {int(step) + 1} val/coupled_strict {r['val/coupled_strict']:.4f}", file=sys.stderr)
    if len(out) == k:
        break
print("\n".join(out))
