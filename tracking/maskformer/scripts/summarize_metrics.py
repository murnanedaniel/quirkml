"""Print a compact table of the key training/validation metrics from a CSVLogger metrics.csv."""
import sys

import pandas as pd

d = pd.read_csv(sys.argv[1])
every = int(sys.argv[2]) if len(sys.argv) > 2 else 1
keys = ["train/loss", "train/grad_norm", "train/steps_per_s", "train/effective_batch_size", "train/gpu_mem_peak_gb",
        "val/loss", "val/mask_acc", "val/hit_recall", "val/p0.5_eff", "val/p1.0_eff", "val/p0.5_quirk_eff",
        "val/p1.0_quirk_eff", "val/p0.5_pur", "val/num_pred", "val/num_true"]
keys += [k for k in d.columns if k.startswith("lr-")]
cols = [k for k in keys if k in d.columns]
g = d.groupby("step")[cols].last().dropna(how="all")
pd.set_option("display.width", 250)
print("train rows:")
print(g[[c for c in cols if not c.startswith("val/")]].dropna(how="all").iloc[::every].tail(12).to_string(float_format=lambda v: f"{v:.4g}"))
print("val rows:")
print(g[[c for c in cols if c.startswith("val/")]].dropna(how="all").to_string(float_format=lambda v: f"{v:.4g}"))
