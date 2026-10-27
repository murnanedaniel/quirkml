"""Summary tables for toy_vs_g4.py output (per point, per time bin, per momentum bin, step-size check)."""
import sys

import numpy as np
import pandas as pd

D = sys.argv[1] if len(sys.argv) > 1 else "/pscratch/sd/d/danieltm/quirk_tracking/params/toy_vs_g4"
h = pd.read_parquet(f"{D}/toy_vs_g4_hits.parquet")
c = pd.read_parquet(f"{D}/toy_vs_g4_conv.parquet")
h["res"] = np.hypot(h.u0, h.u1)
h["adt_ps"] = 1e3 * h.dt.abs()
q = lambda p: (lambda s: np.nanpercentile(s, p))


def table(g):
    return g.agg(events=("ev", "nunique"), hits=("res", "size"), med=("res", "median"), p68=("res", q(68)), p90=("res", q(90)),
                 p99=("res", q(99)), f_gt1mm=("res", lambda s: np.mean(s > 1)), f_gt5mm=("res", lambda s: np.mean(s > 5)),
                 f_nocross=("res", lambda s: np.mean(~np.isfinite(s))), dt_med_ps=("adt_ps", "median"),
                 dt_p90_ps=("adt_ps", q(90)), cost_s=("cost", "median"))


pd.set_option("display.width", 200)
print("== per point (all sim hits of both quirks) ==")
print(table(h.groupby(["m", "lam"])).round(4).to_string())
h["tbin"] = pd.cut(h.t, [0, 2, 5, 10, 20, 100])
print("\n== by hit time [ns], points of the main 3x3 set ==")
main = h[h.lam.isin([167, 774, 3594])]
print(table(main.groupby(["lam", "tbin"], observed=True)).round(4).to_string())
h["bgbin"] = pd.cut(h.bg0, [0, 1, 2, 4, 8, 100]); main = h[h.lam.isin([167, 774, 3594])]
print("\n== by initial beta*gamma of the quirk ==")
print(table(main.groupby(["lam", "bgbin"], observed=True)).round(4).to_string())
pe = h.groupby(["m", "lam", "ev", "quirk"]).res.median().reset_index()
print("\n== per-quirk median residual: fraction of quirks with median < 0.1 / 1 mm ==")
print(pe.groupby(["m", "lam"]).res.agg(lt01=lambda s: np.mean(s < 0.1), lt1=lambda s: np.mean(s < 1), med="median").round(3).to_string())
print("\n== step-size check (default vs 4x finer), events 0-4 ==")
print(c.groupby(["m", "lam"]).agg(max_dpos=("d_pos", "max"), med_dpos=("d_pos", "median"), max_dt_ps=("d_t", lambda s: 1e3 * np.nanmax(np.abs(s)))).to_string())
