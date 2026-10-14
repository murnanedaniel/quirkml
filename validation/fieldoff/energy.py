import numpy as np, pandas as pd
for tag in ("L1000", "L500"):
    for geo in ("empty", "odd"):
        G = pd.read_csv(f"g4_{geo}_{tag}_mb1e-2_steps.csv")
        a = G[G.pdg == 4900101].sort_values("t_ns")
        out = []
        for lo in (0, 5, 10, 20, 30):
            w = a[(a.t_ns >= lo) & (a.t_ns < lo + 2)]
            if len(w): out.append(f"t{lo}-{lo+2}: Emax {w.E_MeV.max()/1e3:.4f} GeV")
        print(tag, geo, " | ".join(out), "| pdgs", sorted(G.pdg.unique()))
