import numpy as np, pandas as pd
for g in ("empty", "odd"):
    c = pd.read_csv(f"chk_{g}_L1000.csv"); c = c[c.pdg == 4900101]
    dx = np.sqrt((c.ft_x-c.st_x)**2 + (c.ft_y-c.st_y)**2 + (c.ft_z-c.st_z)**2)
    dE = c.ft_E_MeV - c.st_E_MeV
    print(f"== {g}: steps {len(c)}, t_max {c.t_ns.max():.1f} ns, geomLimited {c.geomLimited.sum()}, forceStep {c.forceStep.sum()}")
    for lab, m in (("geomLimited", c.geomLimited == 1), ("forceStep", c.forceStep == 1), ("normal", (c.geomLimited == 0) & (c.forceStep == 0))):
        if m.sum():
            print(f"  {lab:12s} n={m.sum():6d}  |dx| max {dx[m].max():.3e} mm median {dx[m].median():.2e}   dE sum {dE[m].sum():+.3f} MeV  |dE| max {dE[m].abs().max():.3e}")
