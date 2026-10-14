"""Quirk energy at each crossing (local maxima). With B = 0, no material: alternates between two exact values."""
import numpy as np, pandas as pd
from scipy.signal import find_peaks
for geo in ("empty", "odd"):
    c = pd.read_csv(f"chk_{geo}_L1000.csv"); c = c[(c.pdg == 4900101) & (c.t_ns < 40)].reset_index(drop=True)
    E = c.ft_E_MeV.values; t = c.t_ns.values
    pk, _ = find_peaks(E, distance=20, prominence=1000)
    tp = t[pk]; Ep = E[pk]
    gl_idx = np.nonzero(c.geomLimited.values == 1)[0]
    # boundary steps between consecutive peaks
    nb = np.array([((gl_idx > pk[i]) & (gl_idx <= pk[i+1])).sum() for i in range(len(pk) - 1)])
    for par in (0, 1):
        e = Ep[par::2]; dd = np.diff(e); nbp = np.array([nb[j:j+2].sum() for j in range(par, len(nb) - 1, 2)])[:len(dd)]
        print(f"== {geo} parity {par}: {len(e)} peaks, first {e[0]:.1f}, last {e[-1]:.1f} MeV; "
              f"|dE| between peaks with boundary steps: mean {np.abs(dd[nbp > 0]).mean() if (nbp > 0).any() else float('nan'):.2f} (n={int((nbp > 0).sum())}), "
              f"without: mean {np.abs(dd[nbp == 0]).mean():.2f} (n={int((nbp == 0).sum())}) MeV")
