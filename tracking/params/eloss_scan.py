"""Does a crude ionisation energy loss in the toy reduce the toy-vs-Geant4 difference?
The toy loses eloss * t_Si/|cos| * BetheShape(beta gamma) MeV at each crossing of a measured module (kernel.py);
eloss = 0.39 MeV/mm is silicon alone, larger values stand in for passive material. Prints residual quantiles.
    python eloss_scan.py OUT.txt
"""
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, "/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/params")
from data import cached_event  # noqa: E402
from model import EventModel, theta_truth  # noqa: E402

ELOSS = [0, 0.39, 2, 4, 8]


def f(a):
    m, lam, ev = a
    E = cached_event(m, lam, ev)
    if E is None:
        return []
    th = theta_truth(E); out = []
    for el in ELOSS:
        M = EventModel(E, hits="sim", eloss=el, c_m=0.02); L = M.local(th)
        r = np.concatenate([np.hypot(a, b) for a, b, c in L]); t = np.concatenate([d["t"] for d in M.H])
        out.append((lam, el, r, t))
    return out


if __name__ == "__main__":
    jobs = [(m, l, e) for m in (100, 200, 300) for l in (167, 774, 3594) for e in range(30)]
    R = {}
    with Pool(120) as p:
        for o in p.imap_unordered(f, jobs):
            for lam, el, r, t in o:
                R.setdefault((lam, el), []).append(np.c_[r, t])
    with open(sys.argv[1], "w") as fo:
        for (lam, el), v in sorted(R.items()):
            a = np.vstack(v); late = a[a[:, 1] > 5]
            fo.write("Lambda %5d eloss %4.2f MeV/mm | all hits: median %.3f p68 %.3f p90 %.3f mm | t > 5 ns: median %.3f p68 %.3f mm (%d hits)\n"
                     % (lam, el, np.median(a[:, 0]), np.percentile(a[:, 0], 68), np.percentile(a[:, 0], 90), np.median(late[:, 0]), np.percentile(late[:, 0], 68), len(late)))
    print(open(sys.argv[1]).read())
