"""Check the cluster-shape direction features on test events 17000-19999 of m200_L774_mu0 v1.

1. Resolution per volume group: line angle between the true direction (first sim hit's tpx, tpy, tpz) and the closer
   of the two candidate axes; |phi_angle| / |eta_angle| residuals; diagonal-sign agreement.
2. Quirk switch problem: pairs of hits from the two different quirks of an event within 10 mm of each other.
   - true and estimated angle between the two quirks' directions there (estimated: smallest line angle over the 2x2
     candidate combinations, i.e. the conservative separation a model could rely on);
   - continuation test: for a quirk-A hit h with a quirk-B hit within 10 mm, is h's direction closer to A's next hit
     (in time) than to that B hit? (closeness = smallest candidate line angle; truth version for reference).
Usage: python check_direction_features.py [first stop]
"""
import sys
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

sys.path.insert(0, "/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking")
DIRF = Path("/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirk_tracking_data/direction/m200_L774_mu0")
PKG = Path("/global/cfs/cdirs/m4958/data/ColliderML/simulation/quirks/m200_L774_mu0/v1/parquet")
QUIRK_PDG = 10000100
first, stop = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (17000, 20000)


def load(path_glob_dir, prefix, cols):
    ts = []
    for b in range(first // 1000, (stop - 1) // 1000 + 1):
        ts.append(pq.read_table(path_glob_dir / f"{prefix}.events{1000 * b}-{1000 * b + 999}.parquet", columns=cols))
    import pyarrow as pa
    t = pa.concat_tables(ts).sort_by("event_id")
    ids = t["event_id"].to_numpy()
    t = t.filter(pc.and_(pc.greater_equal(t["event_id"], first), pc.less(t["event_id"], stop)))
    assert t["event_id"].to_numpy().tolist() == list(range(first, stop))
    return t


F = load(DIRF, "quirks.m200_L774_mu0.v1.direction", None)
H = load(PKG / "reco/tracker_hits", "quirks.m200_L774_mu0.v1.reco.tracker_hits",
         ["event_id", "x", "y", "z", "volume_id", "particle_ids", "simhit_ids"])
S = load(PKG / "truth/tracker_simhits", "quirks.m200_L774_mu0.v1.truth.tracker_simhits",
         ["event_id", "tpx", "tpy", "tpz", "true_time"])
P = load(PKG / "truth/particles", "quirks.m200_L774_mu0.v1.truth.particles", ["event_id", "particle_id", "pdg_id"])


def flat(t, c):
    return np.asarray(pc.list_flatten(t[c]).to_numpy(zero_copy_only=False))


nh = np.asarray(pc.list_value_length(H["x"]))
assert np.array_equal(nh, np.asarray(pc.list_value_length(F["phi_angle_abs"])))
evi = np.repeat(np.arange(len(nh)), nh)
ns = np.asarray(pc.list_value_length(S["tpx"]))
soff = np.concatenate([[0], np.cumsum(ns)[:-1]])
sid = np.asarray(pc.list_element(pc.list_flatten(H["simhit_ids"]), 0)).astype(np.int64) + soff[evi]
T = np.stack([flat(S, "tpx"), flat(S, "tpy"), flat(S, "tpz")], 1)[sid].astype(float)
T /= np.linalg.norm(T, axis=1, keepdims=True)
tt = flat(S, "true_time")[sid]
pid = np.asarray(pc.list_element(pc.list_flatten(H["particle_ids"]), 0)).astype(np.int64)
X = np.stack([flat(H, "x"), flat(H, "y"), flat(H, "z")], 1).astype(float)
vol = flat(H, "volume_id").astype(int)
A = np.stack([flat(F, f"axis_a_{c}") for c in "xyz"], 1).astype(float)
B = np.stack([flat(F, f"axis_b_{c}") for c in "xyz"], 1).astype(float)
pa_, ea_ = flat(F, "phi_angle_abs").astype(float), flat(F, "eta_angle_abs").astype(float)
ds = flat(F, "diag_sign")
npart = np.asarray(pc.list_value_length(P["pdg_id"]))
pev = np.repeat(np.arange(len(npart)), npart)
qmask = np.abs(flat(P, "pdg_id")) == QUIRK_PDG
qkeys = set(zip(pev[qmask].tolist(), flat(P, "particle_id")[qmask].tolist()))
isq = np.array([(e, p) in qkeys for e, p in zip(evi.tolist(), pid.tolist())])


def lang(u, v):  # line angle (deg), sign-free
    return np.degrees(np.arccos(np.clip(np.abs((u * v).sum(-1)), 0, 1)))


def best(i, v):  # smallest line angle between hit i's candidates and direction v
    return np.minimum(lang(A[i], v), lang(B[i], v))


def pairbest(i, j):
    return np.min(np.stack([lang(A[i], A[j]), lang(A[i], B[j]), lang(B[i], A[j]), lang(B[i], B[j])]), 0)


# frames for local truth components (needed for |angle| residuals and diagonal sign)
FR = np.load("/pscratch/sd/d/danieltm/quirk_stageA/direction/frames.npy")
KEY = {(int(r[0]), int(r[1]), int(r[2])): k for k, r in enumerate(FR)}
Hl = load(PKG / "reco/tracker_hits", "quirks.m200_L774_mu0.v1.reco.tracker_hits", ["event_id", "layer_id", "surface_id"])
fi = np.array([KEY[k] for k in zip(vol, flat(Hl, "layer_id").astype(int), flat(Hl, "surface_id").astype(int))])
tu, tv, tw = (T * FR[fi, 6:9]).sum(1), (T * FR[fi, 9:12]).sum(1), (T * FR[fi, 12:15]).sum(1)
tphi, teta = np.arctan2(np.abs(tu), np.abs(tw)), np.arctan2(np.abs(tv), np.abs(tw))

print(f"events {first}-{stop - 1}: {len(vol)} hits, {isq.sum()} quirk hits")
print("\n1. resolution vs truth (line angle to the closer candidate; robust width = IQR/1.349)")
rw = lambda x: np.subtract(*np.nanquantile(x, [.75, .25])) / 1.349
for gname, vols in (("pixel", [16, 17, 18]), ("short strip", [23, 24, 25]), ("long strip", [28, 29, 30])):
    for lab, m0 in (("quirk", isq), ("other", ~isq)):
        m = np.isin(vol, vols) & m0
        line = f"  {gname:11s} {lab:5s} n={m.sum():8d}"
        if gname != "long strip":
            a = best(np.flatnonzero(m), T[m])
            both = m & (ds != 0) & (np.degrees(tphi) > 5) & (np.degrees(teta) > 5)
            agree = np.mean(ds[both] == np.sign(tu[both] * tv[both] * np.sign(tw[both]) ** 2))
            line += (f"  3D angle median {np.median(a):5.1f} deg, 68% {np.quantile(a, .68):5.1f}"
                     f" | |eta_angle| res {np.degrees(rw(ea_[m] - teta[m])):4.1f} deg"
                     f" | diag sign agrees {agree:.3f} (n={both.sum()})")
        line += f" | |phi_angle| res {np.degrees(rw(pa_[m] - tphi[m])):4.1f} deg"
        print(line)

print("\n2. hits of the two different quirks within 10 mm")
rows = []
starts = np.concatenate([[0], np.cumsum(nh)])
for e in range(len(nh)):
    sl = np.arange(starts[e], starts[e + 1])
    q = sl[isq[sl]]
    ids = np.unique(pid[q])
    if len(ids) != 2:
        continue
    ha, hb = q[pid[q] == ids[0]], q[pid[q] == ids[1]]
    for h1, h2 in ((ha, hb), (hb, ha)):
        o = h1[np.argsort(tt[h1])]
        nxt = {o[k]: o[k + 1] for k in range(len(o) - 1)}
        d = np.linalg.norm(X[h1][:, None] - X[h2][None], axis=2)
        for i, j in zip(*np.nonzero(d < 10)):
            h, other = h1[i], h2[j]
            rows.append((h, other, nxt.get(h, -1), d[i, j]))
rows = np.array(rows)
h, o, n_, dist = rows[:, 0].astype(int), rows[:, 1].astype(int), rows[:, 2].astype(int), rows[:, 3]
ok = np.isin(vol[h], [16, 17, 18, 23, 24, 25]) & np.isin(vol[o], [16, 17, 18, 23, 24, 25])
print(f"  pairs (ordered A->B) within 10 mm: {len(h)}; both hits in pixel/short strip: {ok.sum()}")
h, o, n_, dist = h[ok], o[ok], n_[ok], dist[ok]
true_ab = lang(T[h], T[o]); est_ab = pairbest(h, o)
print(f"  true angle between the quirks: median {np.median(true_ab):.1f} deg, frac > 10 deg {np.mean(true_ab > 10):.2f},"
      f" > 20 deg {np.mean(true_ab > 20):.2f}")
print(f"  estimated (min over candidates): median {np.median(est_ab):.1f} deg, frac > 10 deg {np.mean(est_ab > 10):.2f},"
      f" > 20 deg {np.mean(est_ab > 20):.2f}; corr with true {np.corrcoef(true_ab, est_ab)[0, 1]:.2f}")
for lo, hi in ((0, 10), (10, 20), (20, 40), (40, 90)):
    m = (true_ab >= lo) & (true_ab < hi)
    print(f"    true {lo:2d}-{hi:2d} deg: n={m.sum():6d}, estimated median {np.median(est_ab[m]) if m.any() else np.nan:5.1f}")
has = (n_ >= 0) & np.isin(vol[np.maximum(n_, 0)], [16, 17, 18, 23, 24, 25])
hh, oo, nn = h[has], o[has], n_[has]
# continuation test with the direction of the step towards each candidate hit
step_same = X[nn] - X[hh]; step_other = X[oo] - X[hh]
step_same /= np.linalg.norm(step_same, axis=1, keepdims=True); step_other /= np.linalg.norm(step_other, axis=1, keepdims=True)
est_right = best(hh, step_same) < best(hh, step_other)
true_right = lang(T[hh], step_same) < lang(T[hh], step_other)
print(f"  continuation test (h -> next hit of its own quirk vs h -> the other quirk's hit, n={len(hh)}):")
print(f"    chooses own quirk: cluster-shape axis {est_right.mean():.3f}, true direction {true_right.mean():.3f}")
# hit-to-hit axis agreement (both hits' own estimates)
print(f"    pair-axis agreement: min angle own-next {np.median(pairbest(hh, nn)):.1f} deg vs own-other "
      f"{np.median(pairbest(hh, oo)):.1f} deg (medians); own closer in {np.mean(pairbest(hh, nn) < pairbest(hh, oo)):.3f}")
