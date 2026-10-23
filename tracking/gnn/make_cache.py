"""Stage 0: per-event cache (one torch file per 1000-event parquet shard).

Per hit: x, y, z, volume index, layer, cluster shape (size_loc0, size_loc1, n_channels, sum_activation),
particle_id (every hit has one; 0 is a real particle, often a quirk), and truth helpers (hit_quirk, hit_pt, hit_nhits, hit_target). hit i = row i of the
event's per-hit lists (= hit_id in predictions.parquet).

Truth edges ("chain"): for every TARGET particle (charge != 0, >= 3 hits; no pT/eta cut, so quirks are always
kept) its hits are ordered by true sim-hit time and consecutive hits are linked. Quirks bend back, turn inwards
and interleave, so ordering by distance from the production vertex (acorn) is wrong; time is the physical order.
Groups: consecutive (in time) hits of one particle on the same surface within 0.1 ns (split clusters, i.e.
several measurements from one crossing) have no meaningful time order; inside such a group the hits are ordered
greedily by distance from the hit preceding the group, so the chain runs through the group in space. Double-sided
strip modules and layer overlaps are different surfaces, are well separated in time order, and are simply
consecutive chain links. Same-particle pairs that are not chain links (e.g. skipping one hit of an overlap pair)
are given loss weight 0 (masked) in the filter and GNN losses, not counted as fakes for the 'pid purity' column,
and counted as not-true for the strict 'purity' column.

Hit time is used only to define truth; it is never a model input.
"""

import argparse
import glob
import os
import pickle
from multiprocessing import Pool

import numpy as np
import pyarrow.parquet as pq


from common import QUIRK_PDG, VOLUMES

HIT_COLS = ["event_id", "x", "y", "z", "time", "particle_id", "volume_id", "layer_id", "surface_id",
            "size_loc0", "size_loc1", "n_channels", "sum_activation"]
PART_COLS = ["event_id", "particle_id", "pdg_id", "charge", "px", "py"]
VOL_LUT = np.full(256, len(VOLUMES) - 1, np.int64)
for _i, _v in enumerate(VOLUMES):
    VOL_LUT[_v] = _i


def chain_for_particle(idx, t, surf, pos):
    o = idx[np.argsort(t[idx], kind="stable")]
    out = []
    i = 0
    while i < len(o):
        j = i + 1
        while j < len(o) and surf[o[j]] == surf[o[i]] and t[o[j]] - t[o[i]] < 0.1:
            j += 1
        grp = list(o[i:j])
        if len(grp) > 1 and out:
            prev = pos[out[-1]]
            rem = grp
            grp = []
            cur = prev
            while rem:
                d = [np.sum((pos[h] - cur) ** 2) for h in rem]
                k = int(np.argmin(d))
                grp.append(rem[k])
                cur = pos[rem[k]]
                rem.pop(k)
        out.extend(grp)
        i = j
    out = np.asarray(out)
    return np.stack([out[:-1], out[1:]])


def process(args):
    hit_file, part_file, out_file = args
    H = pq.read_table(hit_file, columns=HIT_COLS).to_pydict()
    Pt = pq.read_table(part_file, columns=PART_COLS).to_pydict()
    pidx = {e: i for i, e in enumerate(Pt["event_id"])}
    events = []
    for k, eid in enumerate(H["event_id"]):
        j = pidx[eid]
        ppid = np.asarray(Pt["particle_id"][j], np.int64)
        pdg = np.asarray(Pt["pdg_id"][j])
        charge = np.asarray(Pt["charge"][j], np.float32)
        ppt = np.hypot(np.asarray(Pt["px"][j], np.float32), np.asarray(Pt["py"][j], np.float32))
        row = np.full(int(ppid.max()) + 1, -1, np.int64)
        row[ppid] = np.arange(len(ppid))
        x = np.asarray(H["x"][k], np.float32)
        y = np.asarray(H["y"][k], np.float32)
        z = np.asarray(H["z"][k], np.float32)
        t = np.asarray(H["time"][k], np.float32)
        pid = np.asarray(H["particle_id"][k], np.int64)
        vol = np.asarray(H["volume_id"][k], np.int64)
        lay = np.asarray(H["layer_id"][k], np.int64)
        sid = np.asarray(H["surface_id"][k], np.int64)
        surf = vol * 10 ** 8 + lay * 10 ** 5 + sid
        n = len(x)
        has = pid >= 0  # particle_id 0 is a real particle (often a quirk); every hit has a truth particle
        prow = np.full(n, -1, np.int64)
        prow[has] = row[pid[has]]
        ok = prow >= 0
        hit_quirk = np.zeros(n, bool)
        hit_quirk[ok] = np.abs(pdg[prow[ok]]) == QUIRK_PDG
        hit_pt = np.zeros(n, np.float32)
        hit_pt[ok] = ppt[prow[ok]]
        hit_charge = np.zeros(n, np.float32)
        hit_charge[ok] = charge[prow[ok]]
        u, inv, cnt = np.unique(pid, return_inverse=True, return_counts=True)
        hit_nhits = cnt[inv].astype(np.int32)
        hit_nhits[~has] = 0
        hit_target = ok & (hit_charge != 0) & (hit_nhits >= 3)
        pos = np.stack([x, y, z], 1)
        chains = []
        order = np.argsort(pid, kind="stable")
        bounds = np.flatnonzero(np.diff(pid[order])) + 1
        for grp in np.split(order, bounds):
            if len(grp) >= 3 and hit_target[grp[0]]:
                chains.append(chain_for_particle(grp, t, surf, pos))
        chain = np.concatenate(chains, 1).astype(np.int32) if chains else np.zeros((2, 0), np.int32)
        cl = np.stack([np.asarray(H[c][k], np.float32) for c in ["size_loc0", "size_loc1", "n_channels", "sum_activation"]], 1)
        events.append(dict(event_id=int(eid), x=x, y=y, z=z, vol_idx=VOL_LUT[vol].astype(np.int8),
                           layer=lay.astype(np.int16), cluster=cl, pid=pid, hit_quirk=hit_quirk, hit_pt=hit_pt,
                           hit_nhits=hit_nhits, hit_target=hit_target, chain=chain))
    a, b = min(e["event_id"] for e in events), max(e["event_id"] for e in events)
    path = os.path.join(out_file, f"events{a}-{b}.pkl")
    with open(path, "wb") as fh:
        pickle.dump(events, fh, protocol=5)
    nq = sum(int(e["hit_quirk"].sum()) for e in events)
    return path, len(events), nq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    hits = sorted(glob.glob(os.path.join(a.root, "reco/tracker_hits/*.parquet")))
    jobs = []
    for h in hits:
        tag = h.split(".reco.tracker_hits.")[1]
        p = glob.glob(os.path.join(a.root, "truth/particles/*.truth.particles." + tag))
        assert len(p) == 1, (h, p)
        jobs.append((h, p[0], a.out))
    with Pool(a.workers) as pool:
        for path, n, nq in pool.imap_unordered(process, jobs):
            print(path, n, "events", nq, "quirk hits", flush=True)


if __name__ == "__main__":
    main()
