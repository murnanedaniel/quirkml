"""mu60 background proxy: copy blocks of an R1 mu 60 set and its direction features with every quirk hit and quirk
particle removed (pileup only, mu ~ 60; the quirk set has no other hard scatter). Hits of all other particles are kept in
their original order within each event, so the direction rows stay aligned. Outputs are sorted by event_id.
  python noquirk60.py SRC_SET DST_SET BLOCK...
Writes quirk_tracking_data/parquet_r1_mu60/DST_SET/{reco/tracker_hits,truth/particles} and direction/DST_SET.
"""
import os
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "gnn"))
from common import QUIRK_PDG  # noqa: E402

Q = "/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirk_tracking_data"
src, dst, blocks = sys.argv[1], sys.argv[2], [int(b) for b in sys.argv[3:]]


def flat(col):
    c = col.combine_chunks()
    return c.flatten().to_numpy(zero_copy_only=False), np.asarray(c.offsets)


def filter_lists(tab, mask):
    """mask: flat boolean over the list elements (same lengths for every list column of tab)."""
    cols = {}
    off = None
    for name in tab.column_names:
        col = tab.column(name)
        if pa.types.is_list(col.type):
            c = col.combine_chunks()
            o = np.asarray(c.offsets)
            if off is None:
                kept = np.concatenate([[0], np.cumsum(mask)])
                off = kept[o].astype(np.int32)
            cols[name] = pa.ListArray.from_arrays(pa.array(off, pa.int32()), c.flatten().filter(pa.array(mask)))
        else:
            cols[name] = col.combine_chunks()
    return pa.table(cols).cast(tab.schema)


for b in blocks:
    tag = f"events{b * 1000}-{b * 1000 + 999}"
    fh = f"{Q}/parquet_r1_mu60/{src}/reco/tracker_hits/quirks.{src}.v1.reco.tracker_hits.{tag}.parquet"
    fp = f"{Q}/parquet_r1_mu60/{src}/truth/particles/quirks.{src}.v1.truth.particles.{tag}.parquet"
    fd = f"{Q}/direction/{src}/quirks.{src}.v1.direction.{tag}.parquet"
    # the three files need not share an event order (direction files are sorted by event_id): sort all three
    hits, parts, dirs = (pq.read_table(f).sort_by("event_id") for f in (fh, fp, fd))
    assert hits["event_id"].to_pylist() == parts["event_id"].to_pylist() == dirs["event_id"].to_pylist()
    pp, po = flat(parts["particle_id"])
    pg, _ = flat(parts["pdg_id"])
    ev_p = np.repeat(np.arange(len(po) - 1), np.diff(po))
    isq = np.abs(pg) == QUIRK_PDG
    qkey = set(zip(ev_p[isq].tolist(), pp[isq].tolist()))
    hp, ho = flat(hits["particle_id"])
    ev_h = np.repeat(np.arange(len(ho) - 1), np.diff(ho))
    qpid = np.unique(pp[isq])
    cand = np.isin(hp, qpid)  # quirk particle ids of any event; exact (event, id) check on the candidates only
    idx = np.nonzero(cand)[0]
    drop = np.zeros(len(hp), bool)
    drop[idx] = [(e, p) in qkey for e, p in zip(ev_h[idx].tolist(), hp[idx].tolist())]
    keep_h, keep_p = ~drop, ~isq
    n0, n1 = len(hp), int(keep_h.sum())
    dlen = np.diff(np.asarray(dirs["axis_a_x"].combine_chunks().offsets))
    assert (dlen == np.diff(ho)).all(), "direction rows not aligned with hits"
    for d, t, keep in ((f"{Q}/parquet_r1_mu60/{dst}/reco/tracker_hits", hits, keep_h),
                       (f"{Q}/parquet_r1_mu60/{dst}/truth/particles", parts, keep_p),
                       (f"{Q}/direction/{dst}", dirs, keep_h)):
        os.makedirs(d, exist_ok=True)
        kind = os.path.basename(fh).split(".v1.")[1].rsplit(".events", 1)[0] if "tracker_hits" in d else \
            ("truth.particles" if "particles" in d else "direction")
        out = f"{d}/quirks.{dst}.v1.{kind}.{tag}.parquet"
        pq.write_table(filter_lists(t, keep), out + ".tmp")
        os.replace(out + ".tmp", out)
    print(f"block {b}: hits {n0} -> {n1} (quirk hits removed {n0 - n1}), quirk particles removed "
          f"{int(isq.sum())}", flush=True)
