"""Read ColliderML-format parquet (package_parquet output) into per-event tables for quirk studies.

Per event:
  hits      one row per reco tracker hit: x, y, z, volume_id, layer_id, surface_id, the truth
            particle_id (first contributing particle, 0 if none) and the true time t (ns) of
            its sim hit. ColliderML's geometric digitisation writes no reco time, so t is truth.
  particles one row per truth particle, with is_quirk.
  truth_edges for each quirk, its hits ordered by true time; consecutive pairs are the truth
            edges. Ordering by radius (as for helices) fails for quirks, whose tracks turn back
            inwards; time order follows the trajectory.

    from quirk_loader import QuirkDataset
    ds = QuirkDataset("<base>/<campaign>/<dataset>/<version>/parquet")   # packaged layout
    ds = QuirkDataset("<base>/<campaign>/<dataset>/<version>/runs/<n>")  # per-run layout
    ev = ds.event(0)          # dict of DataFrames: hits, particles, truth_edges
"""
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

C_MM_PER_NS = 299.792458
QUIRK_PDG = 10000100  # quirk-gen; the pilot generator in gen/ uses 4900101


def _read_object(root, group, obj):
    # group is "truth"/"reco" for the packaged layout, "" for a per-run directory
    files = sorted(Path(root, group, obj).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet files in {Path(root, group, obj)}")
    return pd.concat([pq.read_table(f).to_pandas() for f in files], ignore_index=True).set_index("event_id")


def _explode(row, exclude=()):
    cols = {k: v for k, v in row.items() if k not in exclude}
    return pd.DataFrame({k: np.asarray(v) for k, v in cols.items()})


class QuirkDataset:
    def __init__(self, parquet_root, quirk_pdg=QUIRK_PDG):
        self.root = Path(parquet_root)
        self.quirk_pdg = quirk_pdg
        # packaged: <root>/{truth,reco}/<object>; per-run: <root>/<object>
        if (self.root / "reco").is_dir():
            truth, reco = "truth", "reco"
        elif (self.root / "tracker_hits").is_dir():
            truth = reco = ""
        else:
            raise FileNotFoundError(f"{self.root} has neither reco/ (packaged) nor tracker_hits/ (per-run)")
        self.particles = _read_object(self.root, truth, "particles")
        self.simhits = _read_object(self.root, truth, "tracker_simhits")
        self.hits = _read_object(self.root, reco, "tracker_hits")
        if not truth:
            # per-run files store times as c*t in mm; the packaged layout stores ns
            for df, col in ((self.particles, "time"), (self.simhits, "true_time")):
                df[col] = [np.asarray(v, dtype=np.float32) / np.float32(C_MM_PER_NS) for v in df[col]]

    @property
    def event_ids(self):
        return sorted(set(self.hits.index) & set(self.particles.index))

    def event(self, event_id):
        parts = _explode(self.particles.loc[event_id])
        parts["is_quirk"] = np.abs(parts.pdg_id) == self.quirk_pdg
        sim = _explode(self.simhits.loc[event_id])
        h = self.hits.loc[event_id]
        hits = _explode(h, exclude=("particle_ids", "simhit_ids"))
        pid = np.array([int(p[0]) if len(p) else 0 for p in h["particle_ids"]], dtype=np.int64)
        sid = np.array([int(s[0]) if len(s) else -1 for s in h["simhit_ids"]], dtype=np.int64)
        hits["particle_id"] = pid
        hits["simhit_id"] = sid
        # true time of the hit = time of its (first) sim hit; simhit_ids index the event's simhits
        t = np.full(len(hits), np.nan)
        ok = (sid >= 0) & (sid < len(sim))
        t[ok] = sim.true_time.values[sid[ok]]
        hits["t"] = t
        hits["is_quirk"] = np.isin(pid, parts.particle_id[parts.is_quirk].values)
        return dict(hits=hits, particles=parts, truth_edges=self.quirk_truth_edges(hits, parts))

    @staticmethod
    def quirk_truth_edges(hits, parts):
        """(hit_index_a, hit_index_b, particle_id) for consecutive hits of each quirk in true time."""
        rows = []
        for qid in parts.particle_id[parts.is_quirk].values:
            idx = np.nonzero(hits.particle_id.values == qid)[0]
            idx = idx[np.argsort(hits.t.values[idx], kind="stable")]
            rows += [(a, b, qid) for a, b in zip(idx[:-1], idx[1:])]
        return pd.DataFrame(rows, columns=["hit_a", "hit_b", "particle_id"])
