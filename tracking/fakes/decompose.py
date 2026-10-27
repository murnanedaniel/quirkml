"""B1: decompose the strict (> 50%) fakes of built tracks into categories (common.FAKE_CATEGORIES) and give the
fake rate vs number of hits, eta and radius of the first (innermost) hit. Checks the fake rate and technical_sm
efficiency against the ACTS strict rerun of the same predictions.

    python decompose.py OUT_DIR
"""
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

import common as C

Q = "/pscratch/sd/d/danieltm/quirk_tracking"
A = Q + "/strict50/acts"
MODELS = {  # name: (predictions, ACTS strict rerun)
    "gnn_v3_s0_walk": (Q + "/gnn/main/g2_q50_dir_s0/test/pred_stitch0.7_mh4", A + "/g2_s0"),
    "pair_gnn_s0_cc": (Q + "/gnn/main/e2_pair_dir_s0/test/pred_cc0.5_mh4", A + "/e2_s0_cc"),
    "mf_long_ot0.1": (Q + "/maskformer/predictions/t2e_pairdir_long_e049_test_ot0.1", A + "/mf_long_ot0.1"),
    "mf_long_ot0.2": (Q + "/maskformer/predictions/t2e_pairdir_long_e049_test_ot0.2", A + "/mf_long_ot0.2"),
}
FIRST, STOP = 17000, 20000
BINS = {"n_hits": [1, 4, 5, 6, 7, 8, 10, 12, 15, 20, 30, 1000], "eta_first": [-4, -3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3, 4],
        "r_min": [0, 40, 80, 120, 200, 300, 400, 600, 800, 1100]}


def binned(t, col, edges):
    x = t[col].to_numpy()
    fake = (t["cls"] == "fake").to_numpy()
    idx = np.digitize(x, edges) - 1
    out = []
    for i in range(len(edges) - 1):
        s = idx == i
        out.append({"lo": edges[i], "hi": edges[i + 1], "tracks": int(s.sum()), "fakes": int(fake[s].sum()),
                    "fake_rate": float(fake[s].mean()) if s.any() else None})
    return out


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    hits, parts = C.load_truth(C.SRC, FIRST, STOP)
    res = {}
    for name, (pdir, adir) in MODELS.items():
        t = C.fake_category(C.track_table(hits, parts, C.read_preds(pdir, FIRST, STOP)))
        t.write_parquet(out / f"tracks_{name}.parquet")
        s = json.loads((Path(adir) / "summary.json").read_text())
        acts_cls = s["track_classification_all_particles"]["very_loose_strict"]
        n = t.height
        nf = int((t["cls"] == "fake").sum())
        tsm = parts.filter(pl.col("technical_sm")).select("event_id", "particle_id")
        mt = t.filter(pl.col("matched")).select("event_id", pl.col("maj_pid").alias("particle_id")).unique()
        tsm_eff = tsm.join(mt, on=["event_id", "particle_id"], how="semi").height / tsm.height
        r = {"tracks": n, "tracks_per_event": n / (STOP - FIRST), "fakes": nf, "fake_rate": nf / n,
             "acts_fake_rate": acts_cls["fake_rate"], "acts_fakes": acts_cls["fake"], "acts_tracks": acts_cls["tracks"],
             "technical_sm_eff": tsm_eff, "acts_technical_sm_eff": s["technical_sm_efficiency"]["very_loose_strict"]["eff"],
             "cls": dict(t["cls"].value_counts().iter_rows())}
        f = t.filter(pl.col("cls") == "fake")
        cats = {}
        for c in "abcdef":
            fc = f.filter(pl.col("fake_cat") == c)
            cats[c] = {"label": C.FAKE_CATEGORIES[c], "fakes": fc.height, "share": fc.height / max(nf, 1),
                       "per_event": fc.height / (STOP - FIRST), "rate_contrib": fc.height / n,
                       "short_le5": int((fc["n_hits"] <= 5).sum()),
                       "median_hits": float(fc["n_hits"].median()) if fc.height else None,
                       "maj_technical": int(fc["maj_technical"].fill_null(False).sum()),
                       "maj_secondary": int((~fc["maj_primary"].fill_null(True)).sum()),
                       "maj_pt_lt_1": int((fc["maj_pt"] < 1).sum()),
                       "with_quirk_hits": int((fc["n_quirk_hits"] > 0).sum())}
        r["categories"] = cats
        r["short_le5"] = {"tracks": int((t["n_hits"] <= 5).sum()), "fakes": int((f["n_hits"] <= 5).sum())}
        r["vs"] = {k: binned(t, k, e) for k, e in BINS.items()}
        res[name] = r
        print(name, f"fake {r['fake_rate']:.4f} (ACTS {r['acts_fake_rate']:.4f}, {nf} vs {acts_cls['fake']}), "
              f"tracks {n} vs {acts_cls['tracks']}, tech_sm {tsm_eff:.4f} (ACTS {r['acts_technical_sm_eff']:.4f})",
              {c: (v["fakes"], round(v["share"], 3)) for c, v in cats.items()}, flush=True)
    (out / "decompose.json").write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
