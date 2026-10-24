"""Quirk efficiency under the explicit quirk truth rule (Python scoring next to the ACTS evaluator).

The ACTS reader takes one truth particle per hit (Release 1 particle_id = the first entry of the packaged hit's
particle_ids). The rule uses the packaged list instead:
  - a hit belongs to quirk q if q is anywhere in its particle_ids; a hit with both quirks belongs to both;
  - a hit whose particles are all quirk descendants (delta rays and other secondaries of a quirk, found through
    particles.parent_id) is "quirk-secondary": it is in no particle's truth and does not count in a track's purity;
  - every other hit belongs to its first particle that is neither a quirk nor a quirk descendant.

Matching is TrackTruthMatcher's double matching with the ACTS majority step, extended to shared hits: a track T
matches quirk q if q's hit count on T is the largest among T's particles (ties broken as ACTS does: the particle
whose last hit on T comes latest), purity (q's hits on T / T's counted hits) >= r and completeness (q's hits on T /
q's hits) >= r. The other quirk does not compete with q on a track where the two share a hit; when the two quirks
share no hit on T this is exactly the ACTS rule. With Release 1 labels ("r1" mode) the function reproduces ACTS.

Coupled efficiency: the pair (the union of both quirks' hits) is one particle; the pair is found if one track
matches it. Selections: a quirk (or the pair) with at least 3 hits under the labels used.

    python tracking/quirk_rule.py --predictions-dir P --dataset R1_ROOT --packaged PACKAGED_ROOT --events A B \
        --acts-eval EVAL_DIR [--acts-pair-eval PAIR_EVAL_DIR] --out quirk_rule.json

EVAL_DIR / PAIR_EVAL_DIR are evaluate.py outputs of the same predictions on the dataset and on its pair-truth copy
(make_pair_truth.py); their per_event.parquet gives the ACTS counts for the r1-mode checks. evaluate.py
--packaged-dataset runs the same scoring and puts it in summary.json["quirk_rule"].
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r1_data import QUIRK_PDG, read_events  # noqa: E402

WORKING_POINTS = {"very_loose": (0.5, True), "tight": (0.9, True)}
PAIR_LABEL = -2  # label of the pair object in the coupled scoring
MIN_HITS = 3
_KEY = 1 << 40  # (count, last row) packed as count * _KEY + last row


# Section: inputs


def read_r1_hits(r1_root, first: int, stop: int) -> pl.DataFrame:
    """(event_id, hit_id, particle_id) of a Release 1 (or staged) dataset."""
    tab = read_events(r1_root, "tracker_hits", first, stop, columns=["event_id", "particle_id"])
    lists = tab["particle_id"].to_pylist()
    n = [len(v) for v in lists]
    return pl.DataFrame({"event_id": np.repeat(np.arange(first, stop), n),
                         "hit_id": np.concatenate([np.arange(k) for k in n]),
                         "particle_id": np.concatenate([np.asarray(v, dtype=np.int64) for v in lists])})


def read_packaged(packaged, first: int, stop: int) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Packaged hits exploded to (event_id, hit_id, pos, particle_id) and particles (event_id, particle_id, pdg_id,
    parent_id, charge, primary)."""
    hits = pl.from_arrow(read_events(packaged, "tracker_hits", first, stop, columns=["event_id", "particle_ids"]))
    hits = (hits.with_columns(pl.col("event_id").cast(pl.Int64)).explode("particle_ids")
            .with_columns(hit_id=pl.int_range(pl.len()).over("event_id"))
            .explode("particle_ids").rename({"particle_ids": "particle_id"})
            .with_columns(pl.col("particle_id").cast(pl.Int64),
                          pos=pl.int_range(pl.len()).over("event_id", "hit_id")))
    parts = pl.from_arrow(read_events(packaged, "particles", first, stop,
                                      columns=["event_id", "particle_id", "pdg_id", "parent_id", "charge", "primary"]))
    parts = parts.with_columns(pl.col("event_id").cast(pl.Int64)).explode(
        ["particle_id", "pdg_id", "parent_id", "charge", "primary"]).with_columns(
        pl.col("particle_id").cast(pl.Int64), pl.col("parent_id").cast(pl.Int64))
    return hits, parts


def quirk_descendants(parts: pl.DataFrame) -> tuple[pl.DataFrame, int]:
    """(event_id, particle_id, is_quirk, quirk_descendant) and the number of parent chains that leave the table."""
    parts = parts.with_columns(is_quirk=pl.col("pdg_id").abs() == QUIRK_PDG)
    parent = parts.select("event_id", "particle_id", "parent_id", "is_quirk")
    cur = parts.select("event_id", "particle_id", pl.col("parent_id").alias("cur")).with_columns(
        desc=pl.lit(False))
    broken = 0
    for _ in range(500):
        active = cur.filter(pl.col("cur") >= 0)
        if active.is_empty():
            break
        step = active.join(parent.rename({"particle_id": "cur", "parent_id": "next", "is_quirk": "cur_quirk"}),
                           on=["event_id", "cur"], how="left")
        broken += int(step["cur_quirk"].null_count())
        step = step.with_columns(desc=pl.col("desc") | pl.col("cur_quirk").fill_null(False),
                                 cur=pl.when(pl.col("cur_quirk").fill_null(True)).then(-1).otherwise(pl.col("next")))
        cur = pl.concat([cur.filter(pl.col("cur") < 0), step.select("event_id", "particle_id", "cur", "desc")])
    else:
        raise RuntimeError("parent chains longer than 500 steps (a loop?)")
    out = parts.select("event_id", "particle_id", "is_quirk").join(
        cur.select("event_id", "particle_id", pl.col("desc").alias("quirk_descendant")), on=["event_id", "particle_id"])
    return out, broken


def rule_labels(hits: pl.DataFrame, kinds: pl.DataFrame) -> tuple[pl.DataFrame, dict]:
    """(event_id, hit_id, label) under the rule (several rows for a hit of both quirks; none for quirk-secondary
    hits) and label statistics."""
    h = hits.join(kinds, on=["event_id", "particle_id"], how="left").with_columns(
        pl.col("is_quirk").fill_null(False), pl.col("quirk_descendant").fill_null(False))
    per_hit = h.group_by("event_id", "hit_id").agg(
        n_quirks=pl.col("is_quirk").sum(), first_is_quirk=pl.col("is_quirk").sort_by("pos").first(),
        first_pid=pl.col("particle_id").sort_by("pos").first(),
        all_desc=(pl.col("quirk_descendant") & ~pl.col("is_quirk")).all())
    quirk_rows = h.filter(pl.col("is_quirk")).select("event_id", "hit_id", pl.col("particle_id").alias("label"))
    other = (h.join(per_hit.filter(pl.col("n_quirks") == 0).select("event_id", "hit_id"), on=["event_id", "hit_id"])
             .filter(~pl.col("quirk_descendant")).sort("pos").group_by("event_id", "hit_id")
             .agg(pl.col("particle_id").first().alias("label")))
    labels = pl.concat([quirk_rows.unique(), other]).sort("event_id", "hit_id")
    stats = {
        "hits": per_hit.height,
        "quirk_hits_first_particle": int(per_hit["first_is_quirk"].sum()),
        "quirk_hits_rule": int((per_hit["n_quirks"] > 0).sum()),
        "hits_of_both_quirks": int((per_hit["n_quirks"] > 1).sum()),
        "quirk_secondary_hits": int(((per_hit["n_quirks"] == 0) & per_hit["all_desc"]).sum()),
    }
    return labels, stats


# Section: matching


def match_labels(labels: pl.DataFrame, preds: pl.DataFrame, targets: pl.DataFrame, ratio: float,
                 double: bool) -> pl.DataFrame:
    """(event_id, label, matched_tracks) for every target (event_id, label) that is matched by at least one track.

    labels: (event_id, hit_id, label), several rows per hit allowed, hits without rows not counted. preds: rows in
    prototrack order (event_id, hit_id, predicted_track_id). See the module docstring for the rule."""
    targets = targets.select("event_id", "label").unique().with_columns(target=pl.lit(True))
    label_hits = labels.group_by("event_id", "label").agg(pl.len().alias("label_hits"))
    a = (preds.with_row_index("row").filter(pl.col("predicted_track_id") >= 0)
         .join(labels, on=["event_id", "hit_id"], how="inner"))
    keys = ["event_id", "predicted_track_id"]
    track_hits = a.group_by(keys).agg(pl.col("row").n_unique().alias("track_hits"))
    counts = (a.group_by(*keys, "label").agg(pl.len().alias("n"), pl.col("row").max().alias("last"))
              .join(targets, on=["event_id", "label"], how="left").with_columns(pl.col("target").fill_null(False))
              .with_columns(key=pl.col("n") * _KEY + pl.col("last")))
    share = (a.join(targets, on=["event_id", "label"], how="inner").group_by(*keys, "row").len()
             .group_by(keys).agg((pl.col("len") > 1).any().alias("share")))
    best_other = counts.filter(~pl.col("target")).group_by(keys).agg(pl.col("key").max().alias("best_other"))
    tgt = counts.filter(pl.col("target"))
    rival = (tgt.join(tgt.select(*keys, pl.col("label").alias("label_r"), pl.col("key").alias("key_r")), on=keys)
             .filter(pl.col("label") != pl.col("label_r")).group_by(*keys, "label")
             .agg(pl.col("key_r").max().alias("best_target")))
    tgt = (tgt.join(best_other, on=keys, how="left").join(rival, on=[*keys, "label"], how="left")
           .join(share, on=keys, how="left").join(track_hits, on=keys).join(label_hits, on=["event_id", "label"])
           .with_columns(pl.col("share").fill_null(False))
           .with_columns(competitor=pl.max_horizontal(
               pl.col("best_other").fill_null(-1),
               pl.when(pl.col("share")).then(-1).otherwise(pl.col("best_target").fill_null(-1))))
           .with_columns(matched=(pl.col("key") > pl.col("competitor"))
                         & (pl.col("n") / pl.col("track_hits") >= ratio)
                         & (~pl.lit(double) | (pl.col("n") / pl.col("label_hits") >= ratio))))
    return (tgt.filter(pl.col("matched")).group_by("event_id", "label")
            .agg(pl.len().alias("matched_tracks")))


def _ratio(accepted, total):
    if not total:
        return {"eff": None, "err": None, "accepted": accepted, "total": total}
    eff = accepted / total
    return {"eff": eff, "err": (eff * (1 - eff) / total) ** 0.5, "accepted": accepted, "total": total}


def score(labels: pl.DataFrame, quirk_ids: pl.DataFrame, preds: pl.DataFrame, first: int, stop: int) -> tuple[dict, pl.DataFrame]:
    """Per-quirk, quirk-event and coupled efficiency for one labelling. quirk_ids: (event_id, label) of the quirks.
    Returns (numbers, per-event frame)."""
    counts = labels.join(quirk_ids, on=["event_id", "label"]).group_by("event_id", "label").agg(pl.len().alias("nh"))
    quirks_sel = counts.filter(pl.col("nh") >= MIN_HITS).select("event_id", "label")
    pair_labels = (labels.join(quirk_ids.with_columns(q=pl.lit(True)), on=["event_id", "label"], how="left")
                   .with_columns(label=pl.when(pl.col("q").fill_null(False)).then(PAIR_LABEL).otherwise(pl.col("label")))
                   .select("event_id", "hit_id", "label").unique())
    pair_sel = (pair_labels.filter(pl.col("label") == PAIR_LABEL).group_by("event_id").len()
                .filter(pl.col("len") >= MIN_HITS).select("event_id", label=pl.lit(PAIR_LABEL, dtype=pl.Int64)))
    ev = pl.DataFrame({"event_id": np.arange(first, stop, dtype=np.int64)})
    sel_per_event = quirks_sel.group_by("event_id").len().rename({"len": "quirks_selected"})
    ev = ev.join(sel_per_event, on="event_id", how="left").with_columns(pl.col("quirks_selected").fill_null(0))
    ev = ev.join(pair_sel.select("event_id", pair_selected=pl.lit(True)), on="event_id", how="left").with_columns(
        pl.col("pair_selected").fill_null(False))
    out = {"quirks_selected": quirks_sel.height, "pairs_selected": pair_sel.height,
           "events_with_two_quirks_selected": int((ev["quirks_selected"] == 2).sum())}
    for point, (ratio, double) in WORKING_POINTS.items():
        m = match_labels(labels, preds, quirks_sel, ratio, double)
        per = m.group_by("event_id").len().rename({"len": f"quirks_matched_{point}"})
        mp = match_labels(pair_labels, preds, pair_sel, ratio, double)
        ev = (ev.join(per, on="event_id", how="left").with_columns(pl.col(f"quirks_matched_{point}").fill_null(0))
              .join(mp.select("event_id", pl.lit(True).alias(f"pair_matched_{point}")), on="event_id", how="left")
              .with_columns(pl.col(f"pair_matched_{point}").fill_null(False)))
        two = ev.filter(pl.col("quirks_selected") == 2)
        out[point] = {
            "per_quirk_efficiency": _ratio(int(ev[f"quirks_matched_{point}"].sum()), quirks_sel.height),
            "quirk_event_efficiency": _ratio(int((two[f"quirks_matched_{point}"] == 2).sum()), two.height),
            "coupled_efficiency": _ratio(int(ev[f"pair_matched_{point}"].sum()), pair_sel.height),
        }
    return out, ev


def quirk_rule_block(preds: pl.DataFrame, r1_root, packaged, first: int, stop: int,
                     acts_per_event: pl.DataFrame | None = None, acts_pair_per_event: pl.DataFrame | None = None) -> dict:
    """The summary.json "quirk_rule" block. acts_per_event / acts_pair_per_event: evaluate.py per_event frames
    (event_id, quirks_matched_<point>) of the same predictions on the dataset and on its pair-truth copy."""
    r1 = read_r1_hits(r1_root, first, stop)
    hits, parts = read_packaged(packaged, first, stop)
    kinds, broken = quirk_descendants(parts)
    first_pid = hits.filter(pl.col("pos") == 0).select("event_id", "hit_id", pl.col("particle_id").alias("pk"))
    aligned = r1.join(first_pid, on=["event_id", "hit_id"], how="full", coalesce=True)
    checks = {"packaged_hits_equal_r1_hits": aligned.height == r1.height == first_pid.height,
              "packaged_first_particle_equals_r1_particle_id": bool((aligned["particle_id"] == aligned["pk"]).all()),
              "parent_chains_leaving_the_table": broken}
    quirk_ids = kinds.filter(pl.col("is_quirk")).select("event_id", pl.col("particle_id").alias("label"))
    qinfo = parts.filter(pl.col("pdg_id").abs() == QUIRK_PDG)
    checks["quirks_charged_primary"] = bool((qinfo["charge"] != 0).all() and qinfo["primary"].all())

    labels_rule, label_stats = rule_labels(hits, kinds)
    labels_r1 = r1.rename({"particle_id": "label"})
    rule, ev_rule = score(labels_rule, quirk_ids, preds, first, stop)
    r1_mode, ev_r1 = score(labels_r1, quirk_ids, preds, first, stop)

    # Events whose rule labels are the Release 1 labels exactly.
    diff = pl.concat([labels_rule.join(labels_r1, on=["event_id", "hit_id", "label"], how="anti"),
                      labels_r1.join(labels_rule, on=["event_id", "hit_id", "label"], how="anti")])
    differing = set(diff["event_id"].unique().to_list())
    same = [e for e in range(first, stop) if e not in differing]
    checks["events_with_identical_labels"] = len(same)
    for point in WORKING_POINTS:
        col = f"quirks_matched_{point}"
        if acts_per_event is not None:
            acts = acts_per_event.select("event_id", pl.col(col).alias("acts")).with_columns(pl.col("event_id").cast(pl.Int64))
            j = ev_r1.select("event_id", col).join(acts, on="event_id")
            checks[f"{point}_r1_mode_equals_acts_per_event"] = bool((j[col] == j["acts"]).all()) and j.height == stop - first
            jr = ev_rule.select("event_id", col).join(acts, on="event_id").filter(pl.col("event_id").is_in(same))
            checks[f"{point}_rule_equals_acts_on_identical_label_events"] = bool((jr[col] == jr["acts"]).all())
        if acts_pair_per_event is not None:
            acts = acts_pair_per_event.select("event_id", (pl.col(col) > 0).alias("acts")).with_columns(
                pl.col("event_id").cast(pl.Int64))
            j = ev_r1.select("event_id", f"pair_matched_{point}").join(acts, on="event_id")
            checks[f"{point}_r1_mode_coupled_equals_acts_pair_truth_per_event"] = bool(
                (j[f"pair_matched_{point}"] == j["acts"]).all()) and j.height == stop - first
    return {"definition": __doc__.split("\n\n")[1], "label_stats": label_stats, "rule": rule,
            "release1_labels": r1_mode, "checks": checks}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions-dir", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path, help="Release 1 dataset root")
    parser.add_argument("--packaged", required=True, type=Path, help="packaged dataset root (particle_ids lists)")
    parser.add_argument("--events", required=True, type=int, nargs=2, metavar=("A", "B"))
    parser.add_argument("--acts-eval", type=Path, help="evaluate.py output dir of these predictions on --dataset")
    parser.add_argument("--acts-pair-eval", type=Path, help="evaluate.py output dir on the pair-truth copy")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    first, stop = args.events
    preds = (pl.scan_parquet(args.predictions_dir / "predictions.parquet")
             .filter((pl.col("event_id") >= first) & (pl.col("event_id") < stop))
             .select(pl.col("event_id").cast(pl.Int64), pl.col("hit_id").cast(pl.Int64),
                     pl.col("predicted_track_id").cast(pl.Int64)).collect())
    pe = pl.read_parquet(args.acts_eval / "per_event.parquet") if args.acts_eval else None
    pp = pl.read_parquet(args.acts_pair_eval / "per_event.parquet") if args.acts_pair_eval else None
    block = quirk_rule_block(preds, args.dataset, args.packaged, first, stop, pe, pp)
    block["predictions_dir"] = str(args.predictions_dir.resolve())
    block["events"] = [first, stop]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(block, indent=1) + "\n")
    for mode in ("release1_labels", "rule"):
        b = block[mode]
        print(mode, {p: {k: round(v["eff"], 4) if v["eff"] is not None else None for k, v in b[p].items()}
                     for p in WORKING_POINTS}, "quirks", b["quirks_selected"], "pairs", b["pairs_selected"])
    print("checks", block["checks"])
    print("labels", block["label_stats"])


if __name__ == "__main__":
    main()
