"""Evaluate canonical tracking predictions on ColliderML LRT truth."""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from tracking_core.schema import read_predictions

CHUNK_RE = re.compile(r"events(\d+)-(\d+)\.parquet$")
PROMPT_R_MM = 2.0
DISPLACED_R_MM = 10.0  # headline LRT region boundary
R_BINS_MM = np.array([0, .5, 1, 2, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300, np.inf])
PT_BINS_GEV = np.array([.5, 1, 1.5, 2, 3, 5, 10, 20, 50, 100, np.inf])
ABS_ETA_BINS = np.array([0, .5, 1, 1.5, 2, 2.5, 3, 3.5, 4])


@dataclass(frozen=True)
class MatchingConfig:
    majority_fraction: float = 0.5
    min_hits: int = 3

    def __post_init__(self) -> None:
        if not 0 <= self.majority_fraction <= 1:
            raise ValueError("majority_fraction must be in [0, 1]")
        if self.min_hits < 1:
            raise ValueError("min_hits must be at least 1")


@dataclass(frozen=True)
class SelectionConfig:
    pt_min_gev: float
    abs_eta_max: float
    min_distinct_layers: int

    @classmethod
    def fiducial(cls) -> "SelectionConfig":
        return cls(1.0, 2.5, 4)

    @classmethod
    def loose(cls) -> "SelectionConfig":
        return cls(0.5, 4.0, 3)


def _indexed(directory: Path) -> dict[tuple[int, int], Path]:
    result = {}
    for path in directory.glob("*.parquet"):
        match = CHUNK_RE.search(path.name)
        if match:
            token = (int(match.group(1)), int(match.group(2)))
            if token in result:
                raise RuntimeError(f"duplicate event range {token} in {directory}")
            result[token] = path
    return result


def _select_chunks(
    chunks: list[tuple[int, int]],
    n_files: int | None,
    chunk_range: tuple[int, int] | list[int] | None,
) -> list[tuple[int, int]]:
    if chunk_range is not None:
        if len(chunk_range) != 2:
            raise ValueError("chunk_range must contain exactly two indices")
        first, last = chunk_range
        if first > last:
            raise ValueError("chunk_range first index must not exceed last index")
        chunks = [token for token in chunks if first <= token[0] // 1000 <= last]
    elif n_files is None:
        raise ValueError("n_files may be None only when chunk_range is set")
    if n_files is not None:
        if n_files < 1:
            raise ValueError("n_files must be at least 1")
        if len(chunks) < n_files:
            raise RuntimeError(f"requested {n_files} chunks; only {len(chunks)} are paired")
        chunks = chunks[:n_files]
    return chunks


def _row_map(table: object, name: str) -> dict[int, int]:
    ids = table["event_id"].to_numpy(zero_copy_only=False)
    result = {int(value): index for index, value in enumerate(ids)}
    if len(result) != len(ids):
        raise RuntimeError(f"duplicate event_id in {name}")
    return result


def wilson_interval(k: np.ndarray, n: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    k, n = np.broadcast_arrays(np.asarray(k, float), np.asarray(n, float))
    eff = np.full(k.shape, np.nan)
    low = np.full(k.shape, np.nan)
    high = np.full(k.shape, np.nan)
    good = n > 0
    eff[good] = k[good] / n[good]
    denom = 1 + 1 / n[good]
    center = (eff[good] + 1 / (2 * n[good])) / denom
    half = np.sqrt(eff[good] * (1 - eff[good]) / n[good] + 1 / (4 * n[good] ** 2)) / denom
    low[good] = np.clip(center - half, 0, eff[good])
    high[good] = np.clip(center + half, eff[good], 1)
    return eff, low, high


def _scalar(flags: np.ndarray) -> dict[str, object]:
    n, k = len(flags), int(np.count_nonzero(flags))
    eff, low, high = wilson_interval(np.array([k]), np.array([n]))
    return {"numerator": k, "denominator": n, "efficiency": float(eff[0]),
            "wilson_lower": float(low[0]), "wilson_upper": float(high[0])}


def _binned(values: np.ndarray, flags: np.ndarray, edges: np.ndarray) -> dict[str, object]:
    denominator = np.histogram(values, edges)[0]
    numerator = np.histogram(values[flags], edges)[0]
    eff, low, high = wilson_interval(numerator, denominator)
    labels = [f"[{a:g}, {b:g})" if np.isfinite(b) else f"[{a:g}, inf)" for a, b in zip(edges[:-1], edges[1:])]
    return {"bin_edges": edges, "bin_labels": labels, "numerator": numerator,
            "denominator": denominator, "efficiency": eff,
            "wilson_lower": low, "wilson_upper": high}


def _jsonable(value: object) -> object:
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_jsonable(item) for item in list(value)]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    return value


def _plot(result: dict[str, object], xlabel: str, path: Path, log_x: bool = False) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    edges = np.asarray(result["bin_edges"], float).copy()
    if np.isinf(edges[-1]):
        edges[-1] = edges[-2] + edges[-2] - edges[-3]
    x = (edges[:-1] + edges[1:]) / 2
    eff = np.asarray(result["efficiency"], float)
    low = np.asarray(result["wilson_lower"], float)
    high = np.asarray(result["wilson_upper"], float)
    good = np.isfinite(eff)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.errorbar(x[good], eff[good], yerr=[eff[good]-low[good], high[good]-eff[good]], fmt="o-")
    ax.set(xlabel=xlabel, ylabel="Tracking efficiency", ylim=(0, 1.05))
    if log_x:
        ax.set_xscale("log")
    ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def generate_lrt_report(
    predictions_dir: Path | str,
    dataset_root: Path | str,
    n_files: int | None,
    out_dir: Path | str,
    matching: MatchingConfig = MatchingConfig(),
    plots: bool = True,
    chunk_range: tuple[int, int] | list[int] | None = None,
    allow_partial_hits: bool = False,
) -> dict[str, object]:
    """Score predictions against hit-derived truth and write ``summary.json``."""
    predictions, meta = read_predictions(predictions_dir)
    required = {"event_id", "hit_id", "predicted_track_id"}
    if not required.issubset(predictions.columns):
        raise ValueError(f"predictions missing columns {sorted(required-set(predictions.columns))}")
    pred_events: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for event_id, frame in predictions.group_by("event_id", maintain_order=False):
        eid = int(event_id[0]) if isinstance(event_id, tuple) else int(event_id)
        hit_id = frame["hit_id"].to_numpy().astype(np.int64)
        assignment = frame["predicted_track_id"].to_numpy().astype(np.int64)
        if len(np.unique(hit_id)) != len(hit_id):
            raise RuntimeError(f"duplicate prediction hit_id in event {eid}")
        pred_events[eid] = (hit_id, assignment)

    root = Path(dataset_root)
    truth_files = _indexed(root / "truth/particles")
    hit_files = _indexed(root / "reco/tracker_hits")
    selected = _select_chunks(
        sorted(set(truth_files) & set(hit_files)), n_files, chunk_range
    )
    collected = {key: [] for key in ("r", "pt", "abs_eta", "layers", "primary", "charged", "matched")}
    eligible_tracks = fake_tracks = matched_tracks = duplicate_tracks = 0
    seen_events: set[int] = set()
    for chunk_index, token in enumerate(selected, 1):
        print(f"[{chunk_index:02d}/{len(selected):02d}] events{token[0]}-{token[1]}", flush=True)
        truth = pq.read_table(truth_files[token], columns=["event_id", "particle_id", "charge", "vx", "vy", "px", "py", "pz", "primary"])
        hits = pq.read_table(hit_files[token], columns=["event_id", "particle_id", "volume_id", "layer_id"])
        trows, hrows = _row_map(truth, "truth/particles"), _row_map(hits, "reco/tracker_hits")
        if set(trows) != set(hrows):
            raise RuntimeError(f"truth/hit event mismatch in {token}")
        ta = {name: truth[name].combine_chunks() for name in truth.column_names[1:]}
        ha = {name: hits[name].combine_chunks() for name in hits.column_names[1:]}
        for event_id in trows:
            if event_id not in pred_events:
                raise RuntimeError(f"predictions missing event {event_id}")
            seen_events.add(event_id)
            tr, hr = trows[event_id], hrows[event_id]
            truth_pid = np.asarray(ta["particle_id"][tr].as_py(), np.uint64)
            hit_pid = np.asarray(ha["particle_id"][hr].as_py(), np.uint64)
            volume = np.asarray(ha["volume_id"][hr].as_py(), np.uint64)
            layer = np.asarray(ha["layer_id"][hr].as_py(), np.uint64)
            hit_ids, assignments = pred_events[event_id]
            if allow_partial_hits:
                # Hit-filtered predictions: absent hits are unassigned, but
                # ids must still be unique and within the event's hit range.
                if len(np.unique(hit_ids)) != len(hit_ids) or (len(hit_ids) and (hit_ids.min() < 0 or hit_ids.max() >= len(hit_pid))):
                    raise RuntimeError(f"prediction hit ids invalid in event {event_id}")
            elif len(hit_ids) != len(hit_pid) or not np.array_equal(np.sort(hit_ids), np.arange(len(hit_pid))):
                raise RuntimeError(f"prediction hit coverage mismatch in event {event_id}")
            ordered = np.full(len(hit_pid), -1, np.int64)
            ordered[hit_ids] = assignments
            event_matches: list[int] = []
            for track_id in np.unique(ordered[ordered >= 0]):
                members = hit_pid[ordered == track_id]
                if len(members) < matching.min_hits:
                    continue
                eligible_tracks += 1
                pids, counts = np.unique(members, return_counts=True)
                best = int(np.argmax(counts))
                fraction = float(counts[best] / len(members))
                if fraction <= matching.majority_fraction:
                    fake_tracks += 1
                if fraction > matching.majority_fraction:
                    matched_tracks += 1
                    event_matches.append(int(pids[best]))
            duplicate_tracks += len(event_matches) - len(set(event_matches))
            matched_ids = np.asarray(event_matches, np.uint64)
            layer_counts = np.zeros(len(truth_pid), np.int16)
            if len(hit_pid):
                code = (volume << np.uint64(32)) | layer
                pairs = np.unique(np.rec.fromarrays([hit_pid, code], names="pid,layer"))
                lpids, counts = np.unique(pairs.pid, return_counts=True)
                pos = np.searchsorted(lpids, truth_pid)
                valid = pos < len(lpids)
                indices = np.flatnonzero(valid)
                valid[indices] &= lpids[pos[indices]] == truth_pid[indices]
                layer_counts[valid] = counts[pos[valid]]
            px = np.asarray(ta["px"][tr].as_py(), float)
            py = np.asarray(ta["py"][tr].as_py(), float)
            pz = np.asarray(ta["pz"][tr].as_py(), float)
            pt = np.hypot(px, py)
            with np.errstate(divide="ignore", invalid="ignore"):
                eta = np.arctanh(pz / np.sqrt(pt*pt + pz*pz))
            collected["r"].append(np.hypot(np.asarray(ta["vx"][tr].as_py(), float), np.asarray(ta["vy"][tr].as_py(), float)))
            collected["pt"].append(pt)
            collected["abs_eta"].append(np.abs(eta))
            collected["layers"].append(layer_counts)
            collected["primary"].append(np.asarray(ta["primary"][tr].as_py(), bool))
            collected["charged"].append(np.asarray(ta["charge"][tr].as_py(), float) != 0)
            collected["matched"].append(np.isin(truth_pid, matched_ids))
    extra_events = set(pred_events) - seen_events
    if extra_events:
        raise RuntimeError(f"predictions contain {len(extra_events)} events outside selected chunks")

    values = {key: np.concatenate(parts) for key, parts in collected.items()}
    selections = {}
    for name, config in (("fiducial", SelectionConfig.fiducial()), ("loose", SelectionConfig.loose())):
        mask = (values["primary"] & values["charged"] & (values["pt"] > config.pt_min_gev)
                & (values["abs_eta"] < config.abs_eta_max) & (values["layers"] >= config.min_distinct_layers))
        r, pt, eta, matched = (values[key][mask] for key in ("r", "pt", "abs_eta", "matched"))
        prompt = r < PROMPT_R_MM
        displaced = r >= DISPLACED_R_MM
        selections[name] = {"cuts": asdict(config), "overall": _scalar(matched),
            "prompt_r_lt_2_mm": _scalar(matched[prompt]), "displaced_r_ge_2_mm": _scalar(matched[~prompt]),
            # Headline LRT region: r >= 10mm (2-10mm is "intermediate" —
            # ACTS is still ~96% efficient there; not genuinely large-radius).
            "intermediate_r_2_10_mm": _scalar(matched[~prompt & ~displaced]),
            "displaced_r_ge_10_mm": _scalar(matched[displaced]),
            "efficiency_vs_r_mm": _binned(r, matched, R_BINS_MM),
            "efficiency_vs_pt_GeV": _binned(pt, matched, PT_BINS_GEV),
            "efficiency_vs_abs_eta": _binned(eta, matched, ABS_ETA_BINS)}
    rates = {"eligible_predicted_tracks": eligible_tracks, "fake_tracks": fake_tracks,
             "fake_rate": fake_tracks / eligible_tracks if eligible_tracks else None,
             "matched_tracks": matched_tracks, "duplicate_tracks": duplicate_tracks,
             "duplicate_rate": duplicate_tracks / eligible_tracks if eligible_tracks else None}
    summary = {"n_files": len(selected), "n_events": len(seen_events), "prediction_meta": asdict(meta),
               "matching": asdict(matching), "selections": selections, "track_metrics": rates}
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "summary.json").open("w") as stream:
        json.dump(_jsonable(summary), stream, indent=2, allow_nan=False)
        stream.write("\n")
    if plots:
        fid = selections["fiducial"]
        _plot(fid["efficiency_vs_r_mm"], "Truth production radius r [mm]", out / "efficiency_vs_r.png")
        _plot(fid["efficiency_vs_pt_GeV"], "Truth pT [GeV]", out / "efficiency_vs_pt.png", True)
        _plot(fid["efficiency_vs_abs_eta"], "Truth |eta|", out / "efficiency_vs_abs_eta.png")
    return _jsonable(summary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--n-files", type=int)
    parser.add_argument("--chunk-range", type=int, nargs=2, metavar=("FIRST", "LAST"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--majority-fraction", type=float, default=.5)
    parser.add_argument("--min-hits", type=int, default=3)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--allow-partial-hits", action="store_true",
        help="Accept predictions covering a subset of hits (hit-filter front-end); absent hits count as unassigned")
    args = parser.parse_args()
    n_files = args.n_files if args.n_files is not None or args.chunk_range else 50
    result = generate_lrt_report(args.predictions_dir, args.dataset_root, n_files, args.out_dir,
        MatchingConfig(args.majority_fraction, args.min_hits), not args.no_plots,
        chunk_range=args.chunk_range, allow_partial_hits=args.allow_partial_hits)
    fid = result["selections"]["fiducial"]
    def percentage(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.3%}"

    print(f"fiducial prompt: {percentage(fid['prompt_r_lt_2_mm']['efficiency'])}")
    print(f"fiducial displaced(r>=10mm): {percentage(fid['displaced_r_ge_10_mm']['efficiency'])}")
    print(f"fiducial displaced(r>=2mm): {percentage(fid['displaced_r_ge_2_mm']['efficiency'])}")
    print(f"fake rate: {percentage(result['track_metrics']['fake_rate'])}")
    print(f"duplicate rate: {percentage(result['track_metrics']['duplicate_rate'])}")


if __name__ == "__main__":
    main()
