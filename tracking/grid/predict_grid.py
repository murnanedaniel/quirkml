"""Stage F inference (no retraining): one MaskFormer checkpoint on many grid test sets, predictions.parquet per set
and object threshold (mask threshold 0.5, plain mask-argmax hit assignment, as QuirkPredictionWriter).

Usage: python predict_grid.py CKPT RUN_CONFIG OUT_BASE DS [DS ...]
Inputs per set: quirk_tracking_data/testruns/<ds>/parquet_r1 (events 0-1999), direction features from
quirk_tracking_data/direction/<ds>. Writes OUT_BASE/<ds>/ot<thr>/{predictions.parquet,meta.json}; skips sets done.
"""
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import polars as pl
import torch
import yaml
from torch.utils.data import DataLoader

from ml_tracking.experiments.quirk.data import QuirkTrackingDataset
from ml_tracking.experiments.quirk.model import QuirkModel
from tracking_core.schema import RunMeta, write_predictions

THRESHOLDS = (0.1, 0.2)
MASK_THR = 0.5
DATA = Path("/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirk_tracking_data")

ckpt, run_cfg, out_base = sys.argv[1], sys.argv[2], Path(sys.argv[3])
cfg = yaml.safe_load(Path(run_cfg).read_text())["data"]
module = QuirkModel.load_from_checkpoint(ckpt, map_location="cuda")
module.eval()
net = module.model

for ds in sys.argv[4:]:
    outs = {t: out_base / ds / f"ot{t}" for t in THRESHOLDS}
    if all((o / "predictions.parquet").exists() for o in outs.values()):
        print(f"[{ds}] done already", flush=True)
        continue
    try:
        dset = QuirkTrackingDataset(
            str(DATA / "testruns" / ds / "parquet_r1"), (0, 1999), event_max_num_particles=cfg["event_max_num_particles"],
            particle_min_pt=cfg["particle_min_pt"], particle_max_abs_eta=cfg["particle_max_abs_eta"],
            min_num_sihit=cfg["min_num_sihit"], particle_require_primary=cfg["particle_require_primary"],
            include_quirks=cfg["include_quirks"], quirk_pdg_ids=tuple(cfg["quirk_pdg_ids"]),
            quirk_min_num_sihit=cfg["quirk_min_num_sihit"], cluster_features=cfg["cluster_features"],
            row_group_cache_size=16, quirk_pair_target=cfg.get("quirk_pair_target", False),
            direction_dir=str(DATA / "direction" / ds) if cfg.get("direction_dir") else None,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[{ds}] FAILED to open: {exc}", flush=True)
        continue
    loader = DataLoader(dset, batch_size=8, collate_fn=QuirkTrackingDataset.collate_fn, num_workers=4)
    frames = {t: [] for t in THRESHOLDS}
    t0 = datetime.now()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for inputs, targets in loader:
            inputs = {k: v.cuda(non_blocking=True) for k, v in inputs.items()}
            final = net.predict(net(inputs))["final"]
            obj = final["flow_valid"]["flow_valid_prob"].float().cpu()
            masks = final["flow_sihit_assignment"]["flow_sihit_valid_prob"].float().cpu()
            hv = inputs["sihit_valid"].cpu()
            for b in range(obj.shape[0]):
                n = int(hv[b].sum())
                m = masks[b, :, :n]
                for t in THRESHOLDS:
                    valid = obj[b] >= t
                    assign = torch.full((n,), -1, dtype=torch.int64)
                    score = torch.full((n,), float("nan"))
                    if valid.any() and n:
                        q = torch.nonzero(valid).flatten()
                        win, loc = m[valid].max(dim=0)
                        ok = win >= MASK_THR
                        assign[ok] = q[loc[ok]]
                        score[ok] = win[ok]
                    frames[t].append(pl.DataFrame({
                        "event_id": pl.Series([int(targets["event_id"][b])] * n, dtype=pl.UInt64),
                        "hit_id": pl.Series(range(n), dtype=pl.UInt64),
                        "predicted_track_id": pl.Series(assign.numpy(), dtype=pl.Int64),
                        "score": pl.Series(score.numpy(), dtype=pl.Float32)}))
    for t in THRESHOLDS:
        meta = RunMeta(run_id=str(uuid4()), implementation="ml_tracking:quirk_maskformer_pair_direction",
                       config={"ckpt_path": ckpt, "object_threshold": t, "mask_threshold": MASK_THR, "dataset": ds},
                       wrote_at=datetime.now(timezone.utc).isoformat())
        write_predictions(pl.concat(frames[t]), outs[t], meta)
    print(f"[{ds}] {len(dset)} events in {(datetime.now() - t0).total_seconds():.0f} s", flush=True)
