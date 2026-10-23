#!/bin/bash
# Write predictions.parquet for an event_id range with a trained checkpoint.
# Usage (inside an allocation, via srun):
#   predict.sh RUN_CONFIG CKPT OUT_DIR FIRST_EVENT LAST_EVENT [OBJECT_THR=0.1] [MASK_THR=0.5]
# RUN_CONFIG: the resolved config.yaml saved in the training run's log dir
# (it is the truth for what ran). Events are inclusive event_id bounds.
set -euo pipefail
RUN_CONFIG=$1; CKPT=$2; OUT=$3; A=$4; B=$5; OT=${6:-0.1}; MT=${7:-0.5}
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd)
cat > "$OUT/predict_overlay.yaml" <<YML
data:
  test_events: [$A, $B]
  num_test: -1
  allow_split_overlap: true
  num_workers: 4
trainer:
  logger: false
  devices: 1
  callbacks:
    - class_path: ml_tracking.experiments.quirk.prediction_writer.QuirkPredictionWriter
      init_args:
        out_dir: $OUT
        object_threshold: $OT
        mask_threshold: $MT
YML
exec bash "$HERE/run.sh" predict --config "$RUN_CONFIG" --config "$OUT/predict_overlay.yaml" \
  --ckpt_path "$CKPT" --return_predictions false --trainer.default_root_dir "$OUT/logs_$(date +%Y%m%dT%H%M%S)"
