#!/bin/bash
# Package and convert the test runs (0-3, events 0-1999) of grid datasets before the whole dataset is done, for the
# generalisation tests (ROADMAP.md T4). The full package later holds the same events with the same ids.
#   bash tracking/package_test_runs.sh <out_dir> <config_dir>...
# Writes <out_dir>/<dataset>/parquet (ColliderML layout) and <out_dir>/<dataset>/parquet_r1 (convert_to_r1.py).
# Needs a python with pyarrow and yaml (PYTHON, default python3). Skips datasets already done or with runs missing.
set -uo pipefail
OUT=${1:?usage: package_test_runs.sh <out_dir> <config_dir>...}; shift
PY=${PYTHON:-python3}
HERE=$(cd "$(dirname "$0")" && pwd)
for cfg in "$@"; do
  ( source "$cfg/sample.env"
    PROD=$WORK_DIR/colliderml-production/scripts/postprocessing
    d=$OUT/$DATASET
    [ -f "$d/parquet_r1/.done" ] && { echo "$DATASET: done already"; exit 0; }
    for n in 0 1 2 3; do
      [ -f "$OUTPUT_BASE/$RUNS_DIR/$n/.done_digitization" ] || { echo "$DATASET: run $n not digitised yet"; exit 0; }
    done
    mkdir -p "$d"
    sed -e "s|^input_base_dir:.*|input_base_dir: $OUTPUT_BASE|" -e "s|^output_base_dir:.*|output_base_dir: $d/pack|" \
      "$cfg/4_package_parquet.yaml" > "$d/package_test.yaml"
    echo "max_run: 3" >> "$d/package_test.yaml"
    $PY "$PROD/package_native_parquet.py" --config "$d/package_test.yaml" > "$d/package.log" 2>&1 &&
    $PY "$PROD/verify_packaged_parquet.py" --config "$d/package_test.yaml" --chunks 0-1 >> "$d/package.log" 2>&1 ||
      { echo "$DATASET: packaging FAILED ($d/package.log)"; exit 1; }
    [ -e "$d/parquet" ] && mv "$d/parquet" "$d/parquet.old.$(date +%s)"
    mv "$d/pack/$CAMPAIGN/$DATASET/$VERSION/parquet" "$d/parquet"
    $PY "$HERE/convert_to_r1.py" "$d/parquet" "$d/parquet_r1" 2 > "$d/convert.log" 2>&1 && touch "$d/parquet_r1/.done" &&
      echo "$DATASET: packaged and converted" || echo "$DATASET: conversion FAILED ($d/convert.log)"
  )
done
