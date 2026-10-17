#!/bin/bash
# Run every stage of a sample, in order, on this machine: generation, stage1, simulation,
# digitization, package. Each stage goes over all runs before the next starts (colliderml/run_stage.sh).
#   bash colliderml/run_sample.sh <config_dir> [--parallel K]
# Runs that have finished a stage are skipped, so after a failure the same command carries on.
set -euo pipefail
CFG=${1:?usage: bash colliderml/run_sample.sh <config_dir> [--parallel K]}; shift
HERE=$(cd "$(dirname "$0")" && pwd)
for stage in generation stage1 simulation digitization; do
  bash "$HERE/run_stage.sh" "$CFG" "$stage" "$@"
done
P=$(source "$CFG/sample.env"; echo "${OUTPUT_BASE:-$WORK_DIR}/$CAMPAIGN/$DATASET/$VERSION/parquet")
[ -d "$P" ] && echo "already packaged: $P" || bash "$HERE/run_stage.sh" "$CFG" package
