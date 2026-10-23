#!/bin/bash
# Quirk MaskFormer launcher. Run inside an allocation via srun, e.g.
#   env $(env | grep -o '^SLURM_[A-Z_]*' | sed 's/^/-u /') srun --jobid=<id> -n1 -c32 --gpus=1 \
#       bash scripts/run.sh fit --config $CFG/quirk.yaml [--config $CFG/dev_m400.yaml] [flags]
#   ... scripts/run.sh predict --config <run_dir>/config.yaml --ckpt_path <ckpt> \
#       --data.test_events '[A,B]' --data.num_test N \
#       --trainer.callbacks+=... (or edit out_dir via a small overlay)
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
export PYTHONPATH=$HERE/packages/ml_tracking/src:$HERE/packages/tracking_core/src
export WANDB_MODE=online COMET_MODE=disabled PYTHONFAULTHANDLER=1
PY=/pscratch/sd/d/danieltm/envs/ml-tracking-review/bin/python
exec $PY $HERE/packages/ml_tracking/src/ml_tracking/experiments/quirk/main.py "$@"
