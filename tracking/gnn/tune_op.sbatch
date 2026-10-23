#!/bin/bash
#SBATCH -A m4958 -C cpu -q debug -N 1 -t 00:30:00
#SBATCH -o /pscratch/sd/d/danieltm/quirk_tracking/gnn/logs/tune_op_%x_%j.out
# usage: sbatch -J qtrk_tune --export=ALL,SCORED='<glob>',OUT=<json> tune_op.sbatch
cd /global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/gnn
/pscratch/sd/d/danieltm/envs/ml-tracking-review/bin/python tune_op.py --cache /pscratch/sd/d/danieltm/quirk_tracking/gnn/cache/m200_L774_mu0 \
  --scored "$SCORED" --events 16200 16999 --out $OUT --workers 24
