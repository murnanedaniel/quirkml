#!/bin/bash
# usage: run.sh <jobid> <logname> <script.py> [args...]  -- runs on the allocation via srun, full log.
# NO_OVERLAP=1: own GPU + 32 CPUs inside a multi-GPU allocation (two arms side by side); default shares (--overlap).
JOB=$1; LOG=$2; shift 2
cd /global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/gnn
mkdir -p $PSCRATCH/quirk_tracking/gnn/logs
if [ -n "$NO_OVERLAP" ]; then OV="--exact"; else OV="--overlap"; fi
env $(env | grep -o '^SLURM_[A-Z_]*' | sed 's/^/-u /') srun --jobid=$JOB $OV -n 1 -c 32 --gpus=1 \
  bash -c 'echo "host $(hostname) CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"; exec "$@"' _ \
  /pscratch/sd/d/danieltm/envs/ml-tracking-review/bin/python -u "$@" > $PSCRATCH/quirk_tracking/gnn/logs/$LOG.log 2>&1
rc=$?; echo "exit $rc" >> $PSCRATCH/quirk_tracking/gnn/logs/$LOG.log; exit $rc
