#!/bin/bash
#SBATCH -A m5386 -C gpu -q shared --gpus 1 -c 32
#SBATCH -o /pscratch/sd/d/danieltm/quirk_tracking/maskformer/logs/slurm-%x-%j.out
# One resumable training lane on one GPU (gpu_shared batch). Usage:
#   sbatch -t HH:MM:SS -J NAME sbatch_lane.sh TAG LR STEPS VAL_INTERVAL [extra fit args...]
# SKIP_EVAL=1: checkpoint selection and evaluation are done outside (by val coupled efficiency).
TAG=$1; LR=$2; STEPS=$3; VCI=$4; shift 4
set -f  # no globbing of the [a,b] CLI values
MF=/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/maskformer
OUT=/pscratch/sd/d/danieltm/quirk_tracking/maskformer
cd $OUT
echo "[sbatch_lane] job $SLURM_JOB_ID $TAG start $(date -Is)"
srun -n1 -c32 --gpus=1 env SKIP_EVAL=1 MAX_STEPS=$STEPS VAL_INTERVAL=$VCI bash $MF/scripts/lane.sh $TAG $LR "$@" >> $OUT/logs/lanes_$TAG.log 2>&1
echo "[sbatch_lane] $TAG end $(date -Is) rc=$?"
