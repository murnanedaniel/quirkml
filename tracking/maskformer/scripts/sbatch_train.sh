#!/bin/bash
#SBATCH -A m4958
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH --gpus 1
#SBATCH -c 32
#SBATCH -t 06:00:00
#SBATCH -J qtrk_mf
#SBATCH -o /pscratch/sd/d/danieltm/quirk_tracking/maskformer/logs/slurm-%x-%j.out
# Usage: sbatch [--job-name qtrk_<tag>] sbatch_train.sh TAG [extra CLI args...]
# Trains with configs/quirk.yaml (20k m200 set) and writes the run under runs/TAG.
set -uo pipefail
TAG=$1; shift
MF=/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/maskformer
CFG=$MF/packages/ml_tracking/src/ml_tracking/experiments/quirk/configs
OUT=/pscratch/sd/d/danieltm/quirk_tracking/maskformer
cd $OUT
echo "[sbatch_train] job $SLURM_JOB_ID tag $TAG start $(date -Is) host $(hostname) args: $*"
srun -n1 -c32 --gpus=1 bash $MF/scripts/run.sh fit --config $CFG/quirk.yaml --name $TAG \
  --trainer.default_root_dir $OUT/runs/$TAG "$@" > $OUT/logs/$TAG.log 2>&1
rc=$?
echo "[sbatch_train] tag $TAG end $(date -Is) rc=$rc"
exit $rc
