#!/bin/bash
#SBATCH -A m4958
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH --gpus-per-node 4
#SBATCH -c 128
#SBATCH -t 06:00:00
#SBATCH -J qtrk_mf_pack
#SBATCH -o /pscratch/sd/d/danieltm/quirk_tracking/maskformer/logs/slurm-%x-%j.out
# Usage: sbatch sbatch_pack.sh "TAG1|args" "TAG2|args" ...   (up to 4 lanes, 1 GPU + 32 CPUs each)
set -uo pipefail
MF=/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/maskformer
CFG=$MF/packages/ml_tracking/src/ml_tracking/experiments/quirk/configs
OUT=/pscratch/sd/d/danieltm/quirk_tracking/maskformer
cd $OUT
PIDS=()
for spec in "$@"; do
  TAG=${spec%%|*}; ARGS=${spec#*|}
  echo "[pack] job $SLURM_JOB_ID lane $TAG start $(date -Is) args: $ARGS"
  srun --exact -n1 -c32 --gpus=1 bash $MF/scripts/run.sh fit --config $CFG/quirk.yaml --name $TAG \
    --trainer.default_root_dir $OUT/runs/$TAG $ARGS > $OUT/logs/$TAG.log 2>&1 &
  PIDS+=($!)
done
for i in "${!PIDS[@]}"; do wait ${PIDS[$i]}; echo "[pack] lane $i rc=$? $(date -Is)"; done
