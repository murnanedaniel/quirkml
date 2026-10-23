#!/bin/bash
#SBATCH -A m4958
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH --gpus 1
#SBATCH -c 32
#SBATCH -t 01:30:00
#SBATCH -J qtrk_eval_mf
#SBATCH -o /pscratch/sd/d/danieltm/quirk_tracking/maskformer/logs/slurm-%x-%j.out
# Usage: sbatch sbatch_eval.sh TAG DATASET_ROOT FIRST LAST_EXCL [CKPT] [OBJ_THRS="0.1 0.3"]
# Predicts events FIRST..LAST_EXCL-1 with the lowest-val-loss checkpoint of runs/TAG (or CKPT),
# for each object threshold (mask 0.5), then runs quirkml/tracking/evaluate.py.
set -uo pipefail
TAG=$1; ROOT=$2; A=$3; B=$4; CKPT=${5:-}; THRS=${6:-"0.1 0.3"}
MF=/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/maskformer
Q=/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml
OUT=/pscratch/sd/d/danieltm/quirk_tracking/maskformer
RUN=$(ls -d $OUT/runs/$TAG/*/*/version_0 | tail -1)
if [ -z "$CKPT" ]; then
  CKPT=$(ls $RUN/ckpts/*.ckpt | awk -F'val_loss=' '{print $2" "$0}' | sort -g | head -1 | cut -d' ' -f2-)
fi
echo "[sbatch_eval] $TAG ckpt $CKPT events [$A,$B) thresholds $THRS start $(date -Is)"
for OT in $THRS; do
  P=$OUT/predictions/${TAG}_ev${A}-$((B-1))_ot$OT
  srun -n1 -c32 --gpus=1 bash $MF/scripts/predict.sh $RUN/config.yaml "$CKPT" $P $A $((B-1)) $OT 0.5 > $OUT/logs/predict_${TAG}_ot$OT.log 2>&1 || { echo "predict failed ot=$OT"; continue; }
  echo "$CKPT" > $P/ckpt_used.txt
  srun -n1 -c32 --gpus=1 bash -c "cd $Q && /global/cfs/cdirs/m4958/usr/danieltm/ml_tracking_pilot_2026-09/envs/eval48/bin/python tracking/evaluate.py --predictions-dir $P --dataset $ROOT --events $A $B --out $OUT/eval/${TAG}_ev${A}-$((B-1))_ot$OT" > $OUT/logs/eval_${TAG}_ot$OT.log 2>&1
  echo "== ot $OT rc $?"; grep -e "quirk track eff" -e "prototracks" $OUT/logs/eval_${TAG}_ot$OT.log
done
echo "[sbatch_eval] end $(date -Is)"
