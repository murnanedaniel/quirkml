#!/bin/bash
# Predict + evaluate.py for a finished lane on an event range (run via srun on one GPU).
# Usage: eval_split.sh TAG LABEL FIRST LAST_EXCL "OT1 OT2 ..."   (lowest-val-loss checkpoint, mask 0.5)
set -uo pipefail
TAG=$1; LABEL=$2; A=$3; B=$4; THRS=$5
MF=/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/maskformer
Q=/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml
OUT=/pscratch/sd/d/danieltm/quirk_tracking/maskformer
ROOT=/global/cfs/cdirs/m4958/data/ColliderML/simulation/quirks/m200_L774_mu0/v1/parquet_r1
EVPY=/global/cfs/cdirs/m4958/usr/danieltm/ml_tracking_pilot_2026-09/envs/eval48/bin/python
R=$OUT/runs/$TAG
BEST=$(ls $R/*/*/version_0/ckpts/*.ckpt | awk -F'val_loss=' '{print $2" "$0}' | sort -g | head -1 | cut -d' ' -f2-)
RUNCFG=$(ls -t $R/*/*/version_0/config.yaml | tail -1)
for OT in $THRS; do
  P=$OUT/predictions/${TAG}_${LABEL}_ot$OT
  bash $MF/scripts/predict.sh $RUNCFG "$BEST" $P $A $((B-1)) $OT 0.5 > $OUT/logs/predict_${TAG}_${LABEL}_ot$OT.log 2>&1 || { echo "predict failed ot=$OT"; continue; }
  echo "$BEST" > $P/ckpt_used.txt
  (cd $Q && $EVPY tracking/evaluate.py --predictions-dir $P --dataset $ROOT --events $A $B \
     --out $OUT/eval/${TAG}_${LABEL}_ot$OT --jobs 16 --staging-dir $OUT/staging_${TAG}_$LABEL) > $OUT/logs/eval_${TAG}_${LABEL}_ot$OT.log 2>&1
  echo "[$TAG $LABEL] ckpt $(basename $BEST) ot=$OT $(grep '^very_loose' $OUT/logs/eval_${TAG}_${LABEL}_ot$OT.log)"
done
