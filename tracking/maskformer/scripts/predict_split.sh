#!/bin/bash
# GPU: predictions of a finished lane (lowest-val-loss checkpoint) on an event range at several object thresholds.
# Usage: predict_split.sh TAG LABEL FIRST LAST_INCL "OT1 OT2 ..."  -> predictions/<TAG>_<LABEL>_ot<OT>
set -uo pipefail
TAG=$1; LABEL=$2; A=$3; B=$4; THRS=$5
MF=/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/maskformer
OUT=/pscratch/sd/d/danieltm/quirk_tracking/maskformer
R=$OUT/runs/$TAG
BEST=$(ls $R/*/*/version_0/ckpts/*.ckpt | awk -F'val_loss=' '{print $2" "$0}' | sort -g | head -1 | cut -d' ' -f2-)
RUNCFG=$(ls -t $R/*/*/version_0/config.yaml | tail -1)
for OT in $THRS; do
  P=$OUT/predictions/${TAG}_${LABEL}_ot$OT
  [ -f $P/predictions.parquet ] && [ "$(cat $P/ckpt_used.txt 2>/dev/null)" = "$BEST" ] && continue
  bash $MF/scripts/predict.sh $RUNCFG "$BEST" $P $A $B $OT 0.5 > $OUT/logs/predict_${TAG}_${LABEL}_ot$OT.log 2>&1 && echo "$BEST" > $P/ckpt_used.txt
done
