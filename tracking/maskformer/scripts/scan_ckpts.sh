#!/bin/bash
# Checkpoint choice by val coupled efficiency (data-scaling study). For run TAG: candidates = the last NLAST checkpoints
# (by step) plus the lowest-val-loss one; val predictions (16000-16999) at object threshold 0.1 -> coupled_eval.py
# (strict and >=). Writes diag/scan_<TAG>.tsv (ckpt, coupled_strict, coupled_ge) and diag/chosen_<TAG>.txt (best strict).
# Run on one GPU (srun inside a GPU job). Usage: scan_ckpts.sh TAG [NLAST=5]
set -uo pipefail
TAG=$1; NLAST=${2:-5}
MF=/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/maskformer
OUT=/pscratch/sd/d/danieltm/quirk_tracking/maskformer
PY=/pscratch/sd/d/danieltm/envs/ml-tracking-review/bin/python
SRC=/global/cfs/cdirs/m4958/data/ColliderML/simulation/quirks/m200_L774_mu0/v1/parquet_r1
R=$OUT/runs/$TAG; cd $OUT
RUNCFG=$(ls -t $R/*/*/version_0/config.yaml | tail -1)
ALL=$(ls $R/*/*/version_0/ckpts/*.ckpt)
if [ "${CANDIDATES:-last}" = metric ]; then LAST=$($PY $MF/scripts/candidates_by_metric.py $TAG 3); BEST=""; else
LAST=$(for c in $ALL; do e=$(basename $c | sed -E 's/epoch=([0-9]+)-.*/\1/'); echo "$e $(stat -c %Y $c) $c"; done | sort -k1,1n -k2,2n | tail -$NLAST | cut -d' ' -f3)
BEST=$(echo "$ALL" | awk -F'val_loss=' '{print $2" "$0}' | sort -g | head -1 | cut -d' ' -f2-)
fi
fails=0
: > diag/scan_$TAG.tsv
for CK in $(printf '%s\n' $LAST $BEST | sort -u); do
  N=$(basename $CK .ckpt | tr '=' '_')
  P=$OUT/predictions/scan_${TAG}_${N}_val_ot0.1
  if [ ! -f $P/predictions.parquet ]; then
    bash $MF/scripts/predict.sh $RUNCFG $CK $P 16000 16999 0.1 0.5 > $OUT/logs/predict_scan_${TAG}_$N.log 2>&1 || { fails=$((fails+1)); echo "predict failed $CK"; [ $fails -ge 2 ] && exit 1; continue; }
  fi
  $PY $MF/scripts/coupled_eval.py $P $SRC 16000 17000 $P/coupled.json > /dev/null || { echo "coupled_eval failed $CK"; exit 1; }
  $PY -c "import json,sys; d=json.load(open('$P/coupled.json')); print('$CK', d['coupled_eff_strict'], d['coupled_eff'], sep='\t')" >> diag/scan_$TAG.tsv
done
sort -t$'\t' -k2,2gr diag/scan_$TAG.tsv | head -1 | cut -f1 > diag/chosen_$TAG.txt
cat diag/scan_$TAG.tsv; echo "chosen: $(cat diag/chosen_$TAG.txt)"
