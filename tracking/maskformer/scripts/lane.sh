#!/bin/bash
# One T2 LR-scan lane on one GPU (run via srun inside an allocation). Resumable across windows.
# Usage: [MAX_STEPS=30000] [VAL_INTERVAL=1000] lane.sh TAG MAX_LR [extra fit args...]
# 1. trains configs/quirk.yaml + configs/lrscan.yaml with max LR MAX_LR, resuming from the newest checkpoint;
# 2. when 30k steps are done, predicts val 16000-16999 with the lowest-val-loss checkpoint at object thresholds
#    0.1 0.2 0.3 0.5 (mask 0.5) and runs quirkml/tracking/evaluate.py on each.
set -uo pipefail
TAG=$1; LR=$2; shift 2
MF=/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/maskformer
Q=/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml
CFG=$MF/packages/ml_tracking/src/ml_tracking/experiments/quirk/configs
OUT=/pscratch/sd/d/danieltm/quirk_tracking/maskformer
ROOT=/global/cfs/cdirs/m4958/data/ColliderML/simulation/quirks/m200_L774_mu0/v1/parquet_r1
PY=/pscratch/sd/d/danieltm/envs/ml-tracking-review/bin/python
EVPY=/global/cfs/cdirs/m4958/usr/danieltm/ml_tracking_pilot_2026-09/envs/eval48/bin/python
STEPS=${MAX_STEPS:-30000}; VCI=${VAL_INTERVAL:-1000}
R=$OUT/runs/$TAG; mkdir -p $R; cd $OUT
log() { echo "[lane $TAG] $(date -Is) $*"; }

[ -f $R/TRAIN_FAILED ] && { log "TRAIN_FAILED present, not retrying"; exit 1; }
if [ ! -f $R/TRAIN_DONE ]; then
  CKPT=$(ls -t $R/*/*/version_0/ckpts/*.ckpt 2>/dev/null | head -1)
  RESUME=()
  if [ -n "$CKPT" ]; then
    # Restart the interrupted epoch from batch 0 (avoids the slow fast-forward of a non-resumable loader).
    $PY - "$CKPT" <<'PYEOF'
import sys, torch
p = sys.argv[1]; ck = torch.load(p, map_location="cpu", weights_only=False)
bp = ck["loops"]["fit_loop"]["epoch_loop.batch_progress"]
for k in ("ready", "started", "processed", "completed"): bp["current"][k] = 0
torch.save(ck, p); print("patched", p, "global_step", ck["global_step"])
PYEOF
    RESUME=(--ckpt_path "$CKPT")
  fi
  log "fit start lr=$LR resume=${CKPT:-none}"
  bash $MF/scripts/run.sh fit --config $CFG/quirk.yaml --config $CFG/lrscan.yaml --name $TAG \
    --model.lrs_config.max $LR --trainer.max_steps $STEPS --trainer.val_check_interval $VCI --trainer.default_root_dir $R "${RESUME[@]}" "$@" >> $OUT/logs/$TAG.log 2>&1
  rc=$?; log "fit exit rc=$rc"
  if tr "\r" "\n" < $OUT/logs/$TAG.log | grep -q -e "max_steps=$STEPS. reached" -e "did not improve in the last"; then touch $R/TRAIN_DONE; rm -f $R/.fit_fails
  elif [ $rc -ne 0 ]; then
    n=$(( $(cat $R/.fit_fails 2>/dev/null || echo 0) + 1 )); echo $n > $R/.fit_fails
    [ $n -ge 2 ] && { touch $R/TRAIN_FAILED; log "fit failed $n times in a row: TRAIN_FAILED, lane stops"; exit 1; }
  fi
fi

if [ -f $R/TRAIN_DONE ] && [ ! -f $R/EVAL_DONE ] && [ "${SKIP_EVAL:-0}" = 1 ]; then
  touch $R/EVAL_DONE; log "SKIP_EVAL=1: evaluation done outside the lane (checkpoint chosen by val efficiency)"
fi
if [ -f $R/TRAIN_DONE ] && [ ! -f $R/EVAL_DONE ]; then
  BEST=$(ls $R/*/*/version_0/ckpts/*.ckpt | awk -F'val_loss=' '{print $2" "$0}' | sort -g | head -1 | cut -d' ' -f2-)
  RUNCFG=$(ls -t $R/*/*/version_0/config.yaml | tail -1)
  log "eval with $BEST"
  ok=1
  for OT in 0.1 0.2 0.3 0.5; do
    P=$OUT/predictions/${TAG}_val_ot$OT
    if [ -f $OUT/eval/${TAG}_val_ot$OT/summary.json ] && [ "$(cat $P/ckpt_used.txt 2>/dev/null)" = "$BEST" ]; then log "ot=$OT already evaluated"; continue; fi
    bash $MF/scripts/predict.sh $RUNCFG "$BEST" $P 16000 16999 $OT 0.5 > $OUT/logs/predict_${TAG}_val_ot$OT.log 2>&1 || { ok=0; log "predict failed ot=$OT"; continue; }
    echo "$BEST" > $P/ckpt_used.txt
    (cd $Q && $EVPY tracking/evaluate.py --predictions-dir $P --dataset $ROOT --events 16000 17000 \
       --out $OUT/eval/${TAG}_val_ot$OT --jobs 16 --staging-dir $OUT/staging_$TAG) > $OUT/logs/eval_${TAG}_val_ot$OT.log 2>&1 || { ok=0; log "eval failed ot=$OT"; continue; }
    $PY $MF/scripts/coupled_eval.py $P $ROOT 16000 17000 $OUT/eval/${TAG}_val_ot$OT/coupled.json > /dev/null 2>&1
    log "ot=$OT $(grep -e '^very_loose' $OUT/logs/eval_${TAG}_val_ot$OT.log) coupled $(cat $OUT/eval/${TAG}_val_ot$OT/coupled.json 2>/dev/null | tr -d '\n ')"
  done
  [ $ok = 1 ] && touch $R/EVAL_DONE
fi
log "lane end"
