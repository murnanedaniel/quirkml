#!/bin/bash
#SBATCH -A m4958 -C cpu -q debug -N 1 -t 00:30:00 -J qtrk_scal_eval
#SBATCH -o /pscratch/sd/d/danieltm/quirk_tracking/scaling/logs/eval_%j.out
# Strict ACTS evaluation of one scaling arm (or a 16k control): val 16000-16999 (curve) and test 17000-19999 (once).
# KIND=v3 (per-quirk): walk + stitch 0.7, >= 4 hits (the g2-v3-dir builder), normal truth.
# KIND=pair (coupled): components at the cut chosen on val 16000-16499 by the p3 rule (highest coupled eff with fake <=
# walk's and technical within 1 point of walk's), >= 4 hits; pair truth (coupled) + normal truth (fake, technical_sm).
# env: TAG SCORED (dir with scored events16000-19999 parquet) OUTD [SKIP_TEST=1]
umask 007
O=/pscratch/sd/d/danieltm/quirk_tracking/gnn
S=/pscratch/sd/d/danieltm/quirk_tracking/scaling
PY=/pscratch/sd/d/danieltm/envs/ml-tracking-review/bin/python
EPY=/global/cfs/cdirs/m4958/usr/danieltm/ml_tracking_pilot_2026-09/envs/eval48/bin/python
EV=/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/evaluate.py
DATA=/global/cfs/cdirs/m4958/data/ColliderML/simulation/quirks/m200_L774_mu0/v1/parquet_r1
PAIR=/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirk_tracking_data/pair_truth/m200_L774_mu0
cd /global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/gnn
mkdir -p $OUTD
ev() {  # pred dataset out first stop
  $EPY $EV --predictions-dir $1 --dataset $2 --events $4 $5 --out $3 --jobs 60 --staging-dir $S/staging > $3.log 2>&1
  local rc=$?; [ -f $3/summary.json ] || { echo "EVAL FAILED rc=$rc $3"; return 1; }
  echo "ok $3"
}
bt() {  # method cut first last out
  $PY build_tracks.py --cache $O/cache/m200_L774_mu0 --scored "$SCORED/events1[6-9]???-*.parquet" --events $3 $4 --out $5 \
    --run-id ${TAG}_$(basename $5) --workers 64 --method $1 --cc-cut $2 --th-stitch 0.7 --min-hits 4 > $5.log 2>&1 \
    && [ -f $5/predictions.parquet ] || { echo "BUILD FAILED $5"; return 1; }
}
if [ "$KIND" = v3 ]; then
  bt ccwalk_stitch 0.5 16000 16999 $OUTD/val_pred || exit 1
  ev $OUTD/val_pred $DATA $OUTD/val_eval 16000 17000 || exit 1
  [ -n "$SKIP_TEST" ] && exit 0
  bt ccwalk_stitch 0.5 17000 19999 $OUTD/test_pred || exit 1
  ev $OUTD/test_pred $DATA $OUTD/test_eval 17000 20000 || exit 1
else
  if [ ! -f $OUTD/val_builders.jsonl ]; then
    $PY tune_builders.py --cache $O/cache/m200_L774_mu0_dir_pair --scored "$SCORED/events16000-16999.parquet" --events 16000 16499 > $OUTD/val_builders.jsonl || exit 1
  fi
  CUT=$($PY -c "
import json; r=[json.loads(l) for l in open('$OUTD/val_builders.jsonl')]
w=[x for x in r if x['method']=='ccwalk_stitch'][0]
ok=[x for x in r if x['method']=='cc' and x['fake']<=w['fake'] and x['tech']>=w['tech']-0.01]
print(max(ok,key=lambda x:x['quirk_eff'])['cut'] if ok else 0.5)")
  echo "chosen components cut $CUT" | tee $OUTD/cut.txt
  bt cc $CUT 16000 16999 $OUTD/val_pred || exit 1
  ev $OUTD/val_pred $PAIR $OUTD/val_evalpair 16000 17000 || exit 1
  ev $OUTD/val_pred $DATA $OUTD/val_eval 16000 17000 || exit 1
  [ -n "$SKIP_TEST" ] && exit 0
  bt cc $CUT 17000 19999 $OUTD/test_pred || exit 1
  ev $OUTD/test_pred $PAIR $OUTD/test_evalpair 17000 20000 || exit 1
  ev $OUTD/test_pred $DATA $OUTD/test_eval 17000 20000 || exit 1
fi
echo "eval done $TAG"
