#!/bin/bash
# Run predict_grid.py inside allocation $1 on every set whose direction features and pair truth are ready, until all done.
J=$1
G=/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/grid
MF=/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/maskformer
DATA=/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirk_tracking_data
OUTB=$DATA/grid_predictions/mf_pairdir_long
read CK CFG < $G/model_used.txt
MAXFAIL=${MAXFAIL:-3}; fails=0  # consecutive failed inference passes before giving up (exit 2)
[ -f "$CK" ] && [ -f "$CFG" ] || { echo "model_used.txt paths not found (must be absolute): $CK $CFG"; exit 2; }
while true; do
  todo=(); waiting=0
  for ds in $(cat $G/grid_sets_mu0.txt); do
    [ -f $OUTB/$ds/ot0.2/predictions.parquet ] && continue
    if [ -f $DATA/direction/$ds/.done ]; then todo+=($ds); else waiting=$((waiting+1)); fi
  done
  echo "$(date -Is) todo ${#todo[@]} waiting ${waiting}"
  [ ${#todo[@]} -eq 0 ] && [ $waiting -eq 0 ] && { echo all done; exit 0; }
  squeue -h -j $J | grep -q . || { echo "allocation $J gone"; exit 1; }
  if [ ${#todo[@]} -gt 0 ]; then
    env $(env | grep -o '^SLURM_[A-Z_]*' | sed 's/^/-u /') srun --jobid=$J -n1 -c32 --gpus=1 env PYTHONPATH=$MF/packages/ml_tracking/src:$MF/packages/tracking_core/src \
      /pscratch/sd/d/danieltm/envs/ml-tracking-review/bin/python $G/predict_grid.py $CK $CFG $OUTB "${todo[@]}" 2>&1 | grep -v "parsing.py\|save_hyper\|QuirkTrackingDataset"
    rc=${PIPESTATUS[0]}
    if [ $rc -ne 0 ]; then
      fails=$((fails+1)); echo "$(date -Is) inference pass failed (rc $rc), $fails consecutive"
      [ $fails -ge $MAXFAIL ] && { echo "giving up after $fails consecutive failures"; exit 2; }
      sleep 30
    else
      fails=0
    fi
  else
    squeue -h -j $J | grep -q . || { echo "allocation $J gone"; exit 1; }
    sleep 60
  fi
done
