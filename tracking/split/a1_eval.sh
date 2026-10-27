#!/bin/bash
# A1 test: splitter predictions on test 17000-19999 (GPU step on allocation J) replacing the tracks they split in the
# pair GNN + components (0.5) predictions; ACTS on the normal truth with the strict points (CPU, srun on the same
# allocation). usage: a1_eval.sh <jobid> <arm> <data: oracle|cc> <decode>
set -o pipefail
J=$1; ARM=$2; DATA=$3; DEC=$4; T=${TAG:-$DEC}  # env: TAG (output suffix), EXTRA (extra predict args), MODEL (checkpoint dir)
D=/pscratch/sd/d/danieltm/quirk_tracking/split
BASE=/pscratch/sd/d/danieltm/quirk_tracking/gnn/main/e2_pair_dir_s0/test/pred_cc0.5_mh4/predictions.parquet
EPY=/global/cfs/cdirs/m4958/usr/danieltm/ml_tracking_pilot_2026-09/envs/eval48/bin/python
U="env $(env | grep -o '^SLURM_[A-Z_]*' | sed 's/^/-u /' | tr '\n' ' ')"
mkdir -p $D/$ARM; cd /global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/gnn
./run.sh $J a1eval_${ARM}_predict /global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/split/splitter2.py predict \
  --data $D/data/${DATA}_test.pkl --out ${MODEL:-$D/$ARM} --base $BASE --pred-out $D/$ARM/test_pred_$T --decode $DEC $EXTRA || { echo "predict failed"; exit 1; }
$U srun --jobid=$J --overlap -n 1 -c 32 $EPY /global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/evaluate.py \
  --predictions-dir $D/$ARM/test_pred_$T --dataset /global/cfs/cdirs/m4958/data/ColliderML/simulation/quirks/m200_L774_mu0/v1/parquet_r1 \
  --events 17000 20000 --out $D/$ARM/test_eval_$T --jobs 16 > $D/$ARM/test_eval_$T.log 2>&1
[ -f $D/$ARM/test_eval_$T/summary.json ] || { echo "eval failed"; exit 1; }
grep -v WARNING $D/$ARM/test_eval_$T.log | grep "very_loose\|tight"
