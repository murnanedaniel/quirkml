#!/bin/bash
#SBATCH -A m4958 -C cpu -q regular -N 1 -c 256 -t 01:00:00 -J qtrk_pu60_lam_ref774
#SBATCH -o /pscratch/sd/d/danieltm/quirk_pu60/logs/lam_ref774_%j.out
# L774 reference for the string-scale table, rescored with the current py_score.py (unassociated class, all-tracks fake
# denominator) and lam_apply60.py (exact L774 val-B cuts). Existing qw40 test tracks and chain output; no new test use.
PY=/pscratch/sd/d/danieltm/envs/ml-tracking-review/bin/python
T=/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking
Q=/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirk_tracking_data
DS=m200_L774_mu60; R=/pscratch/sd/d/danieltm/quirk_pu60/gnn/pair_50k_qw40_s0; CH=/pscratch/sd/d/danieltm/quirk_pu60/chain/qw40_cc0.5
W=/pscratch/sd/d/danieltm/quirk_pu60/lambda/$DS; mkdir -p $W/test $W/chain
P=$R/test/pred_cc0.5_mh4
$PY $T/mu60_eval/py_score.py $P $Q/pair_truth_mu60/$DS 0 2000 $W/test/pypair_cc0.5_mh4.json &
$PY $T/mu60_eval/py_score.py $P $Q/parquet_r1_mu60/$DS 0 2000 $W/test/py_cc0.5_mh4.json &
$PY $T/mu60_eval/py_score.py $CH/pred_split $Q/pair_truth_mu60/$DS 0 2000 $W/chain/py_split_pair.json &
$PY $T/mu60_eval/py_score.py $CH/pred_split $Q/parquet_r1_mu60/$DS 0 2000 $W/chain/py_split.json &
$PY $T/pileup60/lam_apply60.py $DS qw40_cc0.5 $P $R/scored $W/apply $Q/models/tagger/b3_pair60_50k_qw40_cc0.5 \
  $Q/models/quality_pu60/pair60_50k_qw40_cc0.5 > $W/apply.log 2>&1 &
wait
D=$Q/results/pu60/lambda/$DS; mkdir -p $D; cp $W/test/*.json $D/; cp $W/chain/py_split*.json $D/; cp $W/apply/qw40_cc0.5/{apply.json,roc_test.csv} $D/
ls $D
