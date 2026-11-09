#!/bin/bash
#SBATCH -A m4958 -C cpu -q debug -N 1 -c 256 -t 00:30:00 -J qtrk_pu60_pyscore
#SBATCH -o /pscratch/sd/d/danieltm/quirk_pu60/logs/pyscore_%j.out
# Python strict scoring (mu60_eval/py_score.py, ACTS rules) of the zero-shot test predictions
PY=/pscratch/sd/d/danieltm/envs/ml-tracking-review/bin/python
S=/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/mu60_eval/py_score.py
O=/pscratch/sd/d/danieltm/quirk_pu60/zs/m200_L774_mu60
Q=/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirk_tracking_data
R1=$Q/parquet_r1_mu60/m200_L774_mu60; PT=$Q/pair_truth_mu60/m200_L774_mu60
$PY $S $O/v3/test/pred_stitch0.7_mh4 $R1 0 2000 $O/v3/test/py_stitch0.7_mh4.json &
$PY $S $O/pair/test/pred_cc0.5_mh4 $PT 0 2000 $O/pair/test/pypair_cc0.5_mh4.json &
$PY $S $O/pair/test/pred_cc0.5_mh4 $R1 0 2000 $O/pair/test/py_cc0.5_mh4.json &
wait
for f in $O/v3/test/py_stitch0.7_mh4.json $O/pair/test/pypair_cc0.5_mh4.json $O/pair/test/py_cc0.5_mh4.json; do echo == $f; cat $f | head -c 3000; echo; done
