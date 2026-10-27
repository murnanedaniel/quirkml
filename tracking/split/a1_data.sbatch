#!/bin/bash
#SBATCH -A m4958 -C cpu -q debug -N 1 -t 00:30:00 -J qtrk_a1data
#SBATCH -o /pscratch/sd/d/danieltm/quirk_tracking/split/logs/a1data_%j.out
PY=/pscratch/sd/d/danieltm/envs/ml-tracking-review/bin/python
S=/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/split/split_data.py
D=/pscratch/sd/d/danieltm/quirk_tracking/split/data
$PY $S --mode oracle --events 0 15999 --out $D/oracle_train.pkl &
$PY $S --mode oracle --events 16000 16999 --out $D/oracle_val.pkl &
$PY $S --mode oracle --events 17000 19999 --out $D/oracle_test.pkl &
wait
ls -la $D
