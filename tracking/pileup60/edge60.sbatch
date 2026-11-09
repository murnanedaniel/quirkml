#!/bin/bash
#SBATCH -A m4958 -C cpu -q regular -N 1 -c 256 -t 01:00:00 -J qtrk_pu60_edge
#SBATCH -o /pscratch/sd/d/danieltm/quirk_pu60/logs/edge_%j.out
PY=/pscratch/sd/d/danieltm/envs/ml-tracking-review/bin/python
for s in 0 1; do
  R=/pscratch/sd/d/danieltm/quirk_pu60/gnn/pair_16k_s$s
  $PY /global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/pileup60/edge60.py $R $R/test/edge60.json 0.5 0.6 > $R/test/edge60.log 2>&1 &
done
wait; cat /pscratch/sd/d/danieltm/quirk_pu60/gnn/pair_16k_s0/test/edge60.json
