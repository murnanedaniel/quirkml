#!/bin/bash
#SBATCH -A m4958 -C cpu -q debug -N 1 -c 256 -t 00:30:00 -J qtrk_pu60_edgeval
#SBATCH -o /pscratch/sd/d/danieltm/quirk_pu60/logs/edgeval_%j.out
export EDGE_BLOCKS=2
for R in pair_16k_s0 pair_16k_s1 pair_50k_b30k_s0; do
  D=/pscratch/sd/d/danieltm/quirk_pu60/gnn/$R
  /pscratch/sd/d/danieltm/envs/ml-tracking-review/bin/python /global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/pileup60/edge60.py \
    $D $D/val/edge60_val.json 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 > $D/val/edge60_val.log 2>&1 &
done
wait; echo done
