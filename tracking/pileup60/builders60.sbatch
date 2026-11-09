#!/bin/bash
#SBATCH -A m4958 -C cpu -q debug -N 1 -c 256 -t 00:30:00 -J qtrk_pu60_builders
#SBATCH -o /pscratch/sd/d/danieltm/quirk_pu60/logs/builders_%j.out
G=/pscratch/sd/d/danieltm/quirk_pu60/gnn
SPECS="${SPECS:-cc:0.5 cc:0.6 cc:0.7 tt:0.6:0.3 tt:0.7:0.3 br:0.4:0.8 br:0.5:0.7 br:0.5:0.6 br:0.6:0.8 br:0.5:0.8}"
for R in ${RUNS:-pair_16k_s0 pair_16k_s1 pair_50k_b30k_s0}; do
  echo "== $R"
  /pscratch/sd/d/danieltm/envs/ml-tracking-review/bin/python /global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/pileup60/builders60.py \
    $G/$R $G/$R/val/builders60${TAGJ:-}.json $SPECS
done
