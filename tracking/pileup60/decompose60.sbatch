#!/bin/bash
#SBATCH -A m4958 -C cpu -q debug -N 1 -c 256 -t 00:30:00 -J qtrk_pu60_b1
#SBATCH -o /pscratch/sd/d/danieltm/quirk_pu60/logs/b1_%j.out
G=/pscratch/sd/d/danieltm/quirk_pu60/gnn; Z=/pscratch/sd/d/danieltm/quirk_pu60/zs/m200_L774_mu60
/global/cfs/cdirs/m4958/usr/danieltm/ml_tracking_pilot_2026-09/envs/eval48/bin/python /global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/pileup60/decompose60.py \
  /pscratch/sd/d/danieltm/quirk_pu60/fakes_b1 \
  pair60_s0_cc=$G/pair_16k_s0/test/pred_cc0.5_mh4:$G/pair_16k_s0/test/py_cc0.5_mh4.json \
  pair_zs_cc=$Z/pair/test/pred_cc0.5_mh4:$Z/pair/test/py_cc0.5_mh4.json \
  v3_zs_walk=$Z/v3/test/pred_stitch0.7_mh4:$Z/v3/test/py_stitch0.7_mh4.json
