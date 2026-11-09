#!/bin/bash
#SBATCH -A m4958 -C cpu -q debug -N 1 -c 256 -t 00:30:00 -J qtrk_pu60_ccscan
#SBATCH -o /pscratch/sd/d/danieltm/quirk_pu60/logs/ccscan_%j.out
# Components cut scan on mu 60 val (2000-2999) for the pair GNN seeds: build (>= 4 hits) + Python strict, coupled (pair
# truth) loose/tight and fake. usage: sbatch cc_scan.sbatch
PY=/pscratch/sd/d/danieltm/envs/ml-tracking-review/bin/python
T=/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking
C=/pscratch/sd/d/danieltm/quirk_pu60/cache/m200_L774_mu60_dir_pair
PT=/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirk_tracking_data/pair_truth_mu60/m200_L774_mu60
cd $T/gnn
for R in ${RUNS:-/pscratch/sd/d/danieltm/quirk_pu60/gnn/pair_16k_s0 /pscratch/sd/d/danieltm/quirk_pu60/gnn/pair_16k_s1}; do

  for c in 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
    ( $PY build_tracks.py --cache $C --scored "$R/scored/events2000-2999.parquet" --events 2000 2999 --out $R/val/pred_cc${c}_mh4 \
        --workers 16 --method cc --cc-cut $c --min-hits 4 > /dev/null 2>&1 &&
      $PY $T/mu60_eval/py_score.py $R/val/pred_cc${c}_mh4 $PT 2000 3000 $R/val/pypair_cc${c}_mh4.json > /dev/null 2>&1 ) &
  done
done
wait
for R in ${RUNS:-/pscratch/sd/d/danieltm/quirk_pu60/gnn/pair_16k_s0 /pscratch/sd/d/danieltm/quirk_pu60/gnn/pair_16k_s1}; do for c in 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do $PY -c "
import json; r=json.load(open('$R/val/pypair_cc${c}_mh4.json'))
a,b=r['very_loose_strict'],r['tight_strict']
print('$(basename $R) cut $c', 'coupled %.3f / %.3f'%(a['quirk']['eff'],b['quirk']['eff']), 'fake %.3f'%a['tracks']['fake_rate'], 'tech_sm %.3f'%a['technical_sm']['eff'], 'tracks/event %.0f'%(a['tracks']['tracks']/1000))"; done; done
