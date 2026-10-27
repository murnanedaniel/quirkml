#!/bin/bash
#SBATCH -A m4958 -C cpu -q debug -N 1 -t 00:20:00 -J qtrk_perfcc
#SBATCH -o /pscratch/sd/d/danieltm/quirk_tracking/gnn/logs/perfect_cc_%j.out
# Per-quirk perfect scores on graph v3 with plain connected components (cut 0.5), >= 4 hits; ACTS normal truth.
O=/pscratch/sd/d/danieltm/quirk_tracking/gnn
P=$O/main/g1f_qw10r075_q50/perfect
cd /global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/gnn
/pscratch/sd/d/danieltm/envs/ml-tracking-review/bin/python build_tracks.py --cache $O/cache/m200_L774_mu0 --scored "$P/scored/*.parquet" \
  --events 17000 19999 --out $P/pred_cc_mh4 --workers 64 --method cc --cc-cut 0.5 --min-hits 4 > /dev/null 2>&1
/global/cfs/cdirs/m4958/usr/danieltm/ml_tracking_pilot_2026-09/envs/eval48/bin/python ../evaluate.py --predictions-dir $P/pred_cc_mh4 \
  --dataset /global/cfs/cdirs/m4958/data/ColliderML/simulation/quirks/m200_L774_mu0/v1/parquet_r1 --events 17000 20000 \
  --out $P/eval_cc_mh4 --jobs 60 > $P/eval_cc_mh4.log 2>&1
grep -v WARNING $P/eval_cc_mh4.log | grep "very_loose\|tight \|proto"
