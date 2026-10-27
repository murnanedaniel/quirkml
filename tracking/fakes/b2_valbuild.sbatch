#!/bin/bash
#SBATCH -A m4958 -C cpu -q debug -N 1 -t 00:30:00 -J qtrk_fakes_valbuild
#SBATCH -o /pscratch/sd/d/danieltm/quirk_tracking/fakes/logs/valbuild_%j.out
# B2 prep: GNN tracks on val 16000-16999 with the test settings (existing scored edges, no inference):
# per-quirk v3 GNN seed 0 walk + stitch 0.7 >= 4 hits; pair GNN seed 0 components 0.5 >= 4 hits.
umask 007
PY=/pscratch/sd/d/danieltm/envs/ml-tracking-review/bin/python
G=/pscratch/sd/d/danieltm/quirk_tracking/gnn
O=/pscratch/sd/d/danieltm/quirk_tracking/fakes/val_preds
cd /global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/gnn
srun -n1 -c256 $PY build_tracks.py --cache $G/cache/m200_L774_mu0 --scored "$G/main/g2_q50_dir_s0/gnn/scored/events16*.parquet" \
  --events 16000 16999 --method ccwalk_stitch --score-cut-cc 0.01 --th-min 0.1 --th-add 0.6 --th-stitch 0.7 --min-hits 4 \
  --workers 64 --out $O/g2_s0_walk --run-id g2_q50_dir_s0_val_stitch0.7_mh4
srun -n1 -c256 $PY build_tracks.py --cache $G/cache/m200_L774_mu0 --scored "$G/main/e2_pair_dir_s0/gnn/scored/events16*.parquet" \
  --events 16000 16999 --method cc --cc-cut 0.5 --min-hits 4 --workers 64 --out $O/e2_s0_cc --run-id e2_pair_dir_s0_val_cc0.5_mh4
ls -la $O/*/
