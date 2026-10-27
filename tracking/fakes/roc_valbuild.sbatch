#!/bin/bash
#SBATCH -A m4958 -C cpu -q debug -N 1 -t 00:30:00 -J qtrk_fakes_valbuild2
#SBATCH -o /pscratch/sd/d/danieltm/quirk_tracking/fakes/logs/valbuild2_%j.out
# Val 16000-16999 tracks from existing val edge scores, test settings: pair GNN s1/s2 components 0.5 >= 4 hits;
# pair GNN on graph v3 (p3) components 0.7 >= 4 hits.
umask 007
PY=/pscratch/sd/d/danieltm/envs/ml-tracking-review/bin/python
G=/pscratch/sd/d/danieltm/quirk_tracking/gnn
O=/pscratch/sd/d/danieltm/quirk_tracking/fakes/val_preds
cd /global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/gnn
for s in 1 2; do
  srun -n1 -c256 $PY build_tracks.py --cache $G/cache/m200_L774_mu0 --scored "$G/main/e2_pair_dir_s$s/gnn/scored/events16*.parquet" \
    --events 16000 16999 --method cc --cc-cut 0.5 --min-hits 4 --workers 64 --out $O/e2_s${s}_cc --run-id e2_pair_dir_s${s}_val_cc0.5_mh4
done
srun -n1 -c256 $PY build_tracks.py --cache $G/cache/m200_L774_mu0 --scored "$G/main/p3_pairv3_s0/gnn/scored/events16*.parquet" \
  --events 16000 16999 --method cc --cc-cut 0.7 --min-hits 4 --workers 64 --out $O/p3_cc07 --run-id p3_pairv3_s0_val_cc0.7_mh4
ls -la $O/*/
