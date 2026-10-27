#!/bin/bash
#SBATCH -A m4958 -C cpu -q debug -N 1 -t 00:30:00 -J qtrk_tuneb
#SBATCH -o /pscratch/sd/d/danieltm/quirk_tracking/gnn/logs/tune_builders_%j.out
O=/pscratch/sd/d/danieltm/quirk_tracking/gnn
PY=/pscratch/sd/d/danieltm/envs/ml-tracking-review/bin/python
cd /global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/gnn
echo "== pair GNN s0 (pair cache)"
$PY tune_builders.py --cache $O/cache/m200_L774_mu0_dir_pair --scored "$O/main/e2_pair_dir_s0/gnn/scored/events16000-16999.parquet" &
P1=$!
echo "== v3 GNN s0 (per-quirk)" > $O/logs/tune_b_g2.txt
$PY tune_builders.py --cache $O/cache/m200_L774_mu0 --scored "$O/main/g2_q50_dir_s0/gnn/scored/events16000-16999.parquet" >> $O/logs/tune_b_g2.txt
wait $P1
cat $O/logs/tune_b_g2.txt
