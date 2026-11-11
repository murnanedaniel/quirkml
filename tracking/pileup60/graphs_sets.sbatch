#!/bin/bash
#SBATCH -A m5386 -C gpu -q shared -N 1 --gpus 1 -c 32 -t 04:00:00 -J qtrk_pu60_graphs_sets
#SBATCH -o /pscratch/sd/d/danieltm/quirk_pu60/logs/graphs_sets_%j.out
# graph v3 (mu 0 embedding + filter, r 0.075, cut 0.1) for env JOBS="SET:BLOCK ..." (L774 into graphs/v3_mu0f_cut0.1,
# other sets into graphs/v3_mu0f_cut0.1_<set>); blocks already written are skipped by graph_apply60.py
M=/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirk_tracking_data/models/gnn; S=/pscratch/sd/d/danieltm/quirk_pu60
for j in $JOBS; do DS=${j%%:*}; B=${j#*:}
  G=$S/graphs/v3_mu0f_cut0.1_$DS; [ $DS = m200_L774_mu60 ] && G=$S/graphs/v3_mu0f_cut0.1; mkdir -p $G/logs
  srun /pscratch/sd/d/danieltm/envs/ml-tracking-review/bin/python /global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/pileup60/graph_apply60.py \
    --cache $S/cache/${DS}_dir --emb $M/g1_qw10/emb --radius 0.075 --filter $M/g1f_qw10r075_q50/filter --cut 0.1 --out $G --blocks $B \
    >> $G/logs/apply_$B.log 2>&1 || { echo "$j FAILED"; exit 1; }
  echo "$j done $(date)"
done
