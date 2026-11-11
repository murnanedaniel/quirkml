#!/bin/bash
#SBATCH -A m5386 -C gpu -q shared -N 1 --gpus 1 -c 32 -t 02:00:00 -J qtrk_pu60_infer_sets
#SBATCH -o /pscratch/sd/d/danieltm/quirk_pu60/logs/infer_sets_%j.out
# pair-GNN inference (final.pt) for env JOBS="RUN_NAME:SET:BLOCK:OUT_DIR ..." (OUT_DIR gets scored/events<block>.parquet)
S=/pscratch/sd/d/danieltm/quirk_pu60
for j in $JOBS; do IFS=: read R DS B O <<< "$j"
  G=$S/graphs/v3_mu0f_cut0.1_$DS; [ $DS = m200_L774_mu60 ] && G=$S/graphs/v3_mu0f_cut0.1; mkdir -p $O
  srun /pscratch/sd/d/danieltm/envs/ml-tracking-review/bin/python /global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/pileup60/infer60.py \
    --run $S/gnn/$R --ckpt final.pt --cache $S/cache/${DS}_dir_pair --graphs $G --blocks $B --out $O/scored >> $O/infer.log 2>&1 || { echo "$j FAILED"; exit 1; }
  echo "$j done $(date)"
done
