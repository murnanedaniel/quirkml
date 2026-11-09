#!/bin/bash
#SBATCH -A m5386 -C gpu -q shared -N 1 --gpus 1 -c 32 -t 03:00:00 -J qtrk_pu60_lam_gpu
#SBATCH -o /pscratch/sd/d/danieltm/quirk_pu60/logs/lam_gpu_%j.out
# Other string scales at mu 60: graph v3 (mu 0 embedding + filter, cut 0.1) and pair_50k_qw40_s0 inference on test
# blocks 0-1 of each set in $SETS. Outputs under /pscratch/sd/d/danieltm/quirk_pu60/lambda/<set>/.
PY=/pscratch/sd/d/danieltm/envs/ml-tracking-review/bin/python
P=/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/pileup60
M=/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirk_tracking_data/models/gnn
S=/pscratch/sd/d/danieltm/quirk_pu60
for DS in $SETS; do
  W=$S/lambda/$DS; mkdir -p $W
  srun $PY $P/graph_apply60.py --cache $S/cache/${DS}_dir --emb $M/g1_qw10/emb --radius 0.075 --filter $M/g1f_qw10r075_q50/filter \
    --cut 0.1 --out $S/graphs/v3_mu0f_cut0.1_$DS --blocks 0 1 > $W/graphs.log 2>&1 || { echo "$DS graphs FAILED"; exit 1; }
  srun $PY $P/infer60.py --run $S/gnn/pair_50k_qw40_s0 --ckpt final.pt --cache $S/cache/${DS}_dir_pair \
    --graphs $S/graphs/v3_mu0f_cut0.1_$DS --blocks 0 1 --out $W/scored > $W/infer.log 2>&1 || { echo "$DS infer FAILED"; exit 1; }
  echo "$DS gpu done"
done
