#!/bin/bash
#SBATCH -A m5386 -C gpu&hbm80g -q shared -N 1 --gpus 1 -c 32 -t 04:30:00 -J qtrk_pu60_pairqw
#SBATCH -o /pscratch/sd/d/danieltm/quirk_pu60/logs/pairqw_%j.out
# Pair GNN 16k, seed 0, quirk-edge weight $QW (restores the mu 0 share of quirk-touching edges in the training loss,
# computed from edge counts on the training crops), otherwise identical to pair_16k_s0; then test-set inference.
QW=${QW:-40}
C=/pscratch/sd/d/danieltm/quirk_pu60/cache/m200_L774_mu60_dir_pair
G=/pscratch/sd/d/danieltm/quirk_pu60/graphs/v3_mu0f_cut0.1
O=/pscratch/sd/d/danieltm/quirk_pu60/gnn/pair_16k_qw${QW}_s0
PY=/pscratch/sd/d/danieltm/envs/ml-tracking-review/bin/python
mkdir -p $O
END=$(date -d "$(squeue -h -j $SLURM_JOB_ID -o %e)" +%s); DL=$((END - 1500))
cd /global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/gnn
srun $PY ../pileup60/train_gnn60.py --cache $C --pair-all-true --stop-metric coupled --cc-cut 0.5 --graphs $G --out $O \
  --name pu60_pair_16k_qw${QW}_s0 --train-ranges 3000-18999 --val 2000 2099 --n-val 100 --track-val 2000 2199 --steps 30000 \
  --interval 4000 --batch 1 --lr 5e-4 --warmup 1000 --direction --quirk-weight $QW --workers 6 --buffer 1 --max-edges 400000 \
  --seed 0 --deadline $DL >> $O/train.log 2>&1 || exit 1
[ -f $O/final.pt ] && srun $PY ../pileup60/infer60.py --run $O --ckpt final.pt --cache $C --graphs $G --blocks 0 1 2 --out $O/scored >> $O/infer.log 2>&1
echo "exit $?"
