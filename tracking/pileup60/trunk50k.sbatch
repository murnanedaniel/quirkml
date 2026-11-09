#!/bin/bash
#SBATCH -A m5386 -C gpu&hbm80g -q shared -N 1 --gpus 1 -c 32 -t 08:00:00 -J qtrk_pu60_trunk50k
#SBATCH -o /pscratch/sd/d/danieltm/quirk_pu60/logs/trunk50k_%j.out
# Pair GNN 50k-event trunk (train 3000-52999): same recipe as the 16k arm; the trunk itself decays over 48k-60k
# (the 60k-step branch); the state at step 24000 is kept for the 30k-step branch (decay 24k-30k, compute-matched to
# the 16k arm). Resumable (state.pt); deadline 10 min before the job's end.
C=/pscratch/sd/d/danieltm/quirk_pu60/cache/m200_L774_mu60_dir_pair
G=/pscratch/sd/d/danieltm/quirk_pu60/graphs/v3_mu0f_cut0.1
O=/pscratch/sd/d/danieltm/quirk_pu60/gnn/pair_50k_trunk_s0
mkdir -p $O
END=$(date -d "$(squeue -h -j $SLURM_JOB_ID -o %e)" +%s); DL=$((END - 600))
cd /global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/gnn
srun /pscratch/sd/d/danieltm/envs/ml-tracking-review/bin/python ../pileup60/train_gnn60.py --cache $C --pair-all-true --stop-metric coupled --cc-cut 0.5 \
  --graphs $G --out $O --name pu60_pair_50k_trunk_s0 --train-ranges 3000-52999 --val 2000 2099 --n-val 100 --track-val 2000 2199 \
  --steps 60000 --decay-start 48000 --keep-states 24000 --interval 4000 --batch 1 --lr 5e-4 --warmup 1000 --direction --quirk-weight 10 \
  --workers 6 --buffer 1 --max-edges 400000 --seed 0 --deadline $DL >> $O/train.log 2>&1
echo "exit $?"
