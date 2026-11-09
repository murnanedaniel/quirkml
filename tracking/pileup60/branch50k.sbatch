#!/bin/bash
#SBATCH -A m5386 -C gpu&hbm80g -q shared -N 1 --gpus 1 -c 32 -t 02:00:00 -J qtrk_pu60_branch50k
#SBATCH -o /pscratch/sd/d/danieltm/quirk_pu60/logs/branch50k_%j.out
# 30k-step decay branch of the 50k pair trunk: start from the trunk's state at step 24000, linear decay 24k-30k
# (the same schedule as the 16k arm), then test-set inference (blocks 0, 1) with final.pt.
C=/pscratch/sd/d/danieltm/quirk_pu60/cache/m200_L774_mu60_dir_pair
G=/pscratch/sd/d/danieltm/quirk_pu60/graphs/v3_mu0f_cut0.1
T=/pscratch/sd/d/danieltm/quirk_pu60/gnn/pair_50k_trunk_s0
O=/pscratch/sd/d/danieltm/quirk_pu60/gnn/pair_50k_b30k_s0
PY=/pscratch/sd/d/danieltm/envs/ml-tracking-review/bin/python
mkdir -p $O
END=$(date -d "$(squeue -h -j $SLURM_JOB_ID -o %e)" +%s); DL=$((END - 600))
cd /global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/gnn
srun $PY ../pileup60/train_gnn60.py --cache $C --pair-all-true --stop-metric coupled --cc-cut 0.5 --graphs $G --out $O \
  --name pu60_pair_50k_b30k_s0 --train-ranges 3000-52999 --val 2000 2099 --n-val 100 --track-val 2000 2199 --steps 30000 \
  --decay-start 24000 --branch-from $T/state_step24000.pt --interval 2000 --batch 1 --lr 5e-4 --warmup 1000 --direction \
  --quirk-weight 10 --workers 6 --buffer 1 --max-edges 400000 --seed 0 --deadline $DL >> $O/train.log 2>&1 || exit 1
srun $PY ../pileup60/infer60.py --run $O --ckpt final.pt --cache $C --graphs $G --blocks 0 1 2 --out $O/scored >> $O/infer.log 2>&1
echo "exit $?"
