#!/bin/bash
#SBATCH -A m5386 -C gpu&hbm80g -q shared -N 1 --gpus 1 -c 32 -t 12:00:00
#SBATCH -o /pscratch/sd/d/danieltm/quirk_pu60/logs/arm60_%x_%j.out
# mu 60 pair GNN arm (trunk recipe: warmup 1000, constant LR 5e-4, linear decay over the last 20%, no early stopping,
# final checkpoint), then val-block inference (2000-2999) only. Resumable: resubmitting continues from state.pt.
# env: NAME, TRAIN (e.g. 3000-52999), STEPS (60000), SEED, EXTRA (extra train_gnn60.py args)
C=/pscratch/sd/d/danieltm/quirk_pu60/cache/m200_L774_mu60_dir_pair
G=/pscratch/sd/d/danieltm/quirk_pu60/graphs/v3_mu0f_cut0.1
O=/pscratch/sd/d/danieltm/quirk_pu60/gnn/$NAME
PY=/pscratch/sd/d/danieltm/envs/ml-tracking-review/bin/python
STEPS=${STEPS:-60000}; DS=$((STEPS * 8 / 10))
mkdir -p $O
END=$(date -d "$(squeue -h -j $SLURM_JOB_ID -o %e)" +%s); DL=$((END - 1800))
cd /global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/gnn
srun $PY ../pileup60/train_gnn60.py --cache $C --pair-all-true --stop-metric coupled --cc-cut 0.5 --graphs $G --out $O --name pu60_$NAME \
  --train-ranges $TRAIN --val 2000 2099 --n-val 100 --track-val 2000 2199 --steps $STEPS --decay-start $DS --interval 4000 --batch 1 \
  --lr 5e-4 --warmup 1000 --direction --workers 6 --buffer 1 --max-edges 400000 --seed ${SEED:-0} --deadline $DL $EXTRA >> $O/train.log 2>&1
rc=$?; echo "train exit $rc"
[ -f $O/final.pt ] || { echo "no final.pt (resubmit to resume)"; exit 3; }
srun $PY ../pileup60/infer60.py --run $O --ckpt final.pt --cache $C --graphs $G --blocks 2 --out $O/scored >> $O/infer.log 2>&1
echo "infer exit $?"
