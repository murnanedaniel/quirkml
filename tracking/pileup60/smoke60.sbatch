#!/bin/bash
#SBATCH -A m5386 -C gpu&hbm80g -q shared -N 1 --gpus 1 -c 32 -t 00:30:00 -J qtrk_pu60_smoke
#SBATCH -o /pscratch/sd/d/danieltm/quirk_pu60/logs/smoke60_%j.out
# 60-step smoke tests of the prepared arms: (a) norm3 loss + whole-pair crops, (b) degree-aware aggregation (summean)
C=/pscratch/sd/d/danieltm/quirk_pu60/cache/m200_L774_mu60_dir_pair
G=/pscratch/sd/d/danieltm/quirk_pu60/graphs/v3_mu0f_cut0.1
PY=/pscratch/sd/d/danieltm/envs/ml-tracking-review/bin/python
cd /global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/gnn
COMMON="--cache $C --pair-all-true --stop-metric coupled --cc-cut 0.5 --graphs $G --train-ranges 2000-2999 --val 2000 2019 --n-val 20 --track-val 2000 2019 --steps 60 --interval 30 --batch 1 --lr 5e-4 --warmup 20 --direction --quirk-weight 10 --workers 4 --buffer 1 --max-edges 400000"
for arm in "norm3 --loss-mode norm3 --norm3-w 1 1 1 --crop-mode pair" "summean --agg summean"; do
  set -- $arm; T=$1; shift
  O=/pscratch/sd/d/danieltm/quirk_pu60/gnn/smoke_$T; mkdir -p $O
  srun $PY ../pileup60/train_gnn60.py $COMMON --out $O --name pu60_smoke_$T "$@" > $O/train.log 2>&1
  echo "$T exit $?"; grep -v Warn $O/train.log | grep "val_strict\|step_time\|Error" | tail -n 3 | cut -c1-300
  nvidia-smi --query-gpu=memory.used --format=csv
done
