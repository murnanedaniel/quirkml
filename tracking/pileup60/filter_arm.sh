#!/bin/bash
# mu 60 filter retraining arm: smoke (20 steps) then full run, one GPU of allocation J (srun --exact).
# usage: filter_arm.sh <jobid> <pair|v3> <steps>
set -uo pipefail
J=$1; W=$2; N=$3
M=/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirk_tracking_data/models/gnn
C=/pscratch/sd/d/danieltm/quirk_pu60/cache/m200_L774_mu60_dir
PY=/pscratch/sd/d/danieltm/envs/ml-tracking-review/bin/python
S=/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/pileup60/train_filter_stream.py
U="env $(env | grep -o '^SLURM_[A-Z_]*' | sed 's/^/-u /' | tr '\n' ' ')"
R="$U srun --jobid=$J --exact -n 1 -c 32 --gpus=1"
if [ $W = pair ]; then A="--cache ${C}_pair --emb $M/v1/emb --radius 0.07 --init $M/e2_pair_dir_s0/filter --quirk-weight 10"; O=/pscratch/sd/d/danieltm/quirk_pu60/filter/f60_pair
else A="--cache $C --emb $M/g1_qw10/emb --radius 0.075 --init $M/g1f_qw10r075_q50/filter --quirk-weight 50"; O=/pscratch/sd/d/danieltm/quirk_pu60/filter/f60_v3; fi
mkdir -p $O
[ -f $O/smoke/final.pt ] || $R $PY $S $A --steps 20 --val-every 10 --val 2000 2002 --out $O/smoke --name f60_${W}_smoke > $O/smoke.log 2>&1 || { echo "smoke FAILED"; exit 1; }
[ -f $O/final.pt ] || $R $PY $S $A --steps $N --out $O --name f60_$W > $O/train.log 2>&1 || { echo "train FAILED"; exit 1; }
echo done
