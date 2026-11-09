#!/bin/bash
# mu 60 GNN arm on graph v3 (mu 0 filter, cut 0.1), one GPU of allocation J, resumable across windows (state.pt,
# deadline 10 min before the allocation ends). usage: gnn_arm.sh <jobid> <pair|quirk> <seed> <steps> [extra args]
set -uo pipefail
J=$1; W=$2; SEED=$3; N=$4; shift 4
C=/pscratch/sd/d/danieltm/quirk_pu60/cache/m200_L774_mu60_dir
G=/pscratch/sd/d/danieltm/quirk_pu60/graphs/v3_mu0f_cut0.1
PY=/pscratch/sd/d/danieltm/envs/ml-tracking-review/bin/python
U="env $(env | grep -o '^SLURM_[A-Z_]*' | sed 's/^/-u /' | tr '\n' ' ')"
END=$(date -d "$(squeue -h -j $J -o %e)" +%s); DL=$((END - 600))
if [ $W = pair ]; then A="--cache ${C}_pair --pair-all-true --stop-metric coupled --cc-cut 0.5"; else A="--cache $C --stop-metric perquirk"; fi
O=/pscratch/sd/d/danieltm/quirk_pu60/gnn/${W}_16k_s$SEED
mkdir -p $O
cd /global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/gnn
$U srun --jobid=$J --exact -n 1 -c 32 --gpus=1 $PY ../pileup60/train_gnn60.py $A --graphs $G --out $O --name pu60_${W}_16k_s$SEED \
  --train-ranges 3000-18999 --val 2000 2099 --n-val 100 --track-val 2000 2199 --steps $N --interval 4000 --batch 1 --lr 5e-4 \
  --warmup 1000 --direction --quirk-weight 10 --workers 24 --buffer 1 --max-edges 400000 --seed $SEED --deadline $DL "$@" >> $O/train.log 2>&1
rc=$?; echo "exit $rc $(date)"; exit $rc
