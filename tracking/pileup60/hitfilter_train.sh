#!/bin/bash
# Hit filter: smoke (200 steps) then full training on mu 60 train blocks 3000-12999; then val curve (2000-2999).
# usage: hitfilter_train.sh <jobid> [tag]   (srun's one GPU of the allocation)
set -uo pipefail
J=$1; TAG=${2:-hf_v1}
C=/pscratch/sd/d/danieltm/quirk_pu60/cache/m200_L774_mu60_dir
O=/pscratch/sd/d/danieltm/quirk_pu60/hitfilter/$TAG
PY=/pscratch/sd/d/danieltm/envs/ml-tracking-review/bin/python
S=/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/pileup60/hitfilter.py
U="env $(env | grep -o '^SLURM_[A-Z_]*' | sed 's/^/-u /' | tr '\n' ' ')"
R="$U srun --jobid=$J --exact -n 1 -c 32 --gpus=1"
mkdir -p $O
if [ ! -f $O/smoke/final.pt ]; then
  $R $PY $S train --cache $C --out $O/smoke --steps 200 --val-every 100 --val 2000 2019 --name ${TAG}_smoke > $O/smoke.log 2>&1 || { echo "smoke FAILED"; exit 1; }
fi
if [ ! -f $O/final.pt ]; then
  $R $PY $S train --cache $C --out $O --steps ${STEPS:-20000} --val 2000 2099 --name $TAG ${EXTRA:-} > $O/train.log 2>&1 || { echo "train FAILED"; exit 1; }
fi
$R $PY $S curve --cache $C --out $O --events 2000 2999 > $O/curve.log 2>&1 || { echo "curve FAILED"; exit 1; }
echo done
