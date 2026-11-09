#!/bin/bash
# After pair seed S finishes on allocation J: GNN inference on test+val blocks (0, 1, 2), CPU test job, then the
# per-quirk GNN seed S (30k steps, resumable) on the freed GPU. usage: after_pair.sh <jobid> <seed>
set -uo pipefail
J=$1; S=$2
R=/pscratch/sd/d/danieltm/quirk_pu60/gnn/pair_16k_s$S
P=/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/pileup60
PY=/pscratch/sd/d/danieltm/envs/ml-tracking-review/bin/python
U="env $(env | grep -o '^SLURM_[A-Z_]*' | sed 's/^/-u /' | tr '\n' ' ')"
until [ -f $R/final.pt ]; do
  pgrep -f "gnn_arm.sh $J pair $S" > /dev/null || { [ -f $R/final.pt ] || { echo "pair seed $S driver gone without final.pt"; exit 1; }; }
  sleep 60
done
$U srun --jobid=$J --exact -n 1 -c 32 --gpus=1 $PY $P/infer60.py --run $R --ckpt final.pt --cache /pscratch/sd/d/danieltm/quirk_pu60/cache/m200_L774_mu60_dir_pair \
  --graphs /pscratch/sd/d/danieltm/quirk_pu60/graphs/v3_mu0f_cut0.1 --blocks 0 1 2 --out $R/scored > $R/infer.log 2>&1 || { echo "infer failed"; exit 1; }
env -u SLURM_MEM_PER_CPU -u SLURM_JOB_ID -u SLURM_JOBID sbatch -J qtrk_pu60_test_p$S $P/test_cpu.sbatch $R pair
bash $P/gnn_arm.sh $J quirk $S 30000
