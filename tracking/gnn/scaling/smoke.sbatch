#!/bin/bash
#SBATCH -A m5386 -C gpu -q shared -n 1 -c 32 --gpus=1 -t 00:20:00 -J qtrk_scal_smoke
#SBATCH -o /pscratch/sd/d/danieltm/quirk_tracking/scaling/logs/smoke_%j.out
umask 007
O=/pscratch/sd/d/danieltm/quirk_tracking/gnn
PY=/pscratch/sd/d/danieltm/envs/ml-tracking-review/bin/python
cd /global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/gnn/scaling
/usr/bin/time -v srun -n1 -c 32 --gpus=1 $PY -u train_gnn_stream.py --cache $O/cache/m200_L774_mu0_dir \
  --graphs $O/main/g1f_qw10r075_q50/filter/graphs_cut0.001 --out /pscratch/sd/d/danieltm/quirk_tracking/scaling/runs/smoke \
  --name scal_smoke --train-ranges 0-15999 --val 16000 16199 --n-val 20 --steps 300 --interval 150 --save-every 100 \
  --direction --quirk-weight 10 --warmup 100 --seed 0
echo "exit $?"
