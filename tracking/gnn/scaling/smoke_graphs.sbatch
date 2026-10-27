#!/bin/bash
#SBATCH -A m5386 -C gpu -q shared -n 1 -c 32 --gpus=1 -t 04:45:00 -J qtrk_scal_graphs
#SBATCH -o /pscratch/sd/d/danieltm/quirk_tracking/scaling/logs/smoke_graphs_%j.out
# 1) smoke test of the early-stopping trainer (600 steps, strict val on 100 events, patience 1); 2) graphs.sbatch body.
umask 007
O=/pscratch/sd/d/danieltm/quirk_tracking/gnn
S=/pscratch/sd/d/danieltm/quirk_tracking/scaling
PY=/pscratch/sd/d/danieltm/envs/ml-tracking-review/bin/python
srun -n1 -c 32 --gpus=1 $PY -u /global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/gnn/scaling/train_gnn_stream.py \
  --cache $S/cache/dir --graphs $S/graphs/v3 --out $S/runs/smoke2 --name scal_smoke2 --train-ranges 0-15999 \
  --val 16000 16199 --n-val 20 --steps 900 --interval 200 --save-every 100 --stop-metric perquirk --patience 1 \
  --track-val 16000 16099 --direction --quirk-weight 10 --warmup 100 --seed 0 > $S/logs/smoke2.log 2>&1
echo "smoke exit $?"
srun -n1 -c 32 --gpus=1 $PY -u /global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/gnn/train_gnn.py infer \
  --cache $S/cache/dir --graphs "$S/graphs/v3/events16???-*.parquet" --out $S/runs/smoke2 --ckpt best_sel.pt \
  --ranges 16000-16099 > $S/logs/smoke2_infer.log 2>&1
echo "smoke infer exit $?"
bash /global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/gnn/scaling/graphs.sbatch
