#!/bin/bash
#SBATCH -A m5386 -C gpu -q shared -N 1 --gpus 1 -c 32 -t 04:00:00 -J qtrk_pu60_retro
#SBATCH -o /pscratch/sd/d/danieltm/quirk_pu60/logs/retro_%j.out
# per-class loss terms of RUN's checkpoints on 1000 seen training events, 1000 val events and 1000 unseen
# training-range events (60000-60999), one split at a time (host memory)
D=/pscratch/sd/d/danieltm/quirk_pu60/classlog; mkdir -p $D
for sp in unseen val train; do
  srun /pscratch/sd/d/danieltm/envs/ml-tracking-review/bin/python /global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/pileup60/class_loss_retro.py \
    /pscratch/sd/d/danieltm/quirk_pu60/gnn/$RUN $D/${RUN}_u_$sp.json --splits $sp --per-block 250 --val 2000 2999 --unseen 60000 60999 \
    --steps 4000 20000 36000 52000 60000 >> $D/${RUN}_u.log 2>&1 || { echo "$sp FAILED"; exit 1; }
done
