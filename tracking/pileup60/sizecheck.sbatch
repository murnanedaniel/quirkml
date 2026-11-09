#!/bin/bash
#SBATCH -A m5386 -C gpu -q shared -N 1 --gpus 1 -c 32 -t 01:00:00 -J qtrk_pu60_size
#SBATCH -o /pscratch/sd/d/danieltm/quirk_pu60/logs/size_%j.out
PY=/pscratch/sd/d/danieltm/envs/ml-tracking-review/bin/python
S=/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/pileup60/sizecheck.py
O=/pscratch/sd/d/danieltm/quirk_pu60/size; mkdir -p $O
C=/pscratch/sd/d/danieltm/quirk_tracking/gnn/cache
srun $PY $S --cache-dir $C/m200_L774_mu0_dir --cache-pair $C/m200_L774_mu0_dir_pair --events 17000 17019 --out $O/size_mu0.json
srun $PY $S --cache-dir /pscratch/sd/d/danieltm/quirk_pu60/cache/m200_L774_mu60_dir --cache-pair /pscratch/sd/d/danieltm/quirk_pu60/cache/m200_L774_mu60_dir_pair --events 0 49 --out $O/size_mu60.json
