#!/bin/bash
#SBATCH -A m4958 -C cpu -q debug -N 1 -t 00:30:00 -J qtrk_edge_eval
#SBATCH -o /pscratch/sd/d/danieltm/quirk_tracking/gnn/logs/edge_eval_%j.out
# Edge-level efficiency / purity of the quirk GNNs from saved scored graphs (edge_eval.py). CPU only, read-only on data.
PY=/pscratch/sd/d/danieltm/envs/ml-tracking-review/bin/python
OUT=/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirk_tracking_data/results/edge_eval
cd /global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/gnn
srun -n 1 -c 256 $PY edge_eval.py run --out $OUT --workers 28
srun -n 1 -c 256 $PY edge_eval.py summarize --out $OUT
