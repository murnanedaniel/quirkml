#!/bin/bash
# Compact filtered graphs (graph_apply60.py) for mu 60 blocks on one GPU of allocation J.
# usage: graphs60c.sh <jobid> <cache> <emb dir> <radius> <filter dir> <cut> <graph dir> <block>...
set -uo pipefail
J=$1; C=$2; E=$3; R=$4; F=$5; CUT=$6; G=$7; shift 7
PY=/pscratch/sd/d/danieltm/envs/ml-tracking-review/bin/python
U="env $(env | grep -o '^SLURM_[A-Z_]*' | sed 's/^/-u /' | tr '\n' ' ')"
mkdir -p $G/logs
$U srun --jobid=$J --exact -n 1 -c 32 --gpus=1 $PY /global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/pileup60/graph_apply60.py \
  --cache $C --emb $E --radius $R --filter $F --cut $CUT --out $G --blocks "$@" >> $G/logs/apply_$1.log 2>&1 || { echo "FAILED"; exit 1; }
echo done
