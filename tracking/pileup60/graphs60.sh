#!/bin/bash
# Filtered graphs for mu 60 blocks (train_filter.py apply, graphs on the fly from the embedding), one GPU of allocation J.
# usage: graphs60.sh <jobid> <cache> <emb dir> <radius> <filter dir> <cut> <graph dir> <block>...
set -uo pipefail
J=$1; C=$2; E=$3; R=$4; F=$5; CUT=$6; G=$7; shift 7
PY=/pscratch/sd/d/danieltm/envs/ml-tracking-review/bin/python
U="env $(env | grep -o '^SLURM_[A-Z_]*' | sed 's/^/-u /' | tr '\n' ' ')"
mkdir -p $G/logs
cd /global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/gnn
for b in "$@"; do
  [ -s $G/events$((1000*b))-$((1000*b+999)).parquet ] && continue
  $U srun --jobid=$J --exact -n 1 -c 32 --gpus=1 $PY train_filter.py apply --cache $C --emb-dir $E --radius $R --k-max 512 --out $F \
    --cut $CUT --graph-dir $G --ranges $((1000*b))-$((1000*b+999)) > $G/logs/block$b.log 2>&1 || { echo "block $b FAILED"; exit 1; }
  echo "block $b done $(date +%T)"
done
