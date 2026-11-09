#!/bin/bash
#SBATCH -A m4958 -C cpu -q debug -N 1 -c 256 -t 00:30:00 -J qtrk_pu60_chainin
#SBATCH -o /pscratch/sd/d/danieltm/quirk_pu60/logs/chainin_%j.out
# Gated splitter chain inputs: pair-GNN tracks (RUN, components CC) with mu 60 tagger score >= GATE, 250-event chunks.
R=$1; CC=$2; TAGS=$3; GATE=$4; O=$5
T=/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking
PY=/pscratch/sd/d/danieltm/envs/ml-tracking-review/bin/python
mkdir -p $O/chunks
for a in $(seq 0 250 1750); do
  [ -s $O/chunks/cc_$a.pkl ] || SPLIT_NO_TIME=1 SPLIT_CACHE=${SPLIT_CACHE_DIR:-/pscratch/sd/d/danieltm/quirk_pu60/cache/m200_L774_mu60_dir} $PY $T/split/split_data.py --mode cc \
    --pred $R/test/pred_cc${CC}_mh4/predictions.parquet --events $a $((a + 249)) --gate-file $TAGS --gate-cut $GATE --out $O/chunks/cc_$a.pkl > $O/chunks/log_$a.txt 2>&1 &
done
wait
$PY -c "
import pickle, glob
d=[]
for f in sorted(glob.glob('$O/chunks/cc_*.pkl'), key=lambda f:int(f.split('_')[-1][:-4])): d+=pickle.load(open(f,'rb'))
for t in d: t['x']=t['x'].copy(); t['x'][:,28:38]=0
pickle.dump(d,open('$O/ccnodir_test.pkl','wb'),protocol=5); print('gated tracks',len(d))"
