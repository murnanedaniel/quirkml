#!/bin/bash
#SBATCH -A m4958 -C cpu -q regular -N 1 -c 256 -t 02:00:00 -J qtrk_pu60_splitin
#SBATCH -o /pscratch/sd/d/danieltm/quirk_pu60/logs/splitin_%j.out
# Splitter input tracks for the zero-shot gated chain: pair GNN components tracks with B3 tag >= GATE (mu 0 fit), in
# 250-event chunks in parallel; direction columns zeroed (the 1M splitter was trained without them).
DS=${DS:-m200_L774_mu60}; GATE=${GATE:-0.01869147851668048}
O=/pscratch/sd/d/danieltm/quirk_pu60/zs/$DS
T=/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking
PY=/pscratch/sd/d/danieltm/envs/ml-tracking-review/bin/python
mkdir -p $O/split/chunks
for a in $(seq 0 250 1750); do
  [ -s $O/split/chunks/cc_$a.pkl ] || SPLIT_NO_TIME=1 SPLIT_CACHE=/pscratch/sd/d/danieltm/quirk_pu60/cache/${DS}_dir $PY $T/split/split_data.py --mode cc \
    --pred $O/pair/test/pred_cc0.5_mh4/predictions.parquet --events $a $((a + 249)) --gate-file $O/pair/test/b3_scores.parquet \
    --gate-cut $GATE --out $O/split/chunks/cc_$a.pkl > $O/split/chunks/log_$a.txt 2>&1 &
done
wait
$PY -c "
import pickle, glob
d=[]
for f in sorted(glob.glob('$O/split/chunks/cc_*.pkl'), key=lambda f:int(f.split('_')[-1][:-4])): d+=pickle.load(open(f,'rb'))
for t in d: t['x']=t['x'].copy(); t['x'][:,28:38]=0
pickle.dump(d,open('$O/split/ccnodir_test.pkl','wb'),protocol=5); print('gated tracks',len(d))"
