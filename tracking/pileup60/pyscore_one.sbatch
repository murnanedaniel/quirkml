#!/bin/bash
#SBATCH -A m4958 -C cpu -q debug -N 1 -c 256 -t 00:30:00 -J qtrk_pu60_pyscore
#SBATCH -o /pscratch/sd/d/danieltm/quirk_pu60/logs/pyscore_%j.out
# Python strict scoring of one prediction set: sbatch pyscore_one.sbatch PRED DATASET A B OUT_JSON
/pscratch/sd/d/danieltm/envs/ml-tracking-review/bin/python /global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/mu60_eval/py_score.py "$@"
/pscratch/sd/d/danieltm/envs/ml-tracking-review/bin/python -c "
import json; r=json.load(open('$5'))
for p in ['very_loose_strict','tight_strict']:
    x=r[p]; print(p,'quirk %.3f'%x['quirk']['eff'],'event',x['quirk_event']['eff'],'tech_sm %.3f'%x['technical_sm']['eff'],'fake %.3f'%x['tracks']['fake_rate'])"
