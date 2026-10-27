#!/bin/bash
# First 2-GPU window: lane 0 = trainer smoke test then graph v3 on events 20000-153999 (blocks 20-53 first);
# lane 1 = pair graph (same order). Then the remaining time runs arms (window.sh).
S=/pscratch/sd/d/danieltm/quirk_tracking/scaling
O=/pscratch/sd/d/danieltm/quirk_tracking/gnn
R=/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/gnn/scaling
PY=/pscratch/sd/d/danieltm/envs/ml-tracking-review/bin/python
T0=$(date +%s)
log() { echo "$(date '+%F %T') [win0 $SLURM_JOB_ID] $*" >> $S/logs/windows.log; }
todo() { for b in $(seq $2 $3); do [ -f $1/events$((b*1000))-$((b*1000+999)).parquet ] || echo "$((b*1000))-$((b*1000+999))"; done; }
g() {  # kind cache emb radius filterdir cut b0 b1
  local r=$(todo $S/graphs/$1 $7 $8); [ -z "$r" ] && return 0
  cd /global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/gnn
  srun -n1 -c 32 --gpus=1 --exact $PY -u train_filter.py apply --cache $2 --emb-dir $3 --radius $4 --k-max 512 \
    --out $5 --cut $6 --graph-dir $S/graphs/$1 --ranges $r >> $S/logs/graphs_${1}_$SLURM_JOB_ID.log 2>&1
  local rc=$?; log "graphs $1 blocks $7-$8 exit $rc"; return $rc
}
laneA() {
  srun -n1 -c 32 --gpus=1 --exact $PY -u $R/train_gnn_stream.py --cache $S/cache/dir --graphs $S/graphs/v3 --out $S/runs/smoke2 \
    --name scal_smoke2 --train-ranges 0-15999 --val 16000 16199 --n-val 20 --steps 900 --interval 200 --save-every 100 \
    --stop-metric perquirk --patience 1 --track-val 16000 16099 --direction --quirk-weight 10 --warmup 100 --seed 0 \
    > $S/logs/smoke2.log 2>&1
  log "smoke exit $?"
  A="v3 $S/cache/dir $O/main/g1_qw10/emb 0.075 $O/main/g1f_qw10r075_q50/filter 0.001"
  g $A 20 53 && g $A 54 153 && [ -z "$(todo $S/graphs/v3 20 153)" ] && touch $S/graphs/v3/.complete
}
laneB() {
  B="pair $S/cache/dir_pair $O/main/v1/emb 0.07 $O/main/e2_pair_dir_s0/filter 0.005"
  g $B 20 53 && g $B 54 153 && [ -z "$(todo $S/graphs/pair 20 153)" ] && touch $S/graphs/pair/.complete
}
log "window start"
rem() { echo $(( 14400 - ($(date +%s) - T0) )); }
( laneA; log "lane A graphs done"; LANES=0 WIN_S=$(rem) bash $R/window.sh ) &
( laneB; log "lane B graphs done"; LANES=1 WIN_S=$(rem) bash $R/window.sh ) &
wait
