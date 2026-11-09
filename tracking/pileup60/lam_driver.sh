#!/bin/bash
# Other string scales at mu 60 (no retraining): direction prep jobs given as args -> caches -> graphs + qw40 inference (GPU,
# shared lock + 4-job count) -> track building, scoring, L774 tagger/quality cut, chain inputs -> splitter predict (GPU)
# -> Python strict on the chain. Run detached: setsid nohup bash lam_driver.sh <prep job ids> &
set -uo pipefail
SETS="m200_L278_mu60 m200_L2154_mu60"
P=/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/pileup60
T=/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking
S=/pscratch/sd/d/danieltm/quirk_pu60; L=$S/lambda/driver.log; mkdir -p $S/lambda
Q=/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirk_tracking_data
log() { echo "$(date '+%F %T') $*" >> $L; }
waitok() {  # wait for jobs, abort unless all COMPLETED
  while squeue -h -j "$(echo $* | tr ' ' ,)" 2>/dev/null | grep -q .; do sleep 120; done
  for j in $*; do st=$(sacct -j $j -X -n -o State | head -1 | tr -d ' '); [ "$st" = COMPLETED ] || { log "job $j $st - abort"; exit 1; }; done
  log "jobs $* completed"; }
gpu_submit() {  # sbatch args...; under the shared lock, only when the quirk batch count is below 4
  while true; do
    id=$(flock -x /pscratch/sd/d/danieltm/qtrk_release.lock bash -c '
      n=$(squeue -h -u danieltm -A m5386_g -q gpu_shared -o "%j %T %r" | awk '"'"'($1 ~ /^(qtrk_|pv_)/) && ($2 == "RUNNING" || ($3 != "Dependency" && $3 !~ /^JobHeld/))'"'"' | wc -l)
      if [ "$n" -lt 4 ]; then sbatch --parsable "$@"; sleep 10; fi' _ "$@")
    [ -n "$id" ] && { log "submitted GPU $id ($*)"; echo $id; return; }
    sleep 300
  done; }
log "start; prep jobs $*"
[ $# -gt 0 ] && waitok $@
for DS in $SETS; do for b in 0 1; do [ -f $Q/direction/$DS/quirks.$DS.v1.direction.events$((b*1000))-$((b*1000+999)).parquet ] || { log "direction $DS block $b missing - abort"; exit 1; }; done; done
J=""; for DS in $SETS; do J="$J $(cd $P && sbatch --parsable -J qtrk_pu60_lam_cache --export=ALL,DS=$DS,BLOCKS='0 1',W=2 cache_blocks.sbatch)"; done
log "cache jobs $J"; waitok $J
J=$(gpu_submit --export=ALL,SETS="$SETS" $P/lam_gpu.sbatch); waitok $J
J=""; for DS in $SETS; do J="$J $(sbatch --parsable $P/lam_score.sbatch $DS)"; done
log "score jobs $J"; waitok $J
GATE=$(/pscratch/sd/d/danieltm/envs/ml-tracking-review/bin/python -c "import json;print(json.load(open('$Q/models/tagger/b3_pair60_50k_qw40_cc0.5/tagger.json'))['pair_good']['cut_from_valB'])")
for DS in $SETS; do W=$S/lambda/$DS
  J=$(gpu_submit $P/chain_predict.sbatch $W 0.5 $W/apply/qw40_cc0.5/b3_scores_test.parquet $GATE $W/chain); waitok $J
done
J=""; for DS in $SETS; do W=$S/lambda/$DS
  J="$J $(sbatch --parsable -q regular -t 00:45:00 $P/pyscore_one.sbatch $W/chain/pred_split $Q/pair_truth_mu60/$DS 0 2000 $W/chain/py_split_pair.json)"
  J="$J $(sbatch --parsable -q regular -t 00:45:00 $P/pyscore_one.sbatch $W/chain/pred_split $Q/parquet_r1_mu60/$DS 0 2000 $W/chain/py_split.json)"
done
log "chain pyscore jobs $J"; waitok $J
for DS in $SETS; do W=$S/lambda/$DS; D=$Q/results/pu60/lambda/$DS; mkdir -p $D
  cp $W/test/*.json $D/; cp $W/apply/qw40_cc0.5/{apply.json,roc_test.csv} $D/; cp $W/chain/py_split*.json $D/
done
log "ALL DONE"
