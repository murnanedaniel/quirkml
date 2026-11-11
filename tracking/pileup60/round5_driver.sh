#!/bin/bash
# Round 5 (Fable 5.2 follow-ups), three branches in parallel, each aborts on its own failure:
#  A: pair GNN x40 at 120k steps, 16k and 50k events (resubmitted across the 12 h limit; resumes from state.pt)
#  B: quirk-free mu 60 proxy (noquirk60.py, L774 blocks 60-61) -> background tag rate; unseen block 60 -> per-class
#     loss on unseen / seen / val events for pair_50k_qw40_s0 and the trunk
#  C: Lambda 278 / 2154 training-range prep (blocks 2-19) -> mixed-Lambda pair GNN (60k steps) -> val per Lambda
# GPU jobs go through the shared lock with the 4-job count. Run: setsid nohup bash round5_driver.sh &
set -uo pipefail
P=/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/pileup60
S=/pscratch/sd/d/danieltm/quirk_pu60; L=$S/round5.log
Q=/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirk_tracking_data
log() { echo "$(date '+%F %T') $*" >> $L; }
wait_end() { while squeue -h -j "$(echo $* | tr ' ' ,)" 2>/dev/null | grep -q .; do sleep 300; done; }
waitok() { wait_end $*; for j in $*; do st=$(sacct -j $j -X -n -o State | head -1 | tr -d ' ')
  [ "$st" = COMPLETED ] || { log "job $j $st - branch abort"; return 1; }; done; log "jobs $* completed"; }
gpu_submit() {
  while true; do
    id=$(flock -x /pscratch/sd/d/danieltm/qtrk_release.lock bash -c '
      n=$(squeue -h -u danieltm -A m5386_g -q gpu_shared -o "%j %T %r" | awk '"'"'($1 ~ /^(qtrk_|pv_)/) && ($2 == "RUNNING" || ($3 != "Dependency" && $3 !~ /^JobHeld/))'"'"' | wc -l)
      if [ "$n" -lt 4 ]; then sbatch --parsable "$@"; sleep 10; fi' _ "$@")
    [ -n "$id" ] && { log "GPU $id: $*"; echo $id; return; }
    sleep 300
  done; }
run_arm() {  # NAME JOBNAME EXPORTS
  local N=$1 J=$2 E=$3
  for att in 1 2 3 4; do
    id=$(gpu_submit -J $J "--export=ALL,NAME=$N,$E" $P/arm60.sbatch); wait_end $id  # export quoted: TRAIN / EXTRA hold spaces
    if [ -f $S/gnn/$N/final.pt ] && [ -s $S/gnn/$N/scored/events2000-2999.parquet ]; then log "$N done"; return 0; fi
    log "$N attempt $att ended without final ($(sacct -j $id -X -n -o State,ExitCode | head -1))"
  done; log "$N FAILED after 4 attempts"; return 1; }

branchA() {
  run_arm pair_16k_qw40_120k_s0 qtrk_pu60_arm_16k120k "TRAIN=3000-18999,STEPS=120000,SEED=0,EXTRA=--quirk-weight 40" &
  run_arm pair_50k_qw40_120k_s0 qtrk_pu60_arm_50k120k "TRAIN=3000-52999,STEPS=120000,SEED=0,EXTRA=--quirk-weight 40" &
  wait; log "branch A finished"; }

branchB() {
  local DS=m200_L774_mu60_noquirk J
  J=$(sbatch --parsable -A m4958 -C cpu -q debug -N 1 -c 64 -t 00:30:00 -J qtrk_pu60_noquirk -o $S/logs/noquirk_%j.out \
      --wrap "/pscratch/sd/d/danieltm/envs/ml-tracking-review/bin/python $P/noquirk60.py m200_L774_mu60 $DS 60 61")
  waitok $J || return 1
  J="$(cd $P && sbatch --parsable -J qtrk_pu60_cacheB --export=ALL,DS=$DS,BLOCKS='60 61',W=2 cache_blocks.sbatch)"
  J="$J $(cd $P && sbatch --parsable -J qtrk_pu60_cacheB --export=ALL,DS=m200_L774_mu60,BLOCKS=60,W=1 cache_blocks.sbatch)"
  waitok $J || return 1
  J=$(gpu_submit --export=ALL,JOBS="m200_L774_mu60:60 $DS:60 $DS:61" $P/graphs_sets.sbatch); waitok $J || return 1
  ( J=$(gpu_submit --export=ALL,JOBS="pair_50k_qw40_s0:$DS:60:$S/lambda/$DS pair_50k_qw40_s0:$DS:61:$S/lambda/$DS" $P/infer_sets.sbatch)
    waitok $J && J=$(sbatch --parsable $P/bg_score.sbatch) && waitok $J && log "background done" ) &
  for R in pair_50k_qw40_s0 pair_50k_trunk_s0; do
    ( J=$(gpu_submit -J qtrk_pu60_retro --export=ALL,RUN=$R $P/retro_unseen.sbatch); waitok $J && log "retro $R done" ) &
  done
  wait; log "branch B finished"; }

branchC() {
  local J="" DS b jobs=""
  for DS in m200_L278_mu60 m200_L2154_mu60; do
    J="$J $(cd $P && sbatch --parsable -t 05:00:00 -J qtrk_pu60_dirC --export=ALL,DS=$DS,B0=2,B1=19,P=8 prep_9499.sbatch)"; done
  waitok $J || return 1
  for DS in m200_L278_mu60 m200_L2154_mu60; do for b in $(seq 2 19); do
    [ -f $Q/direction/$DS/quirks.$DS.v1.direction.events$((b*1000))-$((b*1000+999)).parquet ] || { log "direction $DS $b missing"; return 1; }; done; done
  J=""; for DS in m200_L278_mu60 m200_L2154_mu60; do
    J="$J $(cd $P && sbatch --parsable -q regular -t 03:00:00 -J qtrk_pu60_cacheC --export=ALL,DS=$DS,BLOCKS="$(seq -s ' ' 2 19)",W=6 cache_blocks.sbatch)"; done
  waitok $J || return 1
  J=""; for DS in m200_L278_mu60 m200_L2154_mu60; do for g in "2 3 4 5 6 7" "8 9 10 11 12 13" "14 15 16 17 18 19"; do
    jobs=""; for b in $g; do jobs="$jobs $DS:$b"; done
    J="$J $(gpu_submit --export=ALL,JOBS="$jobs" $P/graphs_sets.sbatch)"; done; done
  waitok $J || return 1
  # mixed-Lambda training dirs: L774 blocks as they are, L278 / L2154 renamed with +100000 / +200000 (event ids inside
  # a shard stay those of its own set; shards are self-consistent)
  local MC=$S/cache/mixL_mu60_dir_pair MG=$S/graphs/v3_mu0f_cut0.1_mixL; mkdir -p $MC $MG
  for b in $(seq 2 19); do n=events$((b*1000))-$((b*1000+999))
    ln -sfn $S/cache/m200_L774_mu60_dir_pair/$n.pkl $MC/$n.pkl; ln -sfn $S/graphs/v3_mu0f_cut0.1/$n.d $MG/$n.d
    [ $b -lt 3 ] && continue
    for o in "100000 m200_L278_mu60" "200000 m200_L2154_mu60"; do set -- $o
      m=events$(($1+b*1000))-$(($1+b*1000+999))
      ln -sfn $S/cache/${2}_dir_pair/$n.pkl $MC/$m.pkl; ln -sfn $S/graphs/v3_mu0f_cut0.1_$2/$n.d $MG/$m.d; done
  done
  log "mixed dirs: $(ls $MC | wc -l) cache, $(ls $MG | wc -l) graph links"
  run_arm pair_mixL_qw40_s0 qtrk_pu60_arm_mixL "TRAIN=3000-19999 103000-119999 203000-219999,STEPS=60000,SEED=0,EXTRA=--quirk-weight 40,CACHE=$MC,GRAPHS=$MG" || return 1
  local V=$S/lambda/val; jobs=""
  for DS in m200_L278_mu60 m200_L774_mu60 m200_L2154_mu60; do jobs="$jobs pair_mixL_qw40_s0:$DS:2:$V/mixL_$DS"; done
  for DS in m200_L278_mu60 m200_L2154_mu60; do jobs="$jobs pair_50k_qw40_s0:$DS:2:$V/qw40_$DS"; done
  J=$(gpu_submit --export=ALL,JOBS="$jobs" $P/infer_sets.sbatch); waitok $J || return 1
  mkdir -p $V/qw40_m200_L774_mu60/scored; ln -sfn $S/gnn/pair_50k_qw40_s0/scored/events2000-2999.parquet $V/qw40_m200_L774_mu60/scored/events2000-2999.parquet
  jobs=""; for DS in m200_L278_mu60 m200_L774_mu60 m200_L2154_mu60; do for M in mixL qw40; do
    jobs="$jobs $DS:$V/${M}_$DS:$V/${M}_$DS/builders60_pe.json"; done; done
  J=$(sbatch --parsable --export=ALL,JOBS="$jobs" $P/lam_val.sbatch); waitok $J || return 1
  log "branch C finished"; }

log "start"
branchA > $S/round5_A.out 2>&1 &
branchB > $S/round5_B.out 2>&1 &
branchC > $S/round5_C.out 2>&1 &
wait
log "ALL BRANCHES ENDED"
