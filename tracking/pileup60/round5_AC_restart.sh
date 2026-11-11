#!/bin/bash
# Restart of round-5 branch A and the tail of branch C (first run: run_arm passed --export unquoted, so the spaces in
# TRAIN / EXTRA split it and sbatch failed in a loop; fixed in round5_driver.sh). C resumes at the mixed-Lambda arm:
# direction prep, caches, graphs and the mixed dirs were done in the first run.
# Run: setsid nohup bash round5_AC_restart.sh < /dev/null > /dev/null 2>&1 &
P=/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/pileup60
source <(sed '/^log "start"/,$d' $P/round5_driver.sh)
branchC_tail() {
  local MC=$S/cache/mixL_mu60_dir_pair MG=$S/graphs/v3_mu0f_cut0.1_mixL J DS M jobs
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
log "branch A + C-tail restart (run_arm export quoting fix)"
for x in A C; do [ -f $S/round5_$x.out ] && mv $S/round5_$x.out $S/round5_$x.attempt1.out; done
branchA > $S/round5_A.out 2>&1 &
branchC_tail > $S/round5_C.out 2>&1 &
wait; log "ALL BRANCHES ENDED"
