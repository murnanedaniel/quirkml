#!/bin/bash
# Run one stage of a quirk sample over its runs, inside the ODD image, the way ColliderML
# production's run_stage.py does: each run is a separate job in <runs_dir>/<N>/, the stage script
# gets --config, --output <runs_dir>, --output-subdir <N> and --seed <dataset>_<version>_run<N>,
# and every stage is done for all runs before the next stage starts.
#
#   bash colliderml/run_stage.sh <config_dir> <stage> [--run-range A B | --run-list N ...]
#                                [--parallel K] [--force]
#
# Stages, in order:
#   generation    quirk-gen: builds the MadGraph process once, then generates each run
#   stage1        beam spot (mu = 0, merge_and_smear.py) or Pythia8 pileup + merge (mu > 0, pythia_gen.py)
#   simulation    Geant4 + QuirkDDG4 (ddsim_run.py), sim_threads threads per run
#   digitization  ACTS digitisation + per-run parquet (digi_and_reco.py), digi_threads threads per run
#   package       all runs -> one dataset in the ColliderML layout (package_native_parquet.py),
#                 then verify_packaged_parquet.py; needs every run digitised
#   status        which runs have finished which stage
#
# --run-range A B   runs A to B-1 (as production's run_stage.py); default all runs of the sample
# --run-list N ...  these runs
# --parallel K      K runs at the same time on this machine (keep K x threads within its cores)
# --force           rerun runs that have already finished the stage (default: skip them)
#
# A run has finished a stage when the stage exited with 0 and its output passed the check below;
# the driver then writes <run>/.done_<stage>. Logs: <config_dir>/logs/<stage>/run<N>.log.
# For batch systems, run one task per run with --run-list $RUN (colliderml/batch/).
set -uo pipefail
usage() { sed -n '5,7p' "$0" >&2; exit 2; }
[ $# -ge 2 ] || usage
CFG=$(cd "$1" && pwd) || exit 1
STAGE=$2; shift 2
source "$CFG/sample.env"
QUIRKML=$(cd "$(dirname "$0")/.." && pwd)
RUNS_HOST=$WORK_DIR/$RUNS_DIR
RUN_IDS=($(seq 0 $((RUNS - 1))))
PARALLEL=1 FORCE=0 ONE=
while [ $# -gt 0 ]; do
  case $1 in
    --run-range) RUN_IDS=($(seq "$2" $(($3 - 1)))); shift 3 ;;
    --run-list) shift; RUN_IDS=(); while [ $# -gt 0 ] && [[ $1 != --* ]]; do RUN_IDS+=("$1"); shift; done ;;
    --parallel) PARALLEL=$2; shift 2 ;;
    --force) FORCE=1; shift ;;
    --one) ONE=$2; shift 2 ;;            # internal: one run, no checks of the others
    *) usage ;;
  esac
done
die() { echo "STOP: $*" >&2; exit 1; }
count_events() { grep -c '^E ' "$1" 2>/dev/null || echo 0; }

case $STAGE in
  generation|stage1|simulation|digitization|package|status) ;;
  *) die "unknown stage $STAGE (generation, stage1, simulation, digitization, package, status)" ;;
esac

# --- per-run work -------------------------------------------------------------------------------

# Check a run's output for a stage; prints a one-line summary, exits non-zero if incomplete.
check_run() {   # check_run <stage> <N>
  local d=$RUNS_HOST/$2 n
  case $1 in
    generation)
      n=$(count_events "$d/events_hepmc2.hepmc")
      [ "$n" -eq "$EVENTS_PER_RUN" ] || { echo "events_hepmc2.hepmc has $n events, expected $EVENTS_PER_RUN"; return 1; }
      python3 - "$d/events.json" "$MASS_GEV" "$PDGID" <<'EOF' || return 1
import json, sys
m = json.load(open(sys.argv[1]))
if not (abs(m["mass_gev"] - float(sys.argv[2])) < 1e-6 and m["pdgid"] == int(sys.argv[3]) and m["charge"] == 1):
    sys.exit(f"events.json: mass {m['mass_gev']}, pdgid {m['pdgid']}, charge {m['charge']} don't match the sample")
EOF
      echo "events_hepmc2.hepmc: $n events" ;;
    stage1)
      n=$(count_events "$d/merged_events.hepmc3")
      [ "$n" -eq "$EVENTS_PER_RUN" ] || { echo "merged_events.hepmc3 has $n events, expected $EVENTS_PER_RUN"; return 1; }
      echo "merged_events.hepmc3: $n events" ;;
    simulation)
      [ -s "$d/edm4hep.root" ] || { echo "no edm4hep.root"; return 1; }
      echo "edm4hep.root: $(du -h "$d/edm4hep.root" | cut -f1)" ;;
    digitization)
      local t; for t in tracker_hits tracker_simhits particles; do
        ls "$d/$t"/*.parquet >/dev/null 2>&1 || { echo "no $t parquet"; return 1; }
      done
      echo "parquet: tracker_hits, tracker_simhits, particles, tracks, truth_tracks" ;;
  esac
}

# The command that runs a stage script for one run, inside the image.
stage_command() {   # stage_command <stage> <N>
  local runs=/work/$RUNS_DIR seed="${DATASET}_${VERSION}_run$2"
  local common="--config /work/configs/$DATASET/%s --output $runs --output-subdir $2 --seed $seed"
  case $1 in
    stage1)
      if [ "$STAGE1" = 1_merge_smear.yaml ]; then
        printf "python3 merge_and_smear.py $common" "$STAGE1"
      else
        printf "python3 pythia_gen.py $common --hard-scatter-file $runs/$2/events_hepmc2.hepmc" "$STAGE1"
      fi ;;
    simulation)
      # the edm4hep event count is checked in the image, with podio-dump
      printf "python3 ddsim_run.py $common && n=\$(podio-dump $runs/$2/edm4hep.root | awk '\$1 == \"events\" {print \$2; exit}') && echo \"edm4hep.root: \$n events\" && [ \"\$n\" = $EVENTS_PER_RUN ]" 2_simulation.yaml ;;
    digitization) printf "python3 digi_and_reco.py $common" 3_digitization.yaml ;;
  esac
}

run_one() {   # run_one <stage> <N>: run, check, mark done
  local n=$2 d=$RUNS_HOST/$2 log=$CFG/logs/$1/run$2.log rc
  mkdir -p "$d" "$CFG/logs/$1"
  rm -f "$d/.done_$1"
  if [ "$1" = digitization ]; then
    # digi_and_reco.py doesn't replace old parquet shards (and packaging would read both): move a
    # previous attempt's tables aside; nothing is deleted.
    local t old=$d/.old_digitization_$(date +%Y%m%d_%H%M%S)
    for t in tracker_hits tracker_simhits particles tracks truth_tracks; do
      [ -e "$d/$t" ] && { mkdir -p "$old"; mv "$d/$t" "$old/"; }
    done
  fi
  {
    echo "== $1, run $n, $(date)"
    if [ "$1" = generation ]; then
      bash "$QUIRKML/gen/run_quirkgen.sh" "$CFG" "$n"
    else
      qml_run "cd /workspace/scripts/simulation && $(stage_command "$1" "$n")"
    fi
  } > "$log" 2>&1
  rc=$?
  [ $rc -eq 0 ] || { echo "run $n: FAILED, exit code $rc (log: $log)"; return 1; }
  local msg; msg=$(check_run "$1" "$n") || { echo "run $n: FAILED check: $msg (log: $log)"; return 1; }
  touch "$d/.done_$1"
  echo "run $n: done, $msg"
}

mkdir -p "$CFG/logs"
source "$QUIRKML/colliderml/container.sh" "$WORK_DIR" >/dev/null || exit 1

if [ -n "$ONE" ]; then run_one "$STAGE" "$ONE"; exit; fi

# --- status -------------------------------------------------------------------------------------
STAGES=(generation stage1 simulation digitization)
if [ "$STAGE" = status ]; then
  echo "$DATASET $VERSION: $RUNS runs x $EVENTS_PER_RUN events, runs in $RUNS_HOST"
  for s in "${STAGES[@]}"; do
    done_runs=(); missing=()
    for n in "${RUN_IDS[@]}"; do
      [ -f "$RUNS_HOST/$n/.done_$s" ] && done_runs+=("$n") || missing+=("$n")
    done
    printf "  %-13s %d/%d done" "$s" "${#done_runs[@]}" "${#RUN_IDS[@]}"
    [ ${#missing[@]} -gt 0 ] && [ ${#missing[@]} -le 20 ] && printf "; missing: %s" "${missing[*]}"
    echo
  done
  P=$WORK_DIR/$CAMPAIGN/$DATASET/$VERSION/parquet
  [ -d "$P" ] && echo "  packaged      $P" || echo "  packaged      not yet"
  exit 0
fi

# --- package ------------------------------------------------------------------------------------
if [ "$STAGE" = package ]; then
  missing=()
  for n in $(seq 0 $((RUNS - 1))); do [ -f "$RUNS_HOST/$n/.done_digitization" ] || missing+=("$n"); done
  [ ${#missing[@]} -eq 0 ] || die "runs not digitised: ${missing[*]} (packaging needs runs 0 to $((RUNS - 1)), no gaps)"
  P=$WORK_DIR/$CAMPAIGN/$DATASET/$VERSION/parquet
  [ -e "$P" ] && [ $FORCE -eq 0 ] && die "$P exists; move it aside (or use --force to write into it)"
  mkdir -p "$CFG/logs/package"
  echo "== package: $RUNS runs -> $P"
  qml_run "cd /workspace/scripts/postprocessing &&
           python3 package_native_parquet.py --config /work/configs/$DATASET/4_package_parquet.yaml &&
           python3 verify_packaged_parquet.py --config /work/configs/$DATASET/4_package_parquet.yaml" \
    > "$CFG/logs/package/package.log" 2>&1 || die "packaging failed (log: $CFG/logs/package/package.log)"
  tail -3 "$CFG/logs/package/package.log"
  echo "done: $P"
  exit 0
fi

# --- per-run stages -----------------------------------------------------------------------------
[ "$(cat "$WORK_DIR/plugin/BUILD_STAMP" 2>/dev/null)" = "$(qml_plugin_stamp)" ] || \
  die "the plugin in $WORK_DIR/plugin was not built from this image and QuirkDDG4 source: rerun colliderml/setup.sh $WORK_DIR"

# the previous stage must be done for every requested run
prev=
case $STAGE in stage1) prev=generation ;; simulation) prev=stage1 ;; digitization) prev=simulation ;; esac
todo=()
for n in "${RUN_IDS[@]}"; do
  [ -z "$prev" ] || [ -f "$RUNS_HOST/$n/.done_$prev" ] || die "run $n hasn't finished $prev"
  if [ -f "$RUNS_HOST/$n/.done_$STAGE" ] && [ $FORCE -eq 0 ]; then continue; fi
  todo+=("$n")
done
skipped=$(( ${#RUN_IDS[@]} - ${#todo[@]} ))
echo "== $STAGE: ${#todo[@]} runs to do ($skipped already done), $PARALLEL at a time, $(date +%H:%M:%S)"
[ ${#todo[@]} -eq 0 ] && exit 0

if [ "$STAGE" = generation ]; then
  bash "$QUIRKML/gen/run_quirkgen.sh" "$CFG" init > "$CFG/logs/generation_init.log" 2>&1 \
    || die "building the MadGraph process failed (log: $CFG/logs/generation_init.log)"
fi

printf "%s\n" "${todo[@]}" | xargs -P "$PARALLEL" -I{} bash "$0" "$CFG" "$STAGE" --one {}
failed=0
for n in "${todo[@]}"; do [ -f "$RUNS_HOST/$n/.done_$STAGE" ] || failed=$((failed + 1)); done
echo "== $STAGE: $(( ${#todo[@]} - failed )) of ${#todo[@]} runs done, $failed failed, $(date +%H:%M:%S)"
[ $failed -eq 0 ]
