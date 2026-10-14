#!/bin/bash
# Run the three ColliderML stages of one sample, in order, inside the ODD image, and check each one's
# output before the next. Each stage script runs directly with --config and --output, as in the
# ColliderML library's colliderml.simulate().
#   bash colliderml/run_sample.sh <config_dir> [first_stage]      (first_stage 1-3, default 1)
# Logs: <config_dir>/logs/. Needs gen/run_quirkgen.sh to have filled the run directory first.
# Simulation uses one core; digitisation uses up to 8.
set -uo pipefail
CFG=$(cd "${1:?usage: bash colliderml/run_sample.sh <config_dir> [first_stage]}" && pwd)
FIRST=${2:-1}
source "$CFG/sample.env"
QUIRKML=$(cd "$(dirname "$0")/.." && pwd)
source "$QUIRKML/colliderml/container.sh" "$WORK_DIR" || exit 1
RUN_DIR=$WORK_DIR/runs/$DATASET
[ "$(cat "$WORK_DIR/plugin/IMAGE" 2>/dev/null)" = "$QUIRKML_IMAGE" ] || {
  echo "STOP: the plugin in $WORK_DIR/plugin was not built for $QUIRKML_IMAGE: rerun colliderml/setup.sh $WORK_DIR" >&2; exit 1; }
mkdir -p "$CFG/logs"

die() { echo "STOP: $*" >&2; exit 1; }
count_events() { grep -c '^E ' "$1" 2>/dev/null || echo 0; }

# quirk-gen output: the plugin settings in 2_simulation.yaml assume these values.
[ -f "$RUN_DIR/events.json" ] || die "no $RUN_DIR/events.json: run gen/run_quirkgen.sh $CFG first"
python3 - "$RUN_DIR/events.json" "$MASS_GEV" "$PDGID" <<'EOF' || die "events.json doesn't match this sample"
import json, sys
m = json.load(open(sys.argv[1]))
ok = abs(m["mass_gev"] - float(sys.argv[2])) < 1e-6 and m["pdgid"] == int(sys.argv[3]) and m["charge"] == 1
print(f"quirk-gen: mass {m['mass_gev']} GeV, pdgid {m['pdgid']}, charge {m['charge']}, {m['events']} events")
sys.exit(0 if ok else 1)
EOF
n=$(count_events "$RUN_DIR/events_hepmc2.hepmc")
[ "$n" -ge "$EVENTS" ] || die "events_hepmc2.hepmc has $n events, the sample needs $EVENTS"

script_for() {
  case $1 in
    1_merge_smear.yaml) echo merge_and_smear.py ;;
    1_pythia_pileup_merge.yaml) echo pythia_gen.py ;;
    2_simulation.yaml) echo ddsim_run.py ;;
    3_digitization.yaml) echo digi_and_reco.py ;;
  esac
}

check() {   # check <stage number>: the stage wrote what the next one reads
  case $1 in
    1) n=$(count_events "$RUN_DIR/merged_events.hepmc3")
       [ "$n" -ge "$EVENTS" ] || die "merged_events.hepmc3 has $n events, expected $EVENTS"
       echo "   merged_events.hepmc3: $n events" ;;
    2) [ -s "$RUN_DIR/edm4hep.root" ] || die "no edm4hep.root"
       echo "   edm4hep.root: $(du -h "$RUN_DIR/edm4hep.root" | cut -f1)" ;;
    3) for t in tracker_hits tracker_simhits particles; do
         ls "$RUN_DIR/$t"/*.parquet >/dev/null 2>&1 || die "no $t parquet in $RUN_DIR"
       done
       echo "   parquet: $RUN_DIR/{tracker_hits,tracker_simhits,particles,...}" ;;
  esac
}

i=0
for cfg in $STAGES; do
  i=$((i + 1)); [ "$i" -lt "$FIRST" ] && continue
  echo "== stage $i: $cfg  ($(date +%H:%M:%S))"
  qml_run "cd /workspace/scripts/simulation &&
           python3 $(script_for "$cfg") --config /work/configs/$DATASET/$cfg --output /work/runs/$DATASET" \
    > "$CFG/logs/$cfg.log" 2>&1
  rc=$?
  [ $rc -eq 0 ] || die "stage $i exit code $rc (log: $CFG/logs/$cfg.log)"
  check $i
done
echo "done: $DATASET -> $RUN_DIR"
