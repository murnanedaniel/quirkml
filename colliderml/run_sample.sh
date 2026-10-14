#!/bin/bash
# Run the four ColliderML stages of one sample, in order, and check each one's output before the next.
#   bash colliderml/run_sample.sh <config_dir> [first_stage]      (first_stage 1-4, default 1)
# Run on a compute node (see docs/tutorial.md); simulation uses one core. Logs: <config_dir>/logs/.
# Needs gen/run_quirkgen.sh to have filled the run directory first.
set -uo pipefail
CFG=$(cd "${1:?usage: bash colliderml/run_sample.sh <config_dir> [first_stage]}" && pwd)
FIRST=${2:-1}
source "$CFG/sample.env"
PY=$WORK_DIR/venv/bin/python
mkdir -p "$CFG/logs"

die() { echo "STOP: $*" >&2; exit 1; }
count_events() { grep -c '^E ' "$1" 2>/dev/null || echo 0; }

# quirk-gen output: the plugin settings in 2_simulation.yaml assume these values.
[ -f "$RUN_DIR/events.json" ] || die "no $RUN_DIR/events.json: run gen/run_quirkgen.sh $CFG first"
"$PY" - "$RUN_DIR/events.json" "$MASS_GEV" "$PDGID" <<'EOF' || die "events.json doesn't match this sample"
import json, sys
m = json.load(open(sys.argv[1]))
ok = abs(m["mass_gev"] - float(sys.argv[2])) < 1e-6 and m["pdgid"] == int(sys.argv[3]) and m["charge"] == 1
print(f"quirk-gen: mass {m['mass_gev']} GeV, pdgid {m['pdgid']}, charge {m['charge']}, {m['events']} events")
sys.exit(0 if ok else 1)
EOF
n=$(count_events "$RUN_DIR/events_hepmc2.hepmc")
[ "$n" -ge "$EVENTS" ] || die "events_hepmc2.hepmc has $n events, the sample needs $EVENTS"

check() {   # check <stage number>: the stage wrote what the next one reads
  case $1 in
    1) n=$(count_events "$RUN_DIR/merged_events.hepmc3")
       [ "$n" -ge "$EVENTS" ] || die "merged_events.hepmc3 has $n events, expected $EVENTS"
       echo "   merged_events.hepmc3: $n events" ;;
    2) [ -s "$RUN_DIR/edm4hep.root" ] || die "no edm4hep.root"
       echo "   edm4hep.root: $(du -h "$RUN_DIR/edm4hep.root" | cut -f1)" ;;
    3) ls "$RUN_DIR"/tracker_hits/*.parquet >/dev/null 2>&1 || die "no tracker_hits parquet"
       echo "   tracker_hits: $(ls "$RUN_DIR"/tracker_hits/*.parquet | wc -l) file(s)" ;;
    4) p=$(dirname "$(dirname "$RUN_DIR")")/parquet
       for t in reco/tracker_hits truth/tracker_simhits truth/particles; do
         ls "$p/$t"/*.parquet >/dev/null 2>&1 || die "no $p/$t parquet"
       done
       echo "   parquet: $p" ;;
  esac
}

i=0
for cfg in $STAGES; do
  i=$((i + 1)); [ "$i" -lt "$FIRST" ] && continue
  echo "== stage $i: $cfg  ($(date +%H:%M:%S))"
  ( cd "$COLLIDERML" && "$PY" scripts/cli/run_stage.py "$CFG/$cfg" ) > "$CFG/logs/$cfg.log" 2>&1
  rc=$?
  # The packaging validation wants track files >= 0.1 MB, which small samples don't reach; the
  # output check below decides for stage 4.
  [ $rc -eq 0 ] || [ "$i" -eq 4 ] || die "run_stage exit code $rc (log: $CFG/logs/$cfg.log)"
  check $i
done
echo "done: $DATASET"
