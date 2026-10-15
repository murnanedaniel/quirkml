#!/bin/bash
# quirk-gen for one run, inside an environment that has MadGraph, Pythia8 and HepMC3 python
# (called by gen/run_quirkgen.sh, which picks the environment). Does what quirk-gen's
# scripts/run_chain.sh does, with two changes for many runs, as ColliderML production does for
# MadGraph (madgraph_init once, then a copy of the process per run):
#   - the MadGraph process is built once (init) and each run works in its own copy of it, so runs
#     can go at the same time;
#   - each run limits MadGraph to the given number of cores.
#
#   quirkgen_run.sh init <quirk-gen dir> <procdir> <mass_gev>
#   quirkgen_run.sh run  <quirk-gen dir> <procdir> <run_dir> <mass_gev> <lambda_ev> <events> <seed> <cores> <converter>
#
# run writes to <run_dir>: events.hepmc, events.json (quirk-gen output) and events_hepmc2.hepmc
# (the same events numbered from 0, as the ColliderML stages need; see colliderml/convert_to_hepmc2.py).
set -eo pipefail
MODE=$1 QG=$2 PROC=$3

if [ "$MODE" = init ]; then
  MASS=$4
  [ -d "$PROC" ] && { echo "MadGraph process already built: $PROC"; exit 0; }
  out=$(dirname "$PROC")/init_test
  # quirk-gen builds the process on first use; a 10-event run also checks that it generates.
  python3 "$QG/scripts/generate_lhe.py" --mass "$MASS" --events 10 --seed 1 \
    --out "$out" --procdir "$PROC" --keep-procdir
  echo "MadGraph process built: $PROC"
  exit 0
fi

RUN_DIR=$4 MASS=$5 LAMBDA=$6 EVENTS=$7 SEED=$8 CORES=$9 CONVERT=${10}
[ -d "$PROC" ] || { echo "no MadGraph process at $PROC: run the generation stage's init first" >&2; exit 1; }
TMP=$(mktemp -d "${TMPDIR:-/tmp}/quirkgen.XXXXXX")
cleanup() { find "$TMP" -delete 2>/dev/null || true; }
trap cleanup EXIT

cp -r "$PROC" "$TMP/procdir"
CONF=$TMP/procdir/Cards/me5_configuration.txt
sed -i -e '/^[# ]*run_mode *=/d' -e '/^[# ]*nb_core *=/d' "$CONF"
printf 'run_mode = 2\nnb_core = %s\n' "$CORES" >> "$CONF"

python3 "$QG/scripts/generate_lhe.py" --mass "$MASS" --events "$EVENTS" --seed "$SEED" \
  --out "$TMP" --procdir "$TMP/procdir"
gunzip -fc "$TMP/events.lhe.gz" > "$TMP/events.lhe"
python3 "$QG/scripts/shower.py" --lhe "$TMP/events.lhe" --out "$TMP/events.hepmc" \
  --mass "$MASS" --lambda "$LAMBDA" --seed "$SEED"

mkdir -p "$RUN_DIR"
cp "$TMP/events.hepmc" "$TMP/events.json" "$RUN_DIR/"
python3 "$CONVERT" "$RUN_DIR/events.hepmc" "$RUN_DIR/events_hepmc2.hepmc"
