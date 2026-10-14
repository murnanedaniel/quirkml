#!/bin/bash
# Generate the quirk events of one sample with quirk-gen and put them, converted, in its run directory.
#   bash gen/run_quirkgen.sh <config_dir>          (config_dir from colliderml/make_sample.py)
# Runs quirk-gen (burzynski-lab/quirk-gen at the commit below, unmodified) in the el9 container with
# its LCG_109 view, on $PSCRATCH because MadGraph fails on CFS. A login node is fine for a few hundred
# events (100 events at m = 400 GeV: 1.6 min on 8 cores). Writes to the run directory:
#   events.hepmc, events.json    quirk-gen output
#   events_hepmc2.hepmc          the same events as HepMC2, numbered from 0 (the sim image can't read
#                                quirk-gen's HepMC3 3.03 files; colliderml/convert_quirkgen_hepmc.sh)
set -eo pipefail   # not -u: the LCG setup script uses unset variables
QUIRKGEN_REPO=https://github.com/burzynski-lab/quirk-gen.git
QUIRKGEN_COMMIT=9980db3
CFG=$(cd "${1:?usage: bash gen/run_quirkgen.sh <config_dir>}" && pwd)
source "$CFG/sample.env"
SRC=$QUIRKGEN_SCRATCH/quirk-gen
OUT=$QUIRKGEN_SCRATCH/$DATASET

if ! grep -qi "almalinux" /etc/os-release 2>/dev/null; then
  mkdir -p "$QUIRKGEN_SCRATCH"
  if [ ! -d "$SRC/.git" ]; then
    git clone -q "$QUIRKGEN_REPO" "$SRC" && git -C "$SRC" checkout -q "$QUIRKGEN_COMMIT"
  fi
  exec shifter --image=registry.cern.ch/atlasadc/atlas-grid-almalinux9:latest --module=cvmfs /bin/bash "$0" "$@"
fi

cd "$SRC" && source setup.sh
scripts/run_chain.sh --mass "$MASS_GEV" --events "$EVENTS" --lambda "$LAMBDA_EV" --seed "$SEED" --out "$OUT"
cp "$OUT/events.hepmc" "$OUT/events.json" "$RUN_DIR/"
python3 "$QUIRKML/colliderml/convert_to_hepmc2.py" "$RUN_DIR/events.hepmc" "$RUN_DIR/events_hepmc2.hepmc"
echo "quirk events ready in $RUN_DIR"
