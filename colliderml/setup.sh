#!/bin/bash
# One-time setup for making quirk samples with ColliderML. Run on a Perlmutter login node:
#   bash colliderml/setup.sh <work_dir>
# <work_dir> must be under /global/cfs/cdirs/m4958 (the container sees only m4958 CFS and $PSCRATCH).
# It gets:
#   colliderml-production/   the ColliderML pipeline at the commit below, with scripts/cli/env_setup.yaml
#   venv/                    python for run_stage.py and for reading the parquet (ml/quirk_loader.py)
# and this repo gets sim/QuirkDDG4/build-pr-8/lib/libQuirkDDG4.so, built in the sim image.
# Safe to rerun: existing pieces are kept.
set -euo pipefail
WORK=${1:?usage: bash colliderml/setup.sh <work_dir>}
QUIRKML=$(cd "$(dirname "$0")/.." && pwd)
COLLIDERML_REPO=https://github.com/OpenDataDetector/colliderml-production.git
COLLIDERML_BRANCH=quirk/pilot-integration
COLLIDERML_COMMIT=6495666          # feat/two-container-pipeline + the quirk changes (docs/colliderml.md)
IMAGE=ghcr.io/opendatadetector/sw:pr-8

case "$WORK" in /global/cfs/cdirs/m4958/*) ;; *) echo "work_dir must be under /global/cfs/cdirs/m4958" >&2; exit 1;; esac
case "$QUIRKML" in /global/cfs/cdirs/m4958/*|"${PSCRATCH:-/nonexistent}"/*) ;;
  *) echo "clone quirkml under /global/cfs/cdirs/m4958 (the plugin library must be visible in the container)" >&2; exit 1;; esac
mkdir -p "$WORK"

echo "== 1/4 colliderml-production ($COLLIDERML_BRANCH @ $COLLIDERML_COMMIT)"
if [ ! -d "$WORK/colliderml-production/.git" ]; then
  git clone -q -b "$COLLIDERML_BRANCH" "$COLLIDERML_REPO" "$WORK/colliderml-production"
  git -C "$WORK/colliderml-production" checkout -q "$COLLIDERML_COMMIT"
fi
echo "   at $(git -C "$WORK/colliderml-production" rev-parse --short HEAD)"
ENV_SETUP=$WORK/colliderml-production/scripts/cli/env_setup.yaml
if [ ! -f "$ENV_SETUP" ]; then
  sed -e "s|@COLLIDERML@|$WORK/colliderml-production|g" -e "s|@RUNS_BASE@|$WORK/runs|g" \
      "$QUIRKML/colliderml/env_setup.yaml.in" > "$ENV_SETUP"
  echo "   wrote $ENV_SETUP"
fi

echo "== 2/4 python for run_stage.py"
[ -x "$WORK/venv/bin/python" ] || /usr/bin/python3.11 -m venv "$WORK/venv"
"$WORK/venv/bin/python" -c "import simple_slurm, yaml, pandas, pyarrow" 2>/dev/null ||
  "$WORK/venv/bin/python" -m pip install -q simple-slurm==0.3.5 pyyaml numpy pandas pyarrow
"$WORK/venv/bin/python" -c "import simple_slurm, yaml, pandas, pyarrow; print('   ok')"

echo "== 3/4 sim image ($IMAGE)"
podman-hpc image exists "$IMAGE" || podman-hpc pull "$IMAGE"
echo "   ok"

echo "== 4/4 QuirkDDG4 plugin"
if [ ! -f "$QUIRKML/sim/QuirkDDG4/build-pr-8/lib/libQuirkDDG4.so" ]; then
  COLLIDERML_CONTAINER_SETUP=$WORK/colliderml-production/scripts/cli/setup_container_env.sh \
    bash "$QUIRKML/sim/QuirkDDG4/build_colliderml.sh"
fi
ls "$QUIRKML/sim/QuirkDDG4/build-pr-8/lib/libQuirkDDG4.so"

echo
echo "setup done. Next: copy colliderml/sample.yaml, set work_dir: $WORK, then"
echo "  $WORK/venv/bin/python $QUIRKML/colliderml/make_sample.py my_sample.yaml"
