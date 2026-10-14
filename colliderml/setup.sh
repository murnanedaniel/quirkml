#!/bin/bash
# One-time setup for making quirk samples with the ColliderML pipeline, on any Linux machine with
# Docker, Podman, podman-hpc (NERSC) or Apptainer:
#   bash colliderml/setup.sh <work_dir>
# <work_dir> gets:
#   colliderml-production/   the ColliderML stage scripts, at the commit below
#   sw-<tag>.sif             the ODD software image (Apptainer only; other runtimes keep their own store)
#   plugin/                  QuirkDDG4 built inside that image
#   cache/                   Geant4 data (downloaded on the first simulation, about 2 GB)
# Safe to rerun: existing pieces are kept. Set QUIRKML_RUNTIME to choose the runtime.
set -euo pipefail
WORK=${1:?usage: bash colliderml/setup.sh <work_dir>}
mkdir -p "$WORK"; WORK=$(cd "$WORK" && pwd)
QUIRKML=$(cd "$(dirname "$0")/.." && pwd)
COLLIDERML_REPO=https://github.com/OpenDataDetector/colliderml-production.git
COLLIDERML_BRANCH=quirk/pilot-integration
COLLIDERML_COMMIT=6495666          # feat/two-container-pipeline + the quirk changes (docs/colliderml.md)
source "$QUIRKML/colliderml/container.sh" "$WORK"

echo "== 1/3 colliderml-production ($COLLIDERML_BRANCH @ $COLLIDERML_COMMIT)"
if [ ! -d "$WORK/colliderml-production/.git" ]; then
  git clone -q -b "$COLLIDERML_BRANCH" "$COLLIDERML_REPO" "$WORK/colliderml-production"
  git -C "$WORK/colliderml-production" checkout -q "$COLLIDERML_COMMIT"
fi
echo "   at $(git -C "$WORK/colliderml-production" rev-parse --short HEAD)"

echo "== 2/3 image $QUIRKML_IMAGE via $QML_RT (about 6 GB, pulled once)"
qml_pull
echo "   ok"

echo "== 3/3 QuirkDDG4 plugin"
if [ ! -f "$WORK/plugin/libQuirkDDG4.so" ]; then
  qml_run "cmake -S /quirkml/sim/QuirkDDG4 -B /work/plugin/build -DCMAKE_BUILD_TYPE=Release >/dev/null &&
           cmake --build /work/plugin/build -j8 2>&1 | grep -E 'error|Built target' ;
           cp /work/plugin/build/lib/libQuirkDDG4.so /work/plugin/build/lib/libQuirkDDG4.components /work/plugin/"
fi
ls "$WORK/plugin/libQuirkDDG4.so"

echo
echo "setup done. Next: copy colliderml/sample.yaml next to $(basename "$WORK")/ (or set its work_dir: $WORK), then"
echo "  python3 $QUIRKML/colliderml/make_sample.py my_sample.yaml"
