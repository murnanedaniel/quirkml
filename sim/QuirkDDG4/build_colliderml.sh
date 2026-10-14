#!/bin/bash
# Build QuirkDDG4 inside the ColliderML sim image (podman-hpc), for use with the production
# simulation stage (ddsim_run.py, user_physics). Output: build-<tag>/lib/libQuirkDDG4.so
#   bash build_colliderml.sh [image]      (default ghcr.io/opendatadetector/sw:pr-8)
IMAGE=${1:-ghcr.io/opendatadetector/sw:pr-8}
TAG=$(echo "$IMAGE" | sed 's|.*:||')
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SETUP=${COLLIDERML_CONTAINER_SETUP:-/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/software/colliderml-prod-container/scripts/cli/setup_container_env.sh}
podman-hpc run --rm -v /global/cfs/cdirs/m4958:/global/cfs/cdirs/m4958 --entrypoint /bin/bash "$IMAGE" -c "
  source $SETUP >/dev/null 2>&1
  echo \"Geant4 \$(geant4-config --version 2>/dev/null)  DD4hep \$(dirname \$(dirname \$(which ddsim 2>/dev/null)))\"
  cmake -S $HERE -B $HERE/build-$TAG -DCMAKE_BUILD_TYPE=Release && cmake --build $HERE/build-$TAG -j8 &&
  ls -la $HERE/build-$TAG/lib/ && cat $HERE/build-$TAG/lib/*.components"
