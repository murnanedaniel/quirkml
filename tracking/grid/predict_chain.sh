#!/bin/bash
# Keep a 1-GPU m5386 shared_interactive window running predict_loop.sh until every grid set has predictions.
G=/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/grid
H=/pscratch/sd/d/danieltm/quirk_tracking/maskformer/gpu_hours.txt
J=${1:-}
while true; do
  if [ -n "$J" ]; then
    bash $G/predict_loop.sh $J; rc=$?
    scancel $J 2>/dev/null; sleep 5
    sacct -j $J -X -n -o Start,End,ElapsedRaw | awk -v j=$J '{printf "%s %s %s 1 shared_interactive(m5386) Stage F grid inference gpu_hours=%.3f\n", j, $1, $2, $3/3600}' >> $H
    [ $rc -eq 0 ] && { echo "$(date -Is) all predictions done"; exit 0; }
    [ $rc -eq 2 ] && { echo "$(date -Is) predict_loop gave up (repeated failures); not requesting another window"; exit 2; }
  fi
  J=$(salloc -A m5386 -C gpu -q shared_interactive --gpus 1 -c 32 -t 04:00:00 --no-shell -J qtrk_mf_F 2>&1 | grep -o 'Granted job allocation [0-9]*' | grep -o '[0-9]*$')
  [ -z "$J" ] && { echo "$(date -Is) salloc failed, retry"; sleep 120; continue; }
  echo "$(date -Is) window $J"
done
