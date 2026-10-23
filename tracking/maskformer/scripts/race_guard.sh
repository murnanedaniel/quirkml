#!/bin/bash
# Whichever of the LR-scan submissions starts first wins; cancel the other route (GPU cap 4).
LOG=/pscratch/sd/d/danieltm/quirk_tracking/maskformer/logs/race_guard.log
SHARED="59877252,59877253,59877254"; PACK=59877255
while true; do
  if squeue -h -j $PACK -t R | grep -q .; then scancel $(echo $SHARED | tr ',' ' '); echo "$(date -Is) pack started; cancelled shared jobs" >> $LOG; exit 0; fi
  if squeue -h -j $SHARED -t R | grep -q .; then scancel $PACK; echo "$(date -Is) shared started; cancelled pack" >> $LOG; exit 0; fi
  if ! squeue -h -j $SHARED,$PACK | grep -q .; then echo "$(date -Is) nothing queued" >> $LOG; exit 0; fi
  sleep 20
done
