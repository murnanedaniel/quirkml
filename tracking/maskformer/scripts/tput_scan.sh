#!/bin/bash
# Loader+model throughput at batch sizes 1/4/8 on the 20k set (no validation).
cd /pscratch/sd/d/danieltm/quirk_tracking/maskformer
for bs in 1 4 8; do
  steps=$(( bs == 1 ? 1200 : (bs == 4 ? 500 : 300) ))
  bash /global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/maskformer/scripts/run.sh fit --config /global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/maskformer/packages/ml_tracking/src/ml_tracking/experiments/quirk/configs/quirk.yaml --name tput_bs$bs --data.batch_size $bs     --trainer.max_steps $steps --trainer.val_check_interval 100000 --trainer.num_sanity_val_steps 0 --data.num_val 1     --trainer.default_root_dir /pscratch/sd/d/danieltm/quirk_tracking/maskformer/runs/tput_bs$bs > logs/tput_bs$bs.log 2>&1
  echo "bs $bs rc $?"
done
