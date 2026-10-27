#!/bin/bash
#SBATCH -A m5386 -C gpu -q shared --gpus 1 -c 32 -t 01:00:00 -J qtrk_scan
#SBATCH -o /pscratch/sd/d/danieltm/quirk_tracking/maskformer/logs/slurm-%x-%j.out
# Checkpoint scans (scan_ckpts.sh) for the given runs, then test predictions (17000-19999, ot 0.1 and 0.2) of each chosen checkpoint.
MF=/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/maskformer
OUT=/pscratch/sd/d/danieltm/quirk_tracking/maskformer
for TAG in "$@"; do
  srun -n1 -c32 --gpus=1 env CANDIDATES=${CANDIDATES:-last} bash $MF/scripts/scan_ckpts.sh $TAG || { echo "scan $TAG failed"; continue; }
  CK=$(cat $OUT/diag/chosen_$TAG.txt); RUNCFG=$(ls -t $OUT/runs/$TAG/*/*/version_0/config.yaml | tail -1)
  for OT in 0.1 0.2; do
    for SPLIT in val:16000:16999 test:17000:19999; do
      IFS=: read -r S A B <<< "$SPLIT"; P=$OUT/predictions/${TAG}_chosen_${S}_ot$OT
      srun -n1 -c32 --gpus=1 bash $MF/scripts/predict.sh $RUNCFG $CK $P $A $B $OT 0.5 > $OUT/logs/predict_${TAG}_chosen_${S}_ot$OT.log 2>&1 && echo $CK > $P/ckpt_used.txt || echo "predict $TAG $S $OT failed"
    done
  done
done
