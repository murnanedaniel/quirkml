#!/bin/bash
# One data-scaling arm inside an allocation (one GPU step): streamed training (fixed optimizer steps, resumable), then
# GNN inference on val 16000-16999 and test 17000-19999 with best_sel.pt (best strict val efficiency).
# env: TAG KIND(v3|pair) RANGES STEPS SEED DEADLINE(unix s) [SRUN_OPTS]. Exit: 0 done, 3 deadline (resume later), else failure.
umask 007
S=/pscratch/sd/d/danieltm/quirk_tracking/scaling
PY=/pscratch/sd/d/danieltm/envs/ml-tracking-review/bin/python
G=/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/gnn
if [ "$KIND" = v3 ]; then C=$S/cache/dir; GR=$S/graphs/v3; else C=$S/cache/dir_pair; GR=$S/graphs/pair; fi
M=$S/runs/$TAG
mkdir -p $M
echo "$SLURM_JOB_ID" > $M/RUNNING; trap 'rm -f $M/RUNNING' EXIT
echo "$(date '+%F %T') start job $SLURM_JOB_ID $TAG kind $KIND ranges $RANGES steps $STEPS seed $SEED" >> $M/driver.log
if [ ! -f $M/gnn/cut_scan.json ]; then
  n=$(ls $M/train_part*.log 2>/dev/null | wc -l)
  srun -n1 -c 32 --gpus=1 $SRUN_OPTS $PY -u $G/scaling/train_gnn_stream.py --cache $C --graphs $GR --out $M/gnn --name scal_$TAG \
    --train-ranges $RANGES --val 16000 16199 --n-val 200 --steps $STEPS --interval 4000 --stop-metric $([ "$KIND" = v3 ] && echo perquirk || echo coupled) --patience 3 --direction --cluster node \
    --quirk-weight 10 --warmup 1000 --batch 4 --seed $SEED --deadline $DEADLINE > $M/train_part$n.log 2>&1
  rc=$?; echo "$(date '+%F %T') train exit $rc" >> $M/driver.log
  [ $rc -ne 0 ] && exit $rc
fi
if [ "$(ls $M/gnn/scored 2>/dev/null | wc -l)" -lt 4 ]; then
  srun -n1 -c 32 --gpus=1 $SRUN_OPTS $PY -u $G/train_gnn.py infer --cache $C --graphs "$GR/events1[6-9]???-*.parquet" --out $M/gnn \
    --ckpt best_sel.pt --ranges 16000-16999 17000-19999 > $M/infer.log 2>&1
  rc=$?; echo "$(date '+%F %T') infer exit $rc" >> $M/driver.log
  [ $rc -ne 0 ] && exit $rc
  [ "$(ls $M/gnn/scored | wc -l)" -ge 4 ] || exit 5
fi
echo "$(date '+%F %T') finished" >> $M/driver.log
