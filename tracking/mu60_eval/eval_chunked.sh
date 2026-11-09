#!/bin/bash
# Chunked ACTS evaluation for mu 60 (pattern of the ML-tracking-review thread, reports/newgraphs/scripts): the event
# range [A, B) is split into chunks of CHUNK events; each chunk is its own evaluate.py run (own staging on /pscratch,
# own output dir OUT/chunk-<A'>-<B'>), PAR chunk runs at a time with J workers each; chunks already done are skipped.
# Then combine_eval_chunks.py sums the counts into OUT/summary_combined.json (efficiencies from summed counts).
# Run on one CPU node: srun -N1 -n1 -c 256 --mem=0 bash eval_chunked.sh PRED_DIR DATASET A B OUT [CHUNK PAR J]
# mu 60: ACTS takes ~5 min per event per worker; PAR 4 x J 12 (48 workers) fits one node (96 workers ran out of memory).
set -uo pipefail
umask 007
P=$1; DS=$2; A=$3; B=$4; OUT=$5; CHUNK=${6:-250}; PAR=${7:-4}; J=${8:-8}
Q=/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml
PY=/global/cfs/cdirs/m4958/usr/danieltm/ml_tracking_pilot_2026-09/envs/eval48/bin/python
STG=/pscratch/sd/d/danieltm/quirk_mu60/eval_staging
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p $OUT
cd $Q
run() {
  local a=$1 b=$2 o=$OUT/chunk-$1-$(($2 - 1))
  [ -f $o/summary.json ] && return 0
  $PY tracking/evaluate.py --predictions-dir $P --dataset $DS --events $a $b --out $o --jobs $J \
    --staging-dir $STG/$(basename $DS)_$a > $o.log 2>&1
  echo "chunk $a-$b exit $? $(date -Is)"
}
mkdir -p $OUT
i=0
for ((a = A; a < B; a += CHUNK)); do
  b=$((a + CHUNK)); [ $b -gt $B ] && b=$B
  run $a $b &
  i=$((i + 1)); [ $((i % PAR)) -eq 0 ] && wait
done
wait
$PY $Q/tracking/mu60_eval/combine_eval_chunks.py $OUT $A $B
