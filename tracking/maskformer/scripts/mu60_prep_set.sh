#!/bin/bash
# mu 60 data prep for one set: R1 conversion (convert_to_r1_fast.py) into quirk_tracking_data/parquet_r1_mu60/<set>,
# pair truth for test (0-1999) and, for the 100k sets, val (2000-2999) into quirk_tracking_data/pair_truth_mu60/<set>;
# .done marker when complete. Usage: mu60_prep_set.sh <set> [conversion processes]
set -uo pipefail
ds=$1; NP=${2:-2}
Q=/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking
SRC=/global/cfs/cdirs/m4958/data/ColliderML/simulation/quirks/$ds/v1/parquet
DST=/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirk_tracking_data/parquet_r1_mu60/$ds
PT=/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirk_tracking_data/pair_truth_mu60/$ds
PY=/pscratch/sd/d/danieltm/envs/ml-tracking-review/bin/python
umask 007
log() { echo "[$ds] $(date -Is) $*"; }
[ -f $DST/.done ] && { log "already done"; exit 0; }
[ -d $SRC/reco/tracker_hits ] || { log "MISSING source $SRC"; exit 1; }
nsrc=$(ls $SRC/reco/tracker_hits/*.parquet | wc -l)
$PY $Q/convert_to_r1_fast.py $SRC $DST $NP || { log "conversion FAILED"; exit 1; }
ndst=$(ls $DST/reco/tracker_hits/*.parquet | wc -l); npart=$(ls $DST/truth/particles/*.parquet | wc -l)
[ $ndst -eq $nsrc ] && [ $npart -eq $nsrc ] || { log "file count mismatch: src $nsrc hits $ndst particles $npart"; exit 1; }
log "converted $ndst files"
for rng in "0 2000" "2000 3000"; do
  set -- $rng
  [ -f $DST/reco/tracker_hits/quirks.$ds.v1.reco.tracker_hits.events$1-$(( ($1/1000)*1000+999 )).parquet ] || continue
  [ -f $PT/pair_truth_manifest_events$1-$(( $2 - 1 )).json ] && continue
  $PY $Q/make_pair_truth_fast.py --dataset $DST --events $1 $2 --out $PT --name quirks.$ds.v1.pairtruth > $PT.log.$1 2>&1 \
    || { log "pair truth $1-$2 FAILED"; exit 1; }
done
touch $DST/.done; log done
