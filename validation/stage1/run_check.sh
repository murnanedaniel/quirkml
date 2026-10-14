#!/bin/bash
# Stage 1 diagnostic: instrumented QuirkDDG4 (quirkml build), field off, quirk EM off, 1 keV event 3.
R=/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml; T=/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirk_erc/out/trajcmp
W=$T/../stage1; X=$T/fieldoff/odd_xml
for g in odd empty; do
  [ $g = odd ] && C=$X/OpenDataDetector.xml || C=$X/EmptyWorld.xml
  QUIRK_STEPCHECK_FILE=$W/chk_${g}_L1000.csv QUIRK_DUMP_FILE=$W/steps_${g}_L1000.csv bash $R/sim/run_in_shifter.sh --compact $C \
    --input $T/m400_L1000_ev3.hepmc3 --events 1 --output $W/${g}_L1000.root \
    --pdg 4900101 --mass 400 --string-force 5067.730718 --no-em > $W/${g}_L1000.log 2>&1 &
done
wait; echo ALL_DONE
