#!/bin/bash
# Stage 1: field-on convergence, quirk ionisation off, MaxBoost 1e-3 and 1e-4 (1e-2 is in steps_noioni_fieldon_*).
R=/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml; T=/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirk_erc/out/trajcmp; W=$T/../stage1
ODD_ON=/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/software/OtherLibraries/acts_versions/acts_v4/thirdparty/OpenDataDetector/xml/OpenDataDetector.xml
for mb in 1e-3 1e-4; do
  for ev in "L500 m400_L500_ev1 1266.932679" "L1000 m400_L1000_ev3 5067.730718"; do
    set -- $ev; tag=noioni_fieldon_$1_mb$mb
    QUIRK_RMAX_CM=115 QUIRK_DUMP_FILE=$W/steps_$tag.csv bash $R/sim/run_in_shifter.sh --compact $ODD_ON --input $T/$2.hepmc3 --events 1 \
      --output $W/$tag.root --pdg 4900101 --mass 400 --string-force $3 --no-em --no-ionisation --max-boost $mb > $W/$tag.log 2>&1 &
  done
done
wait; echo ALL_DONE
