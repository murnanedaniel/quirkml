#!/bin/bash
# Stage 1: full ODD with hIoni also inactivated on the quirks (--no-em keeps ionisation; --no-ionisation drops it).
R=/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml; T=/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirk_erc/out/trajcmp; W=$T/../stage1
ODD_ON=/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/software/OtherLibraries/acts_versions/acts_v4/thirdparty/OpenDataDetector/xml/OpenDataDetector.xml
ODD_OFF=$T/fieldoff/odd_xml/OpenDataDetector.xml
for fld in off on; do
  [ $fld = off ] && C=$ODD_OFF || C=$ODD_ON
  for ev in "L500 m400_L500_ev1 1266.932679" "L1000 m400_L1000_ev3 5067.730718"; do
    set -- $ev; tag=noioni_field${fld}_$1
    QUIRK_DUMP_FILE=$W/steps_$tag.csv bash $R/sim/run_in_shifter.sh --compact $C --input $T/$2.hepmc3 --events 1 \
      --output $W/$tag.root --pdg 4900101 --mass 400 --string-force $3 --no-em \
      --no-ionisation \
      > $W/$tag.log 2>&1 &
  done
done
wait; echo ALL_DONE
