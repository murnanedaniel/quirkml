#!/bin/bash
# Field-off closed-form test (approved by Daniel 2026-10-14). Pair-only, EM off for quirks, B = 0.
# geometry: odd = full ODD v4 with the solenoid field set to 0 (copy in odd_xml/); empty = world with no volumes, B = 0.
D=/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirk_erc; F=$D/out/trajcmp/fieldoff; cd $D
for g in odd empty; do
  [ $g = odd ] && C=$F/odd_xml/OpenDataDetector.xml || C=$F/odd_xml/EmptyWorld.xml
  for mb in 1e-2 1e-3; do
    for ev in "L500 m400_L500_ev1 1266.932679" "L1000 m400_L1000_ev3 5067.730718"; do
      set -- $ev; tag=${g}_$1_mb$mb
      [ $g = odd ] && [ $mb = 1e-3 ] && continue
      QUIRK_DUMP_FILE=$F/g4_${tag}_steps.csv bash sim/run_in_shifter.sh --compact $C \
        --input $D/out/trajcmp/$2.hepmc3 --events 1 --output $F/g4_${tag}.root \
        --pdg 4900101 --mass 400 --string-force $3 --no-em --max-boost $mb > $F/g4_${tag}.log 2>&1 &
    done
  done
done
wait
echo ALL_DONE
