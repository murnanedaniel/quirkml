#!/bin/bash
# Stage 4: QuirkDDG4 cost vs string tension (pair-only m400 event 3, full ODD v4, 3 T, EM on as in production).
R=/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml; T=/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirk_erc/out/trajcmp; W=$T/../stage4
ODD=/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/software/OtherLibraries/acts_versions/acts_v4/thirdparty/OpenDataDetector/xml/OpenDataDetector.xml
for lam in 3000 10000 30000 100000 1000000; do
  sf=$(python3 -c "print($lam**2 / 1.973269804e-4 / 1e6)")
  for mb in 1e-2 1e-3; do
    tag=L${lam}_mb$mb
    ( st=$(date +%s.%N)
      QUIRK_RMAX_CM=115 QUIRK_ZMAX_CM=300 timeout 1200 bash $R/sim/run_in_shifter.sh --compact $ODD --input $T/m400_L1000_ev3.hepmc3 --events 1 \
        --output $W/$tag.root --pdg 4900101 --mass 400 --string-force $sf --max-boost $mb > $W/$tag.log 2>&1
      rc=$?; en=$(date +%s.%N)
      echo "$tag lambda_eV=$lam stringforce=$sf maxboost=$mb rc=$rc wall_s=$(python3 -c "print(round($en-$st,1))") finished=$(grep -c 'Finished run' $W/$tag.log)" >> $W/cost.txt ) &
  done
done
wait; echo ALL_DONE
