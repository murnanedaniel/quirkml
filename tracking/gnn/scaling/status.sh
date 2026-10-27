#!/bin/bash
# One line per arm: strict val history (loose) from best_sel.json, best, bad count, stopped.
S=/pscratch/sd/d/danieltm/quirk_tracking/scaling
for d in $S/runs/[qp]_*/; do t=$(basename $d)
  python3 -c "
import json,sys
try: j=json.load(open('$d/gnn/best_sel.json'))
except Exception: print('$t: no val yet'); sys.exit()
h=' '.join(f\"{x['step']//1000}k:{x['val_strict_loose']:.3f}/{x['val_strict_tight']:.3f}\" for x in j['history'])
print(f\"$t best {j['best']:.4f}@{j['best_step']//1000}k bad {j['bad']} stopped {j['stopped']} | {h}\")"
  grep -h "finished\|exit [1-9]" $d/driver.log 2>/dev/null | tail -1
done
tail -2 $S/logs/windows.log; squeue -u danieltm -o "%i %j %T %M" | grep qtrk_scal
