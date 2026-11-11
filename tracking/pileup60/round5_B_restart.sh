#!/bin/bash
# Restart of round-5 branch B only (first attempt failed: noquirk60.py event-order assert; fixed by sorting by event_id).
# Loads the helper and branch definitions from round5_driver.sh without starting any branch, then runs branchB.
# Run: setsid nohup bash round5_B_restart.sh < /dev/null > /dev/null 2>&1 &
P=/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/pileup60
source <(sed '/^log "start"/,$d' $P/round5_driver.sh)
log "branch B restart (noquirk60 sort fix)"
branchB > $S/round5_B.out 2>&1
