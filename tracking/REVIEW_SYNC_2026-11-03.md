# Sync with the ML-tracking review thread (2026-11-03)

Source: message from the ml-tracking-review session. Paths: R = /global/cfs/cdirs/m4958/usr/danieltm/ml_tracking_pilot_2026-09,
repo = /global/cfs/cdirs/m4958/usr/danieltm/Side_Work/ExaTrkX/ML-Tracking-Review, D = /global/cfs/cdirs/m3443/data/ColliderML_Tracking_Review.
Their numbers: ACTS pre-fit, technical targets (pT > 1, |eta| < 3, >= 7 hits), tight = 90% double matching.

## What to adopt for quirks

1. LR schedule: warmup 1.5k, constant, linear decay to 0 over the last 20%. The decay is worth +6 to +8 points tight;
   never early-stop before it. Read several budgets with short decay branches off one constant-LR trunk (fixes our
   data-scaling design).
2. Optimiser: Lion, peak LR 1e-4 at 4 events/update; at 1 event/update 5e-5. Clip 0.1, wd 1e-5, bf16.
3. Spike guard: skip updates with pre-clip grad norm > 20x the median; use their fixed backstop (branch
   guard-backstop-fix, commits 2aa62b6, 352c500).
4. MaskFormer loss: published weights (0.1 object BCE, 2 Dice, 50 focal, 5 null-mask focal), matching cost focal
   normalised by hit count, sparse LAP solver (lap1015_late), fp32 costs. Dice is needed (-6.7 points without).
   Ledger: R/reports/building-blocks/BLOCK_LEDGER.md.
5. Architecture: graph attention along 1-hop module-map edges (EdgeGeo) is their best tracker (PU200: 92.0% tight vs
   GNN4ITk 90.4%, CKF 83.0%); window-mask attention (FlexGeo) 89.5%. Decoder depth 8 > 3 (+5 points at PU50).
   Per-hit cluster information +1.5 to +2 at PU0.
6. High pileup: their edge-aware graph hit filter keeps ~36k hits/event at PU200 with a 99% ceiling (tracker 3.3
   s/update) vs ~110k hits and 80 GB GPUs for the published filter. A PU0 -> PU50 curriculum gave no gain. Synthesised
   pileup levels: repo docs/dataset.md, mltr-prepare-ladder; module-map graphs D/full_odd_new_cuts (not full_odd_meanrms4).
7. Long-strip hits are missed by ML tracks at high pT (hits at strip centres); check for quirks.
8. Reuse: tracking_core eval (`mltr-evaluate --working-points`), pilot experiment code on branch pu0-round2
   (R/worktrees/pu0-round2: fixed guard, PILOT_PATH_MIRROR read mirror to pscratch for CFS stalls), paired event
   bootstrap and seed decision rule R/reports/building-blocks/render_ledger.py.

## GPU coordination

Same user, shared per-user QOS limits: they hold both gpu_interactive and both gpu_shared_interactive slots now
(about two more windows), with regular-QOS jobs queued. Their GPU jobs are moving from m5386_g to m4958_g, so m5386_g
will be free of them within hours. Quirk work: batch (gpu_shared / gpu_regular) on m5386 until a shared slot frees,
or ask Daniel to free one.
