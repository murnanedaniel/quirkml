# Quirk tracking: roadmap

Goal: the highest quirk track efficiency (per quirk, ACTS proto-track level, very loose matching) at the lowest
fake rate, first at one grid point, then across the (m, Lambda) grid, then with pileup. Ledger:
`tracking/EXPERIMENTS.md`. Agreed 2026-10-17. Hard constraint: Perlmutter maintenance 21 Oct 06:00 to 27 Oct;
everything below that needs compute is planned to finish or checkpoint before it.

Resources: GPU on account m5386 (`-A m5386`), CPU simulation on m4958. GPU queues: gpu_shared_interactive (2 jobs,
up to 2 GPUs each: the fast lane), gpu_shared / gpu_regular batch (slow). gpu_interactive and CPU interactive are
left to other projects. Budgets: about 400 GPU-hours, about 1000 CPU node-hours.

## T0. Done (2026-10-15/16)

- Data: m200 Lambda 774 eV mu 0, 20k events (train 0-15999, val 16000-16999, test 17000-19999); pilots; cost study.
- ACTS evaluator with quirk track / quirk event / technical (with and without quirks) efficiency and fake rate,
  validated with truth and broken predictions.
- GNN chain v1/v2 (all tracks in the loss, quirk weight 10): quirk eff 0.72 at fake 0.095 (3000 test events).
  Limit: separating the two interleaved quirks.
- MaskFormer ported; overfit gate passed; LR scan never started (batch queue).

## T1. Quirk-as-signal GNN chain (LRT recipe), m200 Lambda 774 eV

Treat quirks as LRT treated displaced tracks: ordinary tracks stay in the graph but are nearly silent in the loss.
- Metric learning and filter: quirk pairs/edges weight 10, wrong pairs/edges 1, other true pairs/edges 0 (ignored).
- GNN: quirk true edges 10, other true edges 0.02, wrong edges 0.1 (LRT recipe "S").
- A/B on top (not from LRT): wrong edges joining the two quirks weight 10 vs 0.1.
- Checkpoint and operating point chosen by quirk efficiency at matched fake rate on val, not by val loss.
- Reference rows: v2 (all tracks) with cluster features on / off (finish the cluster=none run).
Gate to T3: the better of the two beats v2 at matched fake rate on the 3000 test events, or we record that it doesn't.

## T2. MaskFormer, m200 Lambda 774 eV

- LR scan (batch 8, 30k steps, WSD, max LR 1.25e-5, 5e-5, 1e-4, 2e-4) with object_bce weight 1.0 (LRT objbce1).
  All tracks stay targets; no per-quirk weighting of the mask loss (it lost in LRT).
- Evaluate each at object threshold 0.1, 0.2, 0.3, 0.5 (mask 0.5); compare with T1 at matched fake rate.
- Long run from the best LR (until val loss flattens, checkpointed across windows).
- Result (2026-10-17): best max LR 5e-5, quirk eff 0.180 at fake 0.091 (test, threshold 0.1), far below the GNN.
  2e-4 diverged. Long run at 5e-5 (100k steps) running.
- Quirk-specific variant, started 2026-10-17 because of that gap: quirks x10 in the object and mask losses, all
  tracks still targets, LR re-scanned (5e-5, 1e-4).
- If the objective changes to quirks specifically (quirk-only or quirk-weighted targets or losses), repeat the LR
  scan for that objective: the all-tracks optimum need not carry over (Daniel, 2026-10-17).

## T3. Edge-geo and flex-geo

Edge-geo on the quirk-aware graph from T1 (metric learning + filter), flex-geo with its phi/eta window; same targets
and evaluation as T2. Starts when T1's graphs and T2's training loop are settled.

## T4. Generalisation across the grid (no retraining)

The best T1/T2 models evaluated on the test runs (runs 0-3, 2000 events) of every grid point (S1 below), as maps of
quirk efficiency and fake rate over (m, Lambda), with the three regimes marked.

## T5. Training on the grid, then generalisation again

Train on several grid points (from S2), then on all; repeat T4. Then mu = 60 (S4/S5).

## Simulation (CPU, m4958, regular queue, datasets on CFS `simulation/quirks/`)

Settings as in the pilots: 500 events per run, 8 simulation threads, version v1, seeds `<dataset>_v1_run<N>`,
edm4hep and HepMC kept, packaged at the end of each dataset. Runs are taken from one shared work list by
self-balancing workers (several multi-node jobs), in this priority order:

| Step | Content | Runs | Est. node-hours |
|---|---|---|---|
| S1 | mu 0, every grid point (11 masses 50-300 GeV x 10 Lambda 100-10000 eV), runs 0-3 = 2000 test events | 440 | ~25 |
| S3 | mu 0, m200 Lambda 774 eV extended to 1M events (runs 40-1999) | 1960 | ~20 |
| S2 | mu 0, every grid point, runs 4-43 = 20k training events | 4400 | ~245 |
| S4 | mu 60, every grid point, runs 0-3 = 2000 events | 440 | ~55 |
| S5 | mu 60, m200 at Lambda 278, 774, 2154 eV, 100k each | 600 | ~50 |

Total about 400 node-hours, leaving margin for retries and a later second pass. Each grid dataset is one
`<dataset>/v1` with 44 runs: runs 0-3 are the test runs, never trained on.

## Open items (not blocking)

- Cluster direction columns (local_eta, local_phi, eta_angle, ...) are all zero in the parquet: find where the
  ACTS writer leaves them empty; report before changing anything.
- MaskFormer and edge-geo memory/throughput with mu 60 events.
- About 1% of simulation runs (500 events each) stop with a fatal Geant4 error (em0033,
  G4MuPairProductionModel::DataCorrupted) in the hadron pair-production process the plugin gives quirks
  (EnableEM). Seen so far only in multi-threaded runs. Production retries failed runs once; physics unchanged
  mid-production. Question for Daniel: drop pair production for quirks (negligible for slow heavy quirks?) in a
  later version, or find the cause.
