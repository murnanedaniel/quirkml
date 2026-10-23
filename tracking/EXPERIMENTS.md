# Quirk tracking pilot: experiment ledger

Goal: establish ML track finders on pu0 quirk events and hill-climb to the highest quirk track
efficiency at the lowest fake rate. Started 2026-10-15.

## Metrics (ACTS evaluator, proto-track level; the Kalman fit assumes helices)

1. quirk track efficiency, per quirk: the figure of merit;
2. quirk event efficiency: both quirks of the event matched;
3. technical efficiency for all other tracks (pT > 1 GeV, at least 7 hits, ...);
4. fake rate (and duplicate rate).

## Budget

400 GPU-hours and 100 CPU node-hours, on interactive / shared_interactive (debug for short jobs).

| Date | Job | Resource | Hours | Running total (GPU / node) |
|---|---|---|---|---|

## Data

| Dataset | Point | Events | Use |
|---|---|---|---|
| quirks/m200_L774_mu0/v1 | m = 200 GeV, Lambda = 774 eV, mu = 0 | 40 runs x 500 | first train (0-15999) / val (16000-16999) / test (17000-19999) |
| quirks/m400_L1000_mu0/v1 | m = 400 GeV, Lambda = 1 keV, mu = 0 | 8 runs x 500 | development set |

## Runs

| ID | Model | Data | Change from parent | Quirk track eff | Quirk event eff | Tech eff | Fake | Notes |
|---|---|---|---|---|---|---|---|---|
| mf-smoke2k | MaskFormer (LRT recipe, max LR 1.25e-5, bs 1) | m400 dev, 2000 steps | first run | 0/397 | 0 | 0/3728 | 0.93 | infrastructure check only; overfit of 4 events reached 8/8 quirks; 0.36 GPU-h so far |
| gnn-v1-ccwalk | metric learning r=0.07 -> filter (cut 0.005) -> GNN (128, 8 it.) -> CC+walkthrough | m200, filter/GNN on 4k train events; test 17000-17199 | first run | 0.598 (tight 0.045) | 0.45 | 0.944 | 0.307 | GNN keeps 48% of quirk edges at 0.5; outward-only walk splits quirks that turn inwards; 3.63 GPU-h |
| gnn-v1-cc | same, plain connected components at 0.5 | same | track building | 0.225 | 0.08 | 0.895 | 0.185 | |
| gnn-v1-stitch5 | v1 scores; CC+walk, then join pieces whose ends share an edge > 0.7; >= 5 hits per track | same | track building | 0.588 (tight 0.045) | 0.43 | 0.942 | 0.173 | fakes roughly halved for 1 point of quirk eff; orientation-free greedy walk: 0.388 |
| gnn-v2-node | v1 embedding; filter + GNN retrained on all 16k train events (quirk-edge weight 10), cluster shape on nodes; stitch 0.7, >= 4 hits | m200; test 17000-19999 (3000 ev) | 16k training events | 0.728 (tight 0.070) | 0.571 | 0.965 (sm 0.993) | 0.107 | >= 3 hits: 0.732 / fake 0.126; >= 5: 0.721 / 0.095; ccwalk alone: 0.729 / 0.246 |
| gnn-v2-node_edge | gnn-v2-node + cluster-shape differences as edge features | same | edge cluster features | 0.725 (tight 0.069) | 0.570 | 0.965 (sm 0.993) | 0.106 | >= 3 hits: 0.729 / 0.125; >= 5: 0.717 / 0.095; same as node within noise |
| gnn-v2-none | gnn-v2-node with the 4 cluster-shape columns zeroed | same | no cluster features | 0.707 (tight 0.062) | 0.543 | 0.962 (sm 0.992) | 0.113 | >= 3 hits: 0.711 / 0.133; >= 5: 0.699 / 0.101; cluster shape worth about 2 points |
| mf-t2-lr1.25e-5 | MaskFormer, 10-layer encoder, 128 queries, object_bce 1.0, mask_bce 100 + focal 50; bs 8, 30k steps, WSD max LR 1.25e-5 | m200 train 0-15999; val 16000-16999 (1000 ev) | T2 LR scan | 0.139 (ot 0.1) / 0.138 (0.2) / 0.132 (0.3) / 0.100 (0.5) | 0.016 / 0.015 / 0.013 / 0.005 | 0.786 / 0.773 / 0.719 / 0.284 (sm 0.865 / 0.850 / 0.790 / 0.306) | 0.172 / 0.121 / 0.090 / 0.073 | faulty batched mask loss (fixed 2026-10-17), not a valid baseline; dup <= 0.0006; best val loss 3.81 at 30k, still falling; 1.6 h on 1 GPU |
| mf-t2-lr5e-5 | same, max LR 5e-5 | same | T2 LR scan | 0.179 (ot 0.1) / 0.178 (0.2) / 0.163 (0.3) / 0.090 (0.5) | 0.017 / 0.017 / 0.016 / 0.006 | 0.869 / 0.842 / 0.747 / 0.278 (sm 0.953 / 0.922 / 0.817 / 0.301) | 0.092 / 0.060 / 0.044 / 0.119 | faulty batched mask loss (fixed 2026-10-17), not a valid baseline; dup <= 0.0002; best val loss 2.98 at 30k, still falling; best of the scan |
| mf-t2-lr1e-4 | same, max LR 1e-4 | same | T2 LR scan | 0.167 (ot 0.1) / 0.155 (0.2) / 0.126 (0.3) / 0.069 (0.5) | 0.015 / 0.014 / 0.015 / 0.007 | 0.860 / 0.838 / 0.750 / 0.300 (sm 0.944 / 0.921 / 0.825 / 0.328) | 0.096 / 0.062 / 0.044 / 0.071 | faulty batched mask loss (fixed 2026-10-17), not a valid baseline; one loss spike (grad norm 1.8e6) recovered; best val loss 3.09 at 30k |
| mf-t2-lr2e-4 | same, max LR 2e-4 | same | T2 LR scan | 0.058 (ot 0.1) / 0.048 / 0.027 / 0.000 | 0.004 / 0.002 / 0.001 / 0 | 0.083 / 0.062 / 0.029 / 0.001 | 0.83 / 0.83 / 0.85 / 0.92 | faulty batched mask loss (fixed 2026-10-17), not a valid baseline; diverged after step ~7k (loss spikes to 1e12); evaluated at best val ckpt (step 7k, val loss 6.94) |
| mf-t2-lr5e-5-test | mf-t2-lr5e-5 (best val ckpt, step 30k) | m200 test 17000-19999 (3000 ev; 5962 quirks, 2973 events) | test set | 0.180 (ot 0.1) / 0.176 (0.2) / 0.155 (0.3) / 0.098 (0.5) | 0.023 / 0.021 / 0.018 / 0.011 | 0.871 / 0.843 / 0.755 / 0.290 (sm 0.951 / 0.921 / 0.824 / 0.312) | 0.091 / 0.059 / 0.045 / 0.076 | faulty batched mask loss (fixed 2026-10-17), not a valid baseline; dup <= 0.0003; at matched fake (~0.09-0.11) GNN v2: 0.73 |

## Evaluator validation (m400_L1000_mu0, events 0-3999; very_loose / tight)

| Variant | Quirk track eff | Quirk event eff | Technical eff | Fake |
|---|---|---|---|---|
| oracle | 1 / 1 | 1 / 1 | 1 / 1 | 0 / 0 |
| each quirk split in two halves by time | 1.000 / 0.000 | 1.000 / 0.000 | 1.000 / 0.899 | 0.0015 / 0.0059 |
| 60% of quirk hits dropped | 0 / 0 | 0 / 0 | 0.899 / 0.899 | 0.0029 / 0.0029 |
| each quirk kept with probability 1/2 | 0.499 / 0.499 | 0.240 / 0.240 | 0.949 / 0.949 | 0 / 0 |
| nothing assigned | 0 | 0 | 0 | - |

Denominators: 7943 quirks (charged, primary, >= 3 hits; 57 of 8000 have fewer), 3964 events with both quirks.

## Findings

- 2026-10-17, MaskFormer: the vendored mask losses (mask_bce, mask_focal, mask_dice in
  packages/ml_tracking/src/ml_tracking/models/loss.py) broadcast the [batch, hit] padding mask as [batch, 1, hit]
  against the [object, hit] losses of the valid objects, so at batch size > 1 every object's loss mixed in the
  padding of every event in the batch (hits at high parquet index down-weighted by up to the batch size; batch 1
  unaffected). Evidence (2 events, random logits): batched vs per-event reference 0.400 vs 0.383 (focal), 0.608 vs
  0.480 (dice), 0.9076 vs 0.9064 (bce); after the fix (padding mask gathered per object) they agree and the 199 loss
  tests pass. All mf-t2 rows above trained at batch 8 with the faulty loss; t2fix_lr5e-5 is the fixed-loss control.

- 2026-10-16, m200 Lambda 774 eV (interleaved regime), GNN v1: the limit is separating the two quirks, not
  following turns. Best track per quirk holds a median 55% of its hits at 58% purity; 81% of the wrong hits
  are from the other quirk. Nearest hit of the other quirk: median 22 mm (17% within 5 mm); consecutive hits
  of the same quirk: median 135 mm apart. The GNN scores right and wrong next hits both at about 0.4-0.5.
