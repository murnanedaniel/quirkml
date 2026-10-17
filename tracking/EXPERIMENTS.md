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

- 2026-10-16, m200 Lambda 774 eV (interleaved regime), GNN v1: the limit is separating the two quirks, not
  following turns. Best track per quirk holds a median 55% of its hits at 58% purity; 81% of the wrong hits
  are from the other quirk. Nearest hit of the other quirk: median 22 mm (17% within 5 mm); consecutive hits
  of the same quirk: median 135 mm apart. The GNN scores right and wrong next hits both at about 0.4-0.5.
