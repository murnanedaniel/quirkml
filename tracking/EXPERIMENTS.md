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
| gnn-e1-dir-s0 | gnn-v2-node + cluster direction features (node: phi/eta_angle_abs, abs diag_sign, axis flag; edge: angle of the step to each hit's axis candidates, axis-axis angle); filter + GNN retrained, v1 embedding, seed 0; stitch 0.7, >= 4 hits | m200; test 17000-19999 | direction features (Stage E) | 0.733 (tight 0.072) | 0.573 | 0.965 (sm 0.993) | 0.104 | baseline seed 0 = gnn-v2-node (its rerun reproduced v2 bit for bit at filter epoch 0): 0.728 / 0.070 / fake 0.107. >= 3: 0.736 / 0.123; >= 5: 0.724 / 0.093; >= 6: 0.714 / 0.081. Switches per mixed track 2.95 (v2 2.99) |
| gnn-e1-dir-s1 | gnn-e1-dir-s0 with seed 1 (filter + GNN) | same | seed | 0.740 (tight 0.073) | 0.589 | 0.966 (sm 0.993) | 0.103 | >= 3: 0.744 / 0.121; >= 5: 0.732 / 0.091; >= 6: 0.721 / 0.081; seed spread vs s0 about 0.7 points |
| gnn-e1-base-s1 | gnn-v2-node rerun with seed 1 (filter + GNN; reference arm of the direction A/B) | same | seed | 0.724 (tight 0.069) | 0.566 | 0.965 (sm 0.993) | 0.107 | >= 3: 0.728 / 0.126; >= 5: 0.715 / 0.095; >= 6: 0.705 / 0.083. A/B at >= 4 hits, mean of 2 seeds: direction 0.736 / tight 0.072 / fake 0.104 vs without 0.726 / 0.070 / 0.107 (both direction seeds above both reference seeds; seed spread 0.4-0.7 points) |
| gnn-e2-pair-dir-s0 | coupled task: both quirks one particle, one truth chain over their union in true-time order; otherwise gnn-e1-dir-s0 (all tracks, cluster shape + direction features, filter + GNN retrained, seed 0); stitch 0.7, >= 4 hits (chosen on val) | m200; test 17000-19999; quirk numbers on the pair-truth copy (2991 pairs) | pair target | 0.920 (tight 0.126) coupled | - | 0.965 (sm 0.993) normal truth | 0.104 | ACTS pair truth: >= 3 hits 0.921 / tight 0.126 / fake 0.123; >= 5: 0.916 / 0.126 / 0.092. Same pair truth, gnn-v2-node: 0.606 (coordinator); MaskFormer pair target 0.859. Per-quirk on normal truth (not its target): 0.451 |
| mf-t2-lr1.25e-5 | MaskFormer, 10-layer encoder, 128 queries, object_bce 1.0, mask_bce 100 + focal 50; bs 8, 30k steps, WSD max LR 1.25e-5 | m200 train 0-15999; val 16000-16999 (1000 ev) | T2 LR scan | 0.139 (ot 0.1) / 0.138 (0.2) / 0.132 (0.3) / 0.100 (0.5) | 0.016 / 0.015 / 0.013 / 0.005 | 0.786 / 0.773 / 0.719 / 0.284 (sm 0.865 / 0.850 / 0.790 / 0.306) | 0.172 / 0.121 / 0.090 / 0.073 | faulty batched mask loss (fixed 2026-10-17), not a valid baseline; dup <= 0.0006; best val loss 3.81 at 30k, still falling; 1.6 h on 1 GPU |
| mf-t2-lr5e-5 | same, max LR 5e-5 | same | T2 LR scan | 0.179 (ot 0.1) / 0.178 (0.2) / 0.163 (0.3) / 0.090 (0.5) | 0.017 / 0.017 / 0.016 / 0.006 | 0.869 / 0.842 / 0.747 / 0.278 (sm 0.953 / 0.922 / 0.817 / 0.301) | 0.092 / 0.060 / 0.044 / 0.119 | faulty batched mask loss (fixed 2026-10-17), not a valid baseline; dup <= 0.0002; best val loss 2.98 at 30k, still falling; best of the scan |
| mf-t2-lr1e-4 | same, max LR 1e-4 | same | T2 LR scan | 0.167 (ot 0.1) / 0.155 (0.2) / 0.126 (0.3) / 0.069 (0.5) | 0.015 / 0.014 / 0.015 / 0.007 | 0.860 / 0.838 / 0.750 / 0.300 (sm 0.944 / 0.921 / 0.825 / 0.328) | 0.096 / 0.062 / 0.044 / 0.071 | faulty batched mask loss (fixed 2026-10-17), not a valid baseline; one loss spike (grad norm 1.8e6) recovered; best val loss 3.09 at 30k |
| mf-t2-lr2e-4 | same, max LR 2e-4 | same | T2 LR scan | 0.058 (ot 0.1) / 0.048 / 0.027 / 0.000 | 0.004 / 0.002 / 0.001 / 0 | 0.083 / 0.062 / 0.029 / 0.001 | 0.83 / 0.83 / 0.85 / 0.92 | faulty batched mask loss (fixed 2026-10-17), not a valid baseline; diverged after step ~7k (loss spikes to 1e12); evaluated at best val ckpt (step 7k, val loss 6.94) |
| mf-t2-lr5e-5-test | mf-t2-lr5e-5 (best val ckpt, step 30k) | m200 test 17000-19999 (3000 ev; 5962 quirks, 2973 events) | test set | 0.180 (ot 0.1) / 0.176 (0.2) / 0.155 (0.3) / 0.098 (0.5) | 0.023 / 0.021 / 0.018 / 0.011 | 0.871 / 0.843 / 0.755 / 0.290 (sm 0.951 / 0.921 / 0.824 / 0.312) | 0.091 / 0.059 / 0.045 / 0.076 | faulty batched mask loss (fixed 2026-10-17), not a valid baseline; dup <= 0.0003; at matched fake (~0.09-0.11) GNN v2: 0.73 |
| mf-t2fix-lr5e-5 | MaskFormer as mf-t2-lr5e-5, fixed batched mask loss; bs 8, 30k steps, WSD max LR 5e-5 | m200 train 0-15999; val 16000-16999 | loss fix (control) | 0.168 (ot 0.1) / 0.168 (0.2) / 0.144 (0.3) / 0.084 (0.5); tight 0.0005 | 0.024 / 0.024 / 0.018 / 0.014 | 0.872 / 0.848 / 0.757 / 0.313 (sm 0.957 / 0.930 / 0.831 / 0.340) | 0.093 / 0.059 / 0.043 / 0.060 | dup <= 0.0002; train events 0-999 at ot 0.1: quirk 0.174, tech 0.876, fake 0.088 (no train/val gap) |
| mf-long-lr5e-5 | same, 100k steps, WSD max LR 5e-5, val every 2k | same | 100k steps | 0.159 (ot 0.1) / 0.150 (0.2) / 0.112 (0.3) / 0.060 (0.5); tight 0.0005 | 0.034 / 0.029 / 0.024 / 0.013 | 0.881 / 0.839 / 0.739 / 0.321 (sm 0.968 / 0.922 / 0.814 / 0.352) | 0.107 / 0.057 / 0.035 / 0.050 | longer training helps ordinary tracks only |
| mf-qw10-lr5e-5 | mf-t2fix + quirk targets weight 10 in object BCE and mask BCE/focal (mean-one), LR 5e-5 | same | quirk weighting | 0.135 (ot 0.1) / 0.135 / 0.136 / 0.142 (0.5); tight 0 | 0.005 / 0.005 / 0.006 / 0.006 | 0.835 / 0.821 / 0.761 / 0.301 | 0.158 / 0.129 / 0.122 / 0.280 | worse than unweighted |
| mf-qw10-lr1e-4 | same, LR 1e-4 | same | quirk weighting, LR | 0.163 (ot 0.1) / 0.162 / 0.161 / 0.157 (0.5); tight 0.001 | 0.008 / 0.008 / 0.008 / 0.008 | 0.731 / 0.728 / 0.691 / 0.317 | 0.252 / 0.209 / 0.180 / 0.265 | no gain on quirks, more fakes |
| mf-D-bs1 | mf-t2fix at batch 1, 30k updates (matched in updates, 1/8 of the events) | m200 train; val 16000-16999 | batch 1 control | 0.091 (ot 0.1) / 0.083 / 0.068 / 0.001 (0.5) | 0.008 / 0.007 / 0.002 / 0 | 0.176 / 0.157 / 0.102 / 0.001 | 0.744 / 0.742 / 0.772 / 0.943 | undertrained per update; at matched events bs1 30k (val final-layer loss 1.56) ~ bs8 4k (1.62): batch 8 not suspect; coupled 0.160 |
| mf-D-noorigin | mf-t2fix with hit inputs x, y, z, r, layer + cluster shape; no s/eta/phi features, no (r, eta, phi) positional encoding | same | no origin features (+ cluster columns) | 0.171 (ot 0.1) / 0.169 / 0.160 / 0.119 (0.5) | 0.018 / 0.019 / 0.018 / 0.011 | 0.875 / 0.854 / 0.790 / 0.357 (sm 0.960 / 0.937 / 0.866 / 0.386) | 0.088 / 0.050 / 0.038 / 0.077 | val final-layer loss 0.61 vs 0.69; coupled 0.215 vs 0.187; quirk matched mask: double-majority 0.21, 85% of wrong hits from the other quirk (as t2fix) |
| mf-D-dir | mf-D-noorigin + offline cluster-direction features (M = (aa^T+bb^T)/2, diag_sign (aa^T-bb^T)/2, diag_sign, phi/eta_angle_abs, valid) | same | direction features (control: mf-D-noorigin, same seed) | 0.189 (ot 0.1) / 0.202 / 0.196 / 0.130 (0.5) | 0.027 / 0.023 / 0.027 / 0.014 | 0.881 / 0.867 / 0.816 / 0.345 (sm 0.965 / 0.947 / 0.890 / 0.371) | 0.097 / 0.053 / 0.034 / 0.055 | +0.02-0.04 quirk eff over control (sigma ~0.009); coupled 0.205 / 0.237 / 0.240; matched quirk mask double-majority 0.29 (control 0.21), 88% of wrong hits still from the other quirk |
| mf-D-pair-lr5e-5 | mf-t2fix setting with the quirk PAIR as one target (union of both quirks' hits); ordinary tracks unchanged; bs 8, 30k, WSD 5e-5 | m200 train; val 16000-16999 | pair target | per quirk 0.255 (ot 0.1) / 0.235 / 0.200 / 0.108 (0.5); coupled (pair double 50%) 0.862 / 0.825 / 0.733 / 0.430 | 0.013 / 0.010 / 0.005 / 0 | 0.879 / 0.846 / 0.748 / 0.285 (sm 0.954 / 0.920 / 0.814 / 0.307) | 0.074 / 0.041 / 0.030 / 0.050 | coupled efficiency from maskformer/scripts/coupled_eval.py (Python); per-quirk t2fix reference: coupled 0.187 at ot 0.1 |
| mf-D-pair-lr1e-4 | same, max LR 1e-4 | same | pair target, LR | per quirk 0.245 (ot 0.1) / 0.238 / 0.203; coupled 0.742 / 0.749 / 0.698 | 0.028 / 0.025 / 0.020 | 0.830 / 0.813 / 0.734 (sm 0.902 / 0.883 / 0.798) | 0.125 / 0.079 / 0.056 | 5e-5 better; 2.5e-5 queued |
| mf-D-pair-lr2.5e-5 | same, max LR 2.5e-5 | same | pair target, LR | per quirk 0.253 (ot 0.1) / 0.243 / 0.200 / 0.117; coupled 0.848 / 0.836 / 0.746 / 0.513 | 0.022 / 0.025 / 0.014 / 0.003 | 0.871 / 0.843 / 0.742 / 0.281 (sm 0.945 / 0.915 / 0.807 / 0.301) | 0.086 / 0.049 / 0.035 / 0.060 | LR scan for the pair objective: 5e-5 best (coupled 0.862 at ot 0.1) |
| mf-D-pair-lr5e-5-test | mf-D-pair-lr5e-5 (best val ckpt) | m200 test 17000-19999; truth as is (5962 quirks) and pair truth (pair = one particle, 2991 pair objects with >= 3 hits; quirk_tracking_data/pair_truth, tracking/make_pair_truth.py) | test set, ACTS coupled | ACTS pair-truth (coupled) very_loose 0.859 (ot 0.1) / 0.828 (0.2), tight 0.306 / 0.313; Python coupled 0.861 / 0.830 (2973 events); per quirk 0.252 / 0.234 | 0.016 / 0.010 (per quirk) | 0.881 / 0.848 (sm 0.954 / 0.919) | 0.072 / 0.042 (pair truth: 0.068 / 0.037) | ACTS and Python coupled agree within 0.003 |
| gnn-v2-pairtruth | gnn-v2-node stitch 0.7, >= 4 hits, scored against pair truth | same | coupled comparison | ACTS pair-truth very_loose 0.606, tight 0.006; Python coupled 0.605 (with both quirks >= 3 hits in the track: 0.432); per quirk (Python) 0.740 | - | sm 0.993 | 0.109 (pair truth) | per-quirk GNN tracks rarely hold the whole pair |
| mf-E-pair-s1 | mf-D-pair-lr5e-5 with seed 1 | m200; val 16000-16999, test 17000-19999 | seed | coupled ACTS (pair truth) test very_loose 0.829 (ot 0.1) / 0.804 (0.2), tight 0.251 / 0.265; val 0.834 / 0.809; Python test 0.831 / 0.807; per quirk test 0.250 / 0.229 | 0.021 / 0.013 | sm 0.948 / 0.921 | 0.079 / 0.047 (pair truth 0.075 / 0.042) | seed 0: test coupled 0.859 / 0.828 |
| mf-E-pair-s2 | mf-D-pair-lr5e-5 with seed 2 | same | seed | coupled ACTS test very_loose 0.878 (ot 0.1) / 0.811 (0.2), tight 0.336 / 0.344; val 0.879 / 0.814; Python test 0.880 / 0.814; per quirk test 0.242 / 0.217 | 0.013 / 0.007 | sm 0.959 / 0.934 | 0.071 / 0.040 (pair truth 0.066 / 0.035) | 3 seeds, test coupled very_loose: 0.855 +- 0.025 (ot 0.1), 0.814 +- 0.012 (ot 0.2); tight 0.298 +- 0.043 / 0.307 +- 0.040 (mean +- sample std) |
| mf-E-pairdir | pair target + no-origin inputs + cluster-direction features (quirk_pair + noorigin + direction configs), seed 0, LR 5e-5, 30k | same | direction features on the pair target (control: mf-D-pair-lr5e-5 seed 0; also differs by no-origin inputs) | coupled ACTS test very_loose 0.854 (ot 0.1) / 0.855 (0.2), tight 0.299 / 0.318; val 0.858 / 0.868; Python test 0.856 / 0.858; per quirk test 0.291 / 0.285 | 0.029 / 0.025 | sm 0.964 / 0.941 | 0.071 / 0.035 (pair truth 0.067 / 0.031) | flat in object threshold: at ot 0.2 +0.03 coupled and -0.006 fake vs seed 0; tight technical 0.77 vs 0.64 |
| mf-t2fix-pairtruth | mf-t2fix (per-quirk targets) on the test set, scored against pair truth | test 17000-19999 | coupled reference | coupled ACTS very_loose 0.190 (ot 0.1) / 0.185 (0.2), tight 0.007; Python 0.189 / 0.184; per quirk 0.166 / 0.162 | 0.022 / 0.022 | sm 0.954 / 0.927 | 0.092 / 0.059 | |

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

- 2026-10-23, separability ceilings (m200 L774, test 17000-19999, ACTS): with PERFECT edge scores on the GNN v2
  filter graph and the same builder (stitch 0.7, >= 4 hits), per-quirk 0.910 at 50% and 0.610 at 90% (GNN v2:
  0.728 / 0.070), fake 0.083. The 90% limit is the graph: 93.4% of quirk truth edges are in it (ordinary tracks
  97.5%); quirks with a complete graph reach 0.979 at 90%, one missing edge 0.317, two or more 0. Missing edges are
  long steps (median 212 mm vs 121 mm kept), mostly early in the chain. Timing: close (< 10 mm) hits of the two
  quirks are a median 26 ps apart (11 ps at GNN switch points); 30-50 ps resolution separates 5-9% of switch-point
  pairs. Scripts: tracking/separability/.
