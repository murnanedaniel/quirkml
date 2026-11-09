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
| gnn-e2-pair-dir-s1 | gnn-e2-pair-dir-s0 with seed 1 (filter + GNN) | same | seed | 0.925 (tight 0.132) coupled | - | 0.965 (sm 0.993) normal truth | 0.102 | >= 3 hits: 0.926 / 0.132 / fake 0.121; >= 5: 0.922 / 0.132 / 0.090. Seeds 0 and 1 at >= 4 hits: 0.920, 0.925 (mean 0.922); seed 2 training |
| gnn-e2-pair-dir-s2 | gnn-e2-pair-dir-s0 with seed 2 | same | seed | 0.921 (tight 0.124) coupled; components 0.5: 0.954 (0.689) | - | sm 0.993 / 0.990 | 0.104 / 0.062 | 3 seeds, >= 4 hits: walk + stitch 0.922 +- 0.003 (tight 0.127 +- 0.004); components at 0.5 0.950 +- 0.004 (tight 0.694 +- 0.005), fake 0.060-0.062, technical_sm 0.989-0.990 (seed 1 components: 0.949 / 0.696 / fake 0.060). MaskFormer pair target 0.855 +- 0.025 |
| pair-perfect-v3 | NOT A CEILING (chain-only scores). Perfect PAIR-truth edge scores (1 on the time-ordered chain over both quirks' hits) on graph v3; stitch 0.7, >= 4 hits | m200; test 17000-19999; pair-truth copy (2991 pairs) | coupled ceiling | 0.649 (tight 0.383) coupled | - | sm 1.000 | 0.064 | 88.1% of pair-chain edges in the graph; 36% of pairs have every edge. Not an upper bound for this builder: the trained pair GNN on a similar graph scores 0.920 (see pair-perfect-pairarm) |
| pair-perfect-pairarm | NOT A CEILING (chain-only scores). Same on the pair arm's own graph (gnn-e2-pair-dir-s0 filter, cut 0.005, ~55k edges/event) | same | coupled ceiling | 0.784 (tight 0.453) coupled | - | sm 1.000 | 0.081 | 90.0% of pair-chain edges in the graph; 40% of pairs have every edge. The trained GNN on this graph reaches 0.920 / 0.126: with only the chain edges scored 1, the outward-oriented walk breaks the time-ordered A/B zigzag into pieces, while the GNN also scores other same-pair edges high |
| pair-allsame-v3-cc | coupled ceiling: every graph v3 edge with both hits in the quirk pair scored 1, all else 0; plain connected components (cut 0.5), >= 4 hits | m200; test 17000-19999; pair-truth copy | coupled ceiling | 0.998 (tight 0.975) coupled | - | (others unscored) | (quirk-only) | 166 same-pair edges/event. Same scores with walk + stitch 0.7: 0.959 / 0.435. Fake/technical not meaningful (non-quirk edges set to 0) |
| pair-allsame-pairarm-cc | same on the pair arm's graph (gnn-e2-pair-dir-s0 filter, cut 0.005) | same | coupled ceiling | 0.991 (tight 0.882) coupled | - | (others unscored) | (quirk-only) | 109 same-pair edges/event. Walk + stitch: 0.939 / 0.292. With perfect scores the coupled limit is the track builder (walk vs CC), not the graph |
| gnn-e2-pair-dir-s0-cc | gnn-e2-pair-dir-s0 scores; plain connected components at score > 0.5 (cut chosen on val 16000-16999), >= 4 hits | m200; test 17000-19999; pair-truth copy | track builder | 0.947 (tight 0.698) coupled | - | sm 0.990 (normal truth) | 0.061 | val sweep (Python matching): cc 0.3 0.961 / 0.726 but technical 0.946; cc 0.5 0.948 / 0.705 / fake 0.060 chosen; walk + stitch 0.933 / 0.117. Hybrid (components above 0.5, walk for the rest) on test: 0.948 / 0.698 / fake 0.081 |
| gnn-p3-pairv3-s0 | pair-target GNN on graph v3 (no filter retraining): every same-pair edge true (--pair-all-true), direction features, seed 0; components at 0.7 (rule on val: best coupled eff with fake <= its walk's fake and technical within 1 point) | m200; test 17000-19999; pair-truth copy | pair target on graph v3 | 0.981 (tight 0.821) coupled | - | sm 0.984 (normal truth) | 0.164 | walk + stitch on the same scores: 0.890 / 0.319 / fake 0.187. Not at matched fake with gnn-e2-pair-dir-s0-cc (0.947 / 0.698 at fake 0.061): on val, components at 0.5 gave 0.984 / 0.834 at fake 0.122. The higher fake comes from ordinary tracks (technical 0.965 vs 0.986 on val) |
| graph-v3-perfect-cc | per-quirk perfect scores on graph v3 with plain connected components (cut 0.5), >= 4 hits | m200; test 17000-19999 | track builder (ceiling) | 0.981 (tight 0.859) | 0.967 | sm 1.000 | 0.021 | walk + stitch on the same scores: 0.972 / 0.833 / fake 0.061 |
| gnn-g2-v3-dir-s0-cc | gnn-g2-v3-dir-s0 scores with components (cut 0.5) / hybrid (0.7), cuts chosen on val | m200; test 17000-19999 | track builder | cc 0.470 (tight 0.038); hybrid 0.549 (0.032) | 0.220 / 0.358 | sm 0.991 / 0.996 | 0.052 / 0.104 | components merge the two quirks (and quirk + other tracks) on GNN scores; walk + stitch stays far better for the per-quirk task (0.810 / 0.103) |
| graph-v3-perfect | perfect edge scores on the quirk graph v3: metric learning with quirk chain pairs weight 10 among the positives (r = 0.075, k <= 512), filter with quirk-edge weight 50 + direction features (cut 0.001, ~125k edges/event); stitch 0.7, >= 4 hits | m200; test 17000-19999 | graph construction for quirks | 0.972 (tight 0.833) | 0.951 | 0.998 (sm 1.000) | 0.061 | ceiling only, no GNN. Quirk truth edges in graph 98.1% (v2 graph 93.4%); quirks with every edge 81.6%. >= 3 hits: 0.977 / 0.836 / fake 0.070; >= 5: 0.961 / 0.827 / 0.054. v2 graph ceiling: 0.910 / 0.610 |
| gnn-g2-v3-dir-s0 | GNN (v2 recipe + direction features, seed 0, 16 epochs) on quirk graph v3 (see graph-v3-perfect; ~125k edges/event); stitch 0.7, >= 4 hits | m200; test 17000-19999 | candidate graph v3 | 0.810 (tight 0.103) | 0.688 | 0.974 (sm 0.993) | 0.093 | >= 3 hits: 0.812 / 0.104 / fake 0.107; >= 5: 0.802 / 0.102 / 0.084; >= 6: 0.794 / 0.099 / 0.075. vs gnn-e1-dir (v2 graph) 0.736 / 0.072 at fake 0.104. Python dig: quirk hits on no track 6.2% (v2 10.4%), coupled one track 0.737 (v2 0.605), union at 0.9 0.636 (0.507), switches per mixed track 3.07 (2.99). Seed 1 training |
| gnn-g2-v3-dir-s1 | gnn-g2-v3-dir-s0 with seed 1 | same | seed | 0.817 (tight 0.115) | 0.698 | 0.974 (sm 0.993) | 0.092 | >= 3: 0.820 / 0.115 / 0.106; >= 5: 0.810 / 0.113 / 0.083; >= 6: 0.800 / 0.110 / 0.074. Two seeds at >= 4 hits: 0.810, 0.817 (mean 0.814, tight 0.109) |
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
| mf-E-pairdir-s1 | mf-E-pairdir with seed 1 | same | seed | coupled ACTS test very_loose 0.837 (ot 0.1) / 0.842 (0.2), tight 0.330 / 0.343; val 0.839 / 0.849; Python test 0.839 / 0.846; per quirk test 0.279 / 0.272 | 0.028 / 0.024 | sm 0.963 / 0.942 | 0.075 / 0.038 (pair truth 0.072 / 0.034) | |
| mf-E-pairdir-s2 | mf-E-pairdir with seed 2 (resumed once at ~26k) | same | seed | coupled ACTS test very_loose 0.859 (ot 0.1) / 0.866 (0.2), tight 0.415 / 0.425; val 0.866 / 0.870; Python test 0.862 / 0.869; per quirk test 0.280 / 0.273 | 0.016 / 0.011 | sm 0.964 / 0.940 | 0.062 / 0.030 (pair truth 0.058 / 0.026) | pair+direction, 3 seeds, test coupled very_loose 0.850 +- 0.012 (ot 0.1), 0.854 +- 0.012 (ot 0.2); tight 0.348 +- 0.060 / 0.362 +- 0.056; pair-only seeds 0.855 +- 0.025 / 0.814 +- 0.012 |
| mf-E-pairdir-long | mf-E-pairdir, 100k steps WSD (cooldown from 85k), resumed once at 70k; checkpoint chosen by val coupled efficiency: final (100k) | same | 100k steps | coupled ACTS test very_loose 0.915 (ot 0.1) / 0.905 (0.2), tight 0.564 / 0.567; val 0.911 / 0.905; Python test 0.917 / 0.908; per quirk test 0.319 / 0.313 | 0.011 / 0.010 | sm 0.949 / 0.896 | 0.061 / 0.028 (pair truth 0.057 / 0.024) | val coupled (Python, ot 0.1/0.2) by checkpoint: 60k 0.877/0.874, 92k (best val loss) 0.908/0.898, 100k 0.912/0.906. Final-layer val loss rises 0.581 -> 0.651 in the last 8k steps of the cooldown (LR 3.4e-5 -> 0.9e-5) while train loss falls 2.48 -> 2.21 and val efficiency rises; not linked to the resume at 70k (LR and val loss continuous there) |
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

- 2026-10-24, separability of the two quirks inside a pair component (graph v3, test 17000-19999, 2973 pairs; no
  model, graph only). Within one pair component, the two quirks are joined by a median of 69 graph edges between quirk A
  and quirk B hits (mean 86; 99.9% of pairs have at least one), so the pair component can only be split by scores, not by
  topology. Keeping only same-quirk edges, the largest connected piece holds >= 50% of a quirk's hits for 99.5% of
  quirks (both quirks of the pair: 99.2%), >= 90% for 91.7% (pairs 86.1%) and all hits for 87.6% (pairs 80.0%). So
  with perfect per-quirk scores a split after components is possible for about 86% of pairs at the 90% level (the
  per-quirk perfect-score components ceiling, graph-v3-perfect-cc, gives tight 0.859).

- 2026-10-26, GNN grid (Stage F), no retraining, 110 test sets of 2000 events (tracking/grid/gnn_grid.csv, maps
  tracking/grid/gnn_grid_*.png). Per-quirk: graph v3 + GNN g2_q50_dir_s0, walk + stitch, >= 4 hits; coupled: pair-arm
  graph + pair GNN e2_pair_dir_s0, components at 0.5, >= 4 hits; ACTS. Per-quirk very_loose falls with Lambda and
  slowly with mass: 0.99 at 100 eV, 0.95-0.98 at 167 eV, 0.81-0.92 at 278 eV, 0.73-0.82 at 1292 eV, 0.48-0.51 at
  10 keV; tight is 0.88-0.91 at 100 eV, 0.6-0.7 at 167 eV, 0.33-0.37 at 278 eV and below 0.03 from 1292 eV up.
  Quirk event efficiency 0.97-0.99 at 100 eV, 0.03-0.16 at 10 keV (lowest at m50). Coupled very_loose is >= 0.885
  everywhere from 774 eV up (0.92-0.99 for 774-3594 eV), 0.73-0.96 at 464 eV, and drops to 0.50-0.85 at 167-278 eV,
  where the quirks are far apart and one track rarely holds both (there the per-quirk task is easy). Coupled tight
  peaks at 774-2154 eV (0.59-0.91) and falls towards both ends. (Ranges corrected against gnn_grid.csv, 2026-10-26.) technical_sm 0.991-0.994 and fake 0.089-0.096 (per-quirk), 0.059-0.068
  (coupled) on every set. Rate of change with Lambda is far larger than with mass at fixed Lambda.

## Strict matching (> 50%, > 90%), 2026-10-26

From 2026-10-26 the standard working points are strict, as in ATLAS: a track matches a particle if purity AND
completeness are strictly above 50% (loose) or 90% (tight). ACTS TrackTruthMatcher, and so every number above, uses
>= (acts_v5 TrackTruthMatcher.cpp lines 97-102). The rows above are not changed; they are >= numbers.

- evaluate.py (and quirk_rule.py) now also score the points very_loose_strict (matchingRatio 0.5 + 1e-9) and
  tight_strict (0.9 + 1e-9), in addition to very_loose / tight (unchanged): summary.json, per_event.parquet, quirk event
  efficiency, the Python cross-check and the quirk-rule block cover all four. tracking_core is not changed. Check on
  gnn-g2-v3-dir-s0 (test): ACTS strict 4445/5962 = 0.746 per quirk, event 1844/2973 = 0.620, tight 585 (0.098) /
  event 199, equal to the Python strict numbers per event.
- The table below is Python (strict50.py / strict_all.py: TrackTruthMatcher's rule with evaluate.python_matching's
  tie handling) on the kept predictions. Its >= numbers equal the ACTS numbers in summary.json for all 486 evaluations
  scored (266 ledger and side evaluations, 220 GNN grid). Fake rate and technical efficiency also change with strict
  matching: ACTS reruns with the new evaluate.py for the headline rows are in the subsection "Strict ACTS reruns" at
  the end of this section (only those rows have strict fake / technical numbers).
- What sits at exactly 50%. Every quirk (or pair) matched at >= 50% and not at > 50% has its matching track exactly at
  purity 50% or completeness 50%. Per-quirk GNNs: both kinds (gnn-g2-v3-dir-s0: 383 quirks, 203 at purity 50%, 248
  at completeness 50%, 68 both), typically a track that holds exactly half of a quirk's hits or exactly equal numbers
  of hits from the two quirks. Pair-target models scored per quirk (MaskFormer pair, pair GNN): almost all at purity
  50%, a pair track with equally many hits of each quirk, which >= credited to one quirk through the majority
  tie-break (mf-E-pairdir-long ot 0.1: 622 quirks, 590 at purity 50%). Coupled scores: completeness 50%, a track
  holding exactly half of the pair's hits, e.g. one whole quirk when both quirks have the same hit count (gnn-v2 on
  pair truth: 495 pairs, 0.606 -> 0.440). Perfect-score ceilings: all at completeness 50% (a quirk split into two
  equal pieces). Tight changes by at most 2.2 points (pair-allsame-v3 walk + stitch), mostly 0.5-1.2.
- Headline (test 17000-19999): per-quirk GNN v3 0.814 -> 0.748 (2 seeds; tight 0.109 -> 0.103; event 0.693 ->
  0.622); GNN v2 0.728 -> 0.650; coupled pair GNN components 0.950 -> 0.944 (3 seeds; tight 0.694 -> 0.688);
  MaskFormer pair + direction long 0.915 -> 0.907 (ot 0.1; tight 0.564 -> 0.553); ceilings graph-v3-perfect 0.971 ->
  0.966, -cc 0.981 -> 0.976. Per-quirk tasks lose 6-10 points, coupled scores and ceilings 0.1-2.5 points
  (gnn-v2-pairtruth and pair truth tracks excepted).
- GNN grid (tracking/grid/gnn_grid_strict.csv, all 110 sets; maps gnn_grid_*_strict.png from plot_gnn_grid.py
  --strict): per-quirk loses 0.4 points at 100 eV, 5-7 at 278-774 eV and 10-13 at 1292-10000 eV (strict: 0.98-0.99 at
  100 eV, 0.70-0.80 at 774 eV, 0.37-0.42 at 10 keV); quirk event efficiency loses up to 13 points. Coupled loses 0.3-1.6
  points from 464 eV up but 9-14 points at 100-167 eV (strict 0.70-0.75 at 100 eV, 0.48-0.72 at 167 eV), where the
  pair tracks often hold exactly one whole quirk of two with equal hit counts. Tight changes by at most 1.9 points.
- Files: /pscratch/sd/d/danieltm/quirk_tracking/strict50/ (strict50.py, strict_all.py, make_table.py,
  strict_main.csv, strict_grid.csv, test_eval_g2s0/ = the ACTS strict check); tracking/grid/make_gnn_grid_strict.py.
  "at exactly 50%" = number of quirks (per-quirk) or pairs (coupled) matched at >= 50% but not > 50%; with purity /
  completeness breakdown for single per-quirk rows (a quirk can have both). Seeds: mean +- sample std; at-50% counts per seed.

| Row | Metric | loose >= 50% | loose > 50% | at exactly 50% | tight >= 90% | tight > 90% | event >= 50% | event > 50% |
|---|---|---|---|---|---|---|---|---|
| **Per-quirk task, GNN** | | | | | | | | |
| gnn-v1-ccwalk (200 ev) | per-quirk | 0.598 | 0.495 | 41 (pur 15, comp 32) | 0.045 | 0.045 | 0.450 | 0.360 |
| gnn-v1-cc (200 ev) | per-quirk | 0.225 | 0.165 | 24 (pur 13, comp 14) | 0.018 | 0.018 | 0.080 | 0.060 |
| gnn-v1-stitch5 (200 ev) | per-quirk | 0.588 | 0.485 | 41 (pur 14, comp 32) | 0.045 | 0.045 | 0.430 | 0.340 |
| gnn-v2-node | per-quirk | 0.728 | 0.650 | 467 (pur 180, comp 350) | 0.070 | 0.065 | 0.571 | 0.486 |
| gnn-v2-node_edge | per-quirk | 0.725 | 0.642 | 496 (pur 192, comp 393) | 0.069 | 0.064 | 0.570 | 0.484 |
| gnn-v2-none | per-quirk | 0.707 | 0.627 | 477 (pur 199, comp 346) | 0.062 | 0.057 | 0.543 | 0.467 |
| gnn-e1-base-s1 | per-quirk | 0.724 | 0.641 | 494 (pur 215, comp 374) | 0.069 | 0.064 | 0.566 | 0.480 |
| gnn-e1-dir-s0 | per-quirk | 0.732 | 0.656 | 456 (pur 203, comp 350) | 0.072 | 0.065 | 0.573 | 0.496 |
| gnn-e1-dir-s1 | per-quirk | 0.740 | 0.662 | 463 (pur 157, comp 375) | 0.072 | 0.068 | 0.589 | 0.505 |
| gnn-e1-dir, 2 seeds | per-quirk | 0.736 +- 0.005 | 0.659 +- 0.004 | 456/463 | 0.072 +- 0.001 | 0.066 +- 0.002 | 0.581 +- 0.011 | 0.500 +- 0.006 |
| gnn-g2-v3-dir-s0 | per-quirk | 0.810 | 0.746 | 383 (pur 203, comp 248) | 0.103 | 0.098 | 0.688 | 0.620 |
| gnn-g2-v3-dir-s1 | per-quirk | 0.817 | 0.751 | 397 (pur 188, comp 277) | 0.115 | 0.109 | 0.698 | 0.623 |
| gnn-g2-v3-dir, 2 seeds | per-quirk | 0.814 +- 0.005 | 0.748 +- 0.004 | 383/397 | 0.109 +- 0.008 | 0.103 +- 0.007 | 0.693 +- 0.007 | 0.622 +- 0.002 |
| gnn-g2-v3-dir-s0-cc, components 0.5 | per-quirk | 0.470 | 0.397 | 436 (pur 170, comp 291) | 0.038 | 0.036 | 0.220 | 0.175 |
| gnn-g2-v3-dir-s0-cc, hybrid 0.7 | per-quirk | 0.549 | 0.455 | 561 (pur 126, comp 479) | 0.032 | 0.031 | 0.358 | 0.271 |
| gnn-e2-pair-dir-s0 (pair model, normal truth) | per-quirk | 0.451 | 0.374 | 455 (pur 424, comp 41) | 0.001 | 0.001 | 0.014 | 0.009 |
| gnn-e2-pair-dir-s0-cc (pair model, normal truth) | per-quirk | 0.435 | 0.335 | 595 (pur 576, comp 23) | 0.001 | 0.001 | 0.007 | 0.004 |
| gnn-p3-pairv3-s0 cc 0.7 (normal truth) | per-quirk | 0.440 | 0.340 | 595 (pur 588, comp 10) | 0.001 | 0.001 | 0.003 | 0.002 |
| **Per-quirk ceilings and baselines** | | | | | | | | |
| v2-graph perfect scores, stitch (2026-10-23) | per-quirk | 0.910 | 0.887 | 137 (pur 0, comp 137) | 0.610 | 0.601 | 0.858 | 0.822 |
| graph-v3-perfect, stitch | per-quirk | 0.971 | 0.966 | 32 (pur 0, comp 32) | 0.833 | 0.828 | 0.951 | 0.941 |
| graph-v3-perfect-cc | per-quirk | 0.981 | 0.976 | 25 (pur 0, comp 25) | 0.859 | 0.854 | 0.967 | 0.959 |
| stageC ACTS CKF tracks | per-quirk | 0.002 | 0.001 | 4 (pur 2, comp 2) | 0.000 | 0.000 | 0.000 | 0.000 |
| stageC ACTS truth tracks | per-quirk | 0.276 | 0.271 | 29 (pur 1, comp 28) | 0.135 | 0.131 | 0.193 | 0.188 |
| stageC oracle (R1 labels) | per-quirk | 1.000 | 1.000 | 0 (pur 0, comp 0) | 1.000 | 1.000 | 1.000 | 1.000 |
| stageC oracle (quirk rule labels) | per-quirk | 1.000 | 1.000 | 1 (pur 1, comp 0) | 0.970 | 0.963 | 1.000 | 1.000 |
| **Coupled task (pair truth), GNN** | | | | | | | | |
| gnn-v2-pairtruth | coupled | 0.606 | 0.440 | 495 | 0.006 | 0.006 | - | - |
| gnn-e2-pair-dir-s0, walk + stitch | coupled | 0.920 | 0.909 | 32 | 0.126 | 0.120 | - | - |
| gnn-e2-pair-dir-s1, walk + stitch | coupled | 0.925 | 0.914 | 31 | 0.132 | 0.122 | - | - |
| gnn-e2-pair-dir-s2, walk + stitch | coupled | 0.921 | 0.913 | 23 | 0.124 | 0.114 | - | - |
| gnn-e2-pair-dir walk + stitch, 3 seeds | coupled | 0.922 +- 0.003 | 0.912 +- 0.003 | 32/31/23 | 0.127 +- 0.004 | 0.118 +- 0.004 | - | - |
| gnn-e2-pair-dir-s0-cc | coupled | 0.947 | 0.941 | 15 | 0.698 | 0.692 | - | - |
| gnn-e2-pair-dir-s1, components 0.5 | coupled | 0.949 | 0.944 | 13 | 0.696 | 0.689 | - | - |
| gnn-e2-pair-dir-s2, components 0.5 | coupled | 0.954 | 0.947 | 21 | 0.689 | 0.681 | - | - |
| gnn-e2-pair-dir components 0.5, 3 seeds | coupled | 0.950 +- 0.004 | 0.944 +- 0.003 | 15/13/21 | 0.694 +- 0.005 | 0.688 +- 0.006 | - | - |
| gnn-e2-pair-dir-s0, hybrid 0.5 | coupled | 0.948 | 0.942 | 15 | 0.698 | 0.693 | - | - |
| gnn-p3-pairv3-s0, components 0.7 | coupled | 0.981 | 0.980 | 3 | 0.821 | 0.813 | - | - |
| gnn-p3-pairv3-s0, walk + stitch | coupled | 0.890 | 0.876 | 43 | 0.319 | 0.306 | - | - |
| **Coupled ceilings and baselines** | | | | | | | | |
| pair-perfect-v3 (not a ceiling) | coupled | 0.649 | 0.637 | 38 | 0.383 | 0.381 | - | - |
| pair-perfect-pairarm (not a ceiling) | coupled | 0.784 | 0.772 | 36 | 0.453 | 0.449 | - | - |
| pair-allsame-v3-cc | coupled | 0.998 | 0.998 | 1 | 0.975 | 0.974 | - | - |
| pair-allsame-v3, walk + stitch | coupled | 0.959 | 0.949 | 28 | 0.435 | 0.413 | - | - |
| pair-allsame-pairarm-cc | coupled | 0.991 | 0.989 | 6 | 0.882 | 0.877 | - | - |
| pair-allsame-pairarm, walk + stitch | coupled | 0.939 | 0.929 | 30 | 0.292 | 0.277 | - | - |
| v2-graph perfect scores, stitch, pair truth | coupled | 0.609 | 0.452 | 469 | 0.000 | 0.000 | - | - |
| stageC ACTS truth tracks, pair truth | coupled | 0.113 | 0.065 | 143 | 0.000 | 0.000 | - | - |
| stageC ACTS CKF tracks, pair truth | coupled | 0.001 | 0.001 | 0 | 0.000 | 0.000 | - | - |
| **Coupled task (pair truth), MaskFormer, test, ot 0.1** | | | | | | | | |
| mf-t2fix-pairtruth | coupled | 0.190 | 0.165 | 74 | 0.007 | 0.007 | - | - |
| mf-D-pair-lr5e-5-test (seed 0) | coupled | 0.859 | 0.846 | 38 | 0.306 | 0.299 | - | - |
| mf-E-pair-s1 | coupled | 0.829 | 0.815 | 42 | 0.251 | 0.241 | - | - |
| mf-E-pair-s2 | coupled | 0.878 | 0.866 | 36 | 0.336 | 0.325 | - | - |
| MaskFormer pair, 3 seeds | coupled | 0.855 +- 0.025 | 0.842 +- 0.026 | 38/42/36 | 0.298 +- 0.043 | 0.289 +- 0.043 | - | - |
| mf-E-pairdir | coupled | 0.854 | 0.841 | 40 | 0.299 | 0.289 | - | - |
| mf-E-pairdir-s1 | coupled | 0.837 | 0.815 | 65 | 0.330 | 0.322 | - | - |
| mf-E-pairdir-s2 | coupled | 0.859 | 0.849 | 29 | 0.415 | 0.404 | - | - |
| MaskFormer pair + direction, 3 seeds | coupled | 0.850 +- 0.011 | 0.835 +- 0.017 | 40/65/29 | 0.348 +- 0.060 | 0.338 +- 0.059 | - | - |
| mf-E-pairdir-long (final, 100k) | coupled | 0.915 | 0.907 | 24 | 0.564 | 0.553 | - | - |
| **Coupled task (pair truth), MaskFormer, test, ot 0.2** | | | | | | | | |
| mf-t2fix-pairtruth | coupled | 0.185 | 0.161 | 71 | 0.007 | 0.007 | - | - |
| mf-D-pair-lr5e-5-test (seed 0) | coupled | 0.828 | 0.819 | 27 | 0.313 | 0.305 | - | - |
| mf-E-pair-s1 | coupled | 0.804 | 0.792 | 36 | 0.265 | 0.256 | - | - |
| mf-E-pair-s2 | coupled | 0.811 | 0.803 | 25 | 0.344 | 0.333 | - | - |
| MaskFormer pair, 3 seeds | coupled | 0.814 +- 0.012 | 0.805 +- 0.014 | 27/36/25 | 0.308 +- 0.040 | 0.298 +- 0.039 | - | - |
| mf-E-pairdir | coupled | 0.855 | 0.843 | 36 | 0.318 | 0.308 | - | - |
| mf-E-pairdir-s1 | coupled | 0.842 | 0.825 | 51 | 0.343 | 0.334 | - | - |
| mf-E-pairdir-s2 | coupled | 0.866 | 0.857 | 27 | 0.425 | 0.415 | - | - |
| MaskFormer pair + direction, 3 seeds | coupled | 0.854 +- 0.012 | 0.841 +- 0.016 | 36/51/27 | 0.362 +- 0.056 | 0.352 +- 0.056 | - | - |
| mf-E-pairdir-long (final, 100k) | coupled | 0.905 | 0.900 | 15 | 0.567 | 0.556 | - | - |
| **Per-quirk, MaskFormer, test, ot 0.1** | | | | | | | | |
| mf-t2-lr5e-5-test (faulty loss) | per-quirk | 0.180 | 0.120 | 359 (pur 153, comp 250) | 0.000 | 0.000 | 0.023 | 0.011 |
| mf-t2fix (test) | per-quirk | 0.166 | 0.110 | 337 (pur 144, comp 225) | 0.001 | 0.001 | 0.022 | 0.010 |
| mf-D-pair-lr5e-5-test | per-quirk | 0.252 | 0.170 | 487 (pur 452, comp 47) | 0.000 | 0.000 | 0.016 | 0.007 |
| mf-E-pair-s1 | per-quirk | 0.250 | 0.169 | 486 (pur 438, comp 67) | 0.001 | 0.001 | 0.021 | 0.012 |
| mf-E-pair-s2 | per-quirk | 0.242 | 0.161 | 482 (pur 453, comp 39) | 0.000 | 0.000 | 0.013 | 0.006 |
| mf-E-pairdir | per-quirk | 0.291 | 0.197 | 559 (pur 506, comp 72) | 0.000 | 0.000 | 0.029 | 0.017 |
| mf-E-pairdir-s1 | per-quirk | 0.279 | 0.187 | 552 (pur 500, comp 82) | 0.000 | 0.000 | 0.028 | 0.012 |
| mf-E-pairdir-s2 | per-quirk | 0.280 | 0.196 | 499 (pur 465, comp 50) | 0.000 | 0.000 | 0.016 | 0.008 |
| mf-E-pairdir-long (final) | per-quirk | 0.319 | 0.214 | 622 (pur 590, comp 42) | 0.000 | 0.000 | 0.011 | 0.005 |
| **Per-quirk, MaskFormer, test, ot 0.2** | | | | | | | | |
| mf-t2-lr5e-5-test (faulty loss) | per-quirk | 0.176 | 0.119 | 342 (pur 151, comp 234) | 0.000 | 0.000 | 0.021 | 0.009 |
| mf-t2fix (test) | per-quirk | 0.162 | 0.106 | 332 (pur 142, comp 218) | 0.000 | 0.000 | 0.022 | 0.009 |
| mf-D-pair-lr5e-5-test | per-quirk | 0.234 | 0.156 | 466 (pur 438, comp 38) | 0.000 | 0.000 | 0.010 | 0.005 |
| mf-E-pair-s1 | per-quirk | 0.229 | 0.154 | 445 (pur 417, comp 44) | 0.000 | 0.000 | 0.013 | 0.008 |
| mf-E-pair-s2 | per-quirk | 0.217 | 0.144 | 437 (pur 420, comp 27) | 0.000 | 0.000 | 0.007 | 0.004 |
| mf-E-pairdir | per-quirk | 0.285 | 0.193 | 552 (pur 509, comp 64) | 0.000 | 0.000 | 0.025 | 0.015 |
| mf-E-pairdir-s1 | per-quirk | 0.272 | 0.184 | 528 (pur 485, comp 67) | 0.000 | 0.000 | 0.024 | 0.011 |
| mf-E-pairdir-s2 | per-quirk | 0.273 | 0.192 | 483 (pur 460, comp 34) | 0.000 | 0.000 | 0.011 | 0.005 |
| mf-E-pairdir-long (final) | per-quirk | 0.313 | 0.209 | 615 (pur 592, comp 34) | 0.000 | 0.000 | 0.010 | 0.004 |
| mf-F-grid | mf-E-pairdir-long (100k ckpt), no retraining, on the 110 mu0 grid test sets (runs 0-3, 2000 events each; m 50-400 GeV, Lambda 100-10000 eV) | quirk_tracking_data/testruns/<set>; direction features and pair truth built per set (tracking/grid/grid_prep.sh, hits identical to production) | grid generalisation | coupled (pair truth, strict > 50%) median 0.944 [0.221, 0.985] at ot 0.1 (0.937 at 0.2); tight (> 90%) median 0.563 [0.008, 0.844]; per Lambda (median over m): 100: 0.48, 167: 0.25, 278: 0.36, 464: 0.71, 774: 0.92, >= 1292: 0.95-0.98; per quirk median 0.270 | - | sm median 0.948 [0.937, 0.959] | 0.070 [0.062, 0.088] (ot 0.2: 0.033) | table tracking/grid/stageF_results.{csv,md}, maps stageF_*.png; vs GNN grid (gnn_grid_strict.csv): MF higher coupled at Lambda >= 2154 eV (e.g. 10 keV 0.95 vs 0.89), lower at Lambda <= 1000 eV (167 eV: 0.25 vs 0.57); stageF_vs_gnn.csv / .png. m200_L774 (training point) not in the grid sets |
| **Per-quirk, MaskFormer, val 16000-16999, ot 0.1** | | | | | | | | |
| mf-t2-lr1.25e-5 | per-quirk | 0.139 | 0.094 | 90 (pur 45, comp 55) | 0.001 | 0.001 | 0.016 | 0.010 |
| mf-t2-lr5e-5 | per-quirk | 0.178 | 0.126 | 105 (pur 40, comp 78) | 0.000 | 0.000 | 0.017 | 0.008 |
| mf-t2-lr1e-4 | per-quirk | 0.167 | 0.111 | 111 (pur 63, comp 60) | 0.000 | 0.000 | 0.015 | 0.006 |
| mf-t2-lr2e-4 | per-quirk | 0.058 | 0.035 | 46 (pur 28, comp 23) | 0.000 | 0.000 | 0.004 | 0.001 |
| mf-t2fix-lr5e-5 | per-quirk | 0.168 | 0.109 | 117 (pur 58, comp 76) | 0.001 | 0.001 | 0.024 | 0.015 |
| mf-long-lr5e-5 | per-quirk | 0.159 | 0.110 | 97 (pur 18, comp 84) | 0.001 | 0.001 | 0.034 | 0.023 |
| mf-qw10-lr5e-5 | per-quirk | 0.135 | 0.089 | 90 (pur 28, comp 70) | 0.000 | 0.000 | 0.005 | 0.003 |
| mf-qw10-lr1e-4 | per-quirk | 0.163 | 0.113 | 99 (pur 68, comp 42) | 0.001 | 0.001 | 0.008 | 0.003 |
| mf-D-bs1 | per-quirk | 0.090 | 0.058 | 64 (pur 43, comp 28) | 0.000 | 0.000 | 0.008 | 0.003 |
| mf-D-noorigin | per-quirk | 0.171 | 0.125 | 91 (pur 35, comp 63) | 0.000 | 0.000 | 0.018 | 0.009 |
| mf-D-dir | per-quirk | 0.189 | 0.137 | 104 (pur 43, comp 70) | 0.000 | 0.000 | 0.027 | 0.016 |
| mf-D-pair-lr5e-5 | per-quirk | 0.255 | 0.171 | 168 (pur 156, comp 17) | 0.001 | 0.001 | 0.013 | 0.007 |
| mf-D-pair-lr1e-4 | per-quirk | 0.245 | 0.158 | 173 (pur 137, comp 47) | 0.001 | 0.001 | 0.028 | 0.010 |
| mf-D-pair-lr2.5e-5 | per-quirk | 0.253 | 0.173 | 159 (pur 145, comp 21) | 0.001 | 0.001 | 0.022 | 0.008 |

### Strict ACTS reruns (fake rate, technical_sm), 2026-10-26

evaluate.py with the strict points on the headline test rows (49 evaluations, debug CPU jobs 60337045, 60337341,
60338821; outputs /pscratch/sd/d/danieltm/quirk_tracking/strict50/acts/<name>/, per-run table acts_strict_rerun.csv,
columns acts_* in strict_main.csv). ACTS strict quirk, coupled and event numbers equal the Python strict numbers
above on every run, and the >= numbers equal the original evaluations. Fake rate and technical_sm are from the
normal-truth run; "coupled fake" is the fake rate of the pair-truth run (pair as one particle). Per-quirk numbers of
pair-target models are in brackets (not their task). Strict matching raises the fake rate by 0.004-0.027: a track
whose majority particle holds exactly 50% of its hits (or exactly 50% of that particle's hits) is no longer matched,
so it counts as fake. Technical_sm drops by 0-0.006.

| Model (test 17000-19999) | per-quirk >= / > | quirk event >= / > | coupled >= / > | coupled tight >= / > | fake >= / > | technical_sm >= / > | coupled fake (pair truth) >= / > |
|---|---|---|---|---|---|---|---|
| GNN v3 (g2-dir), walk + stitch, 2 seeds | 0.814 +- 0.005 / 0.748 +- 0.004 | 0.693 +- 0.007 / 0.622 +- 0.002 | - | - | 0.092 +- 0.001 / 0.117 +- 0.001 | 0.993 +- 0.000 / 0.991 +- 0.000 | - |
| GNN v2-node, walk + stitch | 0.728 / 0.650 | 0.571 / 0.486 | - | - | 0.107 / 0.134 | 0.993 / 0.991 | - |
| GNN e1-dir, walk + stitch, 2 seeds | 0.736 +- 0.005 / 0.659 +- 0.004 | 0.581 +- 0.011 / 0.500 +- 0.006 | - | - | 0.104 +- 0.001 / 0.130 +- 0.001 | 0.993 +- 0.000 / 0.991 +- 0.000 | - |
| Pair GNN, components 0.5, 3 seeds | (0.434 +- 0.000 / 0.332 +- 0.002) | 0.008 +- 0.001 / 0.005 +- 0.000 | 0.950 +- 0.004 / 0.944 +- 0.003 | 0.694 +- 0.005 / 0.688 +- 0.006 | 0.061 +- 0.001 / 0.079 +- 0.001 | 0.990 +- 0.001 / 0.986 +- 0.001 | 0.061 +- 0.001 / 0.079 +- 0.001 |
| Pair GNN, walk + stitch, 3 seeds | (0.453 +- 0.003 / 0.377 +- 0.003) | 0.015 +- 0.000 / 0.009 +- 0.001 | 0.922 +- 0.003 / 0.912 +- 0.003 | 0.127 +- 0.004 / 0.118 +- 0.004 | 0.104 +- 0.001 / 0.130 +- 0.001 | 0.993 +- 0.000 / 0.990 +- 0.000 | 0.103 +- 0.001 / 0.129 +- 0.001 |
| Pair GNN on graph v3 (p3), components 0.7 | (0.440 / 0.340) | 0.003 / 0.002 | 0.981 / 0.980 | 0.821 / 0.813 | 0.164 / 0.190 | 0.984 / 0.979 | 0.164 / 0.189 |
| MaskFormer pair, 3 seeds, ot 0.1 | (0.248 +- 0.005 / 0.167 +- 0.005) | 0.016 +- 0.004 / 0.008 +- 0.003 | 0.855 +- 0.025 / 0.842 +- 0.026 | 0.298 +- 0.043 / 0.289 +- 0.043 | 0.074 +- 0.004 / 0.087 +- 0.005 | 0.954 +- 0.006 / 0.948 +- 0.006 | 0.070 +- 0.005 / 0.081 +- 0.005 |
| MaskFormer pair + direction, 3 seeds, ot 0.1 | (0.283 +- 0.006 / 0.193 +- 0.006) | 0.024 +- 0.007 / 0.012 +- 0.004 | 0.850 +- 0.011 / 0.835 +- 0.017 | 0.348 +- 0.060 / 0.338 +- 0.059 | 0.069 +- 0.007 / 0.080 +- 0.008 | 0.964 +- 0.000 / 0.959 +- 0.001 | 0.066 +- 0.007 / 0.074 +- 0.008 |
| MaskFormer pair + direction long, ot 0.1 | (0.319 / 0.214) | 0.011 / 0.005 | 0.915 / 0.907 | 0.564 / 0.553 | 0.061 / 0.071 | 0.949 / 0.944 | 0.057 / 0.065 |
| MaskFormer pair, 3 seeds, ot 0.2 | (0.227 +- 0.009 / 0.151 +- 0.007) | 0.010 +- 0.003 / 0.005 +- 0.002 | 0.814 +- 0.012 / 0.805 +- 0.014 | 0.308 +- 0.040 / 0.298 +- 0.039 | 0.043 +- 0.004 / 0.052 +- 0.004 | 0.925 +- 0.008 / 0.920 +- 0.008 | 0.038 +- 0.004 / 0.046 +- 0.004 |
| MaskFormer pair + direction, 3 seeds, ot 0.2 | (0.277 +- 0.007 / 0.189 +- 0.005) | 0.020 +- 0.008 / 0.010 +- 0.005 | 0.854 +- 0.012 / 0.841 +- 0.016 | 0.362 +- 0.056 / 0.352 +- 0.056 | 0.034 +- 0.004 / 0.042 +- 0.004 | 0.941 +- 0.001 / 0.938 +- 0.000 | 0.030 +- 0.004 / 0.035 +- 0.004 |
| MaskFormer pair + direction long, ot 0.2 | (0.313 / 0.209) | 0.010 / 0.004 | 0.905 / 0.900 | 0.567 / 0.556 | 0.028 / 0.035 | 0.896 / 0.894 | 0.024 / 0.028 |
| graph-v3-perfect, walk + stitch | 0.971 / 0.966 | 0.951 / 0.941 | - | - | 0.061 / 0.067 | 1.000 / 1.000 | - |
| graph-v3-perfect-cc | 0.981 / 0.976 | 0.967 / 0.959 | - | - | 0.021 / 0.025 | 1.000 / 1.000 | - |

- 2026-10-26, A0 crossing ambiguity (test 17000-19999 truth, R1 labels, 2973 pairs; tracking/split/a0_crossing.py,
  /pscratch/sd/d/danieltm/quirk_tracking/split/a0/). Per pair: 24 A-B hit pairs on the same layer (median 17), 7.4 on
  the same surface (95% of pairs have one); within 3 x pitch on the same surface 0.05 (3.6% of pairs); within 1 / 3 /
  10 mm on the same layer 0.45 / 1.6 / 5.8 (21% / 48% / 77% of pairs have one); clusters shared by both quirks 0.18
  (12% of pairs). Along the trajectory (time-interpolated) the two quirks come within 3 mm about 5 times per pair
  (99% of pairs at least once; minimum separation median 0.1 mm, 90% below 0.42 mm): they wind around each other
  between layers. Upper bound for a splitter that sees positions only (every A-B hit pair closer than d on one layer a
  coin flip, everything else right): strict loose 1.000 / 0.997 / 0.988 / 0.946 and tight 0.995 / 0.951 / 0.836 /
  0.568 for d = resolution / 1 / 3 / 10 mm. So hit-level ambiguity caps tight only if the splitter cannot resolve
  hits a few mm apart; the hard part is following each quirk through about 5 crossings between layers.

## Fake-rate reduction (work package B), 2026-10-26

Strict matching (> 50%) throughout; m200_L774_mu0 test 17000-19999; choices on val 16000-16999. Scripts:
tracking/fakes/ (common.py track tables, decompose.py, features.py, curves.py); outputs
/pscratch/sd/d/danieltm/quirk_tracking/fakes/.

### B1: what the fakes are (strict, normal truth)

Per fake track (majority particle in the particle table, not matched at > 50%), exclusive categories in this order:
(b) majority particle a quirk and the other quirk >= 25% of the track's hits; (a) other tracks with a quirk majority
(a piece of a quirk); (c) majority particle a quirk descendant, or > 50% of hits from quirk descendants; (f) majority
particle_id 0; (d) ordinary majority particle with purity <= 50% (merged tracks); (e) ordinary majority particle with
purity > 50% and completeness <= 50% (a piece of an ordinary track). Python fake rate and technical_sm equal the ACTS
strict reruns exactly for all four (track counts and fake counts identical). "low pT / secondary" = majority particle
pT < 1 GeV or not primary.

| Model | fake rate | tracks / ev | fakes / ev | (a) quirk piece | (b) two quirks | (c) quirk secondaries | (d) merged ordinary | (e) ordinary piece | (f) noise | (d)+(e) low pT / secondary | fakes with < 4 hits | fakes with <= 5 hits |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gnn-g2-v3-dir-s0, walk + stitch | 0.1175 | 322.0 | 37.8 | 930 (0.8%) | 1276 (1.1%) | 51 (0.0%) | 36406 (32.1%) | 74789 (65.9%) | 8 | 97.6% of fakes | 0 (min 4 hits) | 41029 (36%) |
| gnn-e2-pair-dir-s0, components 0.5 | 0.0796 | 311.4 | 24.8 | 280 (0.4%) | 1150 (1.5%) | 19 (0.0%) | 12382 (16.6%) | 60536 (81.4%) | 9 | 96.1% | 0 (min 4 hits) | 38183 (51%) |
| mf-E-pairdir-long, ot 0.1 | 0.0715 | 74.7 | 5.3 | 546 (3.4%) | 1894 (11.8%) | 24 (0.1%) | 1711 (10.7%) | 11820 (73.8%) | 15 | 65.1% | 5847 (37%) | 9303 (58%) |
| mf-E-pairdir-long, ot 0.2 | 0.0348 | 66.6 | 2.3 | 364 (5.2%) | 1794 (25.8%) | 15 (0.2%) | 715 (10.3%) | 4057 (58.4%) | 6 | 54.5% | 1926 (28%) | 3190 (46%) |

- GNNs: almost all fakes (97% / 96%) are pieces or merges of low-pT (< 1 GeV) secondary particles (loopers and
  their fragments), not quirks: quirk-related categories (a)-(c) are 1.9% (v3) and 1.9% (pair GNN) of the fakes. The
  fake rate falls with track length (v3: 0.27 at 4 hits, 0.19 at 5-6, 0.11 at 8-9, 0.05 at 12-14 hits, rising again
  to 0.13-0.15 above 20 hits) and rises with the radius of the innermost hit (0.04 below 40 mm, 0.15-0.23 at
  80-600 mm, 0.36 beyond 800 mm) and towards central eta (0.08 at |eta| > 3, 0.20 at |eta| < 0.5).
- MaskFormer (pair target, scored here on normal truth): (b) is the pair track itself, which is a fake on normal
  truth by construction (12% / 26% of its fakes); its coupled fake rate on pair truth is 0.065 / 0.028. Tracks with
  fewer than 4 hits (9237 at ot 0.1, fake rate 0.63) hold 37% of its fakes; GNN tracks are cut at >= 4 hits. Its
  ordinary-track pieces (e) include technical particles (2674 at ot 0.1), unlike the GNNs. Fake rate above 80 mm
  innermost radius is 0.64-0.96 (few tracks), and 0.41-0.59 for tracks with > 20 hits (pair tracks).
- Full numbers: fakes/b1/decompose.json (categories, fake rate vs n hits / eta / r of the innermost hit), per-track
  tables fakes/b1/tracks_<model>.parquet.

### B2: track quality score (strict), and B3: quirk tagger

Protocol (fixed before looking at test): gradient-boosted trees (sklearn HistGradientBoosting) on per-track features
(fakes/features.py: hits, layers, hits per layer, innermost / outermost radius, eta, phi span, step lengths, turning
angles, r-z line and x-y circle residuals, shared hits; GNNs also edge-score statistics along and out of the track;
MaskFormer per-hit mask probabilities, its object score is not saved). There are no predictions on training events,
so the score is trained on val 16000-16499, every cut is chosen on val 16500-16999, and test 17000-19999 is scored
once. GNN val tracks were rebuilt from the existing val edge scores with the test settings (no inference). Label: the
track is not a fake in the model's own truth (per-quirk GNN: normal truth; pair GNN and MaskFormer pair models: pair
truth). Variants: quality_qw = quirk-majority good tracks weighted x30 in training; quality_or_tag = keep if quality
>= cut OR the B3 quirk tag is above its 90% cut. Baseline: a hit-count-only cut. Fake = the model's own truth
(> 50%). Curves (fakes/curves.py, reusable; plots fakes/plot_curves.py) and all operating points:
/pscratch/sd/d/danieltm/quirk_tracking/fakes/b2/<model>/ (eff_vs_fake.png, curve_*_test.csv, result.json,
summary.md).

| Model | selection (cut from val) | fake | per-quirk | quirk event | coupled | technical_sm | tracks / ev |
|---|---|---|---|---|---|---|---|
| gnn-g2-v3-dir-s0 walk + stitch | no cut | 0.117 | 0.746 | 0.620 | 0.572 | 0.991 | 322.0 |
| | hit count | fake 0.05 not reached on val (long tracks are fakes too) | | | | | |
| | quality, fake 0.05 | 0.050 | 0.656 | 0.506 | 0.528 | 0.991 | 284.7 |
| | quality_qw, fake 0.05 | 0.049 | 0.736 | 0.609 | 0.569 | 0.991 | 283.7 |
| | quality_or_tag, fake 0.05 | 0.050 | 0.740 | 0.614 | 0.570 | 0.991 | 282.8 |
| | quality, fake 0.02 | 0.020 | 0.212 | 0.108 | 0.188 | 0.985 | 240.2 |
| | quality_qw, fake 0.02 | 0.020 | 0.683 | 0.545 | 0.545 | 0.987 | 238.2 |
| | quality_or_tag, fake 0.02 | 0.019 | 0.720 | 0.590 | 0.558 | 0.981 | 219.9 |
| gnn-e2-pair-dir-s0-cc (pair truth) | no cut | 0.079 | - | - | 0.941 | 0.986 | 311.4 |
| | hit count >= 6, fake 0.05 | 0.047 | - | - | 0.940 | 0.986 | 248.3 |
| | quality_qw, fake 0.05 | 0.048 | - | - | 0.940 | 0.986 | 288.6 |
| | quality, fake 0.02 | 0.019 | - | - | 0.900 | 0.979 | 244.0 |
| | quality_qw, fake 0.02 | 0.019 | - | - | 0.935 | 0.979 | 242.9 |
| mf-E-pairdir-long ot 0.1 (pair truth) | no cut | 0.065 | - | - | 0.907 | 0.944 | 74.7 |
| | >= 4 hits | 0.041 | - | - | 0.906 | 0.944 | 71.6 |
| | hit count >= 7, fake 0.02 | 0.019 | - | - | 0.903 | 0.940 | 64.7 |
| | quality_qw, fake 0.02 | 0.019 | - | - | 0.901 | 0.941 | 70.7 |
| | quality_or_tag, fake 0.02 | 0.019 | - | - | 0.906 | 0.936 | 70.0 |
| mf-E-pairdir-long ot 0.2 (pair truth) | no cut | 0.028 | - | - | 0.900 | 0.894 | 66.6 |
| | >= 4 hits (= hit-count choice at fake 0.02) | 0.018 | - | - | 0.899 | 0.894 | 65.0 |
| | quality, fake 0.02 | 0.019 | - | - | 0.900 | 0.894 | 66.0 |

- Per-quirk GNN: the plain quality score removes quirk tracks with the fakes (quirk tracks are rare in training and
  look like loopers: per-quirk 0.746 -> 0.656 at fake 0.05, 0.21 at 0.02). Protecting them works: weighting quirk
  tracks x30 or OR-ing with the quirk tag keeps per-quirk 0.736-0.740 at fake 0.05 (from 0.117) and 0.683-0.720 at
  fake 0.02, technical_sm 0.981-0.991. The hit-count cut alone never reaches fake 0.05 on this model.
- Pair GNN: fake 0.079 -> 0.048 at coupled 0.940 (quality_qw or >= 6 hits); -> 0.019 at coupled 0.935
  (quality_qw), technical_sm 0.986 -> 0.979.
- MaskFormer long: requiring >= 4 hits takes the pair-truth fake rate 0.065 -> 0.041 (ot 0.1) and 0.028 -> 0.018
  (ot 0.2) at a coupled cost of 0.001; ot 0.1 reaches fake 0.019 at coupled 0.901-0.906 with >= 7 hits or the
  quality score. At matched fake about 0.02: pair GNN 0.935, MaskFormer long 0.900-0.906 (ot 0.1 / 0.2).
- B3 quirk tagger (label: majority particle a quirk, normal truth; cut at 90% efficiency on val B for quirk tracks
  that are not fakes in the model's own truth): ordinary tracks tagged as quirk per event on test, at quirk-track
  efficiency test 0.88-0.91: per-quirk GNN 0.81 (of 320 ordinary tracks; 49% of events have >= 1), pair GNN 1.58
  (74%), MaskFormer long 0.01 (ot 0.1 and 0.2; 1% of events). If all quirk-majority tracks (also fake pieces) must
  pass at 90%: 3.0, 18.0, 0.35, 0.12 per event. MaskFormer's quirk (pair) tracks are long and separate easily from
  the rest; GNN quirk pieces resemble looper fragments.

### A1 learned splitter (2026-10-27), strict ACTS on test 17000-19999

Splitter: transformer over the hits of one track (4 layers, d 128) whose attention gets a learned bias from pair
geometry (displacement, distance, angle of the step to each hit's cluster-axis candidates, axis-axis angle, same
layer); heads: per-hit "neither" and a symmetric pair affinity (same particle), loss on each hit's 8 nearest hits
(weight 1) and all other pairs (0.1); split = sign of the leading eigenvector of the signed affinity on the 8-NN pairs
plus single-hit flips (decode chosen on val: knn 0.860 / refine 0.889 loose). Inputs: positions, offset from the
same-layer centroid of the track (mm), volume/layer, cluster shape, direction features. A first version (per-hit A/B
logits, min-over-swap loss) and the affinity version without the same-layer offsets did not learn (val loss at
ln 2). Code tracking/split/{split_data.py, splitter2.py, a1_eval.sh}; runs /pscratch/sd/d/danieltm/quirk_tracking/split/.
The split tracks replace the tracks they came from in the pair GNN + components (0.5) predictions; other tracks
unchanged.

| ID | Input tracks | Time input | Strict loose / tight per quirk | Quirk event eff (strict) | technical_sm (strict) | Fake (strict) |
|---|---|---|---|---|---|---|
| a1r-oracle-none | oracle pair (pair truth hits) | none (headline) | 0.878 / 0.123 | 0.807 | 0.987 | 0.079 |
| a1r-oracle-smear30 | oracle pair | sim time + 30 ps Gaussian | 0.889 / 0.139 | 0.825 | 0.987 | 0.079 |
| a1r-oracle-smear100 | oracle pair | sim time + 100 ps Gaussian | 0.881 / 0.133 | 0.814 | 0.987 | 0.079 |
| a1r-oracle-true | oracle pair | true sim time (upper-bound reference only) | 0.887 / 0.134 | 0.823 | 0.987 | 0.079 |
| a1r-cc-none | every pair GNN components track with >= 4 hits (trained on its tracks with quirk hits + 2% others) | none | 0.677 / 0.027 | 0.554 | 0.989 | 0.084 |

Reference (strict): per-quirk GNN v3 0.748 +- 0.004 / 0.103; perfect-score ceiling on graph v3 0.966 / 0.828.
With the pair given exactly, the splitter beats the per-quirk GNN at loose (+13 points) but not at tight (0.12-0.14 vs
0.10): its errors sit at the crossings, a few hits per quirk. Time adds about 1 point and is no better unsmeared than
at 30 ps. On real pair tracks it falls below the per-quirk GNN (0.677): those tracks mix in other particles and
miss pair hits, and ordinary tracks get split.

A1 oracle-split error diagnosis (test, a1r-oracle-none knn_refine; tracking/split/a1_diag.py, truth used only here):
4.1 wrong hits per quirk (of ~14); 8.8% of quirks have no error, 16% at most 10%. Errors come in runs along the
quirk's own (true-time) hit order: about 2 runs per quirk, half of the wrong hits in runs of >= 3 hits, longest run
median 2 / 90th percentile 5, i.e. segments swapped between crossings rather than isolated hits. The wrong hits are
not position-ambiguous: nearest other-quirk hit on the same layer median 15 mm (14% within 3 mm, 4% within 1 mm;
6% have no other-quirk hit on that layer). Errors spread over all volumes (pixel 7.9k, short strips 10.5k, long strips
5.9k of 24.2k). A two-path dynamic-programming decode over layer groups (hits ordered by r^2 + z^2) on the same
affinities is not better on val (0.868 / 0.106; with flips 0.879 / 0.119, vs 0.889 / 0.121), so the limit is the
pairwise affinity itself, not the decode; the training curve overfits after ~25 epochs on 16k pairs (val loss 0.55).

A1 data scaling on oracle pairs (2026-10-27). Training pairs from the 1M-event package, events >= 20000 only
(tracking/split/oracle_from_package.py; val 16000-16999 / test 17000-19999 untouched). Direction features exist only
for events < 20000, so the scaling arms use no cluster-direction inputs (columns zeroed, val too). Augmentation:
random phi rotation and z reflection. No time input. Val numbers are Python strict per quirk (> 50% / > 90%) on the
993 val pairs; all runs still improving at their last epoch.

| Training pairs | Inputs | Epochs | Min val loss | Val strict loose / tight |
|---|---|---|---|---|
| 16k (events 0-15999) | with direction, no augmentation (a1r-oracle-none) | 40 | 0.548 (then overfits) | 0.860 / 0.093 |
| 16k | with direction, augmentation | 40 | 0.487 | 0.889 / 0.155 |
| 16k | no direction, augmentation | 40 | 0.495 | 0.902 / 0.160 |
| 50k (events >= 20000) | no direction, augmentation | 16 | 0.464 | 0.918 / 0.243 |
| 200k | no direction, augmentation | 6 | 0.408 | 0.933 / 0.375 |

Test (ACTS strict, 17000-19999) of the 200k model on oracle pairs (a1s-200k-nodir-aug, knn+flip decode; split tracks
replace their sources in the pair GNN components predictions): 0.939 / 0.403 per quirk, quirk event 0.903 / 0.394,
technical_sm 0.987, fake 0.079 (16k no-augmentation model: 0.878 / 0.123). The 1M-pair run is next.

Gated real-track splitting (2026-10-27). Only pair-candidate tracks are split: pair GNN + components (0.5) tracks
with B3 quirk tag >= 0.0187 (cut chosen on val B by the B3 study for pair tracks good in pair truth: test eff 0.913,
1.6 ordinary tracks tagged per event). A track is split only if both parts get >= 3 hits; "neither" hits keep the
original track id; all other tracks unchanged. Strict ACTS, test 17000-19999.

| ID | Splitter | Strict loose / tight per quirk | Quirk event eff (strict) | technical_sm (strict) | Fake (strict) | Tracks split |
|---|---|---|---|---|---|---|
| a1g-200k-gate-b3 | A1 200k oracle model (no direction, augmentation; never saw real tracks) | 0.739 / 0.165 | 0.639 | 0.987 | 0.083 | 5251 |
| a2-edgehead-gate-b3 | A2 edge head on the pair GNN (components on its first head, gated by B3 on the base tracks by majority of hits) | 0.668 / 0.066 | 0.552 | 0.975 | 0.098 | 3181 |
| (reference) gnn-g2-v3-dir, walk + stitch | per-quirk GNN v3, 2 seeds | 0.748 +- 0.004 / 0.103 | 0.622 | 0.991 | 0.117 | - |

The gated A1 splitter is level with the per-quirk GNN v3 at loose (0.739 vs 0.748), above it at tight (0.165 vs 0.103)
and in quirk event efficiency (0.639 vs 0.622), at a lower fake rate (0.083 vs 0.117). A2 is below both.

A1 on all package pairs (971k oracle pairs, events 20000-999999), no direction, augmentation, d 128, 4 layers. The
6-epoch run stopped in its last epoch (LR schedule step-count overrun, fixed); best.pt is epoch 5 of 6 (LR already
3.7e-5). Val strict loose / tight by epoch: 0.934 / 0.295, 0.942 / 0.500, 0.968 / 0.624, 0.970 / 0.674, 0.973 / 0.705
(min val loss 0.266; 200k: 0.408). Test, strict ACTS:

| ID | Input tracks | Strict loose / tight per quirk | Quirk event eff (strict) | technical_sm (strict) | Fake (strict) |
|---|---|---|---|---|---|
| a1s-1m-nodir-aug (oracle) | oracle pairs | 0.976 / 0.740 | 0.965 / 0.734 | 0.987 | 0.079 |
| a1s-1m-nodir-aug gated | B3-gated pair GNN components tracks (5204 split) | 0.804 / 0.394 | 0.735 / 0.336 | 0.987 | 0.082 |
| a1c-ft200k gated | 200k model fine-tuned 4 epochs on pair GNN components tracks (with "neither" labels) | 0.731 / 0.121 | 0.622 | 0.987 | 0.080 |

The 1M oracle number starts from the exact pair hits, so it is not comparable to the graph-v3 perfect-score ceiling
(0.966 / 0.828); it bounds what splitting can give. On real
tracks the gated 1M splitter gives 0.804 / 0.394 vs per-quirk GNN v3 0.748 / 0.103 at fake 0.082 vs 0.117. The
remaining gap to oracle (0.976) comes from the pair tracks themselves: B3 gate efficiency 0.91 on pair tracks, and pair
GNN components coupled strict 0.944 / 0.688. Fine-tuning on real tracks made it worse (the "neither" head marks many
hits, so only 2999 tracks get two parts of >= 3 hits); not pursued.

GNN-score feature arm (second arm): per-quirk GNN v3 edge scores (g2_q50_dir_s0, inference on train events 0-15999 for
this) on graph-v3 edges inside the track, as two extra pair channels (score, edge present); 16k oracle pairs only
(the package has no direction features, which graph v3 needs). Val strict loose / tight, best epoch:

| Epochs | Without GNN score | With GNN score |
|---|---|---|
| 6 | 0.767 / 0.015 (min val loss 0.622) | 0.867 / 0.065 (0.603) |
| 40 | 0.902 / 0.160 (0.495) | 0.891 / 0.124 (0.476) |

Test with GNN score, 40 epochs, oracle pairs: 0.900 / 0.174, event 0.846. The score speeds up training and lowers the
val loss, but at 40 epochs strict efficiency is not better, and 60x more pairs gives far more (0.973 / 0.705). Not
carried forward.

Longer and larger 1M-pair runs (window 60362820). Val strict loose / tight (best epoch; min val loss):
16 epochs d 128 / 4 layers 0.985 / 0.824 (0.199); 8 epochs d 256 / 6 layers 0.987 / 0.859 (0.161).
Scaling summary (val, no direction, augmentation): 16k 0.902 / 0.160, 50k 0.918 / 0.243, 200k 0.933 / 0.375,
971k (6 ep) 0.973 / 0.705, 971k (16 ep) 0.985 / 0.824, 971k larger model 0.987 / 0.859.

| ID | Input tracks / gate | Strict loose / tight per quirk | Quirk event eff loose / tight (strict) | technical_sm (strict) | Fake (strict) | Tracks split |
|---|---|---|---|---|---|---|
| a1s-1m-e16 (oracle) | oracle pairs | 0.984 / 0.837 | 0.977 / 0.834 | 0.987 | 0.079 | - |
| a1s-1m-d256l6 (oracle) | oracle pairs | 0.986 / 0.865 | 0.980 / 0.860 | 0.987 | 0.079 | - |
| a1s-1m-e16 gated | B3 >= 0.0187 | 0.814 / 0.476 | 0.750 / 0.409 | 0.986 | 0.082 | 5264 |
| a1s-1m-d256l6 gated | B3 >= 0.0187 | 0.828 / 0.505 | 0.768 / 0.432 | 0.987 | 0.082 | 5180 |
| a1s-1m-d256l6 gated loose | B3 >= 0.0018 (quirk-majority cut from val B; 18 ordinary tracks tagged per event) | 0.840 / 0.512 | 0.788 / 0.436 | 0.982 | 0.106 | - |
| a1s-1m-d256l6 ungated | every components track (keep-neither) | 0.861 / 0.529 | 0.822 / 0.448 | 0.857 | 0.527 | - |
| (reference) per-quirk GNN v3, walk + stitch | - | 0.748 / 0.103 | 0.622 | 0.991 | 0.117 | - |

Without a gate the splitter cuts ordinary tracks (fake 0.53), so the gate is needed. At the tight B3 cut the gated
splitter gives 0.828 / 0.505 at fake 0.082 against per-quirk GNN v3 0.748 / 0.103 at fake 0.117; the loose cut gives
0.840 / 0.512 at fake 0.106, still below the GNN v3 fake. The split itself is close to its oracle (0.986); the loss on
real tracks is consistent with pair finding and gating (pair GNN components coupled strict 0.944, B3 efficiency on
pair tracks 0.91; product with the oracle split about 0.85). GPU-hours for A so far: 11.8.

## MaskFormer data scaling (2026-10-27; project paused for stock-take afterwards)

Pair target, standard (origin) hit inputs, no cluster or direction features; budget rule equal epochs (15 epochs at batch
8: 16k 30k steps, 50k 93,750, 150k 281,250), WSD max LR 5e-5, checkpoint by val coupled efficiency (strict > 50%,
maskformer/scripts/scan_ckpts.sh), strict ACTS evaluation (very_loose_strict / tight_strict), test 17000-19999 once.
Extra training events 20000-153999 converted to R1 (quirk_tracking_data/parquet_r1_extra). Status: 50k and 150k arms
cancelled at ~16k / ~15k steps (checkpoints kept; resume plan in maskformer STATE.md).

| ID | Training events | Steps | Checkpoint | Coupled > 50% (ot 0.1 / 0.2) | Coupled > 90% (ot 0.1 / 0.2) | Tech_sm (> 50%) | Fake (> 50%) | Notes |
|---|---|---|---|---|---|---|---|---|
| mf-S-16k (3 seeds) | 16,000 | 30,000 | final in all three (val coupled scan) | 0.846 / 0.815 / 0.866 -> 0.842 +- 0.026 (ot 0.1); 0.819 / 0.792 / 0.803 -> 0.805 +- 0.014 (ot 0.2) | 0.299 / 0.241 / 0.325 -> 0.288 +- 0.043; 0.305 / 0.256 / 0.333 -> 0.298 +- 0.039 | 0.947 / 0.942 / 0.954 (ot 0.1) | 0.085 / 0.092 / 0.083 (ot 0.1); 0.051 / 0.057 / 0.049 (ot 0.2) | seeds t2d_pair_lr5e-5, t2e_pair_lr5e-5_s1, _s2; test, strict matching |
| mf-S-50k (equal-epochs rule) | 50,000 | 93,750 | - | - | - | - | - | abandoned at ~16k steps when the budget rule changed (checkpoints kept) |
| mf-S-150k (equal-epochs rule) | 150,000 | 281,250 | - | - | - | - | - | abandoned at ~15k steps when the budget rule changed |

Common budget rule from 2026-10-27 (with the GNN study): train until the strict val coupled efficiency
(val/coupled_strict, writer-style assignment, events 16000-16999) has not improved for 3 validation points (every
2000 steps), at least 20k and at most 120k steps (4x the 16k control), WSD scheduled over the cap; checkpoint = best of
the 3 top validation points re-scored with coupled_eval.py (strict); one seed each; test once.

| ID | Training events | Steps (stop) | Checkpoint (val coupled strict) | Coupled > 50% test (ot 0.1 / 0.2) | Coupled > 90% test (ot 0.1 / 0.2) | Per quirk (> 50%, ot 0.1) | Tech_sm (ot 0.1 / 0.2) | Fake (ot 0.1 / 0.2) | Notes |
|---|---|---|---|---|---|---|---|---|---|
| mf-S2-16k | 16,000 | 34,000 (plateau) | step 32k (0.823) | 0.817 / 0.756 | 0.245 / 0.244 | 0.159 | 0.918 / 0.894 | 0.110 / 0.069 | val 0.819; stopped in the stable phase, no LR cooldown |
| mf-S2-50k | 50,000 | 30,000 (plateau) | step 24k (0.813) | 0.807 / 0.791 | 0.238 / 0.244 | 0.150 | 0.916 / 0.892 | 0.122 / 0.079 | val 0.810; 3.6 epochs |
| mf-S2-150k | 150,000 | 34,000 (plateau) | step 28k (0.866) | 0.865 / 0.821 | 0.309 / 0.309 | 0.154 | 0.922 / 0.903 | 0.099 / 0.064 | val 0.864; 1.8 epochs |

Statistical errors on the test numbers are about 0.006-0.007 (> 50%) and 0.008 (> 90%); seed spread at 16k (30k-step
runs) is 0.026 / 0.043. Reading: 150k > 16k / 50k by +0.05 (> 50%) and +0.06-0.07 (> 90%) at ot 0.1; 16k and 50k do not
differ. All three stopped at 30-34k steps, before the WSD cooldown (which gave the largest gains in the 100k-step 16k
run: val coupled 0.877 -> 0.912 in the last 40k steps), so the plateau rule as set compares under-annealed models; the
16k 30k-step runs with full cooldown (0.842 +- 0.026) sit above the 16k and 50k plateau arms. Plot:
tracking/maskformer/scaling_coupled.png.

### Coupled efficiency vs background rejection, 2026-10-27

Knob: the same per-track quality score for every model (fakes/coupled_roc.py): gradient-boosted trees on the B2
features, trained on val 16000-16499 with label "not a fake on the pair-truth copy" and quirk-majority good tracks
weighted x30; test 17000-19999 tracks kept if score >= cut, cut swept. Efficiency and fake rate are from the same
tracks on the pair-truth copy (pair = one particle), strict matching; background rejection = 1 / fake rate (kept
fake tracks / kept tracks). Rejection at a fixed efficiency is read off the test curve (best cut keeping at least that
efficiency). The no-cut points equal the ACTS strict pair-truth reruns (coupled efficiency, tight efficiency, fake
and tight fake). MaskFormer at ot 0.1. Seeds: mean +- sample std. Figures:
quirkml/docs/talks/2026-10-26/figs/eff_vs_rejection_coupled.png, rejection_bar_coupled.png (fakes/plot_roc.py);
numbers /pscratch/sd/d/danieltm/quirk_tracking/fakes/coupled/ (roc_test.csv per model, rejection_table.json).

| Model | coupled eff, no cut | rejection, no cut | rejection at 0.50 | at 0.80 | at 0.85 | at 0.90 | technical_sm at 0.80 |
|---|---|---|---|---|---|---|---|
| Pair GNN components, 3 seeds | 0.944 +- 0.003 | 12.7 | 2402 +- 442 | 718 +- 75 | 442 +- 19 | 198 +- 29 | 0.90-0.91 |
| Pair GNN on graph v3 (p3), components 0.7 | 0.980 | 5.3 | 4802 (16 fakes kept) | 2150 | 1921 | 1289 | 0.42 |
| Pair GNN walk + stitch, seed 0 | 0.909 | 7.7 | 2796 | 433 | 173 | 34 | 0.92 |
| MaskFormer pair, 3 seeds | 0.842 +- 0.026 | 12.4 | 1235 +- 496 | 101 +- 55 | 1 of 3 seeds | not reached | 0.91-0.94 |
| MaskFormer pair + direction, 3 seeds | 0.835 +- 0.018 | 13.6 | 1480 +- 942 | 107 +- 58 | not reached | not reached | 0.93-0.95 |
| MaskFormer pair + direction, long | 0.907 | 15.4 | 2381 | 712 | 407 | 54 | 0.85 |
| Per-quirk GNN v3 s0 (reference, pair truth) | 0.572 | 8.3 | 21 | not reached | not reached | not reached | - |

- The first cuts remove some fakes at no efficiency cost: at the no-cut efficiency the rejection rises from 5-15 to
  10-24 (p3 5.3 -> 15, MaskFormer pair + direction 12-15 -> 19-24, pair GNN components 13 -> 13-14), so each curve
  starts above its no-cut dot.
- The knob also removes ordinary tracks: at high rejection the kept sample is close to a quirk-track selection
  (technical_sm at coupled 0.5: 0.25-0.78; p3 keeps technical_sm 0.42 at coupled 0.80 and 0.71 at 0.90). Rejection
  here measures the purity of the kept tracks, not a cut that leaves ordinary tracking intact.
- At low efficiency only tens of fakes are kept (p3 at 0.50: 16, pair GNN 64-93, MaskFormer long 37), so those
  rejections have 10-25% statistical uncertainty on top of the seed spread.
- Ranking at 0.80 (all pair models reach it): p3 2150 > pair GNN components 718 ~ MaskFormer long 712 > walk +
  stitch 433 > MaskFormer pair / pair + direction 100-110. At 0.90 only the pair GNNs (p3 1289, components 198,
  walk + stitch 34) and MaskFormer long (54) remain.
- Tight (> 90%) panel: p3 and the pair GNN components lead (no-cut tight efficiency 0.81 and 0.69); MaskFormer long
  reaches 0.55; the walk + stitch builder and the short MaskFormer runs stay below 0.4.

## Quirk parameters with truth hit assignment (work package C1), 2026-10-27

Question: how well do the hits of one event measure m and Lambda (nuisance: both quirk momenta, vertex) when the
hit-to-quirk assignment is perfect? Code: tracking/params/ (data.py, geometry.py, kernel.py, model.py,
toy_vs_g4.py, fisher.py, fit.py, mcmc.py, eloss_scan.py, analyze.py). Outputs and summary tables:
/pscratch/sd/d/danieltm/quirk_tracking/params/ (toy_vs_g4/summary.txt, summary_fisher.txt, summary_fits.txt,
summary_mcmc.txt, fits/fits_table.csv, eloss_scan.txt). Compute: CPU, m4958, 1 debug node for 30 min plus regular
jobs 60358027 (1 node, 1 h), 60358431 and 60366262 (4 nodes, 3 h each; both hit the time limit, but every fit task
finished: see "What is missing"). About 25 node-hours.

Model: the quirk_sim / fit/q2_kernel.py equations (straight Nambu-Goto string, both ends at equal lab time, RK4)
in the ODD solenoid (3 T, measured from ordinary tracks in these events: 2.98-3.05 T), no energy loss, no
multiple scattering. Residuals are taken on each hit's own planar module (centre and axes fitted from the packaged
loc0/loc1 -> x, y, z of all hits; 18,824 modules, exact to 7e-5 mm; long strips are 1D), with the measurement
errors from var_loc0/var_loc1 (pixels 8-14 um, short strips 17-20 um x 0.10-0.12 mm, long strips 21-28 um).
Parameters theta = (ln m, ln Lambda, ln|p|, theta, phi for each quirk, vertex x, y, z, and t0 when hit times are
used). Step c_m = 0.02 (residuals change by less than 1e-4 mm against 4x finer steps). Events: test runs, events
0-29 of each point.

Earlier study (quirk_erc/fits/REPORT.md, 2026-10-07, m = 400/800 GeV, Lambda = 0.5-4 keV, ODD v4, vertex fixed):
the soft direction is the scaling of m, p and sigma together; the field breaks it at the percent level; local
Hessian errors understate the spread by 2-3x at Lambda >= 1 keV; basins of attraction are narrow; Geant4 vs the
toy differs by 1-4 mm at 10-40 ns. The results below repeat this on the ColliderML grid (ODD v6, measured hits,
vertex free) and add timing, profiles, MCMC and the B = 0 case.

### Step 1 (gate): the toy against Geant4 sim hits, from the truth initial state

16 grid points x 30 events, 16,328 quirk sim hits (toy_vs_g4/summary.txt). The residual is on the module plane,
the time residual is model minus Geant4.

| Hit time | Lambda 167 eV: median / p68 (mm) | 774 eV | 3594 eV |
|---|---|---|---|
| 0-2 ns | 0.002 / 0.003 | 0.010 / 0.026 | 0.055 / 0.10 |
| 2-5 ns | 0.016 / 0.029 | 0.095 / 0.19 | 0.17 / 0.35 |
| 5-10 ns | 0.14 / 0.31 | 0.60 / 1.10 | 0.79 / 1.39 |
| 10-20 ns | 0.79 / 1.85 | 1.81 / 3.03 | 2.98 / 3.95 |

- Over all hits per point: median 0.008-0.34 mm, 90th percentile 0.28-3.8 mm, more than 5 mm for 0.4-7% of hits.
  The fraction of quirks whose median residual is below 0.1 mm is 0.66-0.90 at 100-167 eV and 0.12-0.45 from
  464 eV up. It is larger for slow quirks (p68 0.73-0.98 mm at beta gamma < 1, 0.013-0.30 mm at 4-8).
- Hit times agree to a median 0.2-0.6 ps (90th percentile 1-8 ps), so the time model is not the limit.
- A crude ionisation loss in the toy (Bethe-Bloch shape at each crossing of a measured module, 0.39-8 MeV/mm
  silicon-equivalent) reduces the late-hit (t > 5 ns) p68 by at most 16% at 167 eV and not at all at 774 eV. So in
  this form it does not explain the gap; the string model with the field on (delayed string in QuirkDDG4 vs
  straight string in the toy, docs/validation_status.md) is the other known difference and was not tested here.
- Gate: the toy agrees with Geant4 at the measurement precision only for the first ~2 ns of a quirk (pixel and
  first strip layers); later hits differ by 0.1-3 mm, 10-300 times the hit errors. Fits of Geant4 hits with this toy
  are possible only with an error floor; we use floor(t) = 0.02 + 0.015 t^2 mm (t in ns), added in quadrature,
  which follows the p68 above at low and mid Lambda and is too small at 3594 eV (chi2/ndof 50-100 there).

### Step 2a: Fisher information at the truth (local), 16 points x 30 events

J^T W J of the toy's own crossings at the truth (noise-free closure), measurement errors only (floor 0), vertex
free (summary_fisher.txt). The finite differences are taken directly along the scaling direction (ln m + s,
ln Lambda + s/2, ln p1 + s, ln p2 + s): taken per parameter, truncation errors that do not cancel gave spurious
information along it (sigma(ln m) 0.03 at B = 0 instead of infinite). Medians over events:

| m (GeV) | Lambda (eV) | sigma(ln m) [16-84%] | sigma(ln Lambda) | sigma(ln(sigma/m)) | rho(ln m, ln Lambda) | cos(soft vector, scaling) | sigma(ln m), 10 ps |
|---|---|---|---|---|---|---|---|
| 100 | 167 | 0.017 [0.005, 0.033] | 0.008 | 0.003 | 0.98 | 0.57 | 0.016 |
| 100 | 774 | 0.038 [0.012, 0.12] | 0.018 | 0.008 | 0.90 | 0.98 | 0.022 |
| 100 | 3594 | 0.038 [0.010, 0.30] | 0.021 | 0.006 | 0.99 | 0.97 | 0.027 |
| 200 | 167 | 0.032 [0.009, 0.19] | 0.016 | 0.004 | 1.00 | 1.00 | 0.029 |
| 200 | 774 | 0.043 [0.012, 0.08] | 0.018 | 0.009 | 0.92 | 0.99 | 0.030 |
| 200 | 3594 | 0.040 [0.007, 0.07] | 0.017 | 0.006 | 0.98 | 0.95 | 0.026 |
| 300 | 167 | 0.042 [0.006, 0.08] | 0.020 | 0.003 | 1.00 | 1.00 | 0.039 |
| 300 | 774 | 0.042 [0.013, 0.10] | 0.026 | 0.006 | 0.99 | 1.00 | 0.036 |
| 300 | 3594 | 0.042 [0.015, 0.20] | 0.036 | 0.009 | 0.95 | 0.94 | 0.034 |

- m = 200 GeV along Lambda: sigma(ln m) 0.033 (100 eV), 0.016 (278), 0.012 (464), 0.023 (1292), 0.084 (2154),
  0.035 (5995), 0.013 (10 keV); sigma(ln(sigma/m)) 0.0004-0.026.
- Degeneracy: the softest direction is the scaling of m, p and sigma = Lambda^2 / hbar c together (cosine 0.89-1.00
  at every point except m100 L167, 0.57), and rho(ln m, ln Lambda) is 0.87-1.00. Along it only the field acts.
  With B = 0 the Fisher matrix is singular for every event from 100 to 1292 eV (exact symmetry; residuals change by
  < 1e-6 mm along it). From 2154 eV up 7-83% of B = 0 events are not exactly singular, which is a numerical floor of
  the finite differences (equivalent sigma(ln m) 0.01-0.65), not information.
- The combination the hits measure well is ln(sigma/m) = 2 ln Lambda - ln m (the string acceleration scale):
  0.04-2.6% per event.
- Timing: hit times with 100 / 30 / 10 ps resolution change sigma(ln m) by at most 40% (10 ps; 30 ps: at most 28%) and
  mostly by a few percent. This is expected: under the scaling all velocities are unchanged, so times (like dE/dx) carry no
  information along the soft direction by themselves; only the field (p = gamma m v) breaks it.
- Fixing the vertex at the truth lowers sigma(ln m) by 3-40%.
- Position precision matters roughly linearly: a 0.1 mm floor on every hit gives median sigma(ln m) 0.08-0.52, a
  0.5 mm floor 0.35-2.4.

### Step 2b: fits, profiles and MCMC, 9 points (m 100/200/300 x Lambda 167/774/3594), 30 events each

Fits of all parameters (least squares, vertex free, no times) from the truth perturbed by a factor 1.03 or 1.3 on
m, Lambda, |p| (and f/10 rad on the angles), on two hit sets: "closure" = the toy's own crossings of the measured
modules, smeared with the measurement errors (ceiling within the model), and "Geant4" = the packaged measurements
with the floor above. Fail = chi2 more than 9 above the better of the two starts. res = half the 16-84% width of
(fit - truth) over events; pull = res of (fit - truth) / Hessian sigma (summary_fits.txt, fits/fits_table.csv).

| m | Lambda | closure: res ln m / ln Lambda / ln(sigma/m) | closure pull ln m | closure fail 3% / 30% start | Geant4: bias / res ln m | Geant4: bias / res ln(sigma/m) | Geant4 fail 3% / 30% |
|---|---|---|---|---|---|---|---|
| 100 | 167 | 0.014 / 0.010 / 0.004 | 0.91 | 0.03 / 0.53 | +0.32 / 0.51 | -0.001 / 0.015 | 0.00 / 0.20 |
| 200 | 167 | 0.086 / 0.049 / 0.003 | 0.85 | 0.00 / 0.47 | +0.11 / 0.16 | -0.002 / 0.009 | 0.03 / 0.17 |
| 300 | 167 | 0.060 / 0.029 / 0.007 | 1.41 | 0.00 / 0.53 | +0.04 / 0.28 | +0.004 / 0.011 | 0.07 / 0.20 |
| 100 | 774 | 0.053 / 0.050 / 0.038 | 1.66 | 0.07 / 0.80 | +1.08 / 1.57 | -0.002 / 0.071 | 0.03 / 0.57 |
| 200 | 774 | 0.068 / 0.050 / 0.066 | 2.54 | 0.20 / 0.63 | +0.96 / 1.56 | +0.002 / 0.110 | 0.00 / 0.57 |
| 300 | 774 | 0.049 / 0.028 / 0.031 | 1.00 | 0.10 / 0.70 | +1.01 / 3.38 | -0.003 / 0.078 | 0.07 / 0.67 |
| 100 | 3594 | 0.14 / 0.11 / 0.30 | 17 | 0.27 / 0.67 | +0.03 / 0.42 | +0.12 / 0.65 | 0.30 / 0.60 |
| 200 | 3594 | 0.12 / 0.07 / 0.19 | 13 | 0.23 / 0.73 | +0.03 / 0.19 | +0.02 / 0.29 | 0.40 / 0.60 |
| 300 | 3594 | 0.11 / 0.08 / 0.27 | 6.8 | 0.23 / 0.73 | +0.04 / 0.31 | +0.02 / 0.51 | 0.20 / 0.77 |

- 167 and 774 eV, closure: the fits from a 3% start reach the Fisher widths (pulls 0.9-1.7, 2.5 at m200 L774), and
  profiles along the scaling direction are close to parabolic with half-widths 0.017-0.037 in ln m (Fisher
  0.02-0.04). MCMC (6000 steps, 9-10 events per point) gives 68% half-widths in ln m of 0.004-0.058 at 167 eV and
  0.017-0.038 at 774 eV, within 1.5x of the Gauss-Newton sigma; the 68% intervals hold the truth in 5-8 of 10
  events, the 95% intervals in 8-10 of 10.
- 3594 eV, closure: the likelihood is rugged. Pulls of 7-17, 23-27% of fits from a 3% start end above the better
  start by more than 9 in chi2, and profile scans jump between minima (half-widths 0.001 against Fisher 0.01-0.03).
  MCMC chains did not mix (acceptance 0.005-0.23). The Fisher numbers there are a local lower bound only.
- From a 30% start, 47-80% of closure fits (17-77% of Geant4 fits) end in a different minimum: a fit needs a start within a few
  percent on all parameters, or a global search.
- Geant4 hits: ln(sigma/m) is recovered without visible bias at 167 and 774 eV (median bias below 0.004; event
  spread 0.9-1.5% at 167 eV and 7-11% at 774 eV). m and Lambda separately are not: the fits move along the scaling
  direction, by +0.04 to +0.32 in ln m at 167 eV and by about +1.0 in ln m (m about 2.7x too large) at 774 eV,
  where 71-83% of the profile minima along the scaling direction sit at the scan edge (+0.6). MCMC on Geant4 hits at
  774 eV gives 68% half-widths of 3-7 in ln m (the valley is flat within the floor) but 0.9-4% in ln(sigma/m). At
  3594 eV the floor is too small (chi2/ndof 50-100) and ln(sigma/m) spreads by 29-65%.
- Closure fits with hit times (30 or 100 ps, 20 events, 3% start) have the same resolution in ln m as without
  times within the event-to-event spread, as the Fisher matrix says.

### What is measurable per event, and what this means for C2

1. Per event, with perfect assignment, the hits measure ln(sigma/m) = 2 ln Lambda - ln m: 0.3-0.7% within the toy
   and 1-1.5% on Geant4 hits with the toy as model at 167 eV; 3-7% (toy) and 7-11% (Geant4) at 774 eV; 20-65% at
   3594 eV. This is the acceleration scale of the string, and it is robust to the toy-vs-Geant4 difference.
2. m and Lambda separately are measured only through the field, which breaks the scaling at the level of 1-9% in
   ln m per event within the toy (Fisher and closure fits), but the toy-vs-Geant4 difference (0.1-3 mm after 2-5 ns)
   is much larger than that signal: with the toy as fit model, Geant4 events land 0.1-1 units of ln m away along the
   valley. Hit timing (10-100 ps) and dE/dx do not help along the valley, because all velocities are unchanged by
   the scaling.
3. Separating m from Lambda therefore needs either a fit model as close to Geant4 as the field effect (energy loss
   and the delayed string), or many events: the per-event information along the valley is real but small and
   depends on curvature in the field, which accumulates over events. Dataset-level production kinematics (heavier
   quirks are slower: median beta gamma 3.0 / 2.1 / 1.7 at m = 100 / 200 / 300 GeV) is a separate, model-dependent
   handle on m.
4. For C2 (learned inference on the grid, runs 4-43): train on Geant4 events, not on the toy, so the network learns
   the true dynamics; predict a posterior, not a point, in the coordinates (ln(sigma/m), ln m) rather than
   (m, Lambda); expect per-event ln(sigma/m) at the percent level at Lambda <= 774 eV and a broad posterior in ln m.
   On the grid, points with similar Lambda^2/m are near-degenerate per event (for example m = 100 GeV, 774 eV and
   m = 300 GeV, 1292 eV differ by 8% in Lambda^2/m), so per-event classification over the 110 points will confuse
   such pairs; combining events (product of per-event posteriors, or a set model) is where m should come from.
   The C1 numbers above are the ceiling to compare C2 with: ln(sigma/m) per event, and the ln m width after
   combining N events.

### What is missing or limited

- Everything planned ran: 540 fit tasks (9 points x 30 events x 2 hit sets, 2 starts each, with profiles), 360
  timing fits, Fisher for 1420 of 1440 (16 points x 30 events x 3 floors; 20 high-Lambda tasks hit the time
  limit), MCMC for 118 of 120 low/mid tasks and 30 of 30 at 3594 eV; 12 of these chains failed to start (initial
  proposal covariance not positive definite).
- 5 of 90 events at 3594 eV need more than the step cap (15M steps) at the truth; in the second fit job (which ran
  171 of the 540 tasks) such events count as failed fits.
- The toy has no energy loss and a straight string; the floor in the Geant4 fits is a simple function of hit time,
  fitted by eye to the gate table, and too small at 3594 eV. The Geant4-hit numbers are therefore those of "this toy
  with this floor", not of the best possible fit model.
- Lambda 3594 eV: the likelihood is rugged and the local numbers (Fisher, Hessian, profiles, MCMC) do not describe
  it; a global method would be needed for a statement there.
- Shared hits (2.5% of hits carry more than one particle) are kept for both quirks; vertex free, no beam-spot
  constraint, no recoil constraint.

## GNN data scaling (2026-10-27), PAUSED: the stopping rule makes these arms uninformative

Question: does the GNN improve with more training events than the 16k used so far? Data: m200_L774_mu0 v1 1M-event
package; training sets nested (16k = events 0-15999; 50k adds 20000-53999; 150k adds 20000-153999); val 16000-16999,
test 17000-19999 untouched. Cluster-direction features were rebuilt for events 20000-153999 by re-digitising the kept
edm4hep (every hit identical to production in all 134 blocks; CFS quirk_tracking_data/direction/
m200_L774_mu0_events20000-153999), so every arm keeps the direction inputs and the existing 16k models are the
controls. Graph models are the 16k ones, inference only (graph v3: g1_qw10 embedding + g1f_qw10r075_q50 filter; pair
arm: v1 embedding + e2_pair_dir_s0 filter). Training: streamed trainer (tracking/gnn/scaling/train_gnn_stream.py,
same model / loss / features as train_gnn.py, 0.40 s per step as before), common budget rule of the GNN and MaskFormer
studies: stop when the strict val efficiency (per-quirk: walk + stitch 0.7, >= 4 hits; coupled: components 0.5,
>= 4 hits on the pair cache) has not improved for 3 validation points (every 4000 steps, all 1000 val events), cap
4 x 64k = 256k steps, cosine LR over the cap; test once on the best-val checkpoint. The in-training strict metric
equals the ACTS strict numbers (g2-v3-dir-s0 test: 0.7456 / 0.0981 / fake 0.1175 both ways). Test: strict ACTS as in
the "Strict matching" section; pair arms at the components cut chosen on val by the p3 rule (0.5 for every arm).
Outputs /pscratch/sd/d/danieltm/quirk_tracking/scaling/ (scaling_results.{csv,md,png}, runs/, eval/, STATE.md).

| ID | Task | Training events | Steps run / best-val step | LR at stop (fraction of peak) | Test strict loose / tight | Fake (strict) | technical_sm (strict) | Val strict loose / tight |
|---|---|---|---|---|---|---|---|---|
| gnn-e2-pair-dir, 3 seeds (control) | coupled | 16k | 64k (cosine to 0) / last | 0 | 0.944 +- 0.003 / 0.687 +- 0.006 | 0.079 | 0.986 | 0.947 / 0.703 |
| scal-p-16k-s0 | coupled | 16k | 40k / 28k | 0.94 | 0.925 / 0.624 | 0.096 | 0.977 | 0.933 / 0.627 |
| scal-p-50k-s0 | coupled | 50k | 32k / 20k | 0.96 | 0.915 / 0.586 | 0.109 | 0.975 | 0.922 / 0.580 |
| scal-p-150k-s0 | coupled | 150k | 36k / 24k | 0.95 | 0.918 / 0.619 | 0.102 | 0.982 | 0.923 / 0.615 |
| gnn-g2-v3-dir, 2 seeds (control) | per-quirk | 16k | 64k (cosine to 0) / last | 0 | 0.748 +- 0.004 / 0.103 | 0.117 | 0.991 | 0.748 / 0.097 |
| scal-q-50k-s0 | per-quirk | 50k | 40k / 40k, UNFINISHED | 0.94 | not tested | - | - | 0.715 / 0.068 (Python strict, in training) |
| scal-q-150k-s0 | per-quirk | 150k | 20k / 20k, UNFINISHED | 0.99 | not tested | - | - | 0.630 / 0.026 (Python strict, in training) |

- Every rule arm stopped at 32k-40k steps, with the cosine LR still at 0.94-0.99 of its peak (it decays over the 256k
  cap). At that LR the strict val efficiency moves by 2-4 points between validation points (pair 16k: 0.878, 0.933,
  0.913, 0.924, 0.923), so patience 3 fires on noise before any decay. All pair rule arms sit 2-6 points (loose) and
  6-10 points (tight) below the annealed 16k controls, and are flat in training size (0.925 / 0.915 / 0.918). This
  measures high-LR noise, not data scaling; it says nothing about whether more data helps. The MaskFormer arms under the
  same rule show the same problem (stopped at 30-34k steps, before their decay).
- The per-quirk arms were stopped at the pause, still improving (50k: 0.691 at 32k, 0.715 at 40k), not tested.
  Checkpoints kept (runs/q_*/gnn/step*.pt, state.pt).
- Were the 16k controls trained to plateau by this rule? Not verifiable: they kept no per-epoch checkpoints and no strict
  val. Their edge-level val metrics were flat over the last 3-4 epochs, but at an LR annealed to 0 by the 64k cosine.
- Redo recipe: a fixed schedule with full decay per arm, no early stop on a high-LR plateau. Cosine (or warmup-stable-
  decay) over N steps per arm, with N either fixed (64k, compute-matched, = the controls) or scaled with the data
  (e.g. 64k / 128k / 256k for 16k / 50k / 150k, or a fixed number of epochs capped at 256k); test the final or best-val
  checkpoint of the decayed run; 2 seeds per size (seed spread of the controls 0.3-0.5 points). If early stopping is
  wanted, apply it only after the decay starts (e.g. WSD: decay over the last 20% once val stops improving).
  Estimated cost: per-quirk 0.40 s/step (64k = 7.1 GPU-h), pair 0.18 s/step (64k = 3.2 GPU-h).
- GPU: 17.3 GPU-h (m5386; scaling/gpu_hours.txt): graphs for events 20000-153999 (~3 GPU-h inside window 1), pair arms
  ~5.5, per-quirk arms ~8.5 (unfinished).

## Edge-level efficiency and purity (test), 2026-10-28

Question (Daniel): per-edge efficiency / purity of the quirk GNNs. Until now only the per-epoch val numbers at cut 0.5
in the training logs existed. CPU only, from the saved scored graphs (no new inference): tracking/gnn/edge_eval.py
(+ edge_eval.sbatch, debug CPU, ~5 min), figure and table docs/talks/2026-10-26/scripts/fig_edge_eval.py. Outputs:
quirk_tracking_data/results/edge_eval/ (edge_eval.json, curves.npz, edge_table.json, raw/); figures
docs/talks/2026-10-26/figs/edge_eff_purity.png, edge_cross_kept.png; table in figs/numbers.json["edge_eval"].
Models: per-quirk GNN v2 (graph v2, gnn-v2-node), per-quirk GNN on graph v3 (gnn-g2-v3-dir-s0/s1), pair GNN on its
own graph (gnn-e2-pair-dir-s0/s1/s2), pair GNN on graph v3 (gnn-p3-pairv3-s0); checkpoint best.pt (lowest val loss),
as used for every track-building row above.

Definitions.
- Truth = the training truth: one particle per hit (R1 particle_id), consecutive = neighbouring hits of one particle
  in true sim-hit time order (make_cache.py chain; same-surface groups ordered in space). Pair chain = neighbouring
  hits of the merged, time-ordered sequence of both quirks (make_pair_cache.py, the e2 target). p3 target: every
  edge joining two hits of the pair (--pair-all-true). Same-particle non-consecutive edges have loss weight 0 in all
  trainings.
- Edge classes: ordinary true (consecutive, one ordinary particle), same-quirk true (consecutive, one quirk), cross
  (one hit of quirk A, one of quirk B: false for per-quirk targets, true for pair targets), other false.
- Efficiency of a class = edges of the class with score > cut / all truth edges of the class, including those missing
  from the graph (the training-log convention, so the value at cut 0 is the graph efficiency).
- Purity, "same particle": edges above the cut whose two hits belong to one particle (for pair models the pair is one
  particle) / all edges above the cut (all edges) or / edges with at least one quirk hit (quirk-touching). Purity,
  "consecutive" (the training-log val_purity / val_purity_quirk): only consecutive edges count as true; the log's
  quirk numerator is consecutive edges with both hits quirk hits (pair chain for pair models).
- Training-log keys (train_gnn.evaluate -> common.edge_counts, val events 16000-16199, cut 0.5): val_eff = all
  consecutive edges found / all consecutive edges; val_eff_quirk = same for quirk consecutive edges (pair chain with
  the pair cache); val_eff_tech = same for particles with pT > 1 GeV and >= 7 hits; val_purity = consecutive edges /
  edges above the cut; val_purity_quirk = consecutive edges with both hits quirk hits / edges touching a quirk hit.
- Operating cuts for track building: per-quirk models use walk + stitch (the walk follows the best edge above 0.1 and
  branches on edges above 0.6); pair GNN own graph: components at 0.5; pair GNN graph v3: components at 0.7.
- Quirk-rule variant (quirk_rule.rule_labels on the packaged particle_ids): a hit of both quirks belongs to both,
  quirk-secondary hits (all particles quirk descendants; 0.09% of hits, 13% as many as quirk hits) and their edges are
  excluded. It changes same-quirk efficiency by < 0.001 and raises quirk-touching purity by 1.4-4 points (the
  excluded edges are mostly false); the tables use the training truth.

Consistency with the training logs (cut 0.5, recomputed on val 16000-16199 from the saved scores vs cut_scan.json of
best.pt): all five quantities agree within 0.002 for all 7 models (e.g. v3 s0 eff_quirk 0.6598 vs 0.6589,
purity_quirk 0.7708 vs 0.7709; e2 s0 0.8401 vs 0.8399; p3 0.8471 vs 0.8477). On the full val 16000-16999 and on test
the values move by at most 0.015 (val_eff_quirk) and 0.03 (p3 purity_quirk 0.107 on val vs 0.135 on test).

Test 17000-19999 (3000 events), at the operating cuts (0.5 shown for the per-quirk models as the log reference).
Mean +- sample std over seeds (v3: 2, pair own graph: 3). Cross kept = fraction of the graph's cross-quirk edges above
the cut (heavy-tailed denominator: 85 per event on test, 113 on val for graph v3; per event is more stable).

| Model | Cut | Ordinary eff | Same-quirk eff | Pair-chain eff | Cross: pair-chain eff | Cross kept (of graph) | Cross kept per event | Purity all (same particle) | Purity quirk-touching (same particle) | Purity quirk-touching (consecutive) |
|---|---|---|---|---|---|---|---|---|---|---|
| per-quirk, graph v2 | 0.1 | 0.971 | 0.894 | 0.511 | 0.216 | 0.241 | 13.8 | 0.902 | 0.579 | 0.530 |
| per-quirk, graph v2 | 0.5 | 0.938 | 0.626 | 0.347 | 0.064 | 0.065 | 3.7 | 0.971 | 0.787 | 0.777 |
| per-quirk, graph v2 | 0.6 | 0.928 | 0.520 | 0.296 | 0.026 | 0.024 | 1.4 | 0.979 | 0.879 | 0.871 |
| per-quirk, graph v3 | 0.1 | 0.978 +- 0.000 | 0.937 +- 0.000 | 0.522 | 0.219 | 0.165 +- 0.001 | 14.1 +- 0.1 | 0.899 | 0.586 +- 0.003 | 0.536 |
| per-quirk, graph v3 | 0.5 | 0.943 +- 0.000 | 0.672 +- 0.004 | 0.360 | 0.068 | 0.045 +- 0.001 | 3.8 +- 0.1 | 0.970 | 0.795 +- 0.003 | 0.785 |
| per-quirk, graph v3 | 0.6 | 0.933 +- 0.000 | 0.561 +- 0.004 | 0.308 | 0.028 | 0.016 +- 0.000 | 1.4 +- 0.0 | 0.978 | 0.889 +- 0.000 | 0.880 |
| pair, own graph | 0.5 | 0.939 +- 0.001 | 0.819 +- 0.002 | 0.840 +- 0.002 | 0.790 +- 0.003 | 0.589 +- 0.006 | 34.0 +- 0.4 | 0.972 | 0.977 +- 0.000 | 0.346 +- 0.003 |
| pair, graph v3 | 0.5 | 0.882 | 0.941 | 0.844 | 0.765 | 0.950 | 81.1 | 0.953 | 0.949 | 0.949 (all pair edges true) |
| pair, graph v3 | 0.7 | 0.843 | 0.917 | 0.822 | 0.744 | 0.924 | 78.9 | 0.974 | 0.965 | 0.965 (all pair edges true) |

Truth per test event: 25.6 same-quirk consecutive edges, 15.8 pair-chain cross edges, 3243 ordinary consecutive
edges. Graph efficiency (cut 0): same-quirk 0.934 (v2 graph), 0.981 (v3), 0.918 (pair arm's graph); ordinary 0.984 /
0.993 / 0.985. Any two hits of the pair kept (fraction of those in the graph): pair own graph 0.577, pair graph v3 0.950
at 0.5; per-quirk models keep 0.29 (v2) / 0.22 (v3) of the graph's edges joining any two hits of one quirk at 0.5
(edge_table.json frac_samequirk_any_kept_in_graph).

Findings.
- Per-quirk GNNs: same-quirk efficiency falls from 0.87 at cut 0.3 to 0.67 at 0.5 (graph v3), the score bump at
  0.4-0.5 seen in the 17 Oct score plot. At the cut the walk follows (0.1) 0.937 of same-quirk consecutive edges are
  kept, together with 14 cross-quirk edges per event against 24 kept same-quirk consecutive edges per event; at 0.6
  (branching) 1.4 cross edges per event remain. Quirk-touching purity is 0.59 at 0.1 and 0.89 at 0.6. This is in line
  with the switching seen at track level (about 3 switches per mixed track) but the edge numbers alone do not show
  which edges the walk takes.
- Graph v3 vs v2 (per-quirk): +4-5 points same-quirk efficiency at every cut, mostly from the graph (0.981 vs 0.934
  at cut 0); fewer cross edges kept as a fraction of the graph (0.165 vs 0.241 at 0.1) but the same number per event
  (14). Ordinary edges are unchanged (0.94 at 0.5).
- Pair GNN on its own graph: 0.840 of pair-chain edges at 0.5 with quirk-touching purity 0.977 (pair as one
  particle). Its consecutive-only purity is 0.35 because it also scores non-neighbouring same-pair edges high (those
  are masked in its training): the log's val_purity_quirk is not a measure of wrong edges for this model.
- Pair GNN on graph v3: keeps 0.95 of all same-pair edges at 0.5, but ordinary efficiency drops to 0.88 (0.84 at its
  cut 0.7) against 0.94 for every other model, consistent with its higher track fake rate and lower technical
  efficiency (row gnn-p3-pairv3-s0).
- Seeds: spread <= 0.006 on every edge number.

### Edge ROC AUC (test), 2026-10-28

tracking/gnn/edge_auc.py on the edge_eval score histograms (1000 bins; results/edge_eval/auc_<split>.json). AUC over
the candidate edges in each model's graph. Positives = the training-target edges; negatives = edges joining two
different particles (pair GNNs: the quirk pair is one particle); same-particle non-target edges (zero loss weight) left
out. Quirk A against quirk B (per-quirk GNNs only): same-quirk target edges against edges joining the two quirks.

| Model | All edges | Edges with a quirk hit | Quirk A against quirk B |
|---|---|---|---|
| per-quirk, graph v2 | 0.9992 | 0.989 | 0.940 (val 0.954) |
| per-quirk, graph v3 (2 seeds) | 0.9994 | 0.995 | 0.959 (val 0.967) |
| pair, own graph (3 seeds) | 0.9993 | 0.998 | n/a |
| pair, graph v3 | 0.9977 | 0.997 | n/a |

With "true = any two hits of one particle" (the purity definition) the all-edge AUC drops to 0.77-0.82, because the GNNs
score non-neighbouring same-particle edges (not trained on) low; that version is not used.

## Pileup 60 (from 2026-11-03)

Splits (fixed before any training), for m200_L{278,774,2154}_mu60/v1 (100k events, events 0-99999, 500 per run):
test = events 0-1999 (runs 0-3, the same convention as the grid test sets, which are runs 0-3 of every point);
val = events 2000-2999; train = events 3000-99999 (data-scaling subsets are nested prefixes of this range, e.g.
3000-18999 = 16k, 3000-52999 = 50k, 3000-99999 = 97k). R1-layout copies for evaluate.py and the loaders:
quirk_tracking_data/parquet_r1_mu60/<set>/ (convert_to_r1.py, made by the MaskFormer agent, shared by both).
Size: ~67k tracker hits per event at mu 60 (median; p99 101k, max 120k), against ~4k at mu 0.

### Step 1: size check with the mu 0 graph models (GNN side), 2026-11-03

Inputs: cluster-direction features re-derived for mu 60 exactly as for mu 0 (re-digitisation of the kept edm4hep
with measurements.root, 7.6 min per 500-event run; every hit identical to production; tracking/pileup60/mu60_prep.sh,
mu60_direction.py; quirk_tracking_data/direction/m200_L774_mu60/). Caches: tracking/pileup60/make_cache60.py (one
vectorised pass writing the direction and pair caches; reproduces the mu 0 caches exactly on events 17000-17999:
every field and every chain edge identical), 2 min and ~13 GB memory per 1000-event block, 6 GB per block per cache
(pscratch quirk_pu60/cache/, scratch). Measurement: tracking/pileup60/sizecheck.py, one A100, mu 60 events 0-49
against mu 0 events 17000-17019; truth-edge recall = truth chain edges present / all (quirk: either quirk's chain;
pair graph: the pair chain on the pair cache). Outputs /pscratch/sd/d/danieltm/quirk_pu60/size/size_mu{0,60}.json.

| | mu 0 | mu 60 |
|---|---|---|
| hits per event (max) | 4294 (7487) | 69547 (102334) |
| quirk hits per event | 23.8 | 27.0 |
| graph v3 edges per event, before filter (hits at the k = 512 cap) | 0.54M (8%) | 22.8M (95%) |
| graph v3 after filter cut 0.001: edges / quirk recall / all-edge recall / technical recall | 132k / 0.975 / 0.993 / 1.000 | 12.5M / 0.959 / 0.969 / 0.999 |
| graph v3 at filter cut 0.01 / 0.05 / 0.1: edges, quirk recall | 50k 0.949 / 23k 0.901 / 16k 0.851 | 5.9M 0.930 / 2.6M 0.874 / 1.6M 0.838 |
| pair graph after filter cut 0.005: edges / pair-chain recall | 60k / 0.890 | 7.3M / 0.882 |
| pair graph at cut 0.05 / 0.1: edges, pair-chain recall | 22k 0.741 / 15k 0.659 | 2.4M 0.769 / 1.5M 0.698 |
| GPU time per event: embedding / radius graph / filter | 1 / 2 / 25 ms | 5 / 186 / 934 ms |
| peak GPU memory, graph / filter | 0.3 / 5.3 GB | 5.5 / 6.0 GB |

The mu 0 graph models keep the quirk truth edges at mu 60 (0.959 vs 0.975), but the graphs are ~100 times larger
(12.5M edges per event after the v3 filter; 95% of hits saturate k = 512), so GNN training on them is not practical;
tightening the filter cut loses quirk edges faster than at mu 0. Hit charge does not separate quirk hits
(geometric digitisation: sum_activation medians equal for quirk and other hits in every volume). Decision (with the
coordinator): one shared, graph-based hit filter for both the GNN and MaskFormer, keeping quirk hits and hits of
pT > 1 GeV target particles (acceptance on val: >= 99% quirk-pair hits, >= 98% pT > 1 hits, <= ~15k hits/event).
At mu 60 there are 4.6k pT > 1 GeV target hits and 15.2k pT > 0.5 GeV target hits per event (events 0-199).
B3 quirk tagger: refitted with the fakes/quality.py protocol and saved (quirk_tracking_data/models/tagger/
b3_pair_gnn_s0_cc_mu0/; cuts 0.01869 / 0.00180 and all 934,273 test scores identical to the 26 Oct run).

### Graph route at mu 60: hit filter and filter retraining (GNN side), 2026-11-03

Hit filter (tracking/pileup60/hitfilter.py; runs /pscratch/sd/d/danieltm/quirk_pu60/hitfilter/): per-hit GNN on each
hit's 12 nearest hits in the frozen mu 0 v1 embedding (hit + cluster-direction features, pair features, embedding
distance), target = quirk hit or hit of a pT > 1 GeV target particle (quirk weight 20), trained at mu 60 on events
3000-12999 (warmup / constant / linear decay over the last 20% planned for 20k steps), val events 2000-2099. Both arms
stopped early (no decay reached), because the quirk-hit recall was far from the 99% acceptance:

| Arm | Step | Quirk-hit recall / pT > 1 hit recall / hits kept per event |
|---|---|---|
| hf_v1 (k = 12, hidden 96, 4 iterations) | 6000 | 0.61 / - / 13.7k (cut 0.02); 0.43 / 0.94 / 7.1k (cut 0.1); 99% quirk recall needs 58.6k of 66.7k hits |
| hf_v3 (+ 4 nearest same-layer hits < 50 mm as a second edge type; hidden 128, 6 iterations) | 2000 | 0.71 / - / 27.0k (cut 0.02); 0.49 / 0.94 / 12.6k (cut 0.1) |

pT > 1 GeV hits are easy to keep (0.93-0.94 at 7-13k hits), quirk hits are not: locally they look like pileup hits,
and the same-layer partner edges did not change that. A per-hit filter with >= 99% quirk-hit recall would keep almost
every hit, so it is not used.

Edge filter retrained at mu 60 (tracking/pileup60/train_filter_stream.py): from the mu 0 filters (pair: v1 graph +
e2_pair_dir_s0 filter, pair truth, quirk weight 10; per-quirk: graph v3 + g1f_qw10r075_q50, quirk weight 50),
streamed over events 3000-12999, all positive and quirk-touching edges kept per step and other edges subsampled to 3M
with weight 1/fraction, LR 3e-4 constant (decay planned over the last 20% of 5000 steps). Val (events 2000-2049),
edges per event and quirk truth-edge recall:

| Filter | cut 0.01 | 0.05 | 0.1 | 0.2 |
|---|---|---|---|---|
| per-quirk, mu 0 filter (step 0) | 5.35M 0.939 | 2.33M 0.892 | 1.42M 0.849 | 0.76M 0.781 |
| per-quirk, mu 60 fine-tune step 2000 | 1.11M 0.761 | 0.33M 0.565 | 0.20M 0.489 | 0.12M 0.425 |
| pair, mu 0 filter (step 0) | 5.04M 0.899 | 2.15M 0.787 | 1.29M 0.709 | 0.69M 0.606 |
| pair, mu 60 fine-tune step 2000 | 1.07M 0.600 | 0.33M 0.458 | 0.20M 0.398 | 0.12M 0.343 |

At matched edge count the fine-tuned filters keep fewer quirk edges than the mu 0 filters (per-quirk at ~1.1M: 0.76
vs ~0.82 interpolated) and were flat from step 1000 to 2000; stopped at 2000 steps, before any LR decay. Conclusion:
no gain before the decay; not tested with the decay (a full-schedule arm is worth running when GPU time allows, since
the graph's quirk recall, 0.85, now caps the mu 60 result). Choice for the mu 60 GNNs: graph v3 with the mu 0
filter at cut 0.1 (1.42M edges per event, quirk-edge recall 0.849, all truth edges 0.897 on these 50 val events), one
graph set for both the pair GNN (pair truth, every same-pair edge true, as p3) and the per-quirk GNN.

### MaskFormer side at mu 60: zero-shot and FlexGeo hit filter, 2026-11-03

Scoring at mu 60: tracking/mu60_eval/ (shared). ACTS (evaluate.py) costs ~5 min per event per worker at mu 60
(3.8 GB per worker), so it runs chunked (eval_chunked.sh: 250-event chunks, 4 at a time x 24 workers on one CPU
node, counts summed exactly by combine_eval_chunks.py). Python scoring (py_score.py, ACTS's matching tie rule, track
classes and selections; ~1.5 min per 2000 events) reproduces ACTS exactly at mu 0 (very loose and tight, all counts,
t2e_pairdir_long test) and at mu 60: on the pair+direction zero-shot predictions (test events 0-1999, 8 chunks of 250,
normal and pair truth) every count (quirk, quirk_event, technical, technical_sm accepted/total; tracks, matched,
duplicate, fake) is identical to ACTS in every chunk at all four working points (compare_py_acts.py). Python is the
default at mu 60; ACTS for the headline test numbers only (48 workers per node; 96 run out of memory).
Recheck 2026-11-09: ACTS leaves unassociated the tracks whose majority particle is stored with pT = 0 (infinite eta,
outside the eta axis of the "all" selection histograms; 38-104 tracks per 2000 test events). py_score.py counted them
as matched or fake (efficiencies unaffected; fake rates change in the 6th decimal). With the unassociated class added
to py_score.py, compare_py_acts.py (which now also compares unassociated) gives identical counts in every chunk at
all four working points on pair_50k_qw40_s0 test and the three zero-shot runs, normal and pair truth (logs:
quirk_tracking_data/results/pu60/audit_2026-11-09/). Fake and duplicate rates are over all tracks, as in tracking_core.

Zero-shot, m200_L774_mu60 test (events 0-1999), mu 0 models, object threshold 0.1, Python strict scoring:

| Model | Tracks per event | Coupled vl strict / tight strict | Per quirk vl strict | technical_sm vl strict | Fake rate vl strict / tight strict |
|---|---|---|---|---|---|
| t2d_pair (pair target) | 1.3 | 0/1995 / 0/1995 | 0/3967 | 0.000 | 0.99 / 0.99 |
| t2e_pairdir_long (pair + direction) | 6.7 | 10/1995 (0.005) / 1/1995 | 1/3967 | 0.004 | 0.63 / 0.93 |

On ~67k hits per event the mu 0 models almost stop producing tracks (~200 targets per event).

FlexGeo hit filter (configs/filter_mu60_flex.yaml; review FlexGeo port quirk/flexgeo.py: 10 layers, width 256,
window |dphi| < 0.15, |deta| < 0.30, phi/eta RoPE, graph-free; label = quirk hit or hit of a charged primary
pT > 1 GeV, |eta| < 4, >= 3-hit particle; quirk hits x10; Lion 5e-5, warmup 15%, linear decay over the last 20%,
10k steps at 1 event per update, ~2 steps/s, 1.17 GPU-h; run filter_flex_v3). Val events 2000-2999, final
checkpoint (curve: mu60_eval/filter_curve.py, /pscratch/sd/d/danieltm/quirk_mu60/filter_v3_val/curve_val.json):

| Cut | Hits kept per event (of 66.4k) | Quirk-hit recall | Target-hit recall | Events with >= 99% quirk hits | Pair keeps > 50% / > 90% of its hits |
|---|---|---|---|---|---|
| 0.002 | 59.0k | 0.998 | 1.000 | 0.970 | 1.000 / 0.995 |
| 0.005 | 46.8k | 0.986 | 1.000 | 0.900 | 0.999 / 0.963 |
| 0.01 | 37.2k | 0.967 | 0.999 | 0.832 | 0.993 / 0.910 |
| 0.02 | 28.6k | 0.935 | 0.998 | 0.749 | 0.970 / 0.845 |
| 0.05 | 18.1k | 0.864 | 0.996 | 0.597 | 0.888 / 0.696 |
| 0.075 | 14.3k | 0.819 | 0.994 | 0.544 | 0.819 / 0.645 |
| 0.1 | 12.2k | 0.785 | 0.993 | 0.510 | 0.756 / 0.623 |
| 0.5 | 6.1k | 0.613 | 0.973 | 0.294 | 0.605 / 0.458 |

Better than the GNN per-hit filters at equal hits kept (quirk-hit recall 0.82 vs 0.61-0.65 at ~14k), but the gate
(>= 99% quirk-pair hits at <= ~15k hits) is not met: 99% quirk-hit recall needs ~50k hits. The last two columns are
completeness ceilings on the coupled very loose / tight strict efficiency. Decision on the working point pending.

### Step 2: zero-shot (mu 0 models, mu 0 cuts) on m200_L774_mu60 test events 0-1999, 2026-11-03

One combined pass per event on an 80 GB A100 (tracking/pileup60/zs_fused.py: embedding -> graph -> filter at the mu 0
cut -> GNN; edges with GNN score > 0.01 written, which covers every builder threshold), then the mu 0 builders
(per-quirk GNN v3 g2_q50_dir_s0: walk + stitch 0.7; pair GNN e2_pair_dir_s0: components 0.5; >= 4 hits).
Python strict scoring (tracking/mu60_eval/py_score.py, ACTS rules). Chunked ACTS (mu60_eval/eval_chunked.sh, 250-event
chunks) gives the same counts: per-quirk GNN v3 713/3967 = 0.180 strict loose, 9/3967 tight, event 182/1978, technical_sm
0.966; pair GNN coupled 471/1995 = 0.236, tight 71/1995 (pair truth); the normal-truth ACTS run of the pair GNN is pending.
Outputs /pscratch/sd/d/danieltm/quirk_pu60/zs/m200_L774_mu60/.

| Model | Strict loose / tight | Quirk event (strict loose) | technical_sm (strict) | Fake (strict) | Tracks per event | mu 0 strict loose / tight / fake |
|---|---|---|---|---|---|---|
| per-quirk GNN v3, walk + stitch (per-quirk) | 0.180 / 0.002 | 0.092 | 0.966 | 0.455 | 5630 | 0.748 / 0.103 / 0.117 |
| pair GNN, components 0.5 (coupled, pair truth) | 0.236 / 0.036 | - | 0.964 | 0.319 | 4568 | 0.944 / 0.688 / 0.079 |
| pair GNN, components 0.5 (per-quirk, normal truth) | 0.096 / 0.000 | 0.007 | 0.964 | 0.319 | 4568 | - |
| gated splitter chain: pair GNN components -> B3 tagger (mu 0 fit, cut 0.0187) -> 1M splitter d256 l6 (per-quirk) | 0.127 / 0.017 | 0.083 | 0.961 | 0.368 | - | 0.828 / 0.505 / 0.082 |

Per event (2000 events): filtered edges 12.1M (v3, cut 0.001) / 7.0M (pair, cut 0.005); GNN edge efficiency at 0.5
for quirk truth edges, mean over events, 0.32 (v3) / 0.31 (pair) against 0.67 / 0.84 at mu 0; time graph + filter 1.1 s,
GNN 4.8 s (v3) / 2.7 s (pair); GNN peak memory mean 27 / 16 GB, max 57 / 38 GB. GPU 3.3 + 2.1 h (m5386).
Gated chain at mu 60: the mu 0 B3 tagger tags 740k of 9.1M pair-GNN tracks (370 per event; 1.6 ordinary per event at
mu 0), and the splitter splits 561k of them, mostly ordinary tracks.

FlexMaskFormer training step time (maskformer_mu60.yaml: FlexGeo encoder 10 x 256, 8 decoder layers, 256 queries,
review losses, bf16, one A100, 1 event per update), real mu 60 events 3000-3039 with synthetic hit masks (all target
and quirk hits kept, random other hits up to N), 80 steps through the real training path (job 60747791):
29k hits 2.6 steps/s (6.5 GB peak), 37k hits 2.05 steps/s (8.2 GB), all ~68k hits 1.2 steps/s (18.7 GB). For 30k
updates: 3.2 / 4.1 / 7.0 GPU-h. The data loader needs row_group_cache_size >= 3 (hits, particles, mask files of a
1000-event block); with 1 every event re-reads the 3 GB hit file and training runs at ~0.1-0.4 steps/s.

### Step 3a: pair GNN trained at mu 60 (16k events, 2 seeds), 2026-11-03

Graph: graph v3 (g1_qw10 embedding, r 0.075, k <= 512) + mu 0 filter g1f_qw10r075_q50 at cut 0.1, precomputed per
1000-event block (tracking/pileup60/graph_apply60.py; 1.49M edges/event, ~1100-1200 s per block on one A100, 15 GB
per block, memory-mapped layout; val block: quirk truth-edge recall 0.849, all 0.897, technical 0.984).
Trainer: tracking/pileup60/train_gnn60.py (streamed; p3 target: pair cache, every same-pair edge true; direction
features; quirk weight 10; AdamW LR 5e-4, warmup 1000, constant, linear decay to 0 over the last 20% (24k-30k), no
early stopping, final checkpoint tested). One event per step; training graphs larger than 400k edges are cropped to a
phi window (half the windows centred on a quirk hit; truth used only to choose crops); val and test use full graphs.
0.33 s/step; 30k steps = 2.8 h per seed. Train events 3000-18999, val 2000-2199 (track val: components 0.5, >= 4
hits, Python strict on the pair cache). Test 0-1999: components 0.5, >= 4 hits (the mu 0 cut, not re-tuned), Python
strict scoring (py_score.py; it matched ACTS exactly on the zero-shot runs). Runs /pscratch/sd/d/danieltm/quirk_pu60/gnn/pair_16k_s{0,1}.

Val (coupled strict loose / tight, 200 events) by step, seed 0 / seed 1: 4k 0.21 / 0.23; 8k 0.36 / 0.05; 12k 0.40 / 0.36;
16k 0.41 / 0.37; 20k 0.22 / 0.52; 24k 0.48 / 0.53 (end of constant LR); 28k 0.54 / 0.55; 30k 0.535 / 0.556 (tight 0.11 /
0.11). The constant-LR phase is noisy (0.22-0.52 between points); the decay gives a further +5-6 points loose.

| Model (test 0-1999) | Coupled strict loose / tight | Per-quirk strict loose / tight (normal truth) | technical_sm (strict) | Fake (strict) |
|---|---|---|---|---|
| pair GNN zero-shot (mu 0 model, mu 0 cuts) | 0.236 / 0.036 | 0.096 / 0.000 | 0.964 | 0.319 |
| pair GNN trained at mu 60, 16k, 2 seeds | 0.538 +- 0.004 / 0.097 +- 0.007 | 0.075 / 0.001 | 0.960 | 0.314 |
| (mu 0 reference: pair GNN at mu 0, 3 seeds) | 0.944 / 0.688 | - | 0.986 | 0.079 |

Training at mu 60 more than doubles the coupled efficiency over zero-shot, but it stays far below mu 0. The
edge-level breakdown below shows that the GNN, not the graph, is the limit. The fake rate (0.31, against 0.08 at mu 0)
is broken down below. GPU: 2 x 2.8 h training + 2 x 0.3 h test inference (window 60750403).

Fake breakdown at mu 60 (B1 categories, normal truth, strict; tracking/pileup60/decompose60.py, counts identical to the
Python strict scoring; /pscratch/sd/d/danieltm/quirk_pu60/fakes_b1/decompose.json). Quirk-related fakes (a, b, c) are
< 0.15% of the fakes in every set; the rest are ordinary particles:

| Prediction set (test 0-1999) | Fake rate | (e) piece of an ordinary track | (d) merged ordinary tracks | Fakes with <= 5 hits | (e): majority pT < 1 GeV / secondary | Fake rate at 4 / 6 / 8 / 10 / 12 hits |
|---|---|---|---|---|---|---|
| pair GNN trained at mu 60, s0, components 0.5 | 0.315 | 0.868 | 0.131 | 0.53 | 0.98 / 0.88 | 0.62 / 0.43 / 0.22 / 0.12 / 0.08 |
| pair GNN zero-shot, components 0.5 | 0.319 | 0.840 | 0.159 | 0.55 | 0.98 / 0.88 | 0.65 / 0.44 / 0.20 / 0.12 / 0.08 |
| per-quirk GNN v3 zero-shot, walk + stitch | 0.455 | 0.438 | 0.561 | 0.39 | 0.99 / 0.90 | 0.72 / 0.62 / 0.45 / 0.28 / 0.20 |

As at mu 0, the fakes are short low-pT pieces of ordinary (mostly secondary) particles; the walk also merges ordinary
tracks at mu 60 (56% of its fakes). The fake rate rises again above 15 hits (pair GNN: 0.15 at 15-19, 0.37 at 20-29, 0.46
at >= 30 hits; not broken down by category per bin).

Components cut re-tuned on mu 60 val (2000-2999, pair GNN 16k, Python strict, pair truth; rule: highest mean coupled
loose over the two seeds), then test once:

| Cut | Val coupled loose / tight, s0 | s1 | Val fake (s0 / s1) | Val technical_sm (s0 / s1) |
|---|---|---|---|---|
| 0.3 | 0.325 / 0.047 | 0.355 / 0.056 | 0.318 / 0.314 | 0.914 / 0.925 |
| 0.4 | 0.465 / 0.070 | 0.496 / 0.087 | 0.313 / 0.311 | 0.945 / 0.953 |
| 0.5 | 0.527 / 0.091 | 0.534 / 0.109 | 0.314 / 0.313 | 0.958 / 0.963 |
| 0.6 | 0.526 / 0.108 | 0.538 / 0.120 | 0.317 / 0.317 | 0.962 / 0.966 |
| 0.7 | 0.492 / 0.119 | 0.494 / 0.114 | 0.322 / 0.321 | 0.961 / 0.963 |
| 0.8 | 0.426 / 0.092 | 0.422 / 0.082 | 0.327 / 0.326 | 0.956 / 0.957 |
| 0.9 | 0.305 / 0.035 | 0.294 / 0.030 | 0.333 / 0.330 | 0.942 / 0.944 |

Chosen 0.6 (mean loose 0.532 vs 0.531 at 0.5: a tie within val noise; tight +1.4 points). Test at 0.6: coupled
0.520 +- 0.001 / 0.112 +- 0.005 (s0 0.521 / 0.109, s1 0.519 / 0.116), fake 0.318 / 0.317, technical_sm 0.961 / 0.965
(at 0.5: 0.538 / 0.097, fake 0.314). The cut moves loose and tight in opposite directions by ~1.5 points; it is not
what limits the mu 60 result. Headline (val-chosen): 0.520 / 0.112.

Edge-level breakdown of the mu 60 pair GNN on test (tracking/pileup60/edge60.py, CPU; universe = the GNN input graph,
edges with GNN score <= 0.01 were not stored and count as score 0; truth from the pair cache). Ceilings per pair
(1995 pairs with >= 3 hits): largest connected component of pair hits joined by (i) any pair edge in the graph,
(ii) pair edges with GNN score above the cut; "found" if it holds > 50% (> 90%) of the pair's hits.

| Quantity | mu 60 pair GNN s0 / s1 | mu 0 pair GNN (own graph, 3 seeds; edge_eval) |
|---|---|---|
| AUC, pair edges vs wrong edges touching a quirk hit | 0.953 / 0.952 | 0.998 |
| pair-chain edge efficiency at 0.5 (incl. edges missing from the graph) | 0.424 / 0.417 | 0.840 |
| purity of quirk-touching edges above 0.5 / 0.6 | 0.80 / 0.81, 0.84 / 0.85 | 0.977 (at 0.5) |
| wrong quirk-touching edges kept per event at 0.5 / 0.6 | 11.1 / 9.9, 7.7 / 7.0 | - |
| ceiling (i) graph: pairs connectable at > 50% / > 90% | 0.938 / 0.716 | - |
| ceiling (ii) GNN at 0.5: kept pair edges connect > 50% / > 90% | 0.569 / 0.247, 0.572 / 0.235 | - |
| ceiling (ii) GNN at 0.6 | 0.524 / 0.196, 0.522 / 0.189 | - |
| actual coupled strict loose / tight (components 0.6; 0.5) | 0.521 / 0.109, 0.519 / 0.116 (0.535 / 0.090, 0.541 / 0.104) | 0.944 / 0.688 (0.5) |

The graph (mu 0 filter, cut 0.1) still connects > 50% of the hits for 94% of the pairs (> 90% for 72%), so it is not
the main limit. The GNN keeps only ~60% of the pair edges in the graph at 0.6 and its kept pair edges connect > 50%
of the hits for only 52-57% of the pairs, which is where the built tracks end up (0.52-0.54). The loss is in the GNN's
edge discrimination at mu 60 (AUC 0.95 against 0.998 at mu 0), not in the graph or the track builder.

Quality cut at mu 60 (coupled efficiency vs background rejection; tracking/pileup60/roc60.py = fakes/coupled_roc.py
recipe: B2 track-quality score, gradient-boosted trees on the quality.py features incl. GNN edge-score features,
label "not a fake on the pair truth", quirk-majority good tracks x30; trained on val 2000-2499, operating cuts chosen
on val 2500-2999, curve on test 0-1999 once). Model: pair GNN mu 60 s0, components 0.6. Model saved
(quirk_tracking_data/models/quality_pu60/pair60_s0_cc0.6/quality.pkl; roc_test.csv, roc_valB.csv, roc.json).
Operating points corrected 2026-11-09 (roc_exactcut.json): test rates are now taken at exactly the val-B cut; before,
roc60.py used the next test-quantile cut above it (cuts 0.513 / 0.744 / 0.850, coupled 0.511 at keep 99%, fake 0.072,
tracks per event 3152 / 2754 / 2540).

| Operating point (cut from val B) | Coupled strict loose / tight (test) | Fake (strict) | Rejection 1/fake | technical_sm | Tracks per event |
|---|---|---|---|---|---|
| no cut | 0.521 / 0.109 | 0.318 | 3 | 0.961 | 4751 |
| keep 99% of val-B coupled eff (cut 0.511) | 0.512 / 0.109 | 0.073 | 14 | 0.955 | 3155 |
| keep 95% (cut 0.743) | 0.484 / 0.109 | 0.029 | 34 | 0.946 | 2757 |
| keep 90% (cut 0.849) | 0.451 / 0.107 | 0.015 | 66 | 0.936 | 2543 |

Read off the test curve: coupled 0.502 at fake 0.05, 0.464 at 0.02, 0.421 at 0.01; rejection 22 / 68 / 129 / 304 at
coupled efficiency 0.50 / 0.45 / 0.40 / 0.30 (mu 0: rejection 718 +- 75 at coupled 0.80; coupled 0.935 at fake 0.02).
As at mu 0, the quality cut removes most fakes (0.32 -> 0.07) for ~1 point of coupled efficiency.

### Step 3b: per-quirk GNN trained at mu 60 (16k events, 2 seeds), 2026-11-03

Same graphs and recipe as the pair GNN (graph v3 + mu 0 filter at 0.1; normal cache, per-quirk truth; quirk weight
10; 30k steps, constant LR then linear decay over 24k-30k; final checkpoint). Track val on 2000-2049 (walk + stitch at
mu 60 is slow on CPU). Test 0-1999: walk + stitch 0.7, >= 4 hits (mu 0 builder), Python strict. Models
quirk_tracking_data/models/gnn_pu60/quirk_16k_s{0,1}.

| Model (test 0-1999) | Per-quirk strict loose / tight | Quirk event (strict) | technical_sm | Fake (strict) | Tracks per event |
|---|---|---|---|---|---|
| per-quirk GNN v3 zero-shot (mu 0 model, its own graph at filter cut 0.001) | 0.180 / 0.002 | 0.092 | 0.966 | 0.455 | 5630 |
| per-quirk GNN trained at mu 60, 16k, 2 seeds | 0.166 +- 0.003 / 0.004 | 0.073 | 0.976 | 0.397 | 5810 |
| (mu 0 reference, 2 seeds) | 0.748 / 0.103 | 0.622 | 0.991 | 0.117 | - |

Val strict loose (50 events) rose from 0.08-0.10 (4k) to 0.22-0.23 (28k, in the decay). Training at mu 60 on the
tighter graph gives no gain over zero-shot for the per-quirk task (slightly lower efficiency, lower fake rate); at mu
60 the per-quirk task sits at ~0.17, against ~0.52 for the coupled task. GPU: 2 x ~3.8 h.

Decision (Daniel, 2026-11-03): no hit filter for MaskFormer at mu 60 ("I don't want to have ANY reason to not find a
quirk"); the FlexGeo filter v3 checkpoint and curve are kept as a record.

FlexMaskFormer on all hits, 16k-event trunk (train events 3000-18999, seed 0; maskformer_mu60.yaml with
hit_mask_dir null; constant LR 5e-5 after a 1.5k warmup, 24k steps, 1.14 steps/s, 5.9 GPU-h; spike guard skipped
0-3 updates). Peak GPU memory on the largest events: 25.8 / 25.3 / 28.0 GB at 121.5k / 119.2k / 131.7k hits (A100
40 GB). Decay branches (linear to ~0 over the last 20%, fresh epoch from the trunk checkpoint, seed S), Python strict
scoring on val events 2000-2999 (1000 events; object threshold 0.1; predictions under
/pscratch/sd/d/danieltm/quirk_mu60/predictions/):

| Branch | Coupled vl strict | Coupled tight strict | Per quirk vl strict | technical_sm vl / tight strict | Fake rate vl / tight strict |
|---|---|---|---|---|---|
| 8k to 10k | 0.050 | 0.005 | 0.010 | 0.672 / 0.370 | 0.074 / 0.443 |
| 16k to 20k | 0.133 | 0.011 | 0.022 | 0.722 / 0.465 | 0.038 / 0.346 |
| 24k to 30k | 0.151 | 0.011 | 0.024 | 0.731 / 0.491 | 0.031 / 0.316 |

Ordinary tracks are learned (technical_sm 0.73 very loose strict), the pair slowly: the pu0 16k pair model reached
0.55-0.70 coupled by 10-16k steps. The quirk pair's hits span a median 0.47 in phi and 1.04 in eta (7% of pairs
inside one 0.15 x 0.30 window; 63% of pair hit-pairs within a window, vs 73% for pT > 1 GeV tracks). Wide-window
option measured (|dphi| < 0.5, |deta| < 1.0): 0.97 steps/s on all hits, same peak memory (18.7 / 28.0 GB).

### Step 3c: pair GNN data scaling at mu 60 (16k vs 50k events), 2026-11-03

50k arm: train events 3000-52999 (graphs for 19-52 built for it), one constant-LR trunk (seed 0, same recipe), with
(i) a branch from the trunk state at step 24000 decaying linearly over 24k-30k (compute-matched to the 16k arm) and
(ii) the trunk itself decaying over 48k-60k (pending). The trunk stalled once (~1 h, a track-val worker pool, most
likely a host-memory kill on the 1-GPU batch node; restarted from its step-15500 state with 6 workers).

| Arm | Val coupled strict loose / tight at 0.5 (0.6) | Test at the val-chosen cut | Test at 0.5 |
|---|---|---|---|
| 16k, 30k steps, 2 seeds | 0.531 / 0.100 (0.532 / 0.114) | cut 0.6: 0.520 +- 0.001 / 0.112 +- 0.005 | 0.538 +- 0.004 / 0.097 +- 0.007 |
| 50k, 30k steps (branch), seed 0 | 0.548 / 0.107 (0.526 / 0.123) | cut 0.5: 0.529 / 0.099 | 0.529 / 0.099 |
| 50k, 60k steps (trunk) | pending | | |

Trunk val (constant LR, 200 events): 0.38 (8k), 0.42 (12k), 0.47 (20k), 0.47 (24k), 0.54 (40k); the 24k->30k branch
decay gave 0.50, 0.51, 0.55 at 26k, 28k, 30k. At the same step budget (30k) the 50k arm is not better on test than
the 16k arm (0.529 vs 0.538 at 0.5); fake 0.314, technical_sm 0.960.

### Diagnostics on val (2000-2999), after the second opinions, 2026-11-04 (CPU)

Code check: in gnn/train_gnn.py (batch(), `w = w * where(hit_quirk[src] | hit_quirk[dst], QW, 1)`) and therefore in
pileup60/train_gnn60.py, the quirk weight multiplies every edge touching a quirk hit: true pair edges and wrong
quirk-touching edges alike. The running arm pair_16k_qw40_s0 is "all quirk-touching edges x40". Prepared, not launched:
pileup60/pair_norm3.sbatch (train_gnn60.py --loss-mode norm3: separately normalised BCE for true quirk edges / wrong
quirk-touching edges / ordinary edges, weights --norm3-w; --crop-mode pair: quirk crops centred on the pair's
circular-mean phi and widened to hold the whole pair + 0.1 rad).

Training-crop audit (pileup60/crop_audit.py; 995 quirk-centred crops on train 3000-3199): windows are ~2.0 rad wide
(median pair phi spread 0.30 rad); 96% of the pair's hits are inside on average, 81% of crops hold the whole pair, 11%
hold < 90%, 3% hold <= 50%; the largest connected group of pair hits (pair edges inside the crop) holds 0.877 of the
pair's hits on average, against 0.896 on the full graph. The crops cost little.

Builder study (pileup60/builders60.py, coupled strict on the pair cache, paired event bootstrap of the difference to
components at 0.5, 1000 resamples, 95% interval). Failure classes at > 50% for the 995 val pairs, components 0.5
(s0 / s1 / 50k branch): fragmentation (best track holds <= 50% of the pair's hits) 367+18 / 351+20 / 361+19 (of which
18-20 are a track holding one quirk alone), absorption (> 50% complete, <= 50% pure) 49 / 44 / 33, no pair hit on
any track 37 / 49 / 37. At > 90% (s0): incomplete 222, impure 155. So at loose the loss is fragmentation (missing
pair edges), not wrong bridges; at tight both matter.

| Builder (val) | s0 loose / tight / fake | s1 | 50k branch | d loose, d tight vs cc 0.5 (s0; s1; 50k) |
|---|---|---|---|---|
| components 0.5 | 0.527 / 0.091 / 0.314 | 0.534 / 0.109 / 0.313 | 0.548 / 0.107 / 0.314 | - |
| components 0.6 | 0.526 / 0.108 / 0.317 | 0.538 / 0.120 / 0.317 | 0.526 / 0.123 / 0.318 | -0.001, +0.016*; +0.004, +0.011; -0.022*, +0.016* |
| two-threshold 0.6 / 0.3 (single hits attach) | 0.550 / 0.097 / 0.352 | 0.553 / 0.102 / 0.350 | 0.556 / 0.106 / 0.352 | +0.023*, +0.006; +0.019*, -0.007; +0.008, -0.001 |
| bridge cut 0.5 / 0.6 (drop edges < 0.6 without triangle support) | 0.529 / 0.106 / 0.317 | 0.548 / 0.118 / 0.316 | 0.532 / 0.122 / 0.317 | +0.002, +0.014*; +0.014*, +0.009; -0.016*, +0.015* |
| bridge cut 0.5 / 0.7 | 0.524 / 0.118 / 0.320 | 0.525 / 0.117 / 0.319 | 0.521 / 0.113 / 0.321 | -0.004, +0.026*; -0.010, +0.008; -0.027*, +0.006 |

(* = 95% interval excludes 0; more settings in val/builders60.json of each run.) No builder improves loose and tight
together on all three models: two-threshold buys ~2 points loose at +3.8 points fake; bridge cuts and a higher
component cut buy 1-2.6 points tight at 0-3 points loose. The builders do not recover the gap to the GNN-edge ceiling
(loose 0.57 / tight 0.24 at 0.5) because most failures are missing pair edges. Not tested on the test set.

Edge scan on val (pileup60/edge60.py, EDGE_BLOCKS=2; s0 / s1 / 50k nearly identical): AUC 0.951 / 0.949 / 0.947.
Pair-edge recall (any edge joining two pair hits, of those in the graph) at a fixed number of wrong quirk-touching
edges kept per event (s0): 0.42 at 2, 0.53 at 5, 0.63 at 10, 0.72 at 20, 0.83 at 50, 0.89 at 100 (50k: 0.41 / 0.53 /
0.62 / 0.72 / 0.82 / 0.89). Cut scan (s0): cut 0.05: kept 0.875, purity 0.42, 83 wrong/event, ceiling 0.80 / 0.52;
0.1: 0.829, 0.52, 51, 0.76 / 0.46; 0.3: 0.725, 0.70, 21, 0.64 / 0.33; 0.5: 0.644, 0.79, 11, 0.57 / 0.24; 0.7: 0.533, 0.88,
5, 0.49 / 0.16. Pair-edge recall at 0.5 by the smaller graph degree of its two hits: 0.69 (< 50 edges, 83% of pair
edges), 0.42 (50-100, 17%), 0.23 (100-200, 0.6%); the GNN loses pair edges in dense regions. The 50k data (same steps)
changes none of these numbers: more data at this model and graph does not help.

### Headline candidate: pair GNN 50k trunk, 60k steps; gated splitter chain at mu 60 (2026-11-04)

50k trunk (train 3000-52999, constant LR to 48k, linear decay 48k-60k, final checkpoint). Val (2000-2999) components
cut scan: 0.3 0.441 / 0.090; 0.4 0.561 / 0.115; 0.5 0.600 / 0.152; 0.6 0.581 / 0.147; 0.7 0.544 / 0.143 (fake 0.29-0.33)
-> 0.5. Test once (Python strict; chunked ACTS pending: 60787558 pair truth, 60787559 normal truth):

| Pair GNN (test 0-1999, components 0.5) | Coupled strict loose / tight | Per-quirk (normal truth) | technical_sm | Fake |
|---|---|---|---|---|
| 16k, 30k steps, 2 seeds | 0.538 +- 0.004 / 0.097 +- 0.007 | 0.075 | 0.960 | 0.314 |
| 50k, 30k steps (branch) | 0.529 / 0.099 | 0.081 | 0.960 | 0.314 |
| 50k, 60k steps (trunk) | 0.595 / 0.134 | 0.094 | 0.969 | 0.293 |

The 60k-step trunk is +6 points loose over both 30k-step arms; data alone (16k -> 50k at 30k steps) gave nothing, so the
gain is from the longer schedule (not separated from data: no 16k run at 60k steps).

B3 quirk tagger retrained at mu 60 (pileup60/tagger60.py; fakes/quality.py features; trained on val 2000-2499 with the
positives weighted to balance the classes, because only 514 of 2.4M training tracks are quirk-majority and the
unweighted mu 0 recipe gave a constant score; cut at 90% efficiency for quirk tracks good in the pair truth on val
2500-2999). Models: quirk_tracking_data/models/tagger/b3_pair60_50k_{trunk,b30k}_cc0.5/. On the trunk's test tracks:
cut 0.942, efficiency 0.916 for pair-good quirk tracks, 0.68 tracks tagged per event (0.10 ordinary) (mu 0 tagger at
mu 60: 370 per event).

Gated splitter chain (trunk components 0.5 -> mu 60 tagger -> 1M splitter d256 l6 trained at mu 0, unchanged; split
only if both parts >= 3 hits, "neither" hits stay on the original track): 1365 tracks gated, 1354 split (0.68 per event).

| Per-quirk (test 0-1999, normal truth) | Strict loose / tight | Quirk event (strict) | technical_sm | Fake |
|---|---|---|---|---|
| gated splitter chain, trunk + mu 60 tagger | 0.350 / 0.077 | 0.281 | 0.969 | 0.293 |
| per-quirk GNN trained at mu 60 (16k, 2 seeds) | 0.166 / 0.004 | 0.073 | 0.976 | 0.397 |
| zero-shot chain (mu 0 GNN + mu 0 tagger) | 0.127 / 0.017 | 0.083 | 0.961 | 0.368 |
| (mu 0 chain) | 0.828 / 0.505 | 0.768 | 0.987 | 0.082 |

At mu 60 the splitter route doubles the per-quirk efficiency of the per-quirk GNN (0.35 vs 0.17) at a lower fake rate.
The chain's ceiling is the pair finding: coupled 0.595, of which the tagger keeps ~92%.

ACTS (chunked, 8 x 250 events) on the trunk test predictions, pair truth: coupled strict 1187/1995 = 0.595, tight
268/1995 = 0.134 (equal to the Python strict numbers); normal-truth ACTS run pending (job 60787558).

### Loss-weight arm: pair GNN 16k, all quirk-touching edges x40 (2026-11-04)

Daniel's point: at mu 60 the quirk edges' share of the loss fell. Measured shares of quirk-touching edges in the loss
at weight 10: mu 0 pair GNN training graphs 0.064 (383 quirk-touching vs 56,285 other edges per event); mu 60 full
graph 0.0065 (972 vs 1.48M); mu 60 training crops 0.017 (596 vs 337k per step). Weight 40 restores 0.064 on the crops
(full-graph equivalent ~100). Note: in train_gnn.py / train_gnn60.py the weight multiplies every quirk-touching edge,
true and wrong alike, so this arm is "all quirk-touching edges x40". Otherwise identical to pair_16k_s0 (seed 0, 16k,
30k steps, same schedule). Compared on val only (2000-2999, components 0.5, paired event bootstrap, 2000 resamples):

| Arm (val, components 0.5) | Coupled loose / tight | Fake | vs pair_16k_s0: d loose, d tight [95%] |
|---|---|---|---|
| pair_16k_s0 (weight 10) | 0.527 / 0.091 | 0.314 | - |
| pair_16k_s1 (weight 10) | 0.534 / 0.109 | 0.313 | +0.007, +0.018 |
| pair_16k_qw40_s0 (all quirk-touching x40) | 0.580 / 0.122 | 0.349 | +0.053 [+0.033, +0.072], +0.030 [+0.014, +0.046] |
| pair_50k_b30k_s0 (50k, 30k steps) | 0.548 / 0.107 | 0.314 | +0.021 [+0.002, +0.041], +0.015 [+0.001, +0.029] |
| pair_50k_trunk_s0 (50k, 60k steps) | 0.600 / 0.152 | 0.292 | +0.073 [+0.052, +0.094], +0.060 [+0.042, +0.078] |

(qw40 vs s1: +0.046 [+0.028, +0.064] loose, +0.013 [-0.002, +0.028] tight; trunk vs 50k branch: +0.052 / +0.045.)
The x40 weight gains ~5 points loose at 16k / 30k steps, at +3.5 points fake (0.349). The longer schedule (60k steps,
50k data) gains more at a lower fake rate. Not tested on the test set. Prepared next arms (not run; budget):
norm3 loss + whole-pair crops, degree-aware aggregation (smoke tests passed), and a 16k run at 60k steps to separate
steps from data.

### Arms at 60k steps (no GPU cap from 2026-11-04), val only, paired bootstrap against the trunk

All: pair GNN, graph v3 + mu 0 filter at 0.1, 60k steps, warmup 1000, constant LR 5e-4, linear decay over 48k-60k, final
checkpoint, 400k-edge phi crops (pileup60/arm60.sbatch). Val 2000-2999 (995 pairs), coupled strict, components at the
val-best cut, paired event bootstrap (2000 resamples, 95% interval; pileup60/compare_runs.py). Test untouched.
Per-event JSONs: quirk_tracking_data/results/pu60/gnn/<arm>/val/builders60_pe.json; comparisons:
results/pu60/compare/val_vs_trunk.txt; models: models/gnn_pu60/<arm>/.

| Arm | Change vs trunk | Val loose / tight (cut) | Fake | d loose, d tight vs trunk s0 [95%] |
|---|---|---|---|---|
| pair_50k_trunk_s0 | - | 0.600 / 0.152 (0.5) | 0.292 | - |
| pair_50k_trunk_s1 | seed 1 | 0.605 / 0.144 (0.5) | 0.293 | +0.005 [-0.013, +0.025], -0.008 [-0.025, +0.010] |
| pair_16k_60k_s0 | 16k events instead of 50k | 0.587 / 0.143 (0.5) | 0.293 | -0.013 [-0.032, +0.007], -0.009 [-0.025, +0.008] |
| pair_50k_summean_qw40_s0 | summean aggregation (sum + mean + log degree) and all quirk-touching edges x40 | 0.650 / 0.175 (0.5) | 0.325 | +0.050 [+0.030, +0.070], +0.023 [+0.005, +0.041] (vs s1: +0.045, +0.031) |
| pair_50k_norm3_paircrop_s0 | norm3 loss (1/1/1) + whole-pair crops (capped at 1.5x the budget width after an OOM) | 0.556 / 0.084 (0.9) | 0.485 | -0.044 [-0.066, -0.022], -0.068 [-0.088, -0.049] |
| pair_50k_qw40_s0 | all quirk-touching edges x40 only (trunk aggregation) | 0.655 / 0.183 (0.5) | 0.323 | +0.055 [+0.035, +0.075], +0.031 [+0.013, +0.049] (vs s1: +0.050, +0.039) |

- The trunk's two seeds agree within 0.5 points; 16k events at 60k steps is within noise of 50k: the earlier +6 points
  of the trunk over the 30k-step arms come from the schedule, not the data.
- The x40 quirk weight carries the whole gain: x40 alone gives 0.655 / 0.183 (+5.0 to +5.5 points loose, +3 to +4 tight
  over the two trunk seeds), and summean adds nothing on top (summean + x40 vs x40: -0.005 [-0.023, +0.012] loose,
  -0.008 [-0.024, +0.008] tight). Cost: fake +3 points (0.323 vs 0.292-0.293), as with x40 at 16k / 30k steps.
  Val winner: pair_50k_qw40_s0 (one seed; not yet scored on test).
- norm3 with equal weights shifts the edge scores up (1.0M edges/event above 0.01), the best components cut is 0.9,
  and ordinary tracking degrades (2255 tracks/event, fake 0.49): the ordinary edges (~98% of edges) get a third of the
  loss. Not a fair test of loss balancing; a retry should keep most of the weight on ordinary edges.
- Direction features for blocks 94-99 added (re-digitised, every hit identical to production); all 100 blocks present.

### Test scoring of the val winner pair_50k_qw40_s0 (2026-11-04; the only test use of this round)

Pair GNN 50k events, 60k steps, all quirk-touching edges x40, components 0.5 (val-best). Test 0-1999, Python strict.
Chunked ACTS (60837576 / 60837577) agrees: pair truth very_loose_strict 0.6496 (1296/1995) / tight_strict 0.1694,
technical_sm 0.9579, fake 0.3234; normal (per-quirk) truth 0.0988 / 0.0008; non-strict very_loose 0.6586 pair.
Summaries: results/pu60/gnn/pair_50k_qw40_s0/test/acts_{pair,quirk}_summary_combined.json.
posw40 option (--qw-mode pos) smoke-tested on CPU (job 60840754): true quirk-touching edges weight 40, wrong ones 1,
loss finite over 250 steps.

| Quantity (test) | pair_50k_qw40_s0 | trunk s0 (weight 10) |
|---|---|---|
| coupled strict loose / tight | 0.650 / 0.169 | 0.595 / 0.134 |
| fake (strict) / technical_sm | 0.323 / 0.958 | 0.293 / 0.969 |
| quality cut at val-B fake 0.07 (cut 0.542): coupled / tight / fake / technical_sm | 0.632 / 0.167 / 0.070 / 0.952 | - |
| quality cut at val-B fake 0.02 (cut 0.825): coupled / tight / fake / technical_sm | 0.587 / 0.165 / 0.020 / 0.932 | (trunk-era 16k s0: 0.464 at fake 0.02) |
| mu 60 tagger (retrained on its val tracks): cut / pair-good efficiency / tagged per event (ordinary) | 0.776 / 0.916 / 0.82 (0.19) | 0.942 / 0.916 / 0.68 (0.10) |
| gated splitter chain, per-quirk strict loose / tight | 0.397 / 0.098 | 0.350 / 0.077 |
| chain quirk event / technical_sm / fake; tracks split | 0.330 / 0.958 / 0.323; 1611 | 0.281 / 0.969 / 0.293; 1354 |

Quality model: quirk_tracking_data/models/quality_pu60/pair60_50k_qw40_cc0.5/ (fixed-fake working points chosen on val
2500-2999, pileup60/roc60.py); tagger: models/tagger/b3_pair60_50k_qw40_cc0.5/; chain:
results/pu60/chain/qw40_cc0.5_py_split.json.

Arms a / c / d (2026-11-04), each a 16k-event trunk to 24k steps (constant LR) plus the 24k -> 30k linear decay
branch, everything else as the baseline trunk; val events 2000-2999, Python strict scoring, object threshold 0.1;
paired event bootstrap of the coupled very loose strict efficiency against the baseline 30k branch (0.151 / 0.011):

| Arm | Change | Coupled vl strict / tight strict | technical_sm vl / tight strict | Fake vl / tight strict | vs baseline (vl strict, 95%) |
|---|---|---|---|---|---|
| baseline | - | 0.151 / 0.011 | 0.731 / 0.491 | 0.031 / 0.316 | - |
| a | window 0.5 x 1.0 | 0.138 / 0.015 | 0.736 / 0.578 | 0.028 / 0.228 | -0.013 [-0.031, +0.004] |
| c | pair x10 (object BCE, mask losses, matching cost) | 0.420 / 0.049 | 0.726 / 0.475 | 0.033 / 0.332 | +0.269 [+0.242, +0.298] |
| d | warm start from pu0 s2_pair_n150k | 0.116 / 0.001 | 0.699 / 0.416 | 0.050 / 0.391 | -0.035 [-0.053, -0.017] |

In-training val coupled (100 events) of arm c's trunk: 0.15 at 6k, 0.23 at 10k, 0.32 at 18k, 0.31 at 22k; branch
0.47 at 30k. Arm d ran on the interactive slot in two parts (0-16k, then 16k-24k from a fresh epoch, seed 16000).
pair_diag (val, all four 30k branches): almost every missed pair is missed because no query's own mask (mask prob
>= 0.5, before the exclusive hit assignment) strictly matches it (baseline 842 of 845 missed, a 856 / 861, c 573 / 576,
d 877 / 878); object score and competition between queries account for 0-5 pairs per arm. For 84-88% of missed
pairs no query's mask holds any pair hit, while the query matched to the pair has object prob > 0.5 (90-98%). Found
pairs: matched query own-mask efficiency median 0.71-0.77, purity 0.85-0.89. Results copied to
quirk_tracking_data/results/pu60/maskformer/ (arms_30k_val.tsv, val_scores/, pair_diag/, bootstrap/, runs/).

### Round 2 arms on the qw40 recipe (2026-11-05; val 2000-2999 only, no test use)

Same recipe as qw40_s0: pair GNN, 50k events, 60k steps, warmup + constant + linear decay over the last 20%, final
checkpoint. Builder study on val (builders60.py, coupled strict, best components cut per arm), paired event bootstrap
(95% interval) against trunk s0 and against qw40_s0 at cc 0.5.

| Arm | Best cut | Coupled loose / tight | Fake | B - trunk s0 (loose / tight) | B - qw40_s0 (loose / tight) |
|---|---|---|---|---|---|
| qw40_s0 (reference) | 0.5 | 0.655 / 0.183 | 0.323 | +0.055 / +0.031 | - |
| qw40_s1 (second seed) | 0.5 | 0.647 / 0.173 | 0.324 | +0.047 [+0.029, +0.066] / +0.021 [+0.002, +0.040] | -0.008 [-0.024, +0.008] / -0.010 [-0.028, +0.007] |
| qw100 | 0.5 | 0.654 / 0.174 | 0.345 | +0.054 [+0.033, +0.074] / +0.022 [+0.001, +0.043] | -0.001 [-0.017, +0.015] / -0.009 [-0.028, +0.010] |
| qw100 | 0.6 | 0.615 / 0.184 | 0.340 | +0.015 [-0.005, +0.034] / +0.032 [+0.011, +0.052] | -0.040 / +0.001 |
| posw40 (true quirk edges x40 only) | 0.95 | 0.601 / 0.110 | 0.337 | +0.001 [-0.016, +0.019] / -0.042 [-0.060, -0.025] | -0.054 / -0.073 |
| norm3 1/1/8, pair crops | 0.9 | 0.587 / 0.118 | 0.325 | -0.013 [-0.035, +0.008] / -0.034 [-0.054, -0.015] | -0.069 / -0.066 |

- The x40 gain reproduces with a second seed: qw40_s1 is within noise of s0 and above trunk s0 on both loose and tight.
- x100 matches x40 on coupled efficiency and costs 2 more points of fake (0.345 vs 0.323); no reason to prefer it.
- Weighting only the true quirk edges (posw40) loses: the scores of all quirk-touching edges go up, wrong ones included
  (val, edge cut 0.5: 115 quirk-touching edges kept per event at quirk purity 0.125 and quirk efficiency 0.559, vs 54 at
  0.221 and 0.467 for qw40_s0), so components merge until the components cut reaches 0.95, and the tight
  efficiency drops. Down-weighting the wrong quirk-touching edges relative to the true ones is what x40 on all of them avoids.
- norm3 with weights 1/1/8 and whole-pair crops recovers ordinary tracking (fake 0.325 at cut 0.9, vs 0.485 for equal
  weights) but stays below trunk s0 on tight and below qw40 on both.
- Val winner stays pair_50k_qw40_s0 (already scored on test above). GPU: 26.5 GPU-h for the four arms; running total in
  notes/pu60/gpu_hours.txt is 84.9 GPU-h.
- Files: models/gnn_pu60/<arm>/, results/pu60/gnn/<arm>/val/builders60_pe*.json, results/pu60/compare/val_vs_trunk.txt.

### negw40: only wrong quirk-touching edges x40 (2026-11-05; val only, no test use)

pair_50k_negw40_s0: same recipe as qw40_s0, --qw-mode neg (wrong quirk-touching edges x40, true pair edges at 1).
CPU smoke test 60855172: weights 40 on wrong / 1 on true quirk-touching edges, loss finite over 150 steps.
Scores shift down, so the components scan went to low cuts (0.05-0.6); best is cc 0.1.

| Arm (best cut) | Coupled loose / tight | Fake | Failures on 995 pairs: ok / fragmentation / absorption / no track |
|---|---|---|---|
| qw40_s0 (0.5) | 0.655 / 0.183 | 0.323 | 652 / 279 / 14 / 39 |
| posw40 (0.95) | 0.601 / 0.110 | 0.337 | 598 / 163 / 19 / 212 |
| negw40 (0.1) | 0.382 / 0.083 | 0.250 | 380 / 461 / 126 / 1 |

- negw40 - qw40_s0: loose -0.273 [-0.303, -0.243], tight -0.100 [-0.121, -0.078]; vs trunk s0: -0.218 / -0.069.
- Weighting only the wrong edges is worse than weighting only the true ones, and both are below x40 on all quirk-touching
  edges. negw40 keeps fewer true quirk edges (edge cut 0.1: quirk efficiency 0.357 vs 0.575 for qw40_s0, 39 vs 103
  quirk-touching edges per event) and pairs fragment. posw40 keeps too many wrong ones and pairs merge or are lost.
  The x40 gain needs both sides of the quirk-edge loss raised together.
- GPU: 6.41 GPU-h; running total 91.3 GPU-h. Files: models/gnn_pu60/pair_50k_negw40_s0/,
  results/pu60/gnn/pair_50k_negw40_s0/val/, results/pu60/compare/val_vs_trunk.txt.

Combined arms (Daniel 2026-11-05), 60k steps (warmup 1.5k, constant LR, linear decay over 48k-60k, no early stopping,
final checkpoint), 16k-event training set, configs maskformer_mu60_combined.yaml (pair x10) / _w25.yaml (pair x25) +
maskformer_mu60_long60k.yaml: pair weight in object BCE, mask losses and matching cost; inputs x, y, z, r, layer_id,
volume_id, 4 cluster-shape and 16 cluster-direction columns (no s / eta / phi, no position encoding); positive-hit
weight 16 in the mask focal loss; baseline window 0.15 x 0.30; no warm start. w10 on the gpu_shared_interactive slot
in chained 4 h windows (resume from the latest 500-step checkpoint, fresh epoch, seed = step; Lightning's Timer state
dropped on resume), w25 as one batch job. Val events 2000-2999, Python strict scoring, object threshold 0.1, paired
bootstrap against arm c (0.420 / 0.049):

| Arm | Coupled vl strict / tight strict | technical_sm vl / tight strict | Fake vl / tight strict | vs arm c (vl strict, 95%) | vs arm c (tight strict, 95%) |
|---|---|---|---|---|---|
| comb_w10 | 0.546 / 0.088 | 0.682 / 0.513 | 0.053 / 0.282 | +0.126 [+0.104, +0.148] | +0.039 [+0.025, +0.054] |
| comb_w25 | 0.000 / 0.000 (diverged) | 0.000 / 0.000 | 0.989 / 0.992 | - | - |

In-training val (100 events) coupled: w10 0.15 (4k), 0.32 (8k), 0.38 (14k), 0.42 (20k), 0.47 (30k), 0.50 (36k),
0.53 (42k), 0.54 (46k); w25 0.20 (4k), 0.23 (8k), 0.29 (14k), 0.47 (20k), 0.49 (30k), 0.52 (36k), 0.56 (38k), then 0.
w25 diverged at step ~38.4k: loss 14 -> 4e4 between steps 38359 and 38409, gradient norms 1e10-1e12 afterwards; the
spike guard then skipped every update to step 60k (22,657 skipped in total; the backstop adopts a new scale only for
norms steady within x20, which these were not), so the model stayed in the diverged state. An earlier spike episode at
22-24k (about 1,000 skipped updates) recovered. Last pre-divergence checkpoint: trunk-step=036000. w10 skipped 29+
updates over 60k. Peak GPU memory 33.0 GB in both (A100 40 GB). pair_diag w10: 438 of 448 missed pairs have no query
whose own mask matches the pair, 7 have one below the object threshold, 3 lose it in the final assignment; for 67% of
missed pairs no query's mask holds a pair hit; found pairs: matched query own-mask efficiency median 0.86, purity
0.81. Results: quirk_tracking_data/results/pu60/maskformer/ (val_scores/mf_comb_w*, pair_diag/, bootstrap/armc30k_vs_*,
runs/mf_comb_w*/ with all window configs and metrics).

### Per-class loss: does the quirk part overfit? (2026-11-05; inference only, no test use)

Logging: train_gnn60.py now logs, at every interval, for train batches (crops, as trained) and for the val events (full):
unweighted mean BCE of true pair edges / wrong quirk-touching edges / ordinary edges (live edges, as in norm3), the
weighted contribution of each class to the optimised loss (the three sum to train_loss; checked), and quirk-edge
efficiency and purity at 0.5. Default on (--no-class-log turns it off). Code: pileup60/class_loss60.py. Smoke test
(GPU, 40 steps, seed 0): train_loss with logging on vs off differs by 1e-6, the same as two runs with logging off.

Retrospective (pileup60/class_loss_retro.py): saved checkpoints of pair_50k_qw40_s0 and pair_50k_trunk_s0, full events
(no crop) for both splits, the run's own loss weights; 200 training events (50 each from blocks 3000, 15000, 30000,
45000) vs 200 val events (2000-2199). Gap = val - train, 95% interval from resampling events within each split.

| Step | qw40 true-pair BCE train / val (gap) | qw40 wrong-quirk BCE train / val | qw40 quirk-edge eff train / val | qw40 quirk-edge pur train / val | trunk true-pair BCE gap |
|---|---|---|---|---|---|
| 4000 | 2.404 / 2.546 (+0.14 [-0.00, +0.26]) | 0.0277 / 0.0272 | 0.181 / 0.170 | 0.733 / 0.734 | +0.17 [-0.04, +0.36] |
| 12000 | 1.483 / 1.621 (+0.14) | 0.0263 / 0.0272 | 0.476 / 0.440 | 0.836 / 0.818 | +0.14 |
| 20000 | 1.056 / 1.189 (+0.13) | 0.0300 / 0.0318 | 0.616 / 0.572 | 0.809 / 0.796 | +0.23 |
| 28000 | 1.326 / 1.532 (+0.21) | 0.0228 / 0.0239 | 0.535 / 0.485 | 0.868 / 0.841 | +0.18 |
| 36000 | 1.199 / 1.343 (+0.14) | 0.0291 / 0.0290 | 0.549 / 0.505 | 0.838 / 0.828 | +0.18 |
| 44000 | 0.967 / 1.111 (+0.14) | 0.0280 / 0.0298 | 0.669 / 0.636 | 0.825 / 0.804 | +0.22 |
| 52000 | 1.103 / 1.236 (+0.13) | 0.0240 / 0.0266 | 0.626 / 0.586 | 0.868 / 0.833 | +0.19 |
| 60000 | 0.769 / 0.877 (+0.11 [-0.08, +0.31]) | 0.0242 / 0.0271 (+0.003 [-0.001, +0.007]) | 0.709 / 0.676 | 0.867 / 0.836 (-0.031 [-0.059, -0.005]) | +0.15 [-0.07, +0.38] |

- No sign of the quirk part overfitting: val true-pair BCE falls with train BCE at every checkpoint (qw40 2.55 -> 0.88),
  and the train-val gap is the same at step 4000, when the model had seen at most 4000 of the 50k training events, as at
  60000 (+0.14 vs +0.11). The gap is a difference between the two event samples, not memorisation.
- Wrong-quirk BCE and quirk-edge purity show a small late gap (val purity 3 points lower at 52k-60k, interval just
  excluding 0); the trunk has the same size gap (-0.029 [-0.062, +0.002] at 60k), so the x40 weight does not add to it.
- Ordinary-edge BCE: gap +0.0002 throughout.
- Both curves are non-monotonic (28000 is worse than 20000 for both runs): the constant-LR phase is noisy; the decay
  (48k-60k) gives the lowest val values.
- Caveat: 1.2 passes over 50k events, so each training event was seen once or twice by 60k steps.
- GPU: 2 inference jobs (60888612, 60888613). Files: results/pu60/classlog/{pair_50k_qw40_s0,pair_50k_trunk_s0}.json
  (per-event terms), summary.txt (all gaps with intervals), smoke_classlog_*_log.csv.

Split losses (Daniel 2026-11-05): QuirkModel now logs, for train (every log_every_n_steps) and val, the final-layer
loss components by target type as {stage}/split_<component>_<type>[_w] (quirk/split_losses.py; components object_bce,
mask_dice, mask_focal; types pair, ord, and null for object BCE; plain = unweighted per-target mean, _w = that type's
share of the loss actually optimised, with object weights, null weight and positive-hit weight; no gradient, training
unchanged; on by default, log_split_losses=false turns it off). Check on one event: the _w parts sum to the task losses
(object BCE exact, Dice and focal within 0.1%, fp32 vs bf16).

Retrospective train-vs-val check (no training; mu60_eval/retro_split.py, job 60888608, 0.9 GPU-h): every saved
checkpoint of mf_comb_w10 (4k ... 60k) and arm c (trunk 4k ... 24k, decay branch at 28k), on 200 training events the
model saw (3000-3199) and 200 val events (2000-2199). Pair found = writer assignment, strict > 50% / > 50%.

| Step (w10) | Pair found train / val | Pair focal x1e3 train / val | Pair Dice train / val | Pair own-mask completeness train / val | Ordinary Dice train / val |
|---|---|---|---|---|---|
| 4k | 0.21 / 0.14 | 1.31 / 1.31 | 0.78 / 0.83 | 0.30 / 0.23 | 0.811 / 0.812 |
| 8k | 0.41 / 0.31 | 1.27 / 1.43 | 0.65 / 0.70 | 0.40 / 0.34 | 0.597 / 0.598 |
| 16k | 0.46 / 0.36 | 1.00 / 1.20 | 0.63 / 0.69 | 0.42 / 0.37 | 0.510 / 0.521 |
| 24k | 0.50 / 0.43 | 0.95 / 1.05 | 0.59 / 0.63 | 0.44 / 0.41 | 0.464 / 0.467 |
| 32k | 0.53 / 0.41 | 1.29 / 1.47 | 0.56 / 0.64 | 0.47 / 0.39 | 0.437 / 0.449 |
| 40k | 0.54 / 0.48 | 0.96 / 1.07 | 0.56 / 0.60 | 0.52 / 0.46 | 0.414 / 0.418 |
| 48k | 0.56 / 0.45 | 0.89 / 1.02 | 0.55 / 0.61 | 0.48 / 0.45 | 0.413 / 0.420 |
| 56k | 0.64 / 0.56 | 0.85 / 0.94 | 0.47 / 0.53 | 0.57 / 0.51 | 0.340 / 0.341 |
| 60k | 0.65 / 0.58 | 0.85 / 0.95 | 0.46 / 0.52 | 0.61 / 0.55 | 0.336 / 0.342 |

The pair has a train-val gap at every checkpoint from 4k on (pair found +0.07 to +0.12, Dice +0.03 to +0.08, focal
val above train), while ordinary tracks have none (Dice within 0.01). The gap does not grow with steps, and the pair's
val losses keep falling to the end (focal 1.31 -> 0.95, Dice 0.83 -> 0.52; pair found 0.14 -> 0.58), so by the
"train falling, val flat or rising" criterion the pair does not overfit within 60k steps; it generalises less well
than ordinary tracks from the start. Arm c shows the same pattern (pair found train / val 0.49 / 0.42 at 28k).
Full table (all components, plain and weighted, both runs): quirk_tracking_data/results/pu60/maskformer/retro_split/.
200-event samples: binomial sigma about 0.035 per pair-found value.

More data and the w25 decay (Daniel 2026-11-06). mf_comb_w10_97k: comb_w10 config on training events 3000-99999
(97k events, all with direction features), 100k steps (warmup 1.5k, constant LR, linear decay over 80k-100k), one
batch job (60915583, 25.4 GPU-h, 1.1 steps/s, peak 34.3 GB, 889 updates skipped by the spike guard, most between
60k and 70k; in-training val dipped to 0.37 at 70k and recovered), split-loss logging on. mf_comb_w25_decay36k: decay
branch of mf_comb_w25 from trunk-step=036000 (before its divergence), 12k steps of linear decay to step 48k (20% of
60k), interactive slot (2.9 GPU-h). Val 2000-2999, Python strict, object threshold 0.1, paired bootstrap against
mf_comb_w10 (0.546 / 0.088):

| Run | Coupled vl strict / tight strict | Per quirk vl strict | technical_sm vl / tight strict | Fake vl / tight strict | vs mf_comb_w10 vl strict (95%) | tight strict (95%) |
|---|---|---|---|---|---|---|
| mf_comb_w10 (16k events, 60k steps) | 0.546 / 0.088 | 0.029 | 0.682 / 0.513 | 0.053 / 0.282 | - | - |
| mf_comb_w10_97k (97k events, 100k steps) | 0.578 / 0.117 | 0.036 | 0.699 / 0.547 | 0.046 / 0.253 | +0.032 [+0.015, +0.049] | +0.028 [+0.014, +0.042] |
| mf_comb_w25_decay36k (pair x25, 48k steps) | 0.542 / 0.095 | 0.030 | 0.615 / 0.416 | 0.121 / 0.405 | -0.004 [-0.019, +0.011] | +0.007 [-0.005, +0.019] |

pair_diag: 97k run 409 of 416 missed pairs without a query whose own mask matches the pair (2 object score, 5
competition), found pairs' matched-query own-mask efficiency median 0.75; w25 decay 450 of 460 (0 / 10). Results:
quirk_tracking_data/results/pu60/maskformer/ (val_scores/mf_comb_w10_97k_ev2000-2999, val_scores/
mf_comb_w25_decay36k_ev2000-2999, pair_diag/, bootstrap/comb_w10_vs_*, runs/mf_comb_w10_97k, runs/mf_comb_w25_decay36k).
Corrected 2026-11-09 against val_scores/mf_comb_w25_decay36k_ev2000-2999/pyscore_pair.json: w25 decay tight strict
coupled 0.095 (95/995) and technical_sm 0.416 (were written 0.096 / 0.417); its vl strict fake rate is now the
pair-truth value 0.121 like the other rows (was the normal-truth value 0.124).

### Other string scales at mu 60, no retraining (2026-11-09; test 0-1999 of each set, used once)

pair_50k_qw40_s0 (trained on m200_L774_mu60) applied unchanged to m200_L278_mu60 and m200_L2154_mu60: graph v3, components
0.5, the L774 tagger (cut 0.776) and quality model (cuts 0.542 / 0.825 from L774 val B), the 1M splitter. Direction
features for blocks 0-1 built as for L774 (mu60_prep.sh); R1 copies and pair truth were already made by the MaskFormer
prep. L774 rescored with the same code (py_score.py with the unassociated class and all-tracks fake denominator;
quality cut at the exact val-B cut): identical to the earlier numbers at 3 decimals. Python strict throughout.

| Quantity (test) | L = 278 eV | L = 774 eV (training point) | L = 2154 eV |
|---|---|---|---|
| coupled strict loose / tight | 0.211 / 0.035 | 0.650 / 0.169 | 0.882 / 0.369 |
| fake / technical_sm | 0.323 / 0.959 | 0.323 / 0.958 | 0.323 / 0.959 |
| quality cut fake 0.07: coupled / tight / technical_sm | 0.203 / 0.035 / 0.953 | 0.632 / 0.167 / 0.952 | 0.865 / 0.365 / 0.953 |
| quality cut fake 0.02: coupled / tight / technical_sm | 0.180 / 0.031 / 0.934 | 0.587 / 0.165 / 0.932 | 0.806 / 0.355 / 0.934 |
| per-quirk strict (pair GNN tracks) loose / tight | 0.321 / 0.064 | 0.099 / 0.001 | 0.163 / 0.001 |
| tagger: pair-good efficiency / tagged per event (ordinary) | 0.527 / 0.33 (0.18) | 0.916 / 0.82 (0.19) | 0.872 / 0.98 (0.19) |
| gated splitter chain: per-quirk loose / tight, quirk event; tracks split | 0.364 / 0.086, 0.202; 607 | 0.397 / 0.098, 0.330; 1611 | 0.454 / 0.018, 0.336; 1939 |
| mu 0 reference (gnn_grid_strict.csv, mu 0 GNN): coupled / tight; per-quirk loose / tight | 0.518 / 0.169; 0.790 / 0.339 | (training point, not in grid) | 0.960 / 0.766; 0.610 / 0.008 |

- Ordinary tracking (fake, technical_sm, tracks per event) does not depend on the string scale.
- Coupled efficiency rises with Lambda as at mu 0 (tighter, more collinear pairs): at 2154 eV the mu 60 pair GNN reaches
  0.88 coupled (mu 0: 0.96); at 278 eV it falls to 0.21 (mu 0: 0.52). The mu 60 / mu 0 coupled ratio is 0.92 at 2154 and 0.41 at
  278 eV; at 774 eV the only mu 0 test number at hand is 0.935 at fake 0.02, against 0.587 here at the same fake (0.63).
- At 278 eV the two quirks separate; the pair GNN builds them as separate tracks (per-quirk 0.32, above coupled), the
  tagger keeps only 53% of pair-good quirk tracks at the L774 cut, and the chain adds little (0.32 -> 0.36).
- Files: results/pu60/lambda/<set>/{pypair,py}_cc0.5_mh4.json, py_split{,_pair}.json, apply.json, roc_test.csv.
  Scripts: pileup60/lam_driver.sh, lam_gpu.sbatch, lam_score.sbatch, lam_apply60.py, lam_rescore774.sbatch.

Test (MaskFormer's only test use, 2026-11-09): mf_comb_w10_97k final checkpoint (step 100k) on test events 0-1999,
object threshold 0.1. Chunked ACTS (8 x 250-event chunks, 48 workers, CPU m4958) and Python strict give identical
coupled numbers.

| Truth | vl / tight | vl strict / tight strict | technical_sm vl strict / tight strict | Fake vl strict / tight strict |
|---|---|---|---|---|
| pair (coupled) | 0.6075 / 0.1228 | 0.6005 (1198/1995) / 0.1178 (235/1995) | 0.691 / 0.539 | 0.047 / 0.255 |
| per quirk | 0.0678 / 0.000 | 0.0345 / 0.000 | 0.691 / 0.539 | 0.049 / 0.256 |

No track-quality score exists in the MaskFormer writer, so no quality-cut working points. Results:
quirk_tracking_data/results/pu60/maskformer/test_mf_comb_w10_97k/.

Longer schedule on 97k events (branch from the 97k run's trunk-step=080000, constant LR 5e-5 to 160k, decay to 200k):
diverged twice. Seed 80000: loss from about 12 to 1.5e7 between steps 80,689 and 80,729, gradient norms 1e11-1e15
afterwards, every update skipped by the spike guard (cancelled at 88.8k). Seed 80001: one val event with loss 1.3e4 at
86k (val coupled still 0.51), then the same blow-up at about 88.5k (cancelled at 89k). In-training val before the
blow-up: 0.42 (82k), 0.57 (84k), 0.61 (88k). The 97k run itself had an episode of 650 skipped updates at 60k-70k at
the same LR and recovered; w25 diverged the same way at 38k. Metrics of both attempts:
results/pu60/maskformer/runs/mf_comb_w10_97k_long_diverged/.
