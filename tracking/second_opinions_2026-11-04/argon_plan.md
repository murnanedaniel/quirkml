# Independent Second Opinion: Boosting Quirk Tracking Efficiency at Pileup 60 ($\mu = 60$)

## 1. Why the Quirk Pair Is Much Harder at $\mu = 60$ (and Cheap Diagnostics to Tell Causes Apart)

### A. Pair GNN ($\mu = 0$: `0.944 / 0.688` $\to$ $\mu = 60$: `0.520 / 0.112`)

Four mechanisms compound across the loss, message passing, graph construction, and track building:

1. **Uncompensated 1:15 class imbalance *within* quirk-touching edges ([`train_gnn.py:L64-65`](file:///global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/gnn/train_gnn.py#L64-L65))** *(Proven from code + `edge60` numbers)*:
   - In [`train_gnn.py`](file:///global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/gnn/train_gnn.py#L64-L65), `--quirk-weight` (`QW`) is applied via:
     ```python
     w = w * np.where(hq[g["src"]] | hq[g["dst"]], QW, 1.0).astype(np.float32)
     ```
     with `pos_weight = 1.0`. This multiplies **both** true same-pair edges (`hq[s] & hq[d]`, $y = 1$) **and** wrong quirk-to-pileup edges (`hq[s] ^ hq[d]`, $y = 0$) by `QW`.
   - From $\mu = 0 \to \mu = 60$, the number of true pair edges per event is invariant ($\sim 80\text{--}160$), whereas wrong pileup edges touching the ~27 quirk hits scales with hit density ($\sim 16\times$ higher, $\sim 1,000\text{--}1,500$ per event in `v3_mu0f_cut0.1`).
   - Thus, **within the `QW`-weighted subset**, the positive-to-negative ratio collapsed from $\sim 1:1$ at $\mu = 0$ to $\sim 1:15$ at $\mu = 60$, shifting the Bayes-optimal logit on uncertain quirk edges down by $\Delta \text{logit} \approx -\ln(15) \approx -2.7$.
   - **Implication for the running `QW = 40` arm**: Raising `QW` from 10 to 40 fixes global gradient dilution against ordinary pileup edges, **but leaves the ~1:15 positive-to-negative imbalance among quirk-touching edges unchanged** (both `hq[s] & hq[d]` and `hq[s] ^ hq[d]` are multiplied by 40). This directly explains why ranking (`AUC = 0.953`) is much better than calibration at threshold $0.5$ (`pair-chain eff = 0.424` vs `0.840` at $\mu = 0$, while `purity_quirk_touching = 0.80` at $0.5$ and `0.85` at $0.6$).

2. **Plain connected-components (`cc`) pincer between fragmentation and pileup-bridge absorption ([`build_tracks.py:L128-136`](file:///global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/gnn/build_tracks.py#L128-L136))** *(Proven from [EXPERIMENTS.md:L1106-1136](file:///global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/EXPERIMENTS.md#L1106-L1136))*:
   - At cut $0.5$, `ceiling (ii)` (connected components using *only* true pair edges with score $> 0.5$) is **`0.569 / 0.247`**, whereas actual built tracks at $0.5$ get **`0.535 / 0.090`**—losing **15.7 points of tight efficiency (63% of the tight ceiling!)** and 3.4 points of loose efficiency purely to wrong edges.
   - Why? At cut $0.5$, **11.1 wrong quirk-touching edges per event** (`7.4` at $0.6$) survive. In global unweighted connected components (`cc`), **a single wrong edge** above `cc_cut` between any of the 26 quirk hits and any hit of a 15-hit ordinary track (whose internal consecutive edges are scored $> 0.95$) merges the entire ordinary track into the quirk component.
   - For a 26-hit pair, absorbing even **3 wrong hits** pulls purity to $26/29 = 89.7\% \le 90\%$ (killing tight match); absorbing $\ge 26$ hits (e.g., two ordinary tracks) pulls purity $\le 50\%$ (killing loose match).
   - Lowering `cc_cut` to $0.3$ on `val` ([EXPERIMENTS.md:L1108](file:///global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/EXPERIMENTS.md#L1108)) caused loose efficiency to *drop* from `0.527` to `0.325` because global `cc` percolates into surrounding pileup helices.

3. **Ungated, unnormalized sum message passing in `InteractionGNN` ([`common.py:L203-206`](file:///global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/gnn/common.py#L203-L206))** *(High-confidence architectural diagnosis)*:
   - `agg_in = torch.zeros_like(h).index_add_(0, dst, e)` sums all incident edge embeddings without gating or degree normalization.
   - Because the $\mu = 0$ filter (`g1f_qw10r075_q50`) was trained with `QW = 50` to generously keep quirk-like edges, quirk hits have high node degree in `v3_mu0f_cut0.1` (~30–60 incident edges, ~90% of which are pileup hits at $\mu = 60$). Across 8 message-passing steps, raw sum aggregation at each quirk node is swamped ~10:1 by pileup messages.

4. **Graph construction & $\phi$-cropping losses ([`graph_apply60.py`](file:///global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/pileup60/graph_apply60.py), [`train_gnn60.py:L60-73`](file:///global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/pileup60/train_gnn60.py#L60-L73))**:
   - In the radius graph ($r = 0.075, k_{\max} = 512$), **95% of hits saturate $k_{\max} = 512$** at $\mu = 60$ (vs 8% at $\mu = 0$), and tightening the $\mu = 0$ filter cut $100\times$ ($0.001 \to 0.1$) already loses **15.1% of quirk chain edges**, capping the graph tight ceiling at `0.716` (`ceiling (i)`).
   - During training, `phi_crop` centers on a single random quirk hit only 50% of the time (`quirk_frac = 0.5`, which can sit at the $\phi$-boundary of a wide pair) and uniformly at random the other 50%, slicing through wide pairs during training.

---

### B. MaskFormer ($\mu = 0$: `0.907 / 0.553` at 100k, `0.855` at 30k $\to$ $\mu = 60$: `0.151 / 0.011` at 30k)

1. **Windowed encoder ($0.15 \times 0.30$) + high-frequency $(\phi, \eta)$ RoPE is an inductive bias for radial helices, not wide non-helical pairs ([`flexgeo.py:L28-40, L118`](file:///global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/maskformer/packages/ml_tracking/src/ml_tracking/experiments/quirk/flexgeo.py#L28-L40))**:
   - Only **7% of pairs** fit in one $0.15 \times 0.30$ window ($\text{median } \Delta\phi = 0.47, \Delta\eta = 1.04$).
   - High-frequency RoPE (`PHI_FREQUENCIES` up to $1200$, `ETA_FREQUENCIES` up to $300$) in `flexgeo.py` strongly favors radial trajectories at constant $(\phi, \eta)$ ($\Delta\phi < 0.05, \Delta\eta < 0.02$). Quirk pairs oscillate across $\Delta\phi \sim 0.5, \Delta\eta \sim 1.0$, where frequencies $\omega \ge 22$ wrap phase multiple times between consecutive hits.

2. **Compound $16\times$ object-level and $16\times$ hit-level gradient dilution in the decoder ([`maskformer_mu60.yaml`](file:///global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/maskformer/packages/ml_tracking/src/ml_tracking/experiments/quirk/configs/maskformer_mu60.yaml), [`loss.py:L178-181`](file:///global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/maskformer/packages/ml_tracking/src/ml_tracking/models/loss.py#L178-L181))**:
   - **Object-level**: At $\mu = 0$, `batch_size = 8` with ~60 targets/event gave 8 quirk pairs per step (~1.7% of targets). At $\mu = 60$, `batch_size = 1` with ~200 ordinary targets/event gives **1 quirk pair out of ~200 targets ($0.5\%$)** per step—so 99.5% of decoder gradients train the backbone and queries to find straight $(\phi, \eta)$ helices.
   - **Hit-level**: [`mask_focal_loss`](file:///global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/maskformer/packages/ml_tracking/src/ml_tracking/models/loss.py#L178-L181) (and `normalize_focal_cost` in [`task.py:L646-647`](file:///global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/maskformer/packages/ml_tracking/src/ml_tracking/models/task.py#L646-L647)) divides the focal loss sum over all $H$ hits by `valid_counts = input_pad_mask.sum(-1)` ($H \approx 67,000$ at $\mu = 60$ vs $4,300$ at $\mu = 0$) with `sample_weight = 1.0` (no positive-hit $\alpha$ weight). For a 26-hit pair, the positive-hit gradient from `mask_focal` is diluted by **$67\text{k}/4.3\text{k} \approx 16\times$**.

3. **Feature regression in [`maskformer_mu60.yaml:L87-93`](file:///global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/maskformer/packages/ml_tracking/src/ml_tracking/experiments/quirk/configs/maskformer_mu60.yaml#L87-L93)** *(Proven from config inspection)*:
   - `maskformer_mu60.yaml` **omits `direction_dir`** (the 13 offline cluster-direction features from `direction.yaml` that were generated for $\mu = 60$ in `quirk_tracking_data/direction/m200_L774_mu60` are never loaded!), **omits `volume_id`** (passing only `layer_id`, which collides across barrel/endcap volumes), and **re-enables origin features `s, eta, phi` + `PositionEncoder(r, eta, phi)`** that Stage D (`mf-D-noorigin`) showed hurt quirk learning.

4. **Static queries + hard thresholded masked cross-attention ([`decoder.py:L281-297`](file:///global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/maskformer/packages/ml_tracking/src/ml_tracking/models/decoder.py#L281-L297))**:
   - At each decoder layer, cross-attention is masked to hits with `sigmoid(logit) >= 0.5`. If a query has not locked onto the pair, it either attends to a tiny fragment or (`unmask_all_false: true`) attends via uniform softmax over all **67,000 hits**, where 26 quirk hits are $0.04\%$ of the keys.

---

### C. Cheap Diagnostics on Existing Outputs (0 GPU-h, ~10 min CPU)

1. **GNN — Extended `edge60.py` cut scan + Fragmentation vs. Absorption breakdown on `/pscratch/sd/d/danieltm/quirk_pu60/gnn/pair_16k_s0/scored/`**:
   - Run [`edge60.py`](file:///global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/pileup60/edge60.py) at cuts `0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7` (currently only `0.5, 0.6`).
   - For every test/val pair that fails loose (or tight) matching at `cc_cut = 0.3, 0.5, 0.6`, classify whether its majority-quirk component failed **completeness ($\le 50\%$)** (fragmentation) or **purity ($\le 50\%$ / $\le 90\%$)** (pileup-bridge absorption).
   - Bin pair-edge recall by quirk node degree in `v3_mu0f_cut0.1` and by step length $d_{3\text{D}}$ to quantify message-passing swamping at high-degree nodes.

2. **MaskFormer — Span breakdown & Matched-Query vs. Writer check on `/pscratch/sd/d/danieltm/quirk_mu60/predictions/`**:
   - Bin the 30k-step val predictions (`coupled = 0.151`) by true pair $(\Delta\phi_{\text{span}}, \Delta\eta_{\text{span}})$: if pairs with $\Delta\phi < 0.15, \Delta\eta < 0.30$ have high efficiency (~0.50+) while wider pairs are near 0, window reach dominates; if small-span pairs are also $< 0.25$, loss dilution / RoPE / missing direction features dominate.
   - Compare the Hungarian-matched query's `val/quirk_hit_eff_mean`, `val/quirk_hit_pur_mean`, and `val/p0.5_quirk_eff` in the existing logs against writer-assigned `val/coupled_strict` across `mask_threshold` $\in \{0.2, 0.3, 0.5\}$ and `assignment="object_weighted"`.

---

## 2. Highest Efficiency Gain per GPU-Hour

### (a) Pair GNN (Highest ROI overall)

| Rank | Change | Cost | Expected Gain (Loose / Tight) | Evidence / Mechanism |
|---|---|---|---|---|
| **1** | **Two-threshold / bridge-pruned track builder (`cc_core` / helix-veto agglomeration)** on existing & new GNN scores | **0 GPU-h** (CPU) | **+4–8 pts loose / +8–14 pts tight** | Eliminates the gap between `actual` (`0.535 / 0.090`) and `ceiling (ii)` (`0.569 / 0.247`) at 0.5, and allows collecting internal pair edges down to score `0.25–0.35` without percolating into helical pileup tracks. |
| **2** | **Asymmetric quirk loss weights** (`W_PAIR_POS = 120`, `W_QUIRK_NEG = 12`) + **fixed `phi_crop`** (center on pair mean $\phi$, `quirk_frac = 0.8`, prune degree-0 nodes) | **~2.8 GPU-h** | **+6–12 pts loose / +5–10 pts tight** | Fixes the 1:15 pos:neg ratio within quirk-touching edges (`train_gnn.py:L65`) that even `QW=40` leaves untouched; pruning degree-0 nodes in `phi_crop` cuts ~70% of node-MLP compute, giving ~45k steps in 2.8 GPU-h. |
| **3** | **Edge-gated + degree-normalized `InteractionGNN`** (`sigmoid(gate(e)) * e`, `1/sqrt(deg)`) + **same-layer partner distance features** | **~3.2 GPU-h** | **+5–10 pts loose / +5–10 pts tight** | Stops ~50 false pileup edges per quirk node from swamping `index_add_` sum aggregation (`common.py:L203`); same-layer partner features (`< 30 mm` on consecutive layers) exploit the A0 pair geometry (`hf_v3` gained +10 pts recall in 2k steps). |
| **4** | **Two-stage local Transformer pair cleaner/splitter** (`splitter2.py` on candidate pair components + 1-hop graph neighborhood, ~100–200 hits) | **~1.5–2.0 GPU-h** | **+3–6 pts loose / +10–20 pts tight** (and **per-quirk `0.17 -> 0.50+`**) | Full $O(N^2)$ attention with pairwise geometric + direction bias (`splitter2.py:L44-64`) on ~150 local hits runs in <2 GPU-h for 50k events and prunes attached pileup bridges while splitting A vs B. |

*Notes on other levers for GNN*:
- **More data vs. more steps**: `50k` events at `30k` steps gave no gain (`0.529` vs `0.538`), whereas `50k` at `60k` steps reached `0.576` val. With `batch = 1` and `quirk_frac = 0.5`, 30k steps only see 15k quirk crops total. The GNN is **step-, loss-, and architecture-limited, not data-limited**—reuse the existing 16k–50k precomputed graphs.
- **Hit time**: [`common.py:L49`](file:///global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/gnn/common.py#L49) states `"No truth, no time"`. *If* detector hit time (even 50–100 ps) were permitted by the task rules, $m = 200\text{ GeV}$ quirks have median $\beta\gamma \approx 2.1$ ($\beta \approx 0.90$), arriving **0.3–2 ns late** at outer layers, which would trivially separate quirk hits from $\beta \approx 1$ pileup. Assuming hit time is excluded by benchmark rules, do not use it.

---

### (b) MaskFormer

First, **none of the currently running arms (a), (c), (d) can succeed in isolation**:
- Arm (a) (wide window $0.5 \times 1.0$) still has only 0.5% of targets as the quirk pair and $16\times$ focal dilution.
- Arm (c) (pair weight $\times 10$) still uses the $0.15 \times 0.30$ window that severs 93% of pairs.
- Arm (d) (warm start) uses the $0.15 \times 0.30$ window AND copies non-RoPE $\mu = 0$ weights into a `GeometricFlexAttention` encoder that applies high-frequency RoPE (`flexgeo.py:L99`), scrambling pretrained Q-K dot products (and if warm-started from `mf-E-pairdir-long`, `_warm_start` skips `input_nets.0.net` because `25 > 12`).

| Rank | Change | Cost | Expected Gain | Evidence / Mechanism |
|---|---|---|---|---|
| **1** | **Combine into ONE run**: Wide window (`0.5 x 1.0`) + Quirk weight $\times 25$ (loss & cost) + **Cluster-Direction & No-Origin features (`direction.yaml` + `volume_id`)** + Cap RoPE frequencies ($\omega_\phi \le 60, \omega_\eta \le 30$, drop redundant `PositionEncoder`) | **~6–8 GPU-h** (or **~3.5 GPU-h** with hit filter) | **+25–45 pts loose** (`0.15 -> 0.40–0.60`, *estimate*) | Fixes all four simultaneous blockers (window reach, 1:200 target dilution, missing direction/volume features, and high-frequency origin-angle bias). |
| **2** | **Apply `filter_flex_v3` at cut `0.005` (46.8k hits, 99.9% pair $>50\%$ ceiling) or cut `0.01` (37.2k hits, 99.3% pair $>50\%$ ceiling)** | **Saves 50% GPU-h/step** (`2.05` vs `0.97` steps/s) | **2x steps per GPU-h** + cleaner decoder attention | Refusing a filter that keeps **99.3%–99.9%** of pairs at $>50\%$ completeness when current efficiency is `0.151` sacrifices a $2\times$ training speedup and $45\%$ pileup reduction to protect $0.1\text{--}0.7\%$ of the ceiling ([EXPERIMENTS.md:L1019-1021](file:///global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/EXPERIMENTS.md#L1019-L1021)). |
| **3** | **Positive-hit focal weight / normalization** in [`loss.py:L178`](file:///global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/maskformer/packages/ml_tracking/src/ml_tracking/models/loss.py#L178) (`sample_weight` $\alpha_{\text{pos}} \approx 10\text{--}16$) | **0 extra GPU-h** | **+5–15 pts loose** (*estimate*) | Restores the $\mu = 0$ magnitude of positive-hit `mask_focal` gradients against `valid_counts = 67,000` dilution and `null_mask_weight = 5.0`. |

---

## 3. What Looks Wrong or Suspicious in the Setup

1. **[BUG / PATHOLOGY] `QW` up-weights negative quirk-touching edges equally with positive quirk edges ([`train_gnn.py:L65`](file:///global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/gnn/train_gnn.py#L65), [`train_filter.py:L65`](file:///global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/gnn/train_filter.py#L65))**:
   - In [`train_filter_stream.py:L104`](file:///global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/pileup60/train_filter_stream.py#L104), `must = (y > 0) | hq[s] | hq[t]` keeps all ~14,000 edges touching quirk hits in the 22.8M-edge radius graph (99.8% of which have $y = 0$) and multiplies their loss weight by `QW = 50`. That created a ~250:1 negative-to-positive weighted gradient on quirk hits, teaching the fine-tuned $\mu = 60$ filter to **suppress edges touching quirk hits** and explaining why `train_filter_stream.py` degraded below the zero-shot $\mu = 0$ filter ([EXPERIMENTS.md:L979-985](file:///global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/EXPERIMENTS.md#L979-L985)).
   - In [`train_gnn60.py`](file:///global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/pileup60/train_gnn60.py), the same formula means `QW = 40` does not fix the ~1:15 pos:neg imbalance on quirk-touching edges.

2. **[BUG / WASTE] `phi_crop` in [`train_gnn60.py:L60-73, L313-315`](file:///global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/pileup60/train_gnn60.py#L60-L73)**:
   - `T.batch` computes `dir_edge_features` and `edge_truth` on all 1.49M edges *before* `phi_crop` drops 73% of them.
   - `phi_crop` never slices the node tensor `f` (~70k nodes), so `InteractionGNN` runs its node MLPs across 8 iterations on ~50,000 isolated degree-0 nodes every step.
   - Centering on a single random quirk hit (`quirk_frac = 0.5`) or uniform random $\phi$ (`0.5`) slices through wide pairs during training and leaves half the steps without the pair. (Also, `phi_crop(evs[b0], ...)` is broken if `--batch > 1`.)

3. **[CONFIG REGRESSION] [`maskformer_mu60.yaml:L87-93`](file:///global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/maskformer/packages/ml_tracking/src/ml_tracking/experiments/quirk/configs/maskformer_mu60.yaml#L87-L93) dropped `direction_dir` and `volume_id`, and re-enabled origin features**:
   - The $\mu = 60$ direction features (`quirk_tracking_data/direction/m200_L774_mu60`) are never used in `maskformer_mu60.yaml`.
   - `layer_id` is passed without `volume_id`.
   - Origin features (`s, eta, phi`) and `PositionEncoder(r, eta, phi)` are active alongside `FlexGeo` RoPE on `(phi, eta)`.

4. **[WARM-START MISMATCH] [`QuirkModel._warm_start` (`model.py:L40-75`)](file:///global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/maskformer/packages/ml_tracking/src/ml_tracking/experiments/quirk/model.py#L40-L75)**:
   - Copies non-RoPE $\mu = 0$ encoder weights into a RoPE `GeometricFlexAttention` encoder (frequencies up to 1200), and drops `input_nets.0.net` if the $\mu = 0$ checkpoint had more input fields (25 in `mf-E-pairdir-long` vs 12 in `maskformer_mu60.yaml`).

5. **[VAL SUBSET NOISE] In-training `track_val` in `train_gnn60.py` uses 200 val events (`2000-2199`)**:
   - 200 events have $\pm 2.5\%$ binomial noise; always choose operating cuts and checkpoints on all 1,000 val events (`2000-2999`). Evaluation (`py_score.py` vs ACTS) and dataset splits (`0-1999` / `2000-2999` / `3000-99999`) are otherwise clean and verified.

---

## 4. Prioritized Plan for the Remaining ~28 GPU-h

### Step 0: CPU Diagnostics & Track-Builder Upgrade (0 GPU-h)
1. Run `edge60.py` across cuts `0.05..0.7` and decompose failed pairs into fragmentation vs. pileup-bridge absorption on `pair_16k_s0`, `50k_60k`, and the finishing `QW = 40` arm.
2. Implement **two-threshold / helix-veto agglomerative track building** in `build_tracks.py`:
   - Identify smooth ordinary helical tracks at cut `0.6–0.7` (or veto single-edge bridges into components with low Kasa `circ_rms` / `rz_res_rms` and `hits_per_layer ~ 1`), while clustering the remaining quirk-candidate subgraph down to cut `0.25–0.35` using 2-edge-connected components / multi-edge attachment.
   - **Expected result**: `0.520 / 0.112` $\to$ **`0.58–0.63` loose / `0.19–0.25` tight** at **0 GPU-h**.

### Step 1: Gate on Currently Running Arms (0 new GPU-h)
- Score the finishing GNN `QW = 40` arm with both standard `cc` and the Step 0 builder.
- Check MaskFormer arms (a), (c), (d):
  - **Stopping rule**: Do not spend more than **8–10 GPU-h** further on MaskFormer unless the unified run (Step 3) passes **`val/coupled_strict >= 0.35` by 16k steps**. Reallocate any saved MaskFormer hours to GNN + Local Transformer refinement.

### Step 2: GNN High-Leverage Runs (~8.0 to 12.0 GPU-h)
1. **Run G1 (~3.0 GPU-h) — Asymmetric Quirk Loss + Fixed `phi_crop` + Edge-Gated `InteractionGNN`**:
   - In `train_gnn.py`: set `W_PAIR_POS = 120.0` (`hq[s] & hq[d]`) and `W_QUIRK_NEG = 12.0` (`hq[s] ^ hq[d]`).
   - In `train_gnn60.py`: center `phi_crop` on the pair's circular-mean $\phi$ (`quirk_frac = 0.8`), compact active nodes in `f` before `model(...)` (~35% faster steps $\to$ 45k steps in 3.0 GPU-h).
   - In `common.py`: add scalar edge gate `sigmoid(gate(e)) * e` + `1 / sqrt(deg)` normalization to `InteractionGNN`, plus on-the-fly same-layer partner distance node/edge features.
   - **Expected result**: **`0.68–0.75` loose / `0.28–0.38` tight** (with Step 0 builder).
   - **Stop condition**: If val coupled loose after decay is `< 0.60`, do not run a second seed; pivot to graph filter + local refiner.
2. **Run G2 (~3.0 GPU-h) — Retrain $\mu = 60$ Edge Filter with Fixed Asymmetric Weights + Rebuild Graph**:
   - Fix `train_filter_stream.py` so `W_QUIRK_POS = 100.0` (`y & (hq[s] | hq[d])`) and `W_QUIRK_NEG = 2.0` (`~y & (hq[s] | hq[d])`), train 6k steps with full WSD decay (~1.2 GPU-h), and apply to val/test + 16k train blocks (~1.8 GPU-h) to raise quirk edge recall at 1.4M edges from `0.849` to `~0.93+`.
   - **Expected result**: **+4 to +7 pts loose / +6 to +10 pts tight**.
3. **Run G3 (~2.0 GPU-h) — Local Relational Transformer Cleaner/Splitter (`splitter2.py` on GNN candidate subgraphs)**:
   - Extract candidate quirk components + 1-hop graph neighbors (~100–200 hits/event) on 50k train events; train `splitter2.py` (`neither` head = pileup removal, affinity head = A/B split).
   - **Expected result**: Pushes **coupled tight to `0.40–0.50`** and **per-quirk loose from `0.17` to `0.50–0.60`**.

### Step 3: MaskFormer Single Combined Shot (~4.5 to 9.0 GPU-h, gated at 16k steps)
- Launch **one** combined run on `filter_flex_v3` cut `0.005` (46.8k hits, 99.9% pair $>50\%$ ceiling, ~1.8 steps/s) or cut `0.01` (37.2k hits, 99.3% ceiling, ~2.05 steps/s):
  - `flex: {dphi: 0.5, deta: 1.0}` + `object_weight_alpha: 25.0, object_cost_weight: 20.0` + `direction.yaml` (`direction_dir` for $\mu = 60$, 26 no-origin + direction + `volume_id` fields) + capped RoPE frequencies ($\omega_\phi \le 60, \omega_\eta \le 30$, no additive `PositionEncoder`) + positive-hit weight $\alpha_{\text{pos}} = 10$ in `mask_focal_loss`.
- **Hard Stop Gate at 16k steps (~2.5 GPU-h)**:
  - Check a short 16k$\to$20k decay branch (~0.6 GPU-h). If val coupled loose is **`< 0.35`**, **stop MaskFormer at $\mu = 60$** (total spent ~3.1 GPU-h) and put the remaining ~15 GPU-h into **Step 2 (G1 second seed, G2 full 50k graph rebuild, and G3 local Transformer refiner/splitter)**.
  - If **`>= 0.35`** at 20k steps, continue trunk to 48k + 12k decay (~6 GPU-h more, expected **`0.55–0.70` loose**).
