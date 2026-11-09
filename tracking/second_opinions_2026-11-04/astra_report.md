**I would prioritise supervision and training-context fixes over larger models or more data.** The GNN evidence identifies a real classification problem; the MaskFormer evidence does not yet distinguish weak pair supervision, attention behaviour, and output assignment.

I read the ledgers and relevant code without running jobs or changing files. Numerical gain estimates below are guesses, not forecasts.

**1. Why pileup hurts, and what to check first**

The pair contributes roughly 26/67,000 hits—0.04%—while retaining its unusual geometry. Pileup adds many locally plausible alternatives without adding pair information. Tight matching is particularly unforgiving: a 26-hit pair needs at least 24 recovered hits, and even with all 26 recovered, three contaminants fail purity >90%.

- **GNN: diluted supervision and crowded message passing are the leading explanations.** Quirk-edge ranking deteriorates markedly, and 7–8 false quirk-touching edges/event survive at cut 0.6. The network uses eight iterations of **sum aggregation**, making degree and neighbourhood composition relevant. Training on cropped graphs and evaluating full graphs could amplify that problem, although cropping mainly changes boundary neighbourhoods—not every node’s degree. See [aggregation code](/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/gnn/common.py:184).
- **MaskFormer: the rare pair competes with approximately 180 ordinary targets and overwhelming mask background.** Ordinary efficiency improving while pair efficiency remains low supports task imbalance or an unsuitable representation more than a general optimisation failure.
- **The attention-window argument is suggestive, not decisive.** Ten encoder layers allow multihop communication; decoder queries have spatially unrestricted access gated by predicted masks. “Only 7% fit in one window” is therefore not a reconstruction ceiling. Early predicted masks excluding quirk hits are another plausible bottleneck—the gating mechanism is explicit in [the decoder](/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/maskformer/packages/ml_tracking/src/ml_tracking/models/decoder.py:254) and the [Mask2Former design](https://bowenc0221.github.io/mask2former/).

My first diagnostics would be:

| Diagnostic | What it distinguishes |
|---|---|
| For each failed pair: best-track completeness, purity, unassigned-hit fraction, fragmentation; bin by occupancy, angular span, hit count and strip fraction | Missing hits versus contamination versus fragmentation; locality versus density |
| GNN true-pair recall **at fixed false quirk-touching edges/event**, plus per-event connectivity | Better discrimination versus merely shifting scores; more relevant than global AUC |
| Training crops: retained pair-hit fraction, pair connectivity, positive/negative quirk-edge loss shares and boundary distance | Whether crops truncate the learning target or negative quirk edges dominate |
| MaskFormer best independent query mask versus final exclusive hit assignment | Representation failure versus query competition/output processing |
| MaskFormer matched-pair object score, mask recall/purity and attention coverage by decoder layer | Object rejection versus mask failure versus early attention exclusion |

The first three can largely use saved predictions, graphs and truth. **The writer saves assignments and winning scores, not all query logits**, so the last two may require a small validation inference pass; existing prediction parquet alone cannot answer them.

**2. Changes ranked by likely gain per GPU-hour**

For the **pair GNN**:

1. **Read the weight-40 result before adding another arm.** Crucially, `quirk-weight` multiplies *both positive and negative* quirk-touching edges in [the loss construction](/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/gnn/train_gnn.py:60). Restoring their combined loss share does not necessarily restore the positive learning signal. If needed, separately normalise true-pair, wrong-qu irk-touching, and ordinary-edge losses, preserving hard negatives.
2. **Fix crop coverage if the audit implicates it.** Use pair-containing training crops with surrounding context and jitter, retaining random background crops. Truth-centred training sampling is legitimate; validation must remain full-event. Merely increasing quirk-centred crop frequency does not guarantee the whole pair survives.
3. **Try degree-normalised aggregation only if failures track degree/context.** Compare mean or gated aggregation with an explicit degree feature. This is a better targeted hypothesis than more depth; eight crowded propagation rounds already provide substantial mixing.
4. **Keep track-building work small.** At cut 0.6 there is little loose-efficiency headroom from cleaning existing components. Tight has meaningful headroom: pure-edge connectivity ≈0.19 versus actual ≈0.11. Conservative component cleaning or supported fragment merging could help, but indiscriminately lowering the cut already performed poorly.

For **MaskFormer**:

1. **Prioritise pair weighting, then combine it with whichever of wide attention or warm start actually helps.** Wide attention costs only about 18% more per step in the ledger, making it a reasonable existing experiment.
2. **If decoder diagnostics show exclusion, test unmasked early decoder layers or periodically unmasked attention.** This addresses access directly without replacing the whole encoder.
3. **Verify and potentially restore direction features.** The checked-in [mu60 configuration](/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/maskformer/packages/ml_tracking/src/ml_tracking/experiments/quirk/configs/maskformer_mu60.yaml) includes cluster shape but not the direction inputs of the strong mu0 reference. Check resolved configurations before interpreting this as a missing feature.
4. **Prefer a few wider encoder layers over a new multiscale architecture** if the wide-window arm helps but throughput matters. More width/depth has weaker supporting evidence.

Across both approaches:

- **Timing:** potentially informative, but the stored time is explicitly simulation truth in [split_data.py](/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/split/split_data.py:9). Treat perfect timing as an oracle study; any deployable claim needs a measurement/resolution model.
- **Cluster information:** already present in the GNN; charge is reported non-discriminating under this digitisation. Do not expect a new dE/dx benefit from the existing `sum_activation`.
- **Pair-specific features:** candidate-relative geometry, layerwise two-hit structure and direction consistency are plausible additions to a small refinement model. Avoid hard helical or outward-order assumptions.
- **Two-stage:** promising longer term as permissive fragment proposals followed by pair assembly and hit recovery. Measure proposal coverage first. Requiring an already matched pair as the seed would cap recovery near current efficiency.
- **Curriculum/pretraining/data:** finish the existing warm start; defer a new curriculum or pretraining programme. The 50k comparison establishes no gain at 30k updates, not absence of data limitation. More steps remain plausible, but the loose/tight tradeoff argues against buying them blindly.

**3. What looks suspicious**

- **“Graph/building are not limiting” is too broad.** The graph’s tight connectivity ceiling is 0.716; retained-edge tight connectivity exceeds achieved tight efficiency substantially. Also, pure-pair connectivity is an oracle diagnostic, not a mathematical ceiling for every builder that could use background-mediated paths.
- **Warm-start compatibility needs checking.** Loading equal-shaped tensors does not guarantee compatible feature meanings. The mu0 direction input has 25 fields versus 12 in the mu60 config; the loader can leave incompatible projections fresh. Inspect its copy log before interpreting a negative warm-start result.
- **Evaluation looks substantially checked**, including Python–ACTS agreement. Still use strict, final-assignment metrics: some training diagnostics retain `>=` matching and independent masks.
- **Splits are sensible by event ID**, but verify signal-generation seeds/hard events are not reused across mu0 pretraining and mu60 validation/test. I found no evidence establishing either overlap or independence.
- **Freeze test use now.** Repeated test inspection risks turning it into development data. Compare arms on the same 1,000 validation events with paired event bootstrap.
- **Do not transfer the review thread’s gains numerically.** Its target selection differs. Likewise, GNN and MaskFormer fake fractions are not directly comparable when they reconstruct very different ordinary-track populations; report fake tracks/event alongside fake fraction.

**4. Allocation of the remaining 28 GPU-h**

Assuming the running arms are already funded:

| Budget | Action | Continue/stop rule |
|---|---|---|
| GNN: 3 h | One diagnostic-selected arm: separated loss balancing, crop correction, **or** aggregation—not all three | Require ≳3 percentage points loose gain at comparable fake burden, or a useful tight gain |
| GNN: 3 h | Second seed of the winner | Stop if the gain disappears or is only score recalibration |
| GNN: 2 h | Validation inference, short decay branches, final selected-model inference | Reserve rather than opening another architecture search |
| MaskFormer: 8 h | Continue the best existing arm/justified combination, with completed decay branches | Require ≳5 points over the matched-budget baseline |
| MaskFormer: 6 h | One diagnosis-driven intervention: decoder access **or** direction inputs | Stop after a completed short schedule if pair metrics remain flat |
| MaskFormer: 4 h | Independent-seed confirmation at a useful short budget | Require consistent direction of gain |
| MaskFormer: 2 h | Diagnostic/final inference | Preserve enough budget to evaluate properly |

My speculative upside is **+3–10 absolute points coupled loose for the GNN**, and **+5–15 for MaskFormer** if weighting/access is the main problem; zero gain is entirely plausible. These are not additive experiment promises. I would not budget on recovering mu0 performance, and I would stop spending on an approach when completed decay branches show no meaningful paired-validation improvement at comparable background burden.