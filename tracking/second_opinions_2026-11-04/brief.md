# Second opinion request: boosting quirk tracking efficiency at pileup 60

You are asked for an independent, critical second opinion. Read-only: do not run jobs or modify files. You may read
code and ledgers under /global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml (tracking/EXPERIMENTS.md section
"Pileup 60", tracking/STOCKTAKE_2026-10-27.md, tracking/REVIEW_SYNC_2026-11-03.md, tracking/gnn/, tracking/pileup60/,
tracking/maskformer/, tracking/split/).

## Physics and task
Quirks: heavy charged particles pair-produced and joined by a confining string (scale Lambda). The two quirks
oscillate around each other while flying outward, leaving two interleaved, non-helical tracks. Detector: Open Data
Detector (ODD) silicon tracker, ColliderML simulation (Geant4), 2 T field. Benchmark point m = 200 GeV,
Lambda = 774 eV. Two tasks: "coupled" (find both quirks together as one track) and "per-quirk" (each quirk its own
track). Matching is strict: a track matches a particle if purity AND completeness > 50% ("loose") or > 90% ("tight").
Splits at pileup 60: test events 0-1999, val 2000-2999, train 3000-99999 (100k simulated events available).

## Pileup 0 results (16k training events)
- Pair GNN (metric learning -> edge filter -> interaction GNN, every edge between two hits of the pair labelled true,
  tracks by connected components): coupled 0.944 loose / 0.688 tight, fake rate 0.079.
- MaskFormer (pair as one target, full attention, 100k steps): coupled 0.907 / 0.553.
- Per-quirk: pair GNN -> quirk tagger -> learned splitter (trained on 971k true pairs): 0.828 / 0.505.

## Pileup 60 (mu = 60): what happened so far
- Events have ~67k tracker hits (vs ~4k at mu 0); the quirk pair has a median 26 hits. Only 8.5% of hits belong to
  particles with pT > 1 GeV.
- Per-hit filters (a hit-level GNN on k-nearest neighbours; a FlexGeo window-attention filter) keep pT > 1 GeV hits
  easily but NOT quirk hits: quirk hits look like pileup locally. FlexGeo filter: 99% quirk-hit recall needs ~50k of
  66k hits; at 37k hits quirk recall 0.967. Decision: no hit filter for MaskFormer (Daniel: no reason to lose a quirk).
- Zero-shot (mu 0 models on mu 60): pair GNN coupled 0.236 / 0.036; MaskFormer ~0.
- GNN at mu 60: graph = mu 0 graph-v3 embedding + mu 0 edge filter at a tighter cut (1.4M edges/event; contains a
  path joining > 50% of the pair's hits in 94% of events). Retraining the edge filter at mu 60 gave no gain within
  2000 steps (stopped before LR decay). Pair GNN trained at mu 60 on 400k-edge phi crops (half centred on a quirk hit),
  AdamW 5e-4, warmup, constant, linear decay over the last 20%, quirk edge weight 10, 30k steps:
  coupled 0.520 / 0.112 (test, 2 seeds), fake 0.317. Edge level: AUC (true pair edges vs wrong edges touching a
  quirk hit) 0.953 vs 0.998 at mu 0; pair-edge efficiency at score 0.5 is 0.42 vs 0.84; built tracks equal the
  GNN-edge ceiling (52% of pairs joinable by kept edges). So the GNN edge discrimination is the limit, not the graph.
  50k training events at the same 30k steps: no gain (0.529 / 0.099). 50k events with 60k steps: val 0.576 / 0.086
  (decaying). Per-quirk GNN at mu 60: 0.166 / 0.004 (no better than zero-shot). A track-quality cut gives coupled
  0.464 at fake 0.02 (mu 0: 0.935).
- MaskFormer at mu 60 (FlexGeo windowed attention |dphi| < 0.15, |deta| < 0.30 with RoPE, all hits, 256 queries,
  targets = pT > 1 GeV particles + the quirk pair as one object, Lion 5e-5, 1 event/update, 8 decoder layers,
  published loss weights, decay branches from a constant-LR trunk): coupled (val) 0.050 at 10k steps, 0.133 at 20k,
  0.151 at 30k; tight ~0.01. Ordinary-track efficiency ~0.72. The pair's hits span a median 0.47 in phi and 1.04 in
  eta; only 7% of pairs fit in one attention window. Pair target verified present and matched in the loss.
- Running now: MaskFormer arms (a) wide window 0.5 x 1.0, (c) pair object up-weighted x10 in loss and matching cost,
  (d) warm start from the mu 0 pair MaskFormer (encoder + 4 decoder layers + heads copied); GNN arm with quirk edge
  weight 40 (restores the mu 0 quirk share of the loss in the training crops).
- Budget: ~240 GPU-h used of ~400 overall; ~20 GPU-h left for MaskFormer and ~8 for the GNN after these arms.

## Questions
1. What are the most likely reasons the pair is so much harder at mu 60 for both GNN and MaskFormer, given the
   numbers above? What would you check first to tell them apart (cheap diagnostics on existing outputs)?
2. Which changes would you expect to give the largest efficiency gain per GPU-hour, for (a) the pair GNN, (b)
   MaskFormer? Consider: loss weighting, sampling/cropping, features (hit time, cluster shape/direction, pair-specific
   features), architecture (attention reach, GNN depth/width, multi-scale), two-stage approaches (find quirk-like
   regions or seeds first, then track), curriculum, pretraining, more data or steps, track-building changes.
3. Anything in the setup that looks wrong or suspicious (e.g. evaluation, splits, targets, schedules)?
4. A prioritised plan for the remaining ~28 GPU-h, with expected gains and what result would make you stop.
Answer concisely and concretely; size claims to the evidence; say where you are guessing.
