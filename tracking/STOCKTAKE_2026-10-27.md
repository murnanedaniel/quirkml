# Quirk tracking: stock-take at the pause (2026-10-27)

Project paused 2026-10-27 at Daniel's request; resume planned 2026-10-30. Numbers: m = 200 GeV, Lambda = 774 eV,
mu = 0, test events 17000-19999, ACTS, strict matching (loose: purity and completeness both > 50%; tight: > 90%).
Details and row ids: EXPERIMENTS.md; longer list of findings: STATUS_2026-10-27.md; slides: the 26 Oct deck.

## Where we are

| Task | Best method | Loose / tight | Fake rate | Best case (perfect edge scores) |
|---|---|---|---|---|
| Both quirks found as one track (coupled) | pair GNN, connected components (3 seeds) | 0.944 / 0.688 | 0.079 | 0.998 / 0.974 |
| Each quirk on its own track (per-quirk) | pair GNN -> quirk tagger -> splitter (971k training pairs) | 0.828 / 0.505 | 0.082 | splitter on true pairs: 0.986 / 0.865 |
| Coupled, MaskFormer | pair target, 100k steps | 0.907 / 0.553 | 0.065 | |

At matched fake rate 0.02: pair GNN 0.935, MaskFormer 0.901-0.906 (coupled, loose). Background rejection at coupled
efficiency 0.80: pair GNN 718 +- 75, MaskFormer long 712, pair GNN on graph v3 2150 (but that cut keeps only 42% of
ordinary tracks).

## What we know

1. Finding the pair works at the training point; splitting it is the hard part, and it is not limited by hit
   ambiguity (quirk hits on a layer are usually mm apart).
2. The splitter is strongly data-limited (16k -> 971k training pairs: tight 0.16 -> 0.86 on true pairs). On
   reconstructed tracks the remaining loss is in finding and tagging the pair, not in the split.
3. Track building matters as much as the network (same scores: tight 0.118 with walk + stitch, 0.688 with
   connected components).
4. Fakes are 96-98% ordinary low-pT fragments; a quality cut that protects quirk tracks removes most of them for
   ~1 point of efficiency.
5. Across the grid (no retraining), Lambda decides almost everything: per-quirk works at low Lambda, coupled from
   ~774 eV up. MaskFormer finds about half as many pairs as the pair GNN at 167 eV and up to 1.09 times as many from
   2154 eV.
6. Quirk parameters: with perfect hit labels one event measures Lambda^2/m (sigma/m) to ~1% at 167 eV and 7-11% at
   774 eV, but not m and Lambda separately; m, p and sigma scale together and only the magnetic field breaks this.
   Our fast quirk propagator drifts from Geant4 by 1-3 mm after ~10 ns, more than that field effect.

## What we got wrong (and fixed or recorded)

1. >= 50% matching (the ACTS default) credited exact 50/50 tracks; per-quirk numbers were 6-10 points too high.
   Strict matching is now the standard.
2. Relative checkpoint paths in the MaskFormer grid loop: 45.4 GPU-h of failed retries. Loops now check paths and
   stop after repeated failures.
3. 1-CPU job steps (srun --exact without -c) slowed CPU scoring 30x.
4. The data-scaling stopping rule (stop when validation efficiency stops improving) ended every GNN and MaskFormer
   run at 30-40k steps, before the learning rate decayed. Both scaling studies are therefore inconclusive (the
   MaskFormer 150k arm was ~5 points better than 16k/50k; the GNN arms were flat, which reflects noise at high
   learning rate, not data size).
5. Several wrong numbers in messages and slides were caught and corrected (GPU-hour totals, fake comparisons that
   mixed cut and no-cut, unlike-for-like grid fakes).

## Open questions

1. Are the GNNs and MaskFormer data-limited like the splitter? (Needs the scaling redo.)
2. Does the full splitting chain hold across the grid, and with training on more grid points?
3. Anything at mu = 60.
4. Which efficiency an analysis needs: per-quirk, coupled, or both quirks per event. The parameter study suggests the
   pair-level quantity Lambda^2/m is what one event measures. (Decision for Daniel.)
5. Several headline numbers have one training only (MaskFormer long run, pair GNN on graph v3, 971k splitter).

## Plan on resume (agreed 2026-10-27)

1. Redo data scaling: pair GNN, per-quirk GNN, MaskFormer pair at 16k / 50k / 150k, each with a fixed schedule
   ending in a full learning-rate decay, 2-3 seeds (~60 GPU-h).
2. Run pair GNN -> quirk tagger -> splitter on the 110 grid test sets without retraining.
3. Train on several grid points, splitter first.

## Resources used

GPU (m5386): about 241 of ~400 GPU-h (217 to 26 Oct, plus MaskFormer scaling 9.3 and GNN scaling 17.3, less 2.6 counted twice).
CPU (m4958): simulation ~454 node-h (grid production), plus logged analysis ~80 node-h (several studies did not log
CPU time).
