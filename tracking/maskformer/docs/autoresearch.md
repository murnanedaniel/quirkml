# LRT auto-research loop — protocol

The agentic experiment loop for the ColliderML LRT campaign. Pattern follows the
convergent best practice of AIDE / Karpathy-autoresearch / Deep-Researcher-Agent:
frozen data pipeline, mutable configs, natural-language directives, zero-LLM-cost
monitoring, a registry as the search tree, and noise-aware keep/revert rules.

## Fixed evaluation protocol (never changes between experiments)

- **Test set:** higgs_portal v1 chunks **796–815** (20k events), disjoint from all
  training by construction (`test_chunks` guard).
- **Scorer:** `tracking_core.lrt_report` (majority-from-hits, fiducial selection:
  primary charged, pT>1, |η|<2.5, ≥4 layers). Never compare numbers produced by
  different scorers or selections.
- **Headline metrics** (in priority order):
  1. `eff_displaced` = **r≥10mm** fiducial efficiency (`displaced_r_ge_10_mm`).
     2–10mm is "intermediate" (ACTS ~96% there — not genuinely large-radius);
     r≥2mm retained in reports for continuity only.
  2. per-bin: 30–50, 50–75, **75–100**, 100–150, 150–200 mm
  3. `eff_prompt` (must not regress > 1σ below the best-so-far)
  4. fake rate, duplicate rate (report always; gate if > 2× ACTS×10⁴)
- **Reference:** ACTS baseline on the same chunks: prompt 95.54%, displaced(r≥10mm) **37.20%** (r≥2mm: 73.05%)
  (`outputs/lrt_baseline/acts_testset_796_815/report/summary.json`).

## Decision rules (noise-aware)

- **KEEP** a change iff eff_displaced (r≥10mm) improves by > max(2× Wilson σ, 0.3pp
  absolute) on the fixed protocol AND eff_prompt does not regress > 1pp.
- **REVERT** otherwise; near-ties (within noise) → prefer the simpler config.
- **In-training plateau:** if the best val/eff_r30_75mm + val/eff_r75plus
  (2000-step val cadence) improves < 0.5pp absolute over the last full 4h window,
  stop the chain early and bank the checkpoint (don't burn windows on a plateau).
- **Divergence:** val/loss rising for 3 consecutive vals, or NaN → kill window,
  mark run failed, auto-revert to previous config.
- One mutation per experiment. No simultaneous axis changes (exception:
  documented interactions, e.g. LR × batch-accumulation).

## Loop mechanics (zero-LLM-cost during training)

1. **Think** (agent, at window/run boundaries): read `lrt_runs/EXPERIMENTS.md`
   registry + latest metrics; pick the top hypothesis from the queue in
   `lrt_runs/program.md`; write the registry entry (hypothesis, config diff,
   budget) BEFORE launching.
2. **Execute:** config-only diffs where possible (stacked yaml overlays); chained
   4h `shared_interactive` windows (`chain_driver`), ckpts to CFS.
3. **Monitor:** `scripts/check_run_health.py` parses metrics.csv (slope, plateau,
   divergence verdicts, JSON out); the chain driver calls it between windows and
   stops on `plateau`/`diverged`; the session Monitor watches step progression +
   failure signatures. No LLM calls while training runs.
4. **Evaluate:** on chain end, turnkey predict+score on the fixed protocol
   (`lrt_runs/run_m4_eval.sh` pattern; `--chunk-range 796 815`).
5. **Decide & record:** apply the decision rules, update the registry row
   (metrics, KEEP/REVERT, what next), update the overlay plot, push the branch.

## Budget caps

- Default experiment budget: **3 × 4h windows** (≈130k steps tracker-only).
  Extending requires an explicit registry note (why more compute changes the answer).
- Two concurrent chains max (the two shared_interactive slots), ideally one
  "exploit" (best-known config, longer) + one "explore" (queue head, short).

## Registry & directives

- `lrt_runs/EXPERIMENTS.md` — the experiment ledger (search tree). One row per
  run; append-only; decisions recorded with numbers, not vibes.
- `lrt_runs/program.md` — research directives + prioritized mutation queue;
  the agent edits priorities with rationale as evidence accumulates.
