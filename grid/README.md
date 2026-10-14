# Grid

- `grid.yaml`: proposed mass x Lambda x pileup grid. To be agreed; nothing produced.
- Configs for a grid point: one settings file per point, through `colliderml/make_sample.py` (`docs/tutorial.md`).
- `run_tension_cost.sh`, `cost_vs_tension.txt`: QuirkDDG4 run time against string tension.

## QuirkDDG4 cost against tension

One pair-only event (m = 400 GeV, the 1 keV validation event's momenta), full ODD v4, 3 T, quirk EM on, tracking stopped at r = 1150 mm or |z| = 3000 mm. Wall time for one event on one core, including about 45 s of start-up; 20-minute limit.

| Lambda | MaxBoost 1e-2 | MaxBoost 1e-3 |
|---|---|---|
| 3 keV | 51 s | 65 s |
| 10 keV | 69 s | 158 s |
| 30 keV | 186 s | 1199 s |
| 100 keV | not finished in 20 min | not finished in 20 min |
| 1 MeV | not finished in 20 min | not finished in 20 min |

So QuirkDDG4 in its current form covers the slack regime up to about 10-30 keV. For larger tensions, the quirks' largest separation in the pair rest frame shrinks as 1/sigma (9.5 mm at 1 keV, 0.011 mm at 30 keV and 1e-5 mm at 1 MeV for this event); we expect the boost-limited steps to shrink with it, which fits the run times but is not measured step by step. The unresolved and tight regimes (30 keV-GeV) need a different treatment before they can be produced. One possibility is to move the bound pair as one object once its size is below the detector resolution. That's not decided or implemented.

At Lambda = 1 keV, one pair-only event with MaxBoost 1e-3 took about 40 s including start-up (sw:pr-8, `docs/colliderml.md` test). Full quirk-gen events at mu = 0 took about 7 s each and mu = 200 events 2.6 min each (pilot, Lambda = 1 keV, one core).
