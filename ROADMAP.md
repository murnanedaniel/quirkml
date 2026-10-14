# Roadmap: quirkml handover sprint

Goal: hand Jackson Burzynski and his PhD student a repo and docs they can run without us. That means generating a quirk sample with quirk-gen, simulating it on ODD, putting it through ColliderML (pileup, digitisation, parquet), and training or running a baseline on it. In parallel, Makayla Vessella, Jackson and Daniel agree the mass x tension grid and the plan for the comparison paper.

Each stage ends with a check that a new person can repeat. Nothing gets pushed to a remote or shared outside Perlmutter without Daniel's OK.

## Stage 0: done (2026-10-14)

- Local repo with clean copies of QuirkDDG4, quirk_sim, the pilot generator, the fit code and the validation scripts.
- Inventory of what exists (`quirk_erc/starter_repo_plan/INVENTORY.md`).
- Field-off exact-motion test. Geant4 is accurate without detector volumes and drifts with them (`docs/validation_status.md`).
- ColliderML: optional `user_physics` setting for `ddsim_run.py`, on local branch `feat/user-physics-plugin` of colliderml-production. Mock-tested only.

## Stage 1: understand the Geant4-vs-quirk_sim gap - done (2026-10-14)

- Not a boundary bug. The 25 mm gap at 1 keV is mostly quirk ionisation in the silicon (`--no-em` keeps it). Without ionisation, Geant4 follows the exact field-free motion to 0.016 mm.
- With the field on, the two string models differ by 2-4 mm; both codes are converged in step size.
- Geant4's default `MaxBoost` 1e-2 is off by 1.3 mm at 0.5 keV; 1e-3 is converged to 0.1 mm.
- Open: which string model is closer to Nambu-Goto with a field. One way to settle it is a full Nambu-Goto string simulation for one event. Worth raising with Jackson.
- Decision for Daniel: production `MaxBoost` (1e-2 vs 1e-3; 1e-3 takes about 10 times more steps).

## Stage 2: quirk-gen to ODD simulation - done (2026-10-14)

- quirk-gen runs unmodified in the el9 shifter image with LCG_109, from $PSCRATCH (MadGraph fails on CFS). m = 400 GeV, 100 events: 1.6 min on 8 cores.
- Its HepMC3 3.03.01 files can't be read by the sim image's HepMC3 3.3.0; `colliderml/convert_quirkgen_hepmc.sh` converts them to HepMC2 and renumbers events from 0.
- QuirkDDG4 builds in the ColliderML sim image (`sim/QuirkDDG4/build_colliderml.sh`).
- The pilot generator's duplicate-event bug is a ddsim single-thread bug, now fixed (stage 3).
- Open: tell Jackson about the HepMC version mismatch; writing HepMC2 directly in quirk-gen, or pinning a HepMC3 version, would remove the conversion step.

## Stage 3: ColliderML chain for quirks - pilot done (2026-10-14)

- Pilots at m = 400 GeV, Lambda = 1 keV: mu = 0 (20 events) and mu = 200 (5 events), from quirk-gen output to packaged parquet, with every stage run through `run_stage.py`. Simulation took 2.5 min for 20 events at mu = 0 and 13 min for 5 events at mu = 200 (one core).
- ColliderML changes, each on its own local branch of colliderml-production (`docs/colliderml.md` has the table). Nothing has been pushed; they need Daniel's review before a PR.
- Guide: `docs/colliderml.md`, written from the commands that ran.
- Open:
  1. `run_stage` writes config copies and validation reports under `common.output_base_dir`, which defaults to the public production tree. The pilot configs now set it to the pilot tree. A first run before that fix left a small `quirk_pilot/` folder in the production tree (Daniel to decide).
  2. The packaging validation needs track files of at least 0.1 MB, which small pilots don't reach. This is harmless, but it reports a failure.
  3. Have someone new follow `docs/tutorial.md` on their own account.

## Stage 4: grid definition and production plan - partly done

- Grid proposal in `grid/grid.yaml`; `colliderml/make_sample.py` writes the stage configs for any point from one settings file (stage 7).
- QuirkDDG4 cost versus tension (`grid/README.md`): about 1-3 min per pair-only event up to 10-30 keV; at 100 keV and 1 MeV one event doesn't finish in 20 minutes. The unresolved and tight regimes need a different simulation treatment.
- Open: agree the grid; how to simulate Lambda above ~30 keV (Daniel, Jackson); production `MaxBoost`; storage and batch plan (Daniel).

## Stage 5: ML and non-ML starting points - partly done

- `ml/quirk_loader.py` (time-ordered truth) and `ml/to_quirk_tracking.py` run on the pilot parquet; `docs/ml_baselines.md` lists the status of each starting point.
- Open: load the written files with quirk-tracking itself; ACORN quirk config on the loader; adapter from parquet to the string fit (its kernel uses ODD v4 surfaces); coplanar and dE/dx baselines; one worked example notebook.

## Stage 6: handover

1. `docs/tutorial.md` rerun by someone else on their own account.
2. Docs reviewed by Daniel; nothing ATLAS-internal; licences checked for anything taken from quirk-gen, quirk-tracking or quirkviz (none of them has a licence).
3. Give Jackson and his student access to github.com/murnanedaniel/quirkml (private) and walk the student through the tutorial.

## Stage 7: anyone on m4958 can make a sample - done (2026-10-14)

A new member of the m4958 project goes from nothing to tracker-only quirk parquet with `docs/tutorial.md`: four commands and one settings file. Before this stage, the route worked only from Daniel's account.

1. ColliderML changes on GitHub: five branches of OpenDataDetector/colliderml-production (no PRs), merged in `quirk/pilot-integration` at `6495666`. One was new: the container mounted Daniel's `$PSCRATCH` by name (`fix/pscratch-bind-path`).
2. One settings file, `colliderml/sample.yaml`; `colliderml/make_sample.py` writes the stage configs from templates. The generated configs differ from the pilot's only in paths, seeds and the tension rounding (5067.73 against quirk-gen's 5068.0).
3. Scripts: `colliderml/setup.sh` (colliderml-production at a fixed commit and its `env_setup.yaml`, a python env, the sim image, the QuirkDDG4 build), `gen/run_quirkgen.sh` (quirk-gen at a fixed commit on $PSCRATCH, then the HepMC conversion), `colliderml/run_sample.sh` (the four stages, each output checked before the next).
4. Tested twice from fresh GitHub clones: mu = 0 (20 events, 4.4 min) and mu = 200 (5 events, 16.5 min), then the tutorial followed command by command for mu = 0 in a second new directory. 2 quirks and 25-33 quirk hits per event; nothing written to the production tree.

Not tested: a different user account (their podman image pull, group read access to the shared Geant4 data). That needs Jackson's student or another m4958 member. People outside m4958 would need the containers' mounts generalised.

## Open questions for Daniel / Jackson / Makayla

- Who fixes the Geant4 boundary problem, and should it go back to Athena?
- Whether Qiyu Sha's 2024 quirk data can be shared, or if we regenerate it.
