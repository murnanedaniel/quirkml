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

## Stage 2: quirk-gen to ODD simulation, end to end

1. Run quirk-gen for one mass (it needs LCG_109; check that its container route works on Perlmutter).
2. Simulate it with QuirkDDG4 on ODD using the quick start commands. Check the PDG ID, the charge sign and the event count. Also check the duplicate-event issue from the pilot, where single-thread ddsim wrote each event twice.
3. Add vertex smearing. quirk-gen puts the vertex at the origin; ColliderML smears it with 12.5 um in xy, 55.5 mm in z and 0.185 ns in t.
4. Rebuild QuirkDDG4 inside the current ColliderML sim image (sw:pr-8, ODD v6). The current build is for LCG_107 and ODD v4.

Check: one command per step in the README, run by someone who didn't write it.

## Stage 3: ColliderML chain for quirks

1. Test the `user_physics` setting inside the sim image on one quirk-gen file. Then open it as a PR to colliderml-production, once Daniel gives the OK.
2. Pileup: use `pythia_generation` with the quirk file as the hard scatter and `merge: true`. The `merge_smear` pileup path has a known bug (an undefined variable), so don't use it.
3. Fix the user-facing rough edges a newcomer hits first: `ddsim_run.py` exits with code 0 on failure; check that `run_stage.py` interactive mode works for a new user without Daniel's git and Python setup.
4. Write `docs/colliderml.md`, a step-by-step guide from HepMC3 to parquet using only commands we have run. Include which tables are useful for quirks. The ACTS track fits assume a pion helix, so for quirks the useful tables are `tracker_simhits` and the truth links in `tracker_hits`.
5. Pilot one grid point at mu = 0 and mu = 200 interactively, and check the particles table (the quirks, their status, and their truth links to hits).

Check: a PhD student follows `docs/colliderml.md` on a fresh account and gets parquet.

## Stage 4: grid definition and production plan

1. Agree the grid with Makayla and Jackson. Proposal: masses 200-1500 GeV; sqrt(sigma) across the slack (0.1-30 keV), unresolved (30 keV-5 MeV) and tight (MeV-GeV) regimes; mu = 0, 60, 140, 200.
2. Test QuirkDDG4 above 4 keV. It's untested there, and the step cap cost grows as the string gets shorter.
3. Measure cost per event per point, then the storage estimate. Daniel decides on batch production.

## Stage 5: ML and non-ML starting points

1. Loader from ColliderML parquet to per-event arrays, with truth edges ordered by hit time. Hit order by radius fails for quirks, which turn back inwards. Makayla's acorn fork has this logic for ATLAS input; port it onto acorn's ColliderML reader.
2. Writer for Jackson's `quirk-tracking` per-event parquet format: remap the volume IDs and drop the cluster features ColliderML doesn't have.
3. Non-ML baselines: the coupled string fit (exists), two-helix and 1/r^2 fits (exist), road search in pileup (exists), and the coplanar-hit and ionisation selections (to write).
4. One worked example notebook from parquet to a baseline result.

## Stage 6: handover

1. README quick start rerun from scratch on a clean account.
2. Docs reviewed by Daniel; nothing ATLAS-internal; licences checked for anything taken from quirk-gen, quirk-tracking or quirkviz (none of them has a licence).
3. Push to a remote Daniel chooses, and walk Jackson's student through it.

## Open questions for Daniel / Jackson / Makayla

- Who fixes the Geant4 boundary problem, and should it go back to Athena?
- Remote location for the repo (GitHub org?), and when.
- Whether Qiyu Sha's 2024 quirk data can be shared, or if we regenerate it.
