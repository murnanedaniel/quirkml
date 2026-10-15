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

## Stage 7: anyone on m4958 can make a sample - done (2026-10-14), replaced by stage 8

A new member of the m4958 project goes from nothing to tracker-only quirk parquet with `docs/tutorial.md`: four commands and one settings file. Before this stage, the route worked only from Daniel's account.

1. ColliderML changes on GitHub: five branches of OpenDataDetector/colliderml-production (no PRs), merged in `quirk/pilot-integration` at `6495666`. One was new: the container mounted Daniel's `$PSCRATCH` by name (`fix/pscratch-bind-path`).
2. One settings file, `colliderml/sample.yaml`; `colliderml/make_sample.py` writes the stage configs from templates. The generated configs differ from the pilot's only in paths, seeds and the tension rounding (5067.73 against quirk-gen's 5068.0).
3. Scripts: `colliderml/setup.sh` (colliderml-production at a fixed commit and its `env_setup.yaml`, a python env, the sim image, the QuirkDDG4 build), `gen/run_quirkgen.sh` (quirk-gen at a fixed commit on $PSCRATCH, then the HepMC conversion), `colliderml/run_sample.sh` (the four stages, each output checked before the next).
4. Tested twice from fresh GitHub clones: mu = 0 (20 events, 4.4 min) and mu = 200 (5 events, 16.5 min), then the tutorial followed command by command for mu = 0 in a second new directory. 2 quirks and 25-33 quirk hits per event; nothing written to the production tree.

Not tested: a different user account (their podman image pull, group read access to the shared Geant4 data). That needs Jackson's student or another m4958 member. People outside m4958 would need the containers' mounts generalised.

## Stage 8: anyone with a container runtime can make a sample - done except Docker (2026-10-14)

Goal: the tutorial works for people outside m4958 and outside NERSC (Jackson's group), on any Linux machine with Docker, Podman or Apptainer. Stage 7 drove the stages through `run_stage.py`, the NERSC production driver, which assumes m4958 paths, podman-hpc and a Slurm account.

1. Use the ColliderML container route instead (the one `colliderml.simulate()` uses): each stage script runs directly in the ODD image with `--config` and `--output`, with four host folders mounted (`/work`, `/workspace`, `/cache`, `/quirkml`). A small runner picks podman-hpc, Docker, Podman or Apptainer.
2. Stage configs written with container paths; no `run_stage.py`, no Slurm account, no python env on the host for running. The packaging stage was dropped here and is back in stage 9.
3. Geant4 data downloaded into the user's own cache on first use, not read from m4958.
4. quirk-gen: inside the image (route 1, default), or an LCG_109 view from cvmfs (route 2), entered directly on an el9 host, through shifter on NERSC, or through Apptainer with an el9 image elsewhere.
5. Tested from fresh GitHub clones in `$PSCRATCH` (outside m4958): podman-hpc for mu = 0 (20 events, 3.7 min) and mu = 200 (5 events, 21 min); Apptainer 1.5 (from cvmfs) for mu = 0, including quirk-gen through Apptainer and the LCG view. Both runtimes gave 2 quirks and about 27 quirk hits per event. Two Apptainer problems found and fixed: the Geant4 data download went into the in-memory overlay, and `uv` in the image wrote to the user's home (which went over quota).
6. One image for everything: OpenDataDetector/sw branch `feat/python-pythia8-hepmc3` adds `+python` to pythia8 and hepmc3 (GitHub Actions, manual run on the branch, no PR), image `ghcr.io/opendatadetector/sw:feat-python-pythia8-hepmc3`, now the default. MadGraph was already in the image. The python modules are not on the image's `PYTHONPATH`; `run_quirkgen.sh` adds them. Tested with podman-hpc from a fresh directory, mu = 0, 20 events: quirk-gen 44 s, all stages 2.5 min, 2 quirks and 30 quirk hits per event. The same with Apptainer from cvmfs (setup 19 min, mostly the image conversion; the rest 3 min; 2 quirks and 30.4 quirk hits per event). The counts differ from the earlier 27 because MadGraph 3.5.9 and the image's Pythia give different events from the LCG_109 ones at the same seed).
7. Not tested: Docker (not available here; same command form as the ColliderML library). The image tag follows a feature branch; it changes if the branch is rebuilt.

## Stage 9: large samples in runs, multi-threaded simulation - code done, pilots next (2026-10-15)

Goal: samples of up to about 1M events (e.g. mu = 60), made the way ColliderML production makes them, then three pilots: mu = 0, mu = 10, mu = 60, each after Daniel has seen the previous one and its stage configs.

1. QuirkDDG4 is thread-safe: string state per thread, stacking action on every worker, strings cleared per event (`docs/colliderml.md`, "Multi-threaded simulation"). One thread: identical to before on 20 events. 1 against 8 threads on 400 events: quirk hits per event -0.07 +- 0.16 apart; 5.7 times faster.
2. Runs as in production: `runs` x `events_per_run`, `<work_dir>/quirks/<dataset>/<version>/runs/<N>/`, seed `<dataset>_<version>_run<N>` for every stage of run N (and quirk-gen), each stage over all runs before the next. `colliderml/run_stage.sh` runs one stage over runs (`--run-range`, `--run-list`, `--parallel K`, finished runs skipped, `status`); `colliderml/batch/` has Slurm and HTCondor examples.
3. quirk-gen per run, with the MadGraph process built once and copied per run (production's `madgraph_init` and per-run tarball).
4. Packaging back: production's `package_native_parquet.py` and `verify_packaged_parquet.py`, unchanged, in the image. edm4hep and HepMC kept per run.
5. Tested from a new work directory with podman-hpc: mu = 0, 3 runs x 10 events, 3 at a time (2.6 min); mu = 10, 2 runs x 5 events (2.7 min); both packaged and verified (PASS), loader reads both layouts, packaged event 10 = run 1 event 0.
6. mu = 0 pilot done (2026-10-15): 8 runs x 500 events, m = 400 GeV, Lambda = 1 keV, 8 simulation threads per run. Generation and stage1 on an interactive node (8 runs at a time), simulation and digitisation as debug-QOS jobs with `colliderml/batch/slurm_runs.sh` (one task per run): generation 1.7 min, stage1 20 s, simulation 7.3 min (about 7 core-seconds per event), digitisation 1 min, packaging 34 s. Verifier PASS; 4000 unique events (ids and quirk momenta); 2 quirks in every event; 29.74 +- 0.24 quirk hits per event (run means 28.6-31.6), 15 events with no quirk hit; about 4000 tracker hits per event. Disk: 1.3 MB per event in the runs (edm4hep 0.38 MB, HepMC files 0.64 MB), 0.37 MB per event packaged. Found and fixed: the Slurm example looked for the repo next to sbatch's copy of the script.
7. Next: mu = 10 pilot, then mu = 60, with the stage configs shown to Daniel before each.

## Open questions for Daniel / Jackson / Makayla

- Which string model is closer to the true string dynamics with the field on (QuirkDDG4 and quirk_sim differ by 2-4 mm), and whether Athena shows the same ionisation effect (stage 1).
- Whether Qiyu Sha's 2024 quirk data can be shared, or if we regenerate it.
