# Quirk samples through the ColliderML chain

`docs/tutorial.md` is the how-to: it makes a sample from one settings file. This page explains what each step does, for when something needs changing. The chain is the ColliderML production code ([colliderml-production](https://github.com/OpenDataDetector/colliderml-production)). Each stage script runs directly inside the ODD software image, the way the ColliderML python library's `colliderml.simulate()` runs them, and over runs the way production's `run_stage.py` does: one job per run, with the same arguments and seeds. The scripts here do not use `run_stage.py` itself, `env_setup.yaml` or Slurm, which tie it to NERSC; `colliderml/batch/` has examples for batch systems.

## The container and mounts

`colliderml/container.sh` is sourced by the other scripts. It defines `qml_run "<bash command>"`, which runs one command in the image `ghcr.io/opendatadetector/sw:feat-python-pythia8-hepmc3` (override with `QUIRKML_IMAGE`). This image is built from the OpenDataDetector/sw branch `feat/python-pythia8-hepmc3`: the ODD software image with `+python` on `pythia8` and `hepmc3` in its spack environment, so quirk-gen runs in it too. The stages also run in `sw:pr-8`, the image ColliderML production uses. Before the command it sources `/workspace/scripts/cli/setup_container_env.sh`, `/opt/acts-arrow/setup.sh` and `/opt/odd-install/bin/this_odd.sh`, if they exist.

| Mount point | Host path |
|---|---|
| `/work` | `<work_dir>` |
| `/workspace` | `<work_dir>/colliderml-production` |
| `/cache` | `<work_dir>/cache` (Geant4 data; `COLLIDERML_CACHE=/cache`) |
| `/quirkml` | this repo |

Runtime: `$QUIRKML_RUNTIME` if set. Otherwise the first of `podman-hpc`, `docker`, `podman`, `apptainer`, `singularity` found on the path, then the Apptainer in `/cvmfs/oasis.opensciencegrid.org/mis/apptainer/current/bin/`. With Apptainer the image is pulled to `<work_dir>/sw-<tag>.sif`, and `APPTAINER_CACHEDIR` and `APPTAINER_TMPDIR` default to `<work_dir>/cache/apptainer` and `.../apptainer-tmp`, so the roughly 6 GB image cache does not go to `$HOME`. Apptainer runs with `--cleanenv --writable-tmpfs --home <work_dir>/cache/home`, and `<work_dir>/cache/g4data` is mounted at `/g4data`, where the image keeps its Geant4 data: the download would otherwise go into Apptainer's small in-memory overlay and fail, and tools in the image (`uv`) would write to the user's home. Other runtimes use their own image store.

## Setup

`bash colliderml/setup.sh <work_dir>` can be rerun; it keeps pieces that exist. It does three things.

1. Clones colliderml-production into `<work_dir>/colliderml-production`: branch `quirk/pilot-integration`, checked out at `6495666`.
2. Pulls the image (`qml_pull`).
3. Builds the QuirkDDG4 plugin inside the image from `sim/QuirkDDG4` with cmake, and copies `libQuirkDDG4.so` and `libQuirkDDG4.components` to `<work_dir>/plugin/`. `plugin/BUILD_STAMP` records the image and a checksum of the plugin source; `setup.sh` rebuilds when either changes, and `run_stage.sh` stops if they don't match.

Geant4 data (2-4 GB) is downloaded into `cache/g4data` the first time the image environment is set up, which is during the plugin build in `setup.sh`.

## Settings and configs

`colliderml/sample.yaml` has flat `key: value` lines: `work_dir`, `mass_gev`, `lambda_ev`, `mu`, `runs`, `events_per_run`, `version`, `sim_threads`, `digi_threads`, `chunk_size`, `max_boost`. `colliderml/make_sample.py <settings>` reads them (python3 only, no pyyaml) and fills the files in `colliderml/templates/`.

- Dataset name: `m<mass>_L<lambda>_mu<mu>`; campaign `quirks`; version from the settings (`v1`).
- Configs go to `<work_dir>/configs/<dataset>/`. Runs go to `<work_dir>/quirks/<dataset>/<version>/runs/<N>/`, production's `<campaign>/<dataset>/<version>/runs/<N>` layout, and the packaged dataset to `<work_dir>/quirks/<dataset>/<version>/parquet/`.
- `sample.env` in the config directory holds the resolved settings for `gen/run_quirkgen.sh` and `colliderml/run_stage.sh`.
- Seeds: as in production (`cli_utils.build_stage_command`), every stage of run `N` gets `--seed <dataset>_<version>_run<N>`, which the stage scripts hash to an integer (`hash_seed_string`: md5, first 4 bytes, modulo 900000000). The stage scripts offset it where they need two streams (Pythia8 pileup uses seed + 1000). quirk-gen gets the same integer, computed by `gen/run_quirkgen.sh`. Reruns of a run give the same events; a new `version` gives an independent sample.
- Stage 1 is `1_merge_smear.yaml` for mu = 0 and `1_pythia_pileup_merge.yaml` for mu > 0. `4_package_parquet.yaml` is the packaging config.
- Paths in the configs are paths inside the container.

Plugin values in `2_simulation.yaml`:

| Property | Value |
|---|---|
| `Mass` | `mass_gev` x 1000 (MeV) |
| `Charge`, `PdgId` | 1.0 and 10000100 (quirk-gen: +10000100 has charge +1) |
| `StringForce` | Lambda^2 / (hbar c), with hbar c = 1.973269804e-10 MeV mm and Lambda in eV (MeV/mm) |
| `MaxBoost` | `max_boost` |
| `FirstStringLen`, `MaxMergeT`, `MaxMergeMag` | 1.0e-6 (fixed in the template) |
| `EnableEM` | true |

`docs/validation_status.md` is the source cited in the settings and template comments for `max_boost` (1e-3 agrees with 1e-4 to 0.1 mm; 1e-2 does not converge for slack strings).

## quirk-gen and the HepMC conversion

`gen/run_quirkgen.sh <config_dir> init|<run>` clones `burzynski-lab/quirk-gen` at `9980db3` into `<work_dir>/quirkgen/` and runs `gen/quirkgen_run.sh` in the chosen environment. `QUIRKGEN_ENV=image` or `lcg` forces a route. `run_stage.sh <config_dir> generation` calls `init` once and then each run.

`quirkgen_run.sh` does what quirk-gen's `scripts/run_chain.sh` does (`generate_lhe.py`, then `shower.py`, with the run's mass, events, Lambda and seed), arranged as production arranges MadGraph (`madgraph_init` once, then a copy of the process per run):

- `init` builds the MadGraph process once, in `<work_dir>/quirkgen/procdir_<route>`, with a 10-event test run. The process depends only on the model, not on the mass.
- Each run copies it to a temporary directory, sets `nb_core` to `sim_threads` (or `$QUIRKGEN_CORES`) in its `me5_configuration.txt`, generates, showers, and writes `events.hepmc`, `events.json` and `events_hepmc2.hepmc` to `runs/<N>/`. Runs can therefore go at the same time.

**Route 1: inside the ODD image (default).** The image has MadGraph 3.5.9 in its spack tree, and the default image also has the Pythia8 (8.313) and HepMC3 (3.3.0) python modules. These are not on the image's `PYTHONPATH`, so the script adds `/spack/opt/spack/linux-x86_64/pythia8-*/lib` and `hepmc3-*/lib/python3*/site-packages`. Used when `import pythia8, pyHepMC3` then works. Tested with podman-hpc and Apptainer: 20 events in about 45 s. Without the bindings (e.g. `sw:pr-8`), the script uses route 2.

**Route 2: LCG_109 from cvmfs, for images without the bindings** (`/cvmfs/sft.cern.ch/lcg/views/LCG_109/x86_64-el9-gcc13-opt`, or `$LCG_VIEW`). On an el9 host the script runs the view directly. Otherwise it uses shifter with `registry.cern.ch/atlasadc/atlas-grid-almalinux9` (NERSC) or Apptainer with that image. It stops if none of these is available.

**Why the conversion exists.** quirk-gen numbers events from 1; the ACTS HepMC3Reader used in the pileup merge stops with "Event number mismatch" unless the first event is 0. With route 2 there is a second reason: the LCG_109 HepMC3 is 3.03.01 and writes the quirk-gen files. The image's HepMC3 is 3.3.0. It rejects some events from 3.03 files ("too few vertices were parsed") and returns empty events without a pipeline error. `colliderml/convert_to_hepmc2.py` reads the file with the HepMC3 of the environment that wrote it and writes HepMC2 ASCII numbered from 0. In route 2 the conversion runs inside the LCG environment. `colliderml/convert_quirkgen_hepmc.sh <in> <out>` does the same conversion by hand, using shifter if the host is not AlmaLinux.

## The stages

`colliderml/run_stage.sh <config_dir> <stage> [--run-range A B | --run-list N ...] [--parallel K] [--force]` runs one stage over runs (default all; `--run-range` is A to B-1, as in production). Each run is one command in the image:

```
cd /workspace/scripts/simulation && python3 <script> --config /work/configs/<dataset>/<cfg> \
    --output /work/quirks/<dataset>/<version>/runs --output-subdir <N> --seed <dataset>_<version>_run<N>
```

which is the command production's `build_stage_command` builds. `--parallel K` runs K runs at a time (xargs). A stage starts only for runs that have finished the previous stage. After a run, its output is checked, and if the check passes the driver writes `runs/<N>/.done_<stage>`; finished runs are skipped unless `--force`. Logs go to `<config_dir>/logs/<stage>/run<N>.log`. `status` counts, per stage, the runs done and lists the missing ones. `colliderml/run_sample.sh <config_dir> [--parallel K]` runs all stages in order.

| Stage | Script | Reads | Writes | Check after each run |
|---|---|---|---|---|
| `generation` | quirk-gen (above) | | `events.hepmc`, `events.json`, `events_hepmc2.hepmc` | `events_per_run` events (lines starting `E `); `events.json` has the sample's mass, PDG id and charge 1 |
| `stage1`, mu = 0 | `merge_and_smear.py` | `events_hepmc2.hepmc` | `merged_events.hepmc3` | `events_per_run` events |
| `stage1`, mu > 0 | `pythia_gen.py`, with `--hard-scatter-file runs/<N>/events_hepmc2.hepmc` | same | same | same |
| `simulation` | `ddsim_run.py` | `merged_events.hepmc3` | `edm4hep.root` | exit code 0 and `events_per_run` events (`podio-dump`) |
| `digitization` | `digi_and_reco.py` | `edm4hep.root` | parquet directories | `tracker_hits`, `tracker_simhits`, `particles` each have a parquet file |
| `package` | `package_native_parquet.py`, then `verify_packaged_parquet.py` | all runs' parquet | `parquet/{truth,reco}/<object>/` | the verifier's checks |

Before digitising a run, the driver moves any parquet tables already in the run directory to `runs/<N>/.old_digitization_<time>/`: `digi_and_reco.py` does not replace old shards, and packaging would read both.

**Stage 1.** quirk-gen puts every event at the origin. Both variants smear vertices with a Gaussian beam spot: 12.5 um in x and y, 55.5 mm in z, 0.185 ns in t. For mu > 0, `pythia_generation` has `hard_process` unset and `merge: true`, and gets the run's converted file with `--hard-scatter-file`. Each run generates its own Pythia8 SoftQCD events and merges a Poisson number of them (mean `pileup`) into each quirk event, as production does.

**Stage 2.** The `simulation` stage has a `user_physics` block (from the `feat/user-physics-plugin` branch) that loads `QuirkPhysics/Quirks` from `/work/plugin/libQuirkDDG4.so`. The geometry is the tracker-only ODD v6 setup of the drifting-beamspot campaign: `OpenDataDetectorDefs.xml`, `OpenDataDetectorTracker.xml` and `/workspace/configs_production/drift_beamspot/geometry/tracker_fields_plugins.xml`. The last file holds the field and plugin blocks that the tracker file lacks. Without it ddsim crashes. Geant4 runs `sim_threads` threads, one event per thread, as in production (ttbar uses 8); see "Multi-threaded simulation" below. Vertex smearing in this stage is zero.

**Stage 3.** ACTS geometric digitisation with native parquet output (`output_parquet_arrow`), `digi_threads` threads. Seeding, CKF and truth tracking assume helices. The `reco/tracks` and `reco/truth_tracks` tables are not meaningful for quirks. Use `tracker_simhits` and `tracker_hits` (with `particle_ids` truth links).

**Packaging.** Production's `package_native_parquet.py`, unchanged, joins the per-run parquet of runs 0 to `runs - 1` (it refuses gaps) into `parquet/truth/{particles,tracker_simhits}` and `parquet/reco/{tracker_hits,tracks,truth_tracks}`, in files of `chunk_size` events named `quirks.<dataset>.<version>.<group>.<object>.events<A>-<B>.parquet`. It numbers events over the sample (`event_id = run * events_per_run + local event`, no run column) and converts times from mm (c t) to ns. `verify_packaged_parquet.py` then checks event ranges, row counts and that the tables cover the same events.

## Output layout and reading it

Each run directory has `events.json`, `events.hepmc`, `events_hepmc2.hepmc`, `merged_events.hepmc3`, `edm4hep.root`, and one directory per parquet table (`tracker_hits/`, `tracker_simhits/`, `particles/`, and others). Nothing is deleted. The packaged dataset is in `<work_dir>/quirks/<dataset>/<version>/parquet/`.

`ml/quirk_loader.py` reads either:

```python
from quirk_loader import QuirkDataset
ds = QuirkDataset("<work_dir>/quirks/<dataset>/<version>/parquet")    # packaged
ds = QuirkDataset("<work_dir>/quirks/<dataset>/<version>/runs/0")     # one run
ev = ds.event(ds.event_ids[0])     # dict of DataFrames: hits, particles, truth_edges
```

The per-run files store times as c*t in mm; for them the loader divides `particles.time` and `tracker_simhits.true_time` by 299.792458 mm/ns, so `t` in the output is in ns either way. Hit time `t` is the true time of the hit's first sim hit: the digitisation writes no reco time. `truth_edges` pairs consecutive hits of each quirk in true time.

## Multi-threaded simulation

Geant4 shares particle definitions between its worker threads, while QuirkDDG4 (ported from Athena) kept each quirk pair's string on the quirk's particle definition, and installed its stacking action on the master thread only. The plugin therefore used to run on one thread. Now:

- each thread keeps its own string for each quirk pair (`Quirk::GetStringIn`, a thread-local map), with the string parameters stored on the particle definition;
- `QuirkPhysics::constructProcess` installs `QuirkStackingAction` and the extra waiting stack on every worker thread (and on the only thread in a single-threaded run);
- `QuirkStackingAction::PrepareNewEvent` clears the thread's strings, so no event starts with string left over from the previous event on that thread.

Checks, m = 400 GeV, Lambda = 1 keV, mu = 0, in `sw:feat-python-pythia8-hepmc3`:

- One thread, 20 events: identical to the plugin before the change, event by event (hits, quirk hits, hit positions, MC particles).
- Every worker thread reports the stacking action active on its first quirk (`QuirkStackingAction: first quirk classified on thread N`).
- 1 against 8 threads, the same 400 generated events: per event, quirk hits 32.49 against 32.42 (paired difference -0.07 +- 0.16), largest quirk hit radius 677.4 against 677.0 mm (-0.4 +- 0.5), all hits 4251 against 4262 (11 +- 22), MC particles 2279 against 2272 (-7 +- 8). Wall time 2095 s against 369 s.
- Events are not identical one by one between thread counts, but neither are they without the plugin: plain ddsim on 40 minimum-bias events gave 11 of 40 identical between 1 and 4 threads. This is Geant4's multi-threaded random-number handling, also present in production.

The `QUIRK_DUMP_FILE`, `QUIRK_STEPCHECK_FILE` and step-dump diagnostics write one file from all threads, so use them with one thread (`sim/run_quirk_ddsim.py` runs one).

## colliderml-production branches

All are on GitHub, based on `feat/two-container-pipeline` (`2eb67bc`). None has a PR. `setup.sh` uses `quirk/pilot-integration`, which combines the five fix branches with a worktree fix for `run_stage` (from `f71e7c8`).

| Branch | Change | Used by the container route |
|---|---|---|
| `feat/user-physics-plugin` | `user_physics` block in the simulation stage | yes |
| `fix/ddsim-single-thread-duplicates` | single-thread ddsim wrote each event twice | yes |
| `fix/ddsim-exit-code` | `ddsim_run.py` returned 0 on failure | yes |
| `fix/pythia-hard-scatter-path` | `hard_scatter_file` and `pileup_file` from config crashed | yes |
| `fix/pscratch-bind-path` | the container mounted one fixed user's `$PSCRATCH` | only for `run_stage.py`; not used here |
| `quirk/pilot-integration` | the five above, plus the worktree fix | yes (`6495666`) |

Checks done earlier on these branches: quirk events simulated in sw:pr-8 with quirk hits present; one event gave one entry (was 2) with a single thread, and two threads were unchanged; a missing config gave exit code 1 (was 0); the mu = 200 merge ran.
