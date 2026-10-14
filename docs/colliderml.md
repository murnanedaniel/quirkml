# Quirk samples through the ColliderML chain

`docs/tutorial.md` is the how-to: it makes a sample from one settings file. This page explains what each step does, for when something needs changing. The chain is the ColliderML production code ([colliderml-production](https://github.com/OpenDataDetector/colliderml-production)). Each stage script runs directly inside the ODD software image with `--config` and `--output`, the same way the ColliderML python library's `colliderml.simulate()` runs them. The scripts here do not use `run_stage.py`, `env_setup.yaml`, Slurm or a packaging stage.

## The container and mounts

`colliderml/container.sh` is sourced by the other scripts. It defines `qml_run "<bash command>"`, which runs one command in the image `ghcr.io/opendatadetector/sw:pr-8` (override with `QUIRKML_IMAGE`). Before the command it sources `/workspace/scripts/cli/setup_container_env.sh`, `/opt/acts-arrow/setup.sh` and `/opt/odd-install/bin/this_odd.sh`, if they exist.

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
3. Builds the QuirkDDG4 plugin inside the image from `sim/QuirkDDG4` with cmake, and copies `libQuirkDDG4.so` and `libQuirkDDG4.components` to `<work_dir>/plugin/`.

Geant4 data (2-4 GB) is downloaded into `cache/g4data` the first time the image environment is set up, which is during the plugin build in `setup.sh`.

## Settings and configs

`colliderml/sample.yaml` has flat `key: value` lines: `work_dir`, `mass_gev`, `lambda_ev`, `mu`, `events`, `seed`, `max_boost`. `colliderml/make_sample.py <settings>` reads them (python3 only, no pyyaml) and fills the files in `colliderml/templates/`.

- Dataset name: `m<mass>_L<lambda>_mu<mu>`.
- Configs go to `<work_dir>/configs/<dataset>/`; the run directory is `<work_dir>/runs/<dataset>/`.
- `sample.env` in the config directory holds the resolved settings for `gen/run_quirkgen.sh` and `colliderml/run_sample.sh`.
- Seeds: the settings seed is passed to quirk-gen. The stage seeds are seed + 41 (stage 1), + 42 (simulation), + 43 (digitisation).
- Stage 1 is `1_merge_smear.yaml` for mu = 0 and `1_pythia_pileup_merge.yaml` for mu > 0.
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

`gen/run_quirkgen.sh <config_dir>` clones `burzynski-lab/quirk-gen` at `9980db3` into `<work_dir>/quirkgen/` and runs `scripts/run_chain.sh` with the mass, events, Lambda and seed. It copies `events.hepmc` and `events.json` to the run directory and writes `events_hepmc2.hepmc` there. `QUIRKGEN_ENV=image` or `lcg` forces a route.

**Route 1: inside the ODD image.** Used when `python3 -c 'import pythia8, pyHepMC3'` works in the image. The OpenDataDetector/sw branch `feat/python-pythia8-hepmc3` adds Pythia8 and HepMC3 python bindings to the image. Its build is in progress. Use an image built from it by setting `QUIRKML_IMAGE`. This route has not been tested yet. Without the bindings, the script uses route 2.

**Route 2: LCG_109 from cvmfs** (`/cvmfs/sft.cern.ch/lcg/views/LCG_109/x86_64-el9-gcc13-opt`, or `$LCG_VIEW`). On an el9 host the script runs the view directly. Otherwise it uses shifter with `registry.cern.ch/atlasadc/atlas-grid-almalinux9` (NERSC) or Apptainer with that image. It stops if none of these is available.

**Why the conversion exists.** The LCG_109 HepMC3 is 3.03.01 and writes the quirk-gen files. The image's HepMC3 is 3.3.0. It rejects some events from 3.03 files ("too few vertices were parsed") and returns empty events without a pipeline error. `colliderml/convert_to_hepmc2.py` reads the file with the LCG reader and writes HepMC2 ASCII. It also renumbers events from 0: quirk-gen starts at 1, and the ACTS HepMC3Reader used in the pileup merge stops with "Event number mismatch" unless the first event is 0. In route 2 the conversion runs inside the LCG environment. `colliderml/convert_quirkgen_hepmc.sh <in> <out>` does the same conversion by hand, using shifter if the host is not AlmaLinux.

## The three stages

`colliderml/run_sample.sh <config_dir> [first_stage]` runs the stages in order. Each runs as `cd /workspace/scripts/simulation && python3 <script> --config /work/configs/<dataset>/<cfg> --output /work/runs/<dataset>`. Logs go to `<config_dir>/logs/<cfg>.log`. A non-zero exit code stops the run. Before stage 1 the script checks that `events.json` has the sample's mass and PDG id and charge 1, and that `events_hepmc2.hepmc` has at least `events` events (counted as lines starting `E `).

| Stage | Script | Reads | Writes | Check after the stage |
|---|---|---|---|---|
| 1, mu = 0 | `merge_and_smear.py` | `events_hepmc2.hepmc` | `merged_events.hepmc3` | at least `events` events in it |
| 1, mu > 0 | `pythia_gen.py` | `events_hepmc2.hepmc` | `merged_events.hepmc3` | same |
| 2 | `ddsim_run.py` | `merged_events.hepmc3` | `edm4hep.root` | file exists and is not empty |
| 3 | `digi_and_reco.py` | `edm4hep.root` | parquet directories | `tracker_hits`, `tracker_simhits`, `particles` each have a parquet file |

**Stage 1.** quirk-gen puts every event at the origin. Both variants smear vertices with a Gaussian beam spot: 12.5 um in x and y, 55.5 mm in z, 0.185 ns in t. For mu > 0, `pythia_generation` has `hard_process` unset, `merge: true` and `hard_scatter_file` set to the converted file. It generates Pythia8 SoftQCD events and merges a Poisson number of them (mean `pileup`) into each quirk event.

**Stage 2.** The `simulation` stage has a `user_physics` block (from the `feat/user-physics-plugin` branch) that loads `QuirkPhysics/Quirks` from `/work/plugin/libQuirkDDG4.so`. The geometry is the tracker-only ODD v6 setup of the drifting-beamspot campaign: `OpenDataDetectorDefs.xml`, `OpenDataDetectorTracker.xml` and `/workspace/configs_production/drift_beamspot/geometry/tracker_fields_plugins.xml`. The last file holds the field and plugin blocks that the tracker file lacks. Without it ddsim crashes. The plugin is single-threaded, so the stage uses one core (`threads: 1`, `single_thread: true`). Vertex smearing in this stage is zero.

**Stage 3.** ACTS geometric digitisation with native parquet output (`output_parquet_arrow`), 8 threads. Seeding, CKF and truth tracking assume helices. The `reco/tracks` and `reco/truth_tracks` tables are not meaningful for quirks. Use `tracker_simhits` and `tracker_hits` (with `particle_ids` truth links).

## Output layout and reading it

Everything is in `<work_dir>/runs/<dataset>/`: `events.json`, `events.hepmc`, `events_hepmc2.hepmc`, `merged_events.hepmc3`, `edm4hep.root`, and one directory per parquet table (`tracker_hits/`, `tracker_simhits/`, `particles/`, and others). This is the per-run layout.

`ml/quirk_loader.py` reads it:

```python
from quirk_loader import QuirkDataset
ds = QuirkDataset("<work_dir>/runs/<dataset>")
ev = ds.event(0)     # dict of DataFrames: hits, particles, truth_edges
```

It also reads a packaged layout (`truth/` and `reco/` subdirectories) if the root has `reco/`. The per-run files store times as c*t in mm. For the per-run layout the loader divides `particles.time` and `tracker_simhits.true_time` by 299.792458 mm/ns, so `t` in the output is in ns. Hit time `t` is the true time of the hit's first sim hit: the digitisation writes no reco time. `truth_edges` pairs consecutive hits of each quirk in true time.

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
