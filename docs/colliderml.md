# Quirk samples through the ColliderML chain

This guide takes quirk-gen output to ColliderML-format parquet on Perlmutter. Every command here was run for the pilot in `colliderml/pilot_m400_L1000_mu0/` (m = 400 GeV, Lambda = 1 keV, no pileup) and `colliderml/pilot_m400_L1000_mu200/` (mu = 200).

The chain is the ColliderML production code ([colliderml-production](https://github.com/OpenDataDetector/colliderml-production)), driven stage by stage with `scripts/cli/run_stage.py`. Quirks need four changes to it, which exist as local branches (not yet merged or pushed; see "Status of the ColliderML changes" below).

## 0. One-time setup

1. A checkout of colliderml-production with the quirk changes (today: the local branch `quirk/pilot-integration`).
2. `scripts/cli/env_setup.yaml`. It is not in git (it holds site paths); copy it from an existing checkout, or start from `scripts/cli/env_setup.yaml.container-template`. Without it, `run_stage.py` stops with "Configuration missing 'output_base_dir'".
3. The `collider-env` conda python for `run_stage.py` (`~/.conda/envs/collider-env/bin/python`); the system python3 fails.
4. The QuirkDDG4 plugin built for the sim image: `bash sim/QuirkDDG4/build_colliderml.sh` on a login node (writes `sim/QuirkDDG4/build-pr-8/lib/libQuirkDDG4.so`).

## 1. Make quirk-gen output readable

quirk-gen writes HepMC3 3.03.01 files. The sim image's HepMC3 (3.3.0) rejects their events ("too few vertices were parsed") and returns them empty, without an error from the pipeline. Convert to HepMC2 with quirk-gen's own LCG view:

```bash
bash colliderml/convert_quirkgen_hepmc.sh output/m400/events.hepmc <run_dir>/events_hepmc2.hepmc
cp output/m400/events.json <run_dir>/
```

The converter also renumbers events from 0, which the pileup merge needs.

Each `<run_dir>` is `<base>/<campaign>/<dataset>/<version>/runs/<N>/`; the stage configs point at it with `debug_output_dir`.

## 2. Beam spot and pileup

quirk-gen puts every event at the origin. Pick one of:

- **No pileup:** `merge_smear` (Gaussian beam spot: 12.5 um in x and y, 55.5 mm in z, 0.185 ns in t). Config: `pilot_m400_L1000_mu0/1_merge_smear.yaml`.
- **Pileup:** `pythia_generation` with `hard_process` unset, `merge: true` and `hard_scatter_file` pointing at the converted file. It generates Pythia8 SoftQCD events and merges a Poisson number of them (mean `pileup`) into each quirk event, with the same beam spot. Config: `pilot_m400_L1000_mu200/1_pythia_pileup_merge.yaml`.

Both write `merged_events.hepmc3` into the run directory.

## 3. Simulation with the quirk plugin

The `simulation` stage takes a `user_physics` block (new). Values come from quirk-gen's `events.json`:

```yaml
threads: 1                              # forced to 1 anyway by single_thread
user_physics:
  - type: QuirkPhysics/Quirks
    library: <quirkml>/sim/QuirkDDG4/build-pr-8/lib/libQuirkDDG4.so
    single_thread: true
    properties:
      Mass: 400000.0                    # MeV = mass_gev x 1000
      Charge: 1.0                       # quirk-gen: +10000100 has charge +1
      PdgId: 10000100
      StringForce: 5068.0               # stringforce_mev_per_mm
      MaxBoost: 1.0e-3
      EnableEM: true
```

The rest of the config is the ColliderML tracker-only setup (`odd_compact_files` with ODD v6 Defs + Tracker). The plugin is single-threaded, so a run is one core.

## 4. Digitisation and parquet

`digitization` (ACTS geometric digitisation, native parquet) and `package_parquet` run unchanged; configs in the pilot directories.

What to use for quirks:

- `truth/tracker_simhits` and `reco/tracker_hits` (with `particle_ids` truth links) are the useful tables.
- `reco/tracks` and `reco/truth_tracks` come from helix fits with a pion hypothesis. They are not meaningful for quirks.

## Where run_stage writes

Set `common.output_base_dir` in every config to your own run tree (the pilot configs do). `run_stage.py` writes config copies, logs and validation reports under `<output_base_dir>/<campaign>/<dataset>/<version>/`, and the default is the public ColliderML production tree.

The packaging validation requires track files of at least 0.1 MB. A small pilot (tens of events) fails that check even though every table was written; read the packaging log for the per-table `wrote ...` lines.

## Running a stage

On a login node for light stages (generation, merge), on a compute node for simulation:

```bash
cd <colliderml-production checkout>
~/.conda/envs/collider-env/bin/python scripts/cli/run_stage.py <config.yaml>
```

For a compute node, wrap it: `salloc -A m4958 -C cpu -q shared_interactive -t 120 -c 8 srun -n 1 -c 8 bash -c '...'`. podman-hpc worked on `shared_interactive` in our tests; the ColliderML docs warn that it can fail silently on shared nodes, so check that the stage wrote its output.

## Status of the ColliderML changes

All on local branches of colliderml-production, based on `feat/two-container-pipeline`:

| Branch | Change | Checked |
|---|---|---|
| `feat/user-physics-plugin` | `user_physics` block in the simulation stage | quirk events simulated in sw:pr-8; quirk hits present |
| `fix/ddsim-single-thread-duplicates` | single-thread ddsim wrote each event twice | 1 event -> 1 entry (was 2); 2 threads unchanged |
| `fix/ddsim-exit-code` | `ddsim_run.py` returned 0 on failure | missing config: exit 1 (was 0) |
| `fix/pythia-hard-scatter-path` | `hard_scatter_file` / `pileup_file` from config crashed | mu = 200 merge runs |
| `quirk/pilot-integration` | the four above, plus the worktree fix for `run_stage` (from `f71e7c8`) | used for the pilot |
