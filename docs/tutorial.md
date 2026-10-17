# Tutorial: a quirk sample in ColliderML parquet

This takes you from nothing to ColliderML-format parquet for one quirk sample: quirk-gen events, beam spot and optional pileup, Geant4 with the quirk string (QuirkDDG4) in the ODD tracker, and ACTS digitisation. The calorimeters are not simulated.

Everything runs inside the OpenDataDetector software image, including quirk-gen (MadGraph and Pythia8), the same way the ColliderML library's `colliderml.simulate()` runs the pipeline. `docs/colliderml.md` explains what each step does.

## What you need

- A Linux machine (x86_64) with one of: Docker, Podman, Apptainer (or Singularity), or podman-hpc on NERSC. The scripts take the first one they find; set `QUIRKML_RUNTIME` to choose (for example `QUIRKML_RUNTIME=apptainer`).
- `git` and `python3` (standard library only), and about 15 GB of disk for the work directory.
- For reading the output: python with `numpy`, `pandas` and `pyarrow`.

Work in a directory on a local or scratch file system with that much space (see "Site notes" at the end for NERSC).

## 1. Clone and set up (once)

Run all commands from one directory; it ends up holding `quirkml/`, `quirks/` and your settings file.

```bash
git clone https://github.com/murnanedaniel/quirkml.git
bash quirkml/colliderml/setup.sh quirks
```

`setup.sh` puts into the work directory (`quirks/` here):

- `colliderml-production/`: the ColliderML stage scripts, branch `quirk/pilot-integration` of [colliderml-production](https://github.com/OpenDataDetector/colliderml-production) at a fixed commit;
- the image `ghcr.io/opendatadetector/sw:feat-python-pythia8-hepmc3` (about 6 GB; with Apptainer as `sw-feat-python-pythia8-hepmc3.sif`, otherwise in the runtime's own store). It is a build of the ODD software image with Pythia8's and HepMC3's python bindings added, which quirk-gen needs;
- `plugin/libQuirkDDG4.so`, the quirk plugin built inside that image;
- `cache/`: Geant4 data files (2-4 GB, downloaded the first time the image environment is set up).

It took about 3 minutes in our tests once the image was downloaded; with Apptainer, converting the image to `.sif` adds about 15 minutes the first time. Rerunning it keeps what is already there. On a shared cluster run `setup.sh` on a compute node: pulling and unpacking the 6 GB image on a login node can hit its per-user limits (in our tests it ended the login session).

## 2. Describe the sample (one file)

```bash
cp quirkml/colliderml/sample.yaml my_sample.yaml
```

`work_dir: quirks` already matches the command above (a relative path is relative to the settings file). Set the physics point, `mass_gev`, `lambda_ev` (the string scale) and `mu` (0 for no pileup), and the size, `runs` and `events_per_run`. The defaults, 1 run of 20 events, are a quick first sample. Then:

```bash
python3 quirkml/colliderml/make_sample.py my_sample.yaml
```

This writes the stage configs to `quirks/configs/<dataset>/` (the dataset is named e.g. `m400_L1000_mu0`), creates the runs directory `quirks/quirks/<dataset>/v1/runs/`, and prints the commands below.

The datasets go under `work_dir` unless you set `output_base_dir`, e.g. to a shared data area. They are then in `<output_base_dir>/quirks/<dataset>/<version>/`, the ColliderML production layout (campaign `quirks`); configs, logs, the image and the plugin stay in `work_dir`. The commands below assume the default.

As in ColliderML production, a sample is a set of runs. Run `N` is generated, simulated and digitised on its own, in `runs/<N>/`, with the seed `<dataset>_<version>_run<N>`, so rerunning a run gives the same events and runs can be spread over cores, nodes or batch jobs. At the end, packaging joins all runs into one dataset.

## 3. Run the stages

```bash
bash quirkml/colliderml/run_sample.sh quirks/configs/m400_L1000_mu0
```

Run it where you have 8 cores for up to an hour: a workstation, or an interactive or batch job on a cluster. `run_sample.sh` runs the five stages below in order, each over all runs before the next starts, with `colliderml/run_stage.sh`. To run one stage yourself:

```bash
bash quirkml/colliderml/run_stage.sh quirks/configs/m400_L1000_mu0 simulation
bash quirkml/colliderml/run_stage.sh quirks/configs/m400_L1000_mu0 status
```

| Stage | What it does |
|---|---|
| `generation` | [quirk-gen](https://github.com/burzynski-lab/quirk-gen) at a fixed commit, inside the image: MadGraph builds the process once (in `quirks/quirkgen/`), then each run gets its own events, `events.hepmc`, `events.json`, and `events_hepmc2.hepmc` (the same events numbered from 0, which the next stage reads) |
| `stage1` | beam spot (Gaussian: 12.5 um in x and y, 55.5 mm in z); for `mu > 0` also Pythia8 SoftQCD pileup, generated per run and merged in |
| `simulation` | Geant4 on the ODD v6 tracker with QuirkDDG4, `sim_threads` threads per run (one event per thread); writes `edm4hep.root` |
| `digitization` | ACTS geometric digitisation, written as parquet per run |
| `package` | all runs into one dataset in the published ColliderML layout, then a check of the result (`verify_packaged_parquet.py`) |

After each run, the driver checks the run's output (event counts) and marks it done; finished runs are skipped, so after a failure the same command carries on with the rest. A failure prints `FAILED` with the run's log, in `quirks/configs/<dataset>/logs/<stage>/run<N>.log`.

quirk-gen's `--lambda` is only recorded in `events.json`. The string acts only in the simulation, so the events of one mass serve every `lambda_ev`.

Times on one node of a CPU cluster (podman-hpc), m = 400 GeV, Lambda = 1 keV:

- mu = 0, 3 runs of 10 events, 3 runs at a time with 4 threads each: 2.6 minutes for all five stages (generation 1.5 minutes, of which about 45 s builds the MadGraph process; simulation 50 s). About 30 quirk hits and 4,000 tracker hits per event.
- mu = 10, 2 runs of 5 events: 2.7 minutes; 8,000-17,000 tracker hits per event.
- Simulation of 400 events at mu = 0 in one run: 6 minutes with 8 threads, 35 minutes with 1.
- mu = 200, 5 events in one thread (earlier image `sw:pr-8`): 21 minutes; an event then has about 200,000 tracker hits.

## 4. Read it

The packaged dataset is in `quirks/quirks/<dataset>/v1/parquet/`, in the layout of the published ColliderML datasets: `truth/particles`, `truth/tracker_simhits`, `reco/tracker_hits`, `reco/tracks`, `reco/truth_tracks`, files of `chunk_size` events named `quirks.<dataset>.v1.<group>.<object>.events<A>-<B>.parquet`. Event ids run over the whole sample (run `N`, event `i` is `N * events_per_run + i`) and times are in ns.

```python
import sys; sys.path.insert(0, "quirkml/ml")
from quirk_loader import QuirkDataset
ds = QuirkDataset("quirks/quirks/m400_L1000_mu0/v1/parquet")
ev = ds.event(ds.event_ids[0])     # hits, particles, truth_edges
quirks = ev["particles"][ev["particles"].is_quirk]
print(len(quirks), "quirks,", ev["hits"].is_quirk.sum(), "quirk hits")
```

`QuirkDataset` also reads one run's own parquet (`.../runs/<N>/`), whose event ids start at 0 and whose times are in mm (c t). `docs/ml_baselines.md` describes the tables and what is usable for quirks. In short: use `tracker_hits` and `tracker_simhits`; the `tracks` tables come from helix fits and are not meaningful for quirks. Each run also keeps its `edm4hep.root` (the full Geant4 output) and its HepMC files.

## 5. Large samples

The same commands make large samples; set `runs` and `events_per_run` in the settings file. ColliderML production uses runs of 1000-1300 events for ttbar-like samples; for quirks with pileup, pick `events_per_run` so that one run's simulation takes a few hours (see the times above).

- **One machine.** `--parallel K` runs `K` runs of a stage at a time, e.g. on a 128-core node with `sim_threads: 8`, `--parallel 16`. Keep `K` times the stage's threads within the cores. `stage1` with pileup is single-threaded, so it can take a larger `K`.
- **A batch system.** Each run is an independent job: `run_stage.sh <config_dir> <stage> --run-list <N>`. Submit one stage over all runs, wait for it to finish, then the next stage, then `package` once. `colliderml/batch/` has examples for Slurm (one task per run, as production's `multi_node_slurm` mode) and HTCondor (one job per run). Build the MadGraph process first, with `run_stage.sh <config_dir> generation --run-list 0`, so that the generation jobs don't all build it at once.
- **Checking.** `run_stage.sh <config_dir> status` lists, per stage, how many runs are done and which are missing; rerun those with `--run-list`. `package` refuses to start until every run is digitised, because the packaged event ids assume runs 0 to `runs - 1` with no gaps.
- **Disk.** Per run, `edm4hep.root` and the HepMC files are kept; they take more space than the parquet (at mu = 200, about 24 MB and 19 MB per event against 21 MB of parquet).

## Things that go wrong

- **No container runtime found.** Install one, or set `QUIRKML_RUNTIME` to its full path.
- **Apptainer fills your home directory.** The scripts put Apptainer's cache in `<work_dir>/cache/apptainer`; if you run Apptainer by hand, set `APPTAINER_CACHEDIR` yourself.
- **Changing the mass of an existing sample.** The generation check compares each run's `events.json` with the settings file and marks the run failed. Make a new sample (or a new `version`) instead.
- **Using another image.** Set `QUIRKML_IMAGE` before `setup.sh` (it rebuilds the plugin for that image). If the image has no Pythia8 python bindings, for example `ghcr.io/opendatadetector/sw:pr-8`, the generation stage falls back to the LCG_109 view from cvmfs (`docs/colliderml.md`).
- **A high string tension.** Simulation cost grows with the tension: about 1-3 minutes per pair-only event on one thread at Lambda = 10-30 keV, and at 100 keV one event didn't finish in 20 minutes (`grid/README.md`). Start with tens of events.

## Site notes

- **NERSC (Perlmutter).** podman-hpc is found first. On m4958, set `output_base_dir: /global/cfs/cdirs/m4958/data/ColliderML/simulation`, where the ColliderML datasets live. Work on `$PSCRATCH` (MadGraph fails on CFS, and home is small); it is purged after 8 weeks without access, so copy the parquet you want to keep. Run steps 1 and 3 in a compute allocation, not on a login node (see step 1), e.g. `salloc -A <account> -C cpu -q shared_interactive -t 60 -c 8`, then `srun -n 1 -c 8 bash quirkml/colliderml/setup.sh quirks` and the same `srun` prefix for `run_sample.sh`. For large samples, `colliderml/batch/slurm_runs.sh` runs one task per run; the debug QOS (`-q debug`, up to 30 minutes) usually starts at once and suits pilots. Slurm counts hardware threads as CPUs on Perlmutter, so give `--cpus-per-task` twice the stage's threads (16 for `sim_threads: 8`). A pilot of 8 runs x 500 events at mu = 0 took 1.7 minutes for generation, 20 s for stage1, 7 minutes for simulation (one debug job, 8 tasks on one node), 1 minute for digitisation and 34 s for packaging.
- **lxplus and other hosts with Apptainer.** Use Apptainer (`QUIRKML_RUNTIME=apptainer`); keep the work directory off AFS home, which is too small.

