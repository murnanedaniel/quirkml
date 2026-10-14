# Tutorial: a quirk sample in ColliderML parquet

This takes you from nothing to ColliderML-format parquet for one quirk sample: quirk-gen events, beam spot and optional pileup, Geant4 with the quirk string (QuirkDDG4) in the ODD tracker, and ACTS digitisation. The calorimeters are not simulated.

Everything except quirk-gen runs inside the OpenDataDetector software image, the same way the ColliderML library's `colliderml.simulate()` runs the pipeline. `docs/colliderml.md` explains what each step does.

## What you need

- A Linux machine (x86_64) with one of: Docker, Podman, Apptainer (or Singularity), or podman-hpc on NERSC. The scripts take the first one they find; set `QUIRKML_RUNTIME` to choose (for example `QUIRKML_RUNTIME=apptainer`).
- cvmfs with `sft.cern.ch`, for quirk-gen (MadGraph + Pythia8 from the LCG_109 view), unless your image has Pythia8's python bindings (see `docs/colliderml.md`). lxplus and most HEP clusters have it. If the host isn't el9, the scripts enter an el9 image through shifter (NERSC) or Apptainer.
- `git` and `python3` (standard library only), and about 15 GB of disk for the work directory.
- For reading the output: python with `numpy`, `pandas` and `pyarrow`.

Work in a directory on a local or scratch file system with that much space (see "Site notes" at the end for NERSC).

## 1. Clone and set up (once)

```bash
git clone https://github.com/murnanedaniel/quirkml.git
bash quirkml/colliderml/setup.sh quirks
```

`setup.sh` puts into the work directory (`quirks/` here):

- `colliderml-production/`: the ColliderML stage scripts, branch `quirk/pilot-integration` of [colliderml-production](https://github.com/OpenDataDetector/colliderml-production) at a fixed commit;
- the image `ghcr.io/opendatadetector/sw:pr-8` (about 6 GB; with Apptainer as `sw-pr-8.sif`, otherwise in the runtime's own store);
- `plugin/libQuirkDDG4.so`, the quirk plugin built inside that image;
- `cache/`: Geant4 data files (2-4 GB, downloaded the first time the image environment is set up).

It took 1-4 minutes in our tests, plus the image and Geant4 data downloads. Rerunning it keeps what is already there.

## 2. Describe the sample (one file)

```bash
cp quirkml/colliderml/sample.yaml my_sample.yaml
```

`work_dir: quirks` already matches the command above (a relative path is relative to the settings file). Set the physics point: `mass_gev`, `lambda_ev` (the string scale), `mu` (0 for no pileup) and `events`. Then:

```bash
python3 quirkml/colliderml/make_sample.py my_sample.yaml
```

This writes the stage configs to `quirks/configs/<dataset>/` (the dataset is named e.g. `m400_L1000_mu0`), creates the run directory `quirks/runs/<dataset>/`, and prints the next two commands.

## 3. Generate the quirk events

```bash
bash quirkml/gen/run_quirkgen.sh quirks/configs/m400_L1000_mu0
```

This runs [quirk-gen](https://github.com/burzynski-lab/quirk-gen) at a fixed commit in `quirks/quirkgen/` and copies `events.hepmc` and `events.json` into the run directory. It also writes `events_hepmc2.hepmc`, the same events as HepMC2: the image's HepMC3 (3.3) can't read quirk-gen's HepMC3 3.03 files and would see empty events without an error. 20 events took 2-3 minutes.

quirk-gen's `--lambda` is only recorded in `events.json`. The string acts only in the simulation, so the events of one mass serve every `lambda_ev`.

## 4. Run the ColliderML stages

```bash
bash quirkml/colliderml/run_sample.sh quirks/configs/m400_L1000_mu0
```

Run it where you have 8 cores for up to an hour: a workstation, or an interactive or batch job on a cluster.

`run_sample.sh` runs three stages in the image, checks that each wrote its output before starting the next, and stops with a `STOP:` line naming what is missing. Logs are in `quirks/configs/<dataset>/logs/`. To restart from stage `n`, add `n` after the config directory.

| Stage | What it does |
|---|---|
| 1 | beam spot (Gaussian: 12.5 um in x and y, 55.5 mm in z); for `mu > 0` also Pythia8 SoftQCD pileup, merged in |
| 2 | Geant4 on the ODD v6 tracker with QuirkDDG4 (one core: the two quirks share one string object) |
| 3 | ACTS geometric digitisation, written as parquet |

Times on one node of a CPU cluster (podman-hpc), m = 400 GeV, Lambda = 1 keV: mu = 0 with 20 events took 3.7 minutes in all (simulation 2.9 minutes); mu = 200 with 5 events took 21 minutes. Each event had two quirks with about 27 hits from them; at mu = 200 an event has about 200,000 tracker hits.

## 5. Read it

The parquet tables are in `quirks/runs/<dataset>/` (`tracker_hits/`, `tracker_simhits/`, `particles/`, ...):

```python
import sys; sys.path.insert(0, "quirkml/ml")
from quirk_loader import QuirkDataset
ds = QuirkDataset("quirks/runs/m400_L1000_mu0")
ev = ds.event(ds.event_ids[0])     # hits, particles, truth_edges
```

`docs/ml_baselines.md` describes the tables and what is usable for quirks. In short: use `tracker_hits` and `tracker_simhits`; the `tracks` tables come from helix fits and are not meaningful for quirks.

## Things that go wrong

- **No container runtime found.** Install one, or set `QUIRKML_RUNTIME` to its full path.
- **Apptainer fills your home directory.** The scripts put Apptainer's cache in `<work_dir>/cache/apptainer`; if you run Apptainer by hand, set `APPTAINER_CACHEDIR` yourself.
- **Changing the mass without rerunning step 3.** `run_sample.sh` compares `events.json` with the settings file and stops.
- **Many events, or a high string tension.** Simulation is single-threaded and its cost grows with the tension: about 1-3 minutes per pair-only event at Lambda = 10-30 keV, and at 100 keV one event didn't finish in 20 minutes (`grid/README.md`). Start with tens of events.

## Site notes

- **NERSC (Perlmutter).** podman-hpc is found first. Work on `$PSCRATCH` (MadGraph fails on CFS, and home is small); it is purged after 8 weeks without access, so copy the parquet you want to keep. quirk-gen enters the LCG view through shifter. Run step 4 in a compute allocation, e.g. `salloc -A <account> -C cpu -q shared_interactive -t 60 -c 8 srun -n 1 -c 8 bash quirkml/colliderml/run_sample.sh ...`.
- **lxplus and other el9 hosts with cvmfs.** quirk-gen uses the LCG view directly. For the image, use Apptainer (`QUIRKML_RUNTIME=apptainer`); keep the work directory off AFS home, which is too small.

