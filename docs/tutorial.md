# Tutorial: a quirk sample in ColliderML parquet

This takes you from nothing to ColliderML-format parquet for one quirk sample: quirk-gen events, beam spot and optional pileup, Geant4 with the quirk string (QuirkDDG4) in the ODD tracker, digitisation, and packaging. The calorimeters are not simulated.

You need a NERSC account in the m4958 project, and read access to this repo on GitHub. The commands below were run as written on 2026-10-14 from fresh clones (times are from that run).

## 1. Clone and set up (once, about 4 minutes, login node)

Everything the containers read or write has to be under `/global/cfs/cdirs/m4958` or your `$PSCRATCH`, because those are the only host paths mounted inside them. So clone and work under m4958:

```bash
mkdir -p /global/cfs/cdirs/m4958/usr/$USER && cd /global/cfs/cdirs/m4958/usr/$USER
git clone https://github.com/murnanedaniel/quirkml.git
bash quirkml/colliderml/setup.sh /global/cfs/cdirs/m4958/usr/$USER/quirks
```

`setup.sh` puts two things in the work directory (`quirks/` here), pulls the sim image and builds the plugin:

- `colliderml-production/`: the ColliderML pipeline, branch `quirk/pilot-integration` of [colliderml-production](https://github.com/OpenDataDetector/colliderml-production) at a fixed commit, with an `env_setup.yaml` for tracker-only stages;
- `venv/`: a python for the pipeline driver (`run_stage.py`) and for reading the output;
- the sim image `ghcr.io/opendatadetector/sw:pr-8` in your podman-hpc store (pulled only if missing);
- `quirkml/sim/QuirkDDG4/build-pr-8/lib/libQuirkDDG4.so`, built inside that image.

Rerunning `setup.sh` keeps what is already there.

## 2. Describe the sample (one file)

```bash
cp quirkml/colliderml/sample.yaml my_sample.yaml
```

`work_dir` is already `/global/cfs/cdirs/m4958/usr/$USER/quirks`; change it if you gave `setup.sh` another directory. Then set the physics point: `mass_gev`, `lambda_ev` (the string scale), `mu` (0 for no pileup) and `events`. Then:

```bash
quirks/venv/bin/python quirkml/colliderml/make_sample.py my_sample.yaml
```

This writes the stage configs to `quirks/configs/<dataset>/` (dataset is e.g. `m400_L1000_mu0`), creates the run directory, and prints the next two commands.

## 3. Generate the quirk events (login node; 20 events took 1.6 minutes)

```bash
bash quirkml/gen/run_quirkgen.sh quirks/configs/m400_L1000_mu0
```

This runs [quirk-gen](https://github.com/burzynski-lab/quirk-gen) (MadGraph + Pythia8) at a fixed commit, on `$PSCRATCH`, and copies `events.hepmc` and `events.json` into the run directory. It also writes `events_hepmc2.hepmc`, the same events as HepMC2: the sim image cannot read quirk-gen's HepMC3 files (version 3.03) and would otherwise see empty events without an error.

quirk-gen's `--lambda` is only recorded in `events.json`. The string acts only in the simulation, so the events of one mass serve every `lambda_ev`.

## 4. Run the ColliderML stages (compute node)

```bash
salloc -A m4958 -C cpu -q shared_interactive -t 60 -c 8 \
    srun -n 1 -c 8 bash quirkml/colliderml/run_sample.sh quirks/configs/m400_L1000_mu0
```

`run_sample.sh` runs the four stages in order, checks that each wrote its output before starting the next, and stops with a `STOP:` line naming what is missing. Logs are in `quirks/configs/<dataset>/logs/`. To restart from stage `n`, add `n` after the config directory.

| Stage | What it does |
|---|---|
| 1 | beam spot (Gaussian, 12.5 um in x and y, 55.5 mm in z); for `mu > 0` also Pythia8 SoftQCD pileup, merged in |
| 2 | Geant4 on the ODD v6 tracker with QuirkDDG4 (one core: the two quirks share one string object) |
| 3 | ACTS geometric digitisation; writes per-run parquet |
| 4 | packages the run into the released ColliderML layout |

Times from the test run (m = 400 GeV, Lambda = 1 keV, one shared node):

| Sample | Stage 1 | Stage 2 | Stages 3-4 | Total |
|---|---|---|---|---|
| mu = 0, 20 events | 0.7 min | 2.6 min | 1.1 min | 4.4 min |
| mu = 200, 5 events | 1.3 min | 13.6 min | 1.6 min | 16.5 min |

Each event had two quirks with 25-33 hits from them; at mu = 200 an event has about 200,000 tracker hits in all.

The output is in `quirks/runs/quirk/<dataset>/v0/parquet/`. The packaging step reports a failed validation for small samples: it wants track files of at least 0.1 MB. `run_sample.sh` checks the tables directly instead.

## 5. Read it

```python
# quirks/venv/bin/python
import sys; sys.path.insert(0, "quirkml/ml")
from quirk_loader import QuirkDataset
ds = QuirkDataset("quirks/runs/quirk/m400_L1000_mu0/v0/parquet")
ev = ds.event(ds.event_ids[0])     # hits, particles, truth_edges
```

`docs/ml_baselines.md` describes the tables and what is usable for quirks. In short: use `reco/tracker_hits` and `truth/tracker_simhits`; the `tracks` tables come from helix fits and are not meaningful for quirks.

## Things that go wrong

- **A path outside m4958 or `$PSCRATCH`.** The container can't see it. `make_sample.py` and `setup.sh` refuse such paths.
- **Changing the mass without rerunning step 3.** `run_sample.sh` compares `events.json` with the settings file and stops.
- **Many events.** Simulation is single-threaded and its cost grows with the string tension: about 1-3 minutes per pair-only event at Lambda = 10-30 keV, and at 100 keV one event doesn't finish in 20 minutes (`grid/README.md`). Talk to Daniel before running more than a few hundred events, or anything in batch.
- **Geant4 data.** `cache_dir` in the settings file points at a shared copy of the Geant4 data files, readable by the m4958 group. The simulation links to it and doesn't write there.
- **Interactive allocation refused.** The `interactive` queue allows two jobs per user; `shared_interactive` is the other option.
