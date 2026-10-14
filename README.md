# quirkml

Full simulation of quirk pairs in the OpenDataDetector (ODD), and the route from there to a ColliderML-format dataset and to reconstruction studies, ML and non-ML.

Quirks are heavy charged particles held together by a long, unbreakable string of a new confining force. When sqrt(sigma) is below about 30 keV, the string is macroscopic and the pair leaves tracks that are neither helices nor straight lines. Standard helix-based track finding isn't designed for them.

**Status (2026-10-14): work in progress, local only.** See `ROADMAP.md` for what works today and what's planned before handover.

## How this fits with quirk-gen

We don't generate events here. [quirk-gen](https://github.com/burzynski-lab/quirk-gen) produces the hard process (MadGraph + Pythia8, `p p > qk qk~ j`, 13.6 TeV) and writes HepMC3 with the quirks as stable particles. The string only acts in the detector simulation, so one generation per mass serves every string tension.

This repo takes that HepMC3 and does:

1. string transport in Geant4 on ODD (`sim/`, QuirkDDG4);
2. pileup merging, digitisation and parquet output through the ColliderML production chain (`colliderml/`, `docs/colliderml.md`);
3. reconstruction starting points, ML and non-ML (`ml/`, `fit/`, `docs/ml_baselines.md`).

## Layout

| Path | Contents |
|---|---|
| `sim/QuirkDDG4/` | DDG4 plugin: Athena's quirk string transport ported to DD4hep/ddsim (Geant4 11.3). `build.sh` builds for LCG_107, `build_colliderml.sh` for the ColliderML sim image |
| `sim/run_quirk_ddsim.py`, `sim/run_in_shifter.sh` | stand-alone ddsim driver for the plugin, and a wrapper for the el9 container on Perlmutter |
| `colliderml/` | quirk-gen file converter and worked stage configs for the ColliderML chain (`pilot_m400_L1000_mu0`, `pilot_m400_L1000_mu200`) |
| `grid/` | grid proposal and a generator for per-point stage configs |
| `ml/` | parquet loader with time-ordered quirk truth, writer for quirk-tracking, coplanarity and ionisation baselines |
| `quirk_sim/` | fast quirk-pair model in Python/numba: straight string, both ends at the same lab time, RK4 in lab time, no energy loss |
| `fit/` | coupled two-quirk fit of mass, tension and momenta to hits, with two-helix and 1/r^2 alternatives, and a road search kernel for pileup |
| `gen/` | small Pythia8 Hidden Valley generator for quick tests (quirk-gen is the production generator) |
| `validation/` | Geant4 vs quirk_sim vs exact-motion comparisons |
| `docs/` | `colliderml.md` (the chain, step by step), `validation_status.md`, `ml_baselines.md` |

## Quick start (Perlmutter)

QuirkDDG4 builds and runs inside the el9 container with the LCG_107 view (Geant4 11.3, DD4hep 1.32). Both scripts re-exec themselves into shifter. Paths default to the m4958 project area; override them with `COLLIDERML_SW`, `DD4HEP_DIR` and `QUIRK_LIB`.

```bash
bash sim/QuirkDDG4/build.sh            # -> sim/QuirkDDG4/build/lib/libQuirkDDG4.so

# one mass from quirk-gen; mass and tension come from its events.json
bash sim/run_in_shifter.sh --compact $ODD_XML \
    --input events.hepmc --output quirks_edm4hep.root --events 10 \
    --pdg 10000100 --charge 1 --mass 400 --string-force 5067.73
```

`--string-force` is the tension Lambda^2/hbar c in MeV/mm (Lambda = 1 keV gives 5067.73); quirk-gen writes the same number as `stringforce_mev_per_mm`. quirk-gen's quirk is PDG +10000100 with charge +1, and the plugin gives +PDG the charge `--charge`. The plugin is single-threaded, because the two quirks share one string object.

These commands are for a quick stand-alone run and have not been tried on a quirk-gen file. quirk-gen files need converting first (`colliderml/convert_quirkgen_hepmc.sh`); the converted file did run through the ColliderML chain, which is the route for samples (`docs/colliderml.md`).

## What has been checked

- A 16,000-event pilot (masses 400 and 800 GeV, sqrt(sigma) = 0.5, 1, 2, 4 keV) ran through QuirkDDG4 on full ODD v4. It used `gen/`, not quirk-gen.
- With no field and no energy loss, QuirkDDG4 follows the exact closed-form motion to 0.016 mm over 40 ns (`docs/validation_status.md`).
- QuirkDDG4 builds inside the current ColliderML sim image (`sim/QuirkDDG4/build_colliderml.sh`, sw:pr-8).
- The coupled fit recovers the tension to a few percent on hits from its own model.

## Things to know before using the samples

- `--no-em` keeps ionisation on the quirks, because ionisation makes the tracker hits. `--no-ionisation` also drops it, for validation only, since there are then no hits.
- Ionisation changes the quirk oscillation noticeably: up to 24 mm by 40 ns at Lambda = 1 keV in our test event. quirk_sim leaves it out.
- With the field on, QuirkDDG4 and quirk_sim differ by 2-4 mm even without ionisation, because they model the string differently. Which is closer to the true string dynamics is open.
- Use `--max-boost 1e-3` or smaller for slack strings: the default 1e-2 is off by 1.3 mm at 0.5 keV.

## Credits and licence

Apache-2.0 (`LICENSE`). QuirkDDG4 is a port of the Athena `Simulation/G4Extensions/Quirks` package (Copyright CERN for the benefit of the ATLAS collaboration, Apache-2.0); see `NOTICE`.
