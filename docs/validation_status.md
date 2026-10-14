# Validation status of the quirk transport

Two codes move quirk pairs in this repo:

- **QuirkDDG4** (`sim/QuirkDDG4`): Athena's Geant4 quirk transport. Each quirk follows hyperbolic arcs. The string is a queue of 4-vectors pushed by the partner, so the force a quirk feels is delayed by light travel along the string. Step size is capped by `MaxBoost`.
- **quirk_sim** (`quirk_sim/`): a straight string whose two ends are taken at the same lab time, with the Nambu-Goto endpoint force plus the Lorentz force, integrated with RK4 in lab time.

## Exact reference: no field, no material

With no field and no energy loss, the pair rest frame is inertial and the two quirks move back and forth along one line. Each feels a constant force sigma towards the other, so the motion has a closed form. Boosted to the lab, this gives an exact trajectory to compare both codes against (`validation/fieldoff/closed_form.py`).

Two pair-only events at m = 400 GeV (Lambda = 0.5 keV and 1 keV), quirk EM processes off, B = 0. Max distance to the exact trajectory (mm):

| Run | t < 2 ns | t < 10 ns | t < 40 ns |
|---|---|---|---|
| 1 keV, Geant4, empty world | 0.002 | 0.03 | 0.50 |
| 1 keV, Geant4, full ODD volumes | 0.002 | 2.7 | 23.6 |
| 1 keV, quirk_sim (step 0.02 mm) | 0.003 | 0.23 | 2.3 |
| 0.5 keV, Geant4, empty world | 0.002 | 0.015 | exits at 8.5 ns |
| 0.5 keV, Geant4, full ODD volumes | 0.014 | 0.13 | exits at 8.5 ns |
| 0.5 keV, quirk_sim (step 0.005 mm) | 0.001 | 0.007 | exits at 8.5 ns |

The Geant4 numbers in the empty world don't change between `MaxBoost` 1e-2 and 1e-3.

## What this shows

1. Both codes reproduce the exact motion when there are no detector volumes.
2. In the full ODD geometry, Geant4 drifts by 24 mm at 1 keV with no physics process acting on the quirks. The quirk energy changes by GeV over 30 ns there, against 5 MeV in the empty world (`validation/fieldoff/energy_summary.txt`). The earlier Geant4-vs-quirk_sim gap with the field on (25 mm at 1 keV, `validation/trajcmp/`) is about the same size, so most of it comes from this.
3. quirk_sim at 1 keV is 1-2 mm off by 40 ns and doesn't improve steadily with a smaller step. With no field, the two quirks pass through the same point at each of about 75 crossings, where the string direction is undefined. With the field on they don't exactly collide, so this may be specific to the test.

## Not yet known

- Which part of the Geant4 port breaks energy conservation at volume boundaries. A candidate is the branch in `HyperbolaStepper::Update` that advances momentum and string state while the position stays fixed. The port is verbatim from Athena, so Athena may share the behaviour. Not checked.
- How well each code does with the field on. There is no closed form then. A Geant4 run in an empty world with the field on would isolate the field part.

The scripts in `validation/` record how these numbers were made. They still point at the original working area (`quirk_erc`) on Perlmutter.
