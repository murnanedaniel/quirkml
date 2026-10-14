# Validation status of the quirk transport

Two codes move quirk pairs in this repo:

- **QuirkDDG4** (`sim/QuirkDDG4`): Athena's Geant4 quirk transport. Each quirk follows hyperbolic arcs. The string is a queue of 4-vectors pushed by the partner, so the force a quirk feels is delayed by light travel along the string. Step size is capped by `MaxBoost`.
- **quirk_sim** (`quirk_sim/`): a straight string whose two ends are taken at the same lab time, with the Nambu-Goto endpoint force plus the Lorentz force, integrated with RK4 in lab time. It has no energy loss in material.

All numbers below are for two pair-only events at m = 400 GeV (Lambda = 0.5 keV event 1 and 1 keV event 3), on full ODD unless stated. They give the largest 3D distance between trajectories at equal lab time, inside r < 1150 mm and |z| < 3000 mm. The 0.5 keV pair leaves that region at 8.5 ns.

## 1. No field, no energy loss: both codes match the exact motion

With no field and no energy loss, the pair moves back and forth along one line in its rest frame, under a constant force sigma. That motion has a closed form, which we boost to the lab (`validation/fieldoff/closed_form.py`).

| Run | 1 keV, t < 10 ns | 1 keV, t < 40 ns | 0.5 keV, t < 8.5 ns |
|---|---|---|---|
| Geant4, ionisation off (`--no-em --no-ionisation`) | 0.002 mm | 0.016 mm | 0.011 mm |
| quirk_sim, step 0.005 mm | 0.46 mm | 1.2 mm | 0.007 mm |

The 1 keV quirk_sim number doesn't improve steadily with a smaller step. With no field, the two quirks pass through the same point at each of about 75 crossings, where the string direction is undefined.

## 2. Ionisation is most of the earlier Geant4-vs-quirk_sim gap

`--no-em` turns off multiple scattering, bremsstrahlung and pair production for the quirks, but keeps ionisation, because ionisation makes the tracker hits. With ionisation on, the quirks lose energy and emit delta rays in the silicon, which changes the oscillation:

| Run (B = 0) | 1 keV, t < 40 ns | 0.5 keV, t < 8.5 ns |
|---|---|---|
| Geant4, ionisation off, vs exact | 0.016 mm | 0.011 mm |
| Geant4, ionisation on, vs exact | 23.6 mm | 0.14 mm |

This is real detector physics, and quirk_sim leaves it out. The earlier gap of about 25 mm at 1 keV (`validation/trajcmp/`) is mostly this.

## 3. With the 3 T field, the two string models differ by a few mm

Ionisation off, field on:

| Comparison | 1 keV, t < 10 / 20 / 40 ns | 0.5 keV, t < 8.5 ns |
|---|---|---|
| Geant4 vs converged quirk_sim (step 0.00125 mm) | 0.25 / 1.0 / 3.7 mm | 2.0 mm |
| Geant4 MaxBoost 1e-2 vs 1e-4 | 0.004 / 0.03 / 0.07 mm | 1.3 mm (1e-3 vs 1e-4: 0.10 mm) |
| quirk_sim step 0.005 vs 0.00125 mm | 0.07 / 0.07 / 0.8 mm | 0.004 mm |

Both codes are converged in step size, and they agree without the field. So the 2-4 mm with the field comes from the two string models: delayed string vs straight string at equal lab time. Which one is closer to Nambu-Goto dynamics with a field is open. We posit the delayed string is closer, because quirk_sim assumes the string stays straight. That isn't tested.

At 0.5 keV, Geant4's default `MaxBoost` of 1e-2 is not converged (1.3 mm); 1e-3 is.

## What this means for users

- Geant4 (QuirkDDG4) is the reference for detector-level samples. Use `MaxBoost` 1e-3 or smaller for slack strings, since 1e-2 is off by 1.3 mm at 0.5 keV.
- quirk_sim has no energy loss. Using it as a fit model leaves an error of up to about 25 mm by 40 ns at 1 keV for this event, mostly from ionisation and partly from the string model.
- quirk_sim's 0.02 mm default step is not converged at 1 keV (0.4 mm at 10 ns, 2.5 mm at 40 ns). The fit kernel in `fit/q2_kernel.py` uses its own adaptive step and was checked separately.

The scripts in `validation/` record how these numbers were made. They still point at the original working area (`quirk_erc`) on Perlmutter.
