# `quirk_sim` — toy infracolor-string quirk trajectory simulator

A standalone, NumPy/Numba reimplementation of the quirk-pair trajectory
generator behind *"Learning to Reconstruct Quirky Tracks"* (Sha, Murnane et
al., [arXiv:2410.00269](https://arxiv.org/abs/2410.00269)).  It recreates the
physics of the ATLAS Athena `G4Extensions/Quirks` full-simulation generator —
without Geant4 — and is validated **per event** against the reference truth
data in `data/For_Daniel/`.

## Physics

A quirk and its antiquirk are pair-produced and joined by an unbreakable
**infracolor string** of constant tension

    sigma = Lambda^2          (confinement scale Lambda; macroscopic regime)

Because the tension is constant (separation-independent, like quark
confinement) the pair **oscillates** while its centre of mass drifts outward,
giving non-helical tracks that cross some tracker layers several times and —
crucially — make *both* partners travel along the **total-momentum (COM)**
direction rather than their individual momenta.  This co-moving, braided
signature is exactly what the toy reproduces.

The force on each quirk is the relativistic Nambu-Goto straight-string
endpoint force (arXiv:2002.07503) plus the solenoid Lorentz force:

    dp_i/dt = sigma [ sqrt(1 - v_perp^2) u + (v_par / sqrt(1 - v_perp^2)) v_perp ]
              + q_i * kappa * (v_i x B)
    dr_i/dt = v_i = p_i / E_i

with `u` the unit vector from quirk `i` toward its partner, `v_par = v.u`,
`v_perp = v - v_par u`, `kappa = 0.2998 MeV/(mm*T)`.  The transverse-velocity
correction is essential: the rigid `F = sigma u` approximation over-oscillates
(see `string_mode` below).  Integrated with RK4 in lab time (`c = 1`).

Internal units: **mm, MeV, tesla, e** (the For_Daniel data is cm/GeV; see
`units.py`).

## Detector

Cylindrical barrel reconstructed directly from the truth hits: 31 layers at
r = 3.1, 5.1, 7.1, 9.1, 11.1 cm then 12, 14, … 62 cm; uniform **2 T** solenoid
along +z.  A hit is recorded at every layer-radius crossing, so re-crossings
(the multi-hit-per-layer quirk signature) appear naturally (`detector.py`).

## Validation result

Running every event in `mass_1000_Lambda_1000` (132 events, 262 quirks) at the
**theory** tension `sigma = Lambda^2 / hbar*c = 5068 MeV/mm` for Λ = 1000 eV,
with **no per-event tuning**:

| metric | value |
|---|---|
| median per-hit spatial residual | **0.18 cm** (1.8 mm) |
| 95th-percentile hit residual | 1.16 cm |
| hit-count ratio n_sim / n_data | 1.05 (10–90%: 1.00–1.11) |
| quirks with median residual < 1 cm | 98.9% (100% < 2 cm) |

i.e. the toy reproduces the recorded truth hits to well within the 2 cm layer
spacing, event by event, from first principles.  The agreement at the *a
priori* `Lambda^2/hbar*c` tension independently confirms the Λ→tension mapping,
the 2 T field, and the layer geometry.

## Usage

```bash
# from the QuirkTracking repo root, in the acorn-test conda env
conda run -n acorn-test python -m quirk_sim.validate          # full validation + summary plot
conda run -n acorn-test python -m quirk_sim.plot_events --events 4 8 7   # overlay figures
conda run -n acorn-test python -m quirk_sim.fit_sigma         # per-event best-fit tension vs theory
```

```python
from quirk_sim import QuirkPair, Barrel
from quirk_sim.units import string_tension_from_lambda_eV
sim = QuirkPair(mass_mev=1e6, sigma_mev_per_mm=string_tension_from_lambda_eV(1000))
res = sim.integrate(r1, p1, q1, r2, p2, q2, dt=1.0, r_stop_mm=Barrel().r_max*1.2)
hits = Barrel().crossings(res["traj1"])
```

## Files

| file | role |
|---|---|
| `units.py` | unit system + `sigma = Lambda^2/hbar*c` conversion |
| `pair.py` | coupled quirk-pair RK4 integrator (`QuirkPair`) |
| `_kernel.py` | Numba-jitted RK4 inner loop (≈0.9 ms / event) |
| `detector.py` | cylindrical `Barrel` + trajectory→hit crossings |
| `hyperbola.py` | faithful port of the Athena `StringVector` reflect primitive |
| `dataio.py` | loading of For_Daniel truth events |
| `validate.py` | per-event validation + summary plot |
| `plot_events.py` | data-vs-toy trajectory overlays |
| `fit_sigma.py` | per-event best-fit tension (cross-check of the Λ mapping) |

`string_mode`: 0 = rigid straight string, 1 = relativistic (default, used for
validation), 2 = tension-reduction only.

The exact Geant4 hyperbola-stepper generator (which drives the same string
physics through Geant4's stacking, for full detector simulation) is ported
separately in `../QuirkDDG4/`.
