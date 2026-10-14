# Reconstruction starting points

Status of each starting point, as of 2026-10-14. "Runs" means we ran it on the ColliderML pilot parquet (`colliderml/pilot_*`).

## Data access

| Code | What it does | Status |
|---|---|---|
| `ml/quirk_loader.py` | ColliderML parquet -> per-event `hits`, `particles`, `truth_edges` | runs |
| `ml/to_quirk_tracking.py` | writes the per-event layout read by [quirk-tracking](https://github.com/burzynski-lab/quirk-tracking) | runs; not yet loaded by quirk-tracking itself |

Things to know about the data (checked on the pilot):

- Hit times. ColliderML's geometric digitisation writes no reconstructed time. The loader gives each hit the true time of its sim hit (`simhit_ids` -> `tracker_simhits.true_time`).
- Truth edges. They follow each quirk's hits in time order. Radial order, the usual choice for helices, fails for quirks because their tracks turn back inwards.
- Positions. Pixel hits (ODD volumes 16-18) sit 10-15 um from their sim hit, short strips (23-25) about 0.1 mm, and long strips (28-30) 30-40 mm. Long-strip positions are only nominal along the strip, so treat them as lines, not points.
- Truth record. `particles` keeps one entry per final-state quirk, with `pdg_id` +/-10000100 for quirk-gen samples. The simulation output (edm4hep) also contains Pythia's intermediate quirk copies.
- Tracks. The ACTS `tracks` and `truth_tracks` tables come from helix fits with a pion hypothesis and are not meaningful for quirks.

## ML

- **quirk-tracking** (Jackson Burzynski's group; MaskFormer tracking with a GPL-3.0 hepattn fork; no licence on the repo itself). It trains on ATLAS-internal data today. `ml/to_quirk_tracking.py` writes our samples in its format. Its configuration expects ATLAS cluster-shape features, which ColliderML doesn't have: the writer fills them with zeros, and dropping them from its config is the better fix. Its own notes list a plane-normal target that never trained and duplicate predictions in about half the events.
- **GNN tracking (ACORN).** Upstream ACORN has a ColliderML reader, but its truth edges are ordered by radius. Makayla Vessella's ACORN fork has time-ordered quirk truth for ATLAS input; that logic, moved onto the ColliderML reader (or replaced by `quirk_loader.truth_edges`), is the starting point. Not done yet.
- **2024 quirk GNN** (Sha, Murnane et al., arXiv:2410.00269). The code exists on Perlmutter as an unversioned Exa.TrkX copy, with ITk and LRT configs mixed in. We haven't ported it.

## Non-ML

| Method | Code | Status |
|---|---|---|
| Coupled two-quirk string fit (mass, tension, momenta) | `fit/q2.py`, `fit/q2_kernel.py` | runs on the earlier pilot's compact hits; not yet on ColliderML parquet |
| Two-helix and 1/r^2 alternatives | `fit/q2_alt.py` | same |
| Road search for hit assignment in pileup | `fit/road_kernel.py` | same |
| Coplanar-hit selection (Knapen et al., arXiv:1708.02243) | - | to write |
| Ionisation (dE/dx) selection (Li et al., arXiv:1911.02223) | - | to write; ColliderML tracker hits have no charge information, but `tracker_simhits.dE` does |

The fit code uses quirk_sim's straight-string model without energy loss. On Geant4 hits it carries a model error of a few mm (field) up to about 25 mm (ionisation, 1 keV, 40 ns); see `docs/validation_status.md`.
