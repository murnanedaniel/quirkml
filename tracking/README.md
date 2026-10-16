# Quirk tracking pilot

Files:

- `convert_to_r1.py`: packaged quirk dataset -> ColliderML Release 1 layout.
- `r1_data.py`: read or stage an event range of an R1 dataset.
- `oracle_predictions.py`: truth-based predictions for an event range (the reference for model outputs).
- `evaluate.py`: score predictions with the ACTS evaluator.
- `tracking_core/`, `config/`: the ACTS evaluator and its geometry files, copied from ML-Tracking-Review
  (MIT; see `tracking_core/NOTICE` for what changed).

## Evaluation

Python with pyacts 48: `/global/cfs/cdirs/m4958/usr/danieltm/ml_tracking_pilot_2026-09/envs/eval48/bin/python`.
Run on a compute node, not a login node.

```bash
PY=/global/cfs/cdirs/m4958/usr/danieltm/ml_tracking_pilot_2026-09/envs/eval48/bin/python
DATA=$PSCRATCH/quirk_pilots/r1/m400_L1000_mu0

# reference predictions: each hit assigned to its truth particle
$PY tracking/oracle_predictions.py --dataset $DATA --events 0 4000 --out $PSCRATCH/quirk_tracking/pred_oracle

$PY tracking/evaluate.py --predictions-dir $PSCRATCH/quirk_tracking/pred_oracle --dataset $DATA \
    --events 0 4000 --out $PSCRATCH/quirk_tracking/eval_oracle --jobs 64
```

`--events A B` scores events A <= event_id < B. The predictions directory may hold more events than that, but
must hold every event of the range, with one row per hit (`predicted_track_id` -1 when unassigned). `hit_id` is the
position of the hit in its event's lists in `reco/tracker_hits`. Columns: `event_id`, `hit_id`,
`predicted_track_id`, `score` (NaN if unused), plus `meta.json` (see `tracking_core/schema.py`).

`evaluate.py` copies the event range, sorted by event_id, to `$PSCRATCH/quirk_tracking/staging/` (the ACTS reader
reads events by row position, and rows in the R1 files are not in event order), then runs ACTS in `--jobs`
single-threaded passes over consecutive event chunks and adds their histograms. About 1 s per event per pass, and
about 5 GB of memory per pass: keep `--jobs` at 60 or below on one 512 GB Perlmutter CPU node (100 passes ran out
of memory).

Outputs in `--out`: `summary.json` (the numbers below), `acts_summary.json` (all ACTS efficiency histogram sums, in
the tracking_core summary layout), `histograms.pkl` (all ACTS histograms), `per_event.parquet` (quirks selected and
matched per event, from ACTS and from the Python cross-check).

## Metrics

All at the proto-track level: the predicted hit sets as given, before any fit. Matching is ACTS
TrackTruthMatcher with double matching at two working points:

- `very_loose` (default): a track matches its majority particle if at least 50% of the track's hits belong to that
  particle and at least 50% of that particle's hits are on the track;
- `tight`: the same with 90%.

| Key in summary.json | Meaning |
|---|---|
| `quirk_track_efficiency` | Matched quirks / quirks in the quirk selection. The primary figure of merit. |
| `quirk_event_efficiency` | Events where both quirks are matched / events with both quirks in the quirk selection. |
| `technical_efficiency` | Matched / selected particles of the technical selection (quirks included). |
| `technical_sm_efficiency` | The same without quirks, so quirk failures do not enter it. |
| `fake_rate` | Fake prototracks / all prototracks. A fake track's majority particle fails the matching cut. |
| `duplicate_rate` | Duplicate prototracks / all prototracks: extra matched tracks of an already matched particle. |
| `prototracks_per_event` | Mean number of predicted tracks per event. |
| `kf_not_meaningful_for_quirks` | Efficiencies after the ACTS Kalman fit. The fit assumes a helix, so these numbers do not describe quirk tracking. |
| `checks` | Consistency checks (below). |

Selections (efficiency denominators, ACTS ParticleSelector):

- quirk: mass > 10 GeV (`mMin`), charged, primary, at least 3 measurements; no pT, eta or production-vertex cut.
  At mu = 0 only the quirks (|pdg| 10000100, mass 200-400 GeV) pass the mass cut; `checks` confirms the pdg of
  every selected particle.
- technical: pT > 1 GeV, |eta| < 3, charged, primary, production vertex rho < 2 mm and |z| < 200 mm, at least 7
  measurements.
- technical_sm: technical without quirks (ParticleSelector `excludeAbsPdgs` [10000100]).
- all (fake and duplicate rates only): every particle with at least one hit, so that every track with a truth hit is
  matched, duplicate or fake.

The per-event quirk counts are read from the ACTS writers after every event. `checks` holds:

- the ACTS measurement count of each event equals the number of hits in the dataset (event alignment);
- the ACTS quirk selection equals the quirks with at least 3 hits in the dataset, event by event;
- the per-event ACTS counts add up to the inclusive ACTS numbers;
- a Python version of the matching rule gives the same matched quirks in every event.

`evaluate.py` exits with an error if any check fails.

## Validation (m400_L1000_mu0, events 0-3999)

Predictions from `oracle_predictions.py --variant ...`, outputs in `$PSCRATCH/quirk_tracking/validation_m400/`.
7943 quirks pass the quirk selection (57 of 8000 have fewer than 3 hits); 3964 events have both quirks selected.

| Variant | Quirk track eff (very_loose / tight) | Quirk event eff | Technical eff | Fake rate (very_loose / tight) | Duplicate rate (very_loose) |
|---|---|---|---|---|---|
| oracle | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 0 / 0 | 0 |
| quirk_halves_time | 1.000 / 0.000 | 1.000 / 0.000 | 1.000 / 0.899 | 0.0015 / 0.0059 | 0.0015 |
| quirk_drop60 | 0.000 / 0.000 | 0.000 / 0.000 | 0.899 / 0.899 | 0.0029 / 0.0029 | 0 |
| quirk_coinflip | 0.499 / 0.499 | 0.240 / 0.240 | 0.949 / 0.949 | 0 / 0 | 0 |
| unassigned | 0 / 0 | 0 / 0 | 0 / 0 | no tracks | no tracks |

All consistency checks pass in every run. For quirk_coinflip the matched quirks equal the kept quirks event by event
(3961 quirks, 952 of 3964 events with both kept). The technical selection contains quirks (7420 of 73102 selected
particles), so breaking quirk tracks also lowers technical efficiency; `technical_sm_efficiency` excludes them.
Rerun with `technical_sm`: oracle 1.000 (65682/65682); quirk_drop60 1.000 while technical is 0.899.

m200_L774_mu0, oracle: validation range 16000-16999 gives quirk track, quirk event, technical and technical_sm
efficiency 1.000 (1989/1989 quirks, 993/993 events, 17053 and 15231 particles), fake and duplicate rate 0; test range
17000-19999 gives 1.000 for all four (5962/5962, 2973/2973, 52631, 47215), fake and duplicate rate 0.

