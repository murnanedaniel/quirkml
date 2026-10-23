# Predictions schema — the on-disk contract

Both `packages/ml_tracking/` (the lego framework) and every
`baselines/<name>/` Docker image emit predictions in the form below.
`tracking_core.eval.score(...)` reads exactly this form. Nothing else
crosses the boundary — no Python imports, no shared in-memory tensors.

## Per-run layout

```
<outdir>/
  predictions.parquet   # one row per hit
  meta.json             # run-level sidecar
```

## `predictions.parquet`

| column                | type      | meaning                                                |
|-----------------------|-----------|--------------------------------------------------------|
| `event_id`            | `uint64`  | groups hits into one event                             |
| `hit_id`              | `uint64`  | index of the hit within its event's per-hit list cols  |
| `predicted_track_id`  | `int64`   | `-1` ⇒ not assigned to any track                       |
| `score`               | `float32` | optional per-hit confidence; `NaN` if unused           |

### `hit_id` convention

ColliderML's tracker_hits / calo_hits tables don't ship an explicit
`hit_id` column; hits are identified by their position in their
event's per-hit list columns (`x[i]`, `y[i]`, …). The contract uses
that index directly: `hit_id = i` for the i-th element of any per-hit
list in event `event_id`. The `tracks` table's `hit_ids: list<list<u32>>`
column already follows this convention.

### `predicted_track_id`

A non-negative integer assigning the hit to a predicted track. Track
ids are unique within `event_id` but need not be unique across events.
`-1` means the model did not assign this hit to any track.

### `score`

Per-hit confidence on a `[0, 1]`-ish scale (the exact interpretation is
up to the implementation). `NaN` if the implementation doesn't produce
hit-level scores.

## `meta.json`

```json
{
  "run_id": "20260508T143012Z",
  "implementation": "baselines:trackformers",
  "config": { "config_id": "ttbar_pu0", "object_level": "tracker_hits", "...": "..." },
  "wrote_at": "2026-05-08T14:30:12+00:00"
}
```

| field            | meaning                                                       |
|------------------|---------------------------------------------------------------|
| `run_id`         | unique per run; W&B run id, ablation-cell tag, or timestamp.  |
| `implementation` | `"ml_tracking:<pipeline>"` or `"baselines:<name>"`. Used by   |
|                  | eval / aggregation to attribute metrics back to a method.     |
| `config`         | resolved config used for this run; opaque to the contract     |
|                  | but written verbatim into reports.                            |
| `wrote_at`       | ISO-8601 UTC timestamp.                                       |

## Helpers

`tracking_core.schema` exposes the exact pair:

```python
from tracking_core.schema import RunMeta, write_predictions, read_predictions

write_predictions(df, outdir, RunMeta(...))   # writes both files
df, meta = read_predictions(outdir)           # round-trips
```

`write_predictions` validates that `df` has the four required columns
and selects exactly those (extras are dropped).

## Scoring

```python
from tracking_core.eval import score
metrics = score(outdir, hit_shard, particle_shard, surface_map, odd_json)
# {"efficiency_total": ..., "fake_rate": ..., "efficiency_pt_gt_1GeV": ..., ...}
```

The eval routine joins predictions to ColliderML truth by
`(event_id, hit_id)`, then hands the pair to pyacts to compute
ATLAS-pedigree tracking metrics. Exact metric set is whatever pyacts
surfaces for hit-association tracking.
