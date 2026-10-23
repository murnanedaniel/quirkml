# tracking_core

Shared contract between baselines and the lego framework in the ML
Tracking Review monorepo. Tiny on purpose: **no torch, no lightning**.
Anything you'd need to consume ColliderML data or score predictions in
a form both sides agree on lives here.

## What's inside

- `schema` — the on-disk predictions contract (`predictions.parquet` +
  `meta.json`); see [`docs/predictions_schema.md`](../../docs/predictions_schema.md).
- `io` — ColliderML loader with deterministic per-event train/val/test
  splits.
- `eval` — pyacts-backed scorer (`score(predictions_dir, …) → dict[metric, value]`).
  Requires the `eval` extra: `uv pip install -e packages/tracking_core[eval]`.
- `physics` — kinematic helpers (`qop`, `theta`, `pt`, `eta`) that broadcast
  over numpy or torch.

## Why a separate package

Baseline Docker images (`baselines/<name>/`) `pip install tracking_core`
to read events and write predictions in the canonical format. They get
~200 lines of Python, polars, and pyarrow — and crucially **never have
to install our torch / lightning stack** just to talk to the rest of
the monorepo.
