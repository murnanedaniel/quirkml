# Vendored hepattn framework

- Upstream: https://github.com/samvanstroud/hepattn
- Commit: `1df05ccb00c1a4e5a22a7b76f6182c955c2d9def`
- Vendored: 2026-07-16
- Rename mapping: Python distribution and import package `hepattn` → `ml_tracking`; `src/hepattn` → `packages/ml_tracking/src/ml_tracking`; `tests` → `packages/ml_tracking/tests`; `src/lap1015` → `packages/ml_tracking/src/lap1015`.

The upstream `data/` directory was intentionally skipped. It contains 10 test-data files totalling 16,020,644 bytes, including large compressed TrackML events and a ROOT fixture. Tests that need these absent fixtures are marked `requiresdata` upstream.

The `lap1015` sources are retained, but the main package uses a pure-Python build so default installation does not compile or install the extension. The matcher preserves upstream's guarded import and falls back to SciPy when `lap1015` is unavailable. To build the optional solver explicitly, install `src/lap1015` as a separate local project.

Modifications are tracked in this repo's history; upstream patches are ported via `scripts/port_upstream.sh`.
