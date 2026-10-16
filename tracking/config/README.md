# config/

Copied on 2026-10-16 from ML-Tracking-Review/config (MIT, Daniel Murnane). Inputs of the ACTS evaluator
(tracking_core/eval.py, evaluate.py), read with pyacts 48.

| File | What |
|---|---|
| `odd.json.zst` | ODD gen3 tracking geometry (JSON export, ACTS main `b3dabcdbea`, 2026-09-28), the format pyacts >= 48.0.0 reads. |
| `material_map.json.zst` | gen3 material map (used by the Kalman fit only). |
| `geoid_map.csv.zst` | ColliderML (gen1) -> gen3 geometry-id map. |
