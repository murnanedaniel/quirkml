# Cluster-shape direction features (offline)

ACTS does not fill the cluster direction columns of tracker_hits (upstream TODO in Cluster.hpp), so they are all
zero in our parquet. These scripts derive a direction axis per hit from the cluster's cells, with no origin
information and no entry/exit orientation:

1. `redigi_one.sh <run>`: re-digitise one run from its kept edm4hep.root with the production settings plus
   `output_measurements_root: True` (`3_digi_measroot.yaml`), to get the cell lists. Measurements equal production.
2. `build_direction_features.py <block>...`: per 1000-event block, list columns in production hit order:
   `axis_a_*`, `axis_b_*` (the two mirror candidates of the cluster axis in the global frame), `phi_angle_abs`,
   `eta_angle_abs`, `diag_sign` (sign of the cells' loc0-loc1 covariance; 0 = undetermined; resolves the mirror
   ambiguity). NaN for long strips except `phi_angle_abs`. Needs `frames.npy` (surface frames).
3. `check_direction_features.py [A B]`: resolution against the sim-hit direction, and the close-quirk tests.

m200_L774_mu0 (2026-10-23): pixels 6.5 deg median 3D angle to truth, short strips 16.5; `diag_sign` agrees with
truth in 99.8% where defined. Where the two quirks are within 10 mm, the hit's axis picks its own quirk's next hit
over the other quirk's hit in 89% of cases (true direction: 91%). Paths are NERSC/m4958 ones.
