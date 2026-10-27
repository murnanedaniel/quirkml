#!/bin/bash
# Figures reused unchanged from earlier work, copied into figs/. Each line names the script that makes it.
set -e
HERE=$(cd "$(dirname "$0")/.." && pwd)
OLD=/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/docs/talks/2026-10-17/figs
# event displays of the three string regimes, m = 175 GeV: docs/talks/2026-10-17/scripts/fig01_evd_regimes.py
cp $OLD/evd_regimes.png $HERE/figs/evd_regimes.png
# GNN v1/v2 edge-score distributions: docs/talks/2026-10-17/scripts/fig11_gnn_edge_scores.py
cp $OLD/gnn_edge_scores.png $HERE/figs/gnn_edge_scores.png
# timing separability vs hit time resolution: tracking/separability/timing.py
cp /pscratch/sd/d/danieltm/quirk_tracking/separability/frac_separable_vs_sigma_t.png $HERE/figs/timing_separable.png
