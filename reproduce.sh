#!/usr/bin/env bash
set -euo pipefail

export MPLCONFIGDIR="${TMPDIR:-/tmp}/water-workshop-matplotlib"
export SOURCE_DATE_EPOCH="${SOURCE_DATE_EPOCH:-0}"
python paper/build_tables.py
python paper/build_figures.py
python paper/plot_density_times_viscosity.py
python paper/plot_transport_spacing_sensitivity.py \
  --states 0.5=md_simulations/data/transport_sensitivity/states_0p5fs.csv \
  --states 1=md_simulations/data/transport_sensitivity/states_1fs.csv \
  --states 2=md_simulations/data/transport_sensitivity/states_2fs.csv \
  --states 5=md_simulations/data/transport_sensitivity/states_5fs.csv \
  --output-dir paper/figures/transport_sensitivity
python tools/validate_submission.py
