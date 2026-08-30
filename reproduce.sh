#!/usr/bin/env bash
set -euo pipefail

export MPLCONFIGDIR="${TMPDIR:-/tmp}/water-workshop-matplotlib"
export SOURCE_DATE_EPOCH="${SOURCE_DATE_EPOCH:-0}"
python analysis/build_paper_tables.py
python analysis/build_paper_figures.py
python analysis/plot_density_times_viscosity.py
python analysis/plot_transport_spacing_sensitivity.py \
  --states 0.5=data/transport_sensitivity/states_0p5fs.csv \
  --states 1=data/transport_sensitivity/states_1fs.csv \
  --states 2=data/transport_sensitivity/states_2fs.csv \
  --states 5=data/transport_sensitivity/states_5fs.csv \
  --output-dir figures/transport_sensitivity
python tools/validate_submission.py
