> Archived 700 ps workshop snapshot. These bulk results are superseded. [Use the final release](../../README.md). Commands below run from the repository root.

# Paper-output builders

This directory contains only manuscript assembly and generated figures:

- `build_tables.py`: parameter and BEC comparison tables;
- `build_figures.py`: Figures 1--3;
- `plot_density_times_viscosity.py`: density multiplied by viscosity;
- `plot_transport_spacing_sensitivity.py`: 0.5/1/2/5 fs correlation-grid
  sensitivity;
- `figures/`: committed PNG, PDF, and supplementary outputs.

Scientific calculations live beside their source data under
`../model_training/analysis/` and `../md_simulations/analysis/`. Run
`bash reproduce.sh` from the repository root to rebuild every paper output.
