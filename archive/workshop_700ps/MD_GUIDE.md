> Archived 700 ps workshop snapshot. These bulk results are superseded. [Use the final release](../../README.md). Commands below run from the repository root.

# Molecular-dynamics simulations and liquid analysis

## Data

- `data/initial_configuration/liquid-64.xyz`: exact 64-water configuration
  from Bingqing Cheng's public `cacefit/md-water` example.
- `data/initial_configuration/SOURCE.md`: pinned upstream commits, direct URLs,
  checksum, and lineage from Bingqing Cheng's 300 ps NPT example to this 1 ns
  implementation.
- `data/published_results/`: compact density, RDF, oxygen-diffusion,
  viscosity, running-integral, and stress-decomposition values used in Figures
  2 and 3.
- `data/transport_sensitivity/`: 0.5/1/2/5 fs correlation-grid summaries.

## Simulations

`run_npt.py` runs one model/temperature/replica lane for 1 ns at 1.01325 bar
with a 1 fs step. The four temperatures are 275, 295, 315, and 336 K.

`run_nvt.py` starts from the representative NPT frame and runs 700 ps at fixed
volume with a 0.5 fs step. Oxygen velocities and all five shear-stress channels
are recorded every 0.5 fs. The trajectory is restart-safe across 500, 100, and
100 ps segments.

```bash
python md_simulations/run_npt.py --model maceles-off-g1-g6 --temperature 295 --replica 1
python md_simulations/analysis/density_rdf.py --model maceles-off-g1-g6 --temperature 295 --replica 1
python md_simulations/run_nvt.py --model maceles-off-g1-g6 --temperature 295 --replica 1
```

`run_campaign.py` enumerates the 72 combinations of six models, four
temperatures, and three replicas. Use `--dry-run` on a lane launcher or omit
`--execute` from the campaign launcher to inspect commands without running MD.

## Analysis

- `analysis/density_rdf.py`: density over 500--1,000 ps, intermolecular O--O
  and O--H RDFs, and representative-frame selection for one lane.
- `analysis/aggregate_density_rdf.py`: mean, sample SD, and SEM over three NPT
  replicas.
- `analysis/diffusion.py`: short public entry point for oxygen VACF/MSD
  diffusion, including the lane-wise Yeh--Hummer correction.
- `analysis/viscosity.py`: short public entry point for five-channel
  Green--Kubo shear viscosity.
- `analysis/diffusion_and_viscosity.py`: the shared campaign pass used by both
  short entry points so velocity and stress data are frozen consistently.
- `analysis/internal/transport_estimators.py`: audited VACF, MSD,
  Green--Kubo, uncertainty, and finite-size implementation.
- `analysis/package_transport.py`: compact Figure 3 tables and curves.

Diffusion and viscosity deliberately share one trajectory pass because the
finite-size diffusion correction uses the viscosity from the same lane.
