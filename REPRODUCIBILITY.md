# Reproducibility contract

## What is reproduced from this archive alone

`bash reproduce.sh` rebuilds all manuscript tables and Figures 1--3 from the
committed numerical source data. It also rebuilds the density--viscosity
product and the 0.5/1/2/5 fs correlation-grid sensitivity analysis. The
validator checks split hashes, expected record counts, public model names,
table coverage, response-data dimensions, and generated output presence.

The plotting inputs are compact derivatives of much larger simulation
trajectories. Keeping them makes the numerical paper claims auditable inside a
single sub-100-MB workshop artifact.

## What requires separately supplied inputs

Re-running checkpoint-dependent evaluation, forces, response tensors, or MD
requires the files listed in `models/manifest.json`; retraining an adaptation
requires its corresponding released foundation. They are omitted pending
explicit redistribution authorization, as documented in `MODEL_ACCESS.md`.
Raw trajectories are also not included: the 72 one-nanosecond NPT lanes and
72 700 ps NVT lanes with native 0.5 fs velocity and stress records are too
large for a code supplement.
The exact generation and analysis programs are included, so authorized
checkpoints can recreate those trajectories from
`md_simulations/data/initial_configuration/liquid-64.xyz`.

This boundary is intentional and testable:

1. analysis-only reproduction must succeed in a fresh directory;
2. every unavailable model is named and checksum-identified;
3. every production command has a `--dry-run` path;
4. all expensive-stage settings live in `configs/study.yaml`;
5. NVT analysis must use exactly the configured 700 ps endpoint.

## Numerical definitions

- Figure 1: held-out parent validation families; mean and sample SD over three
  training seeds; adapted checkpoints at update 123,200.
- Figure 2: three NPT replicas; density/RDF statistics over 500--1,000 ps.
- Figure 3: three NVT replicas; native 0.5 fs records; 5--10 ps running-integral
  window; complete 700 ps trajectory.
- Corrected diffusion: lane-wise Yeh--Hummer correction using the same lane's
  viscosity and box length.
- Table 2: population SD across atoms in one frame, not uncertainty over
  independent liquid configurations.

After re-running all three training seeds, pass their `evaluation.csv` files
plus one baseline evaluation for each foundation to
`model_training/analysis/aggregate_force_energy.py`. That command regenerates the
mean/SD and deployed-seed CSVs consumed by Figure 1.

## Clean-room command

From an unpacked copy:

```bash
conda env create -f environment/environment.yml
conda run -n water-deployment-analysis bash reproduce.sh
sha256sum -c SHA256SUMS
```

The last command verifies the released archive itself and should be run before
rebuilding, because generated PDF/PNG metadata can vary slightly by platform.
