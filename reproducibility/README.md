# 2.4 Software and reproducibility

This section collects the code, configurations, geometries, reference labels,
model identities, and compact results needed to reproduce the reported figures,
including the **corrected 2 ns results for all 168 trajectories**.

The corrected LES simulations use native single-cell strain. They supersede
the historical LES bulk results elsewhere in this repository. The original
2.4-fold viscosity split and persistent-mode interpretation are withdrawn.
The corrected G1–G6 LES/OFF viscosity ratio at 295 K is approximately 1.207.
The original molecular validation results are unchanged.

## Materials required by Section 2.4

| Required material | Included location |
|---|---|
| Water-cluster geometries and adaptation reference labels | [G1–G2 and G1–G6 train/validation/test data](../model_training/data/molecular_clusters_1000/), [dataset configuration](../model_training/configs/dataset.yaml), [data card](../DATA_CARD.md) |
| Pinned software environment | [Environment and version provenance](environment/README.md), [analysis requirements](environment/requirements-analysis.txt), [production requirements](environment/requirements-production.txt), [exact corrected MD source snapshot](md/vendor/) |
| Training and MD protocols | [Adaptation settings](../model_training/configs/adaptation.yaml), [training study configuration](../configs/study.yaml), [corrected 2 ns protocol](configs/protocol_2ns.json), [168-lane manifest](configs/lanes.csv) |
| Training scripts | [Training launcher](../model_training/train.py), [G1–G2 implementation](../model_training/internal/training/g1-g2/), [G1–G6 implementation](../model_training/internal/training/g1-g6/) |
| Adapter fusion | [G1–G2 export/fusion](../model_training/internal/training/g1-g2/export_model.py), [G1–G6 evaluation/fusion](../model_training/internal/training/g1-g6/evaluate.py) |
| Corrected molecular dynamics | [MD instructions](md/README.md), [lane launcher](md/run_lane.py), [NPT runner](md/internal/run_density.py), [restart-corrected NVT runner](md/internal/run_nvt_segment.py) |
| Observable analysis | [Molecular energy/force analysis](../model_training/analysis/force_energy_rmse.py), [NPT density/RDF](md/internal/analyze_npt.py), [2 ns transport analysis](md/analyze_transport.py), [frozen VACF/MSD/Green–Kubo estimators](md/internal/analyze_preliminary_nvt_dynamics.py), [portable aggregation and plots](scripts/) |
| Model and checkpoint manifests | [All 14 model identities, all three training seeds, checkpoint and fused-model hashes](models/manifest.json), [adapter target modules](results/2ns/data/lora_target_parameters.csv), [model access](../MODEL_ACCESS.md) |
| Per-seed molecular results | [Molecular validation values for seeds 17, 29, 43](results/2ns/data/cluster_rmse_all_seeds.csv) |
| Per-replica and per-seed 2 ns results | [168 lane estimates](results/2ns/data/prefix2000ps/lanes.csv), [56 seed/temperature states](results/2ns/data/prefix2000ps/states_by_training_seed.csv), [seed means and MD variation](results/2ns/data/seed_means_and_md_sd.csv), [24 model/temperature summaries](results/2ns/data/model_summary_2ns.csv) |
| Figure-generation scripts and numerical inputs | [Rebuild command](run_reproduce.sh), [main figure builder](scripts/build_results.py), [diagnostic figure builder](scripts/build_diagnostics.py), [compact correlation arrays](results/2ns/data/curves/), [stress diagnostics](results/2ns/data/stress_diagnostics/) |
| Provenance and integrity | [Release file manifest](manifest.json), [checksums](SHA256SUMS), [completion records](results/2ns/data/trajectory_completion.csv), [label provenance](results/2ns/provenance/label_source_audit.json), [reference-force invariance audit](results/2ns/provenance/monomer_reference_invariance.json) |

## Reproduce the corrected figures without a GPU

From the repository root, using Python 3.12:

```bash
python -m venv .venv-reproduce
source .venv-reproduce/bin/activate
python -m pip install -r reproducibility/environment/requirements-analysis.txt
python reproducibility/scripts/verify.py
bash reproducibility/run_reproduce.sh
```

The rebuild writes to `runs/reproducibility-2ns/` and leaves the released
inputs unchanged. It regenerates the tables, Figures 1 and 2, Figure A1,
Figures S1–S11, a 24-state appendix, and a combined paper-figure PDF. It checks
the rebuilt numerical tables against the frozen values. An alternative output
directory can be supplied with `--output /path/to/new/output`.

The analysis environment needs no MACE, PyTorch, model binary, raw trajectory,
scheduler, or access to the original simulation filesystem.

## Corrected 2 ns results

| Figure or analysis | Numerical inputs | Released figure |
|---|---|---|
| Figure 1: equilibrium density and O–O RDF | [Density](results/2ns/data/density_summary.csv), [168 density lanes](results/2ns/data/density_all168_lanes.csv), [RDF mean/SD](results/2ns/data/rdf_295K.csv), [all 42 RDF replicas](results/2ns/data/rdf_295K_per_replica.csv) | [PDF](results/2ns/figures/figure1_structure.pdf) |
| Figure 2: corrected diffusion and viscosity | [All 168 replicas](results/2ns/data/prefix2000ps/lanes.csv), [24 states](results/2ns/data/model_summary_2ns.csv) | [PDF](results/2ns/figures/figure2_transport.pdf) |
| Figure A1: molecular validation | [All training seeds](results/2ns/data/cluster_rmse_all_seeds.csv) | [PDF](results/2ns/figures/figureA1_molecular.pdf) |
| Prefix sensitivity: 1.0, 1.5, 2.0 ns | [1 ns](results/2ns/data/prefix1000ps/), [1.5 ns](results/2ns/data/prefix1500ps/), [2 ns](results/2ns/data/prefix2000ps/) | Rebuilt Figure S11 |
| Disjoint first and second nanoseconds | [0–1 ns](results/2ns/data/prefix1000ps/), [1–2 ns](results/2ns/data/window1000_2000ps/) | Rebuilt Figures S8–S9 |
| Stress/VACF, running integrals, lag/block sensitivity | [Stress arrays](results/2ns/data/stress_diagnostics/), [curves](results/2ns/data/curves/), [block values](results/2ns/data/block_viscosity.csv), [lag values](results/2ns/data/lane_lag_sensitivity.csv) | Rebuilt Figures S2–S7 and S10 |

The study contains two foundations, three coverage conditions, and four
temperatures (275, 295, 315, 336 K). Each adapted condition has three training
seeds (17, 29, 43), each with three MD replicas. Each baseline has three MD
replicas and no training seed: 144 adapted + 24 baseline = **168 trajectories**.
Each trajectory has 1 ns NPT followed by 2 ns NVT; the total NVT sampling is
336 ns. NPT density and RDF use the final 500 ps.

Dynamics use float64, a 0.5 fs NVT step, and native 0.5 fs velocity/stress
records. Diffusion and viscosity use the prespecified 5–10 ps running-integral
mean. The Yeh–Hummer correction is applied per lane before averaging.
Adapted error bars are sample SD across three training-seed means after
averaging the three MD replicas within each seed. Baseline error bars are
MD-replica sample SD. These are not confidence intervals; 2 ns completion does
not establish size or time convergence.

Compact arrays are thinned for plotting; scalar values were calculated from
the full native-cadence records. The compact rebuild does not re-integrate
thinned curves as a substitute for the original estimator. Archived native-data
audit results are distinguished from checks performed by the portable validator.
Historical archive locations in provenance are logical `archive/` references,
not local file dependencies.

## Software versions and remaining inputs

The manuscript specifies **MACE 0.3.16, PyTorch 2.7.1, ASE 3.29.0, float64**.
The corrected 2 ns campaign's frozen MACE source declares **0.3.17**; the
actual recorded PyTorch build is **2.7.1+cu128**. See the
[version reconciliation](environment/README.md) before describing the two runs
as using one identical environment. The exact corrected source is included.

Checkpoint binaries and raw MD trajectories are not bundled. Model identities
and checksums are included for all 14 models, together with the training,
fusion, simulation, and analysis sources. Checkpoint-dependent reproduction
requires the separately obtained models described in [MODEL_ACCESS.md](../MODEL_ACCESS.md).
The repository's existing distribution terms and post-review release policy
still apply. The compact figure-reproduction route above is complete without
these binaries.

G2–G6 retain the documented raw total quantum labels. G1 uses separately
aligned fragment energies and geometric-distortion forces; its source lacks
explicit method metadata and its validation forces contain a rotational
component of about 0.897 meV/Å. The provenance audits document these limitations;
this release does not relabel or change the historical training data.

The earlier 700 ps artifact remains available through [the original rebuild](../reproduce.sh)
and [historical figure index](../FIGURES.md). Its LES bulk values are superseded
by this corrected section.
