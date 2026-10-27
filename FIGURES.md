# Figures

[Download the main figures as one PDF](reproducibility/results/2ns/paper_figures.pdf).

| Figure | Content | Plot | Numerical inputs |
|---|---|---|---|
| 1 | Equilibrium structure and density | [PDF](reproducibility/results/2ns/figures/figure1_structure.pdf) | [Density](reproducibility/results/2ns/data/density_summary.csv) · [RDF replicas](reproducibility/results/2ns/data/rdf_295K_per_replica.csv) |
| 2 | Diffusion and viscosity | [PDF](reproducibility/results/2ns/figures/figure2_transport.pdf) | [2 ns state summaries](reproducibility/results/2ns/data/model_summary_2ns.csv) |
| A1 | Molecular validation | [PDF](reproducibility/results/2ns/figures/figureA1_molecular.pdf) | [Per-seed errors](reproducibility/results/2ns/data/cluster_rmse_all_seeds.csv) |

The [rebuild command](reproducibility/README.md#reproduce) also generates
Figures S1–S11 and the 24-state appendix. Diffusion uses the Yeh–Hummer
finite-size adjustment; error bars are sample SD as defined in the
[analysis protocol](reproducibility/configs/protocol_2ns.json).
