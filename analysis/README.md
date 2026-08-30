# Analysis entry points

- `build_paper_figures.py`: Figures 1--3.
- `aggregate_cluster_evaluations.py`: three-seed Figure 1 mean/SD tables.
- `build_paper_tables.py`: parameter and BEC tables.
- `plot_density_times_viscosity.py`: Figure 2 density times Figure 3 viscosity.
- `plot_transport_spacing_sensitivity.py`: correlation-grid sensitivity.
- `transport.py`: manifest builder and exact trajectory-level transport analysis.
- `aggregate_npt.py`: three-replica density/RDF aggregation for Figure 2.
- `package_transport.py`: compact Figure 3 states, curves, and stress decomposition.
- `extract_les_latent_charges.py`: raw LES latent outputs for one frame.
- `extract_les_bec.py`: native atom-wise 3x3 BEC tensors for one frame.

`internal/analyze_transport.py` contains the longer audited estimators used by
the concise `transport.py` interface. Keeping that implementation intact makes
the Green--Kubo, VACF, MSD, and Yeh--Hummer calculations inspectable without
making the normal command line complicated.
