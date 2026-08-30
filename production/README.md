# Production entry points

- `train.py`: fixed-update G1-G2 or G1-G6 LoRA training and fused export.
- `evaluate_clusters.py`: Figure 1 energy/force RMSE on G1--G6 clusters.
- `run_npt.py`: one 1 ns density lane (or short gate).
- `analyze_npt.py`: density, RDF, and representative-frame selection.
- `run_nvt.py`: one restart-safe, six-segment 1 ns transport lane.
- `run_campaign.py`: enumerate all 72 lanes or select one scheduler-array index.

All paths and scientific settings come from `configs/study.yaml`. Each public
entry point stays short; audited low-level implementations are isolated under
`production/internal`. Use `--dry-run` to inspect resolved commands without
creating an expensive job.
