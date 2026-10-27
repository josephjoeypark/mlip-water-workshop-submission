# Software and reproducibility

Materials for the 2 ns water study, corresponding to manuscript Section 2.4.

## Reproduce

From the repository root, using Python 3.12:

```bash
python -m venv .venv-reproduce
source .venv-reproduce/bin/activate
python -m pip install -r reproducibility/environment/requirements-analysis.txt
bash reproducibility/run_reproduce.sh
```

The command verifies the inputs and rebuilds the tables, Figures 1–2,
Figure A1, Figures S1–S11, and the 24-state appendix in
`runs/reproducibility-2ns/`. Use `--output PATH` to choose another destination.
No GPU, checkpoint, or raw trajectory is required for this step.

## Release contents

| Material | Files |
|---|---|
| Geometries and reference labels | [Train/validation/test data](../model_training/data/molecular_clusters_1000/) · [Data card](../DATA_CARD.md) |
| Pinned environment and protocols | [Environment](environment/README.md) · [Training](../model_training/configs/adaptation.yaml) · [MD protocol](configs/protocol_2ns.json) · [Lane identities](configs/lanes.csv) |
| Training and adapter fusion | [Launcher](../model_training/train.py) · [G1–G2 export](../model_training/internal/training/g1-g2/export_model.py) · [G1–G6 export](../model_training/internal/training/g1-g6/evaluate.py) |
| MD and observable analysis | [MD guide](md/README.md) · [Molecular evaluation](../model_training/analysis/force_energy_rmse.py) · [Transport analysis](md/analyze_transport.py) |
| Models and checkpoints | [14-model manifest](models/manifest.json) · [Adapter targets](results/2ns/data/lora_target_parameters.csv) · [Access](../MODEL_ACCESS.md) |
| Per-seed and per-replica results | [Molecular](results/2ns/data/cluster_rmse_all_seeds.csv) · [2 ns lanes](results/2ns/data/prefix2000ps/lanes.csv) · [Seed summaries](results/2ns/data/seed_means_and_md_sd.csv) · [RDF replicas](results/2ns/data/rdf_295K_per_replica.csv) |
| Figure generation | [Scripts](scripts/) · [Figure index](../FIGURES.md) · [All compact inputs](results/2ns/data/) |

## 2 ns results

The release includes 24 model/temperature states at 275, 295, 315, and 336 K.
Adapted models use training seeds 17, 29, and 43, with three MD replicas per
seed; baselines have three MD replicas. Each lane has 1 ns NPT and 2 ns NVT.

[State summaries](results/2ns/data/model_summary_2ns.csv) ·
[1 ns prefix](results/2ns/data/prefix1000ps/) ·
[1.5 ns prefix](results/2ns/data/prefix1500ps/) ·
[2 ns prefix](results/2ns/data/prefix2000ps/) ·
[Second nanosecond](results/2ns/data/window1000_2000ps/)

Adapted error bars are sample SD across training-seed means after averaging
MD replicas; baseline bars are MD-replica sample SD. Diffusion uses a lane-wise
Yeh–Hummer finite-size adjustment. Compact curves are thinned for display;
reported scalar estimates use the native 0.5 fs records.

[Checksums](SHA256SUMS) · [File manifest](manifest.json) · [Detailed provenance](PROVENANCE.md)
