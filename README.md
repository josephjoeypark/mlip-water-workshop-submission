# Molecular-to-liquid deployment audit

Code and data for water-cluster adaptation and liquid-water simulations with
MACEOFF23(M) and MACELES-OFF. The study covers three coverage conditions,
four temperatures, and **168 trajectories with 2 ns NVT sampling each**.

## Reproduce the figures

With Python 3.12, run from the repository root:

```bash
python -m pip install -r reproducibility/environment/requirements-analysis.txt
bash reproducibility/run_reproduce.sh
```

Figures and tables are written to `runs/reproducibility-2ns/`.
This uses the included compact data; no GPU or model checkpoint is needed.

## Contents

| Material | Location |
|---|---|
| Software, protocols, and reproduction guide | [reproducibility/](reproducibility/README.md) |
| Cluster geometries and reference labels | [Training data](model_training/data/molecular_clusters_1000/) · [Data card](DATA_CARD.md) |
| Training, adapter fusion, and MD | [Training](model_training/README.md) · [MD](reproducibility/md/README.md) |
| Per-seed and per-replica results | [Results](RESULTS.md) · [Figures](FIGURES.md) |
| Model identities and checkpoint access | [Manifest](reproducibility/models/manifest.json) · [Access](MODEL_ACCESS.md) |

[License](LICENSE) · [Third-party notices](THIRD_PARTY_NOTICES.md) · [Provenance](PROVENANCE.md)
