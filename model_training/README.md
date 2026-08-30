# Model training and molecular evaluation

## Data

- `data/molecular_clusters_1000/`: the fixed 1,000-parent dataset, with
  train/validation/test products and manifests for G1-G2 and G1-G6.
- `data/bec_frame0/liquid-64.xyz`: frame 0 used for LES latent-charge and
  Born-effective-charge evaluation.
- `data/bec_frame0/*.csv`: atom-wise latent outputs, raw 3x3 BEC tensors, and
  the compact comparison table.
- `data/force_energy_results.csv`: Figure 1 mean and sample SD over seeds
  17, 29, and 43 at optimizer update 123,200.
- `data/force_energy_seed17.csv`: deployed-seed Figure 1 values.

## Training

`train.py` is the single public launcher for both foundations and both
adaptation conditions. The low-level frozen implementations remain under
`internal/training/`.

```bash
python model_training/train.py --foundation maceoff23m --condition g1-g2 --seed 17
python model_training/train.py --foundation maceles-off --condition g1-g6 --seed 17
```

Use `--dry-run` to resolve every input, update count, checkpoint, evaluation,
and fused-export command without starting GPU work.

## Analysis

- `analysis/force_energy_rmse.py`: energy and force-component RMSE by cluster
  size G1--G6 for one fused checkpoint.
- `analysis/aggregate_force_energy.py`: combines the three training seeds into
  the Figure 1 mean/SD table.
- `analysis/evaluate_bec.py`: atom-wise native 3x3 BEC tensors for the three
  MACELES-OFF checkpoints.
- `analysis/extract_latent_charges.py`: raw LES latent outputs for the same
  frame; these are not calibrated physical partial charges.

The model files are intentionally absent. Exact names and hashes are in
`../models/manifest.json`; see `../MODEL_ACCESS.md`.
