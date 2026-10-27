# Training and molecular evaluation

The [dataset](data/molecular_clusters_1000/) contains G1–G2 and G1–G6
train/validation/test splits. See the [data card](../DATA_CARD.md) for labels,
split counts, and provenance.

## Train and fuse adapters

```bash
python model_training/train.py --foundation maceoff23m --condition g1-g2 --seed 17
python model_training/train.py --foundation maceles-off --condition g1-g6 --seed 17
```

Run both coverage conditions for each foundation with seeds 17, 29, and 43.
The launcher trains to 123,200 updates, exports the fused model, and evaluates
it. Use `--dry-run` to inspect commands. Foundation checkpoint paths are in
[`configs/study.yaml`](../configs/study.yaml); see [checkpoint access](../MODEL_ACCESS.md).

## Evaluation

[Force/energy evaluation](analysis/force_energy_rmse.py) and
[seed aggregation](analysis/aggregate_force_energy.py) produce molecular error
tables. The release's [per-seed values](../reproducibility/results/2ns/data/cluster_rmse_all_seeds.csv)
underlie Figure A1 and use parent-held-out validation configurations.

The `analysis/` and `data/bec_frame0/` directories also retain the one-frame
latent-charge and Born-effective-charge analysis described in the
[workshop archive](../archive/workshop_700ps/RESULTS.md).
