# Molecular-to-liquid deployment audit

Anonymous NeurIPS workshop artifact for a six-model water study. The repository
contains the full 1,000-parent molecular dataset, exact fixed-update training
launchers, restart-safe NPT/NVT production code, response analysis, compact
paper data, and scripts that rebuild every reported table and figure.

## Models and names

Two foundations are crossed with three conditions:

| Foundation | Baseline | G1-G2 | G1-G6 |
|---|---|---|---|
| `MACEOFF23(M)` | released checkpoint | LoRA on monomers/dimers | LoRA on monomers--hexamers |
| `MACELES-OFF` | released checkpoint | LoRA on monomers/dimers | LoRA on monomers--hexamers |

`Baseline` is not trained in this study. Both adaptations use rank-32 LoRA
with $\alpha=32$ and are evaluated at exactly 123,200 optimizer updates. The
paper reports seeds 17, 29, and 43 for molecular validation; seed 17 is the MD
deployment model.

## Rebuild the paper outputs

The analysis-only route needs no GPU, checkpoint, or raw trajectory:

```bash
conda env create -f environment/environment.yml
conda activate water-deployment-analysis
bash reproduce.sh
```

This regenerates Tables 1--2, Figures 1--3, the density--viscosity product,
and the correlation-grid sensitivity outputs. Compact CSV/NPZ inputs are
committed under [`data/paper_figures`](data/paper_figures/), and the generated
files are written to [`figures`](figures/). See [`FIGURES.md`](FIGURES.md) and
[`RESULTS.md`](RESULTS.md) for captions and interpretation.

## Run the full pipeline

Install the production dependencies after creating the analysis environment:

```bash
python -m pip install -r environment/requirements-production.txt
```

Place the six checkpoint files named in [`models/README.md`](models/README.md)
under `models/`, then verify their SHA-256 values against
[`models/manifest.json`](models/manifest.json). The checkpoints are not in the
submission artifact because redistribution authorization for the adapted and
MACELES-derived binaries has not been confirmed. NeurIPS review confidentiality
does not replace upstream permission. See [`MODEL_ACCESS.md`](MODEL_ACCESS.md)
for the exact reproducibility boundary and the condition for adding them.

Train one adaptation (repeat with seeds 17, 29, and 43 where required):

```bash
python production/train.py --foundation maceoff23m --condition g1-g2 --seed 17
python production/train.py --foundation maceoff23m --condition g1-g6 --seed 17
python production/train.py --foundation maceles-off --condition g1-g2 --seed 17
python production/train.py --foundation maceles-off --condition g1-g6 --seed 17
```

Run one of the 72 model/temperature/replica lanes:

```bash
python production/run_npt.py --model maceles-off-g1-g6 --temperature 295 --replica 1
python production/analyze_npt.py --model maceles-off-g1-g6 --temperature 295 --replica 1
python production/run_nvt.py --model maceles-off-g1-g6 --temperature 295 --replica 1
```

For scheduler arrays, `python production/run_campaign.py npt --lane-index N`
prints lane `N` (`0 <= N < 72`); add `--execute` inside the allocated job. Use
the same interface with `analyze-npt` and `nvt`.

The NPT protocol is 1 ns at 1.01325 bar with a 1 fs step; density and RDF use
the final 500 ps. NVT is 1 ns at fixed volume with a 0.5 fs step and native
0.5 fs velocity/stress recording. After all lanes complete:

```bash
python analysis/aggregate_npt.py
python analysis/transport.py --prefix-ps 1000
python analysis/package_transport.py --analysis runs/transport --output runs/figure3_inputs
```

The frozen Figure 3 values use the common 700 ps prefix available for every
lane at the manuscript snapshot. The runner and endpoint analysis are set to
the planned 1 ns protocol; regenerating from 1 ns trajectories can therefore
change Figure 3 values and must be reported as a new analysis version.

Extract LES latent charges and Born effective charge tensors from frame 0:

```bash
python analysis/extract_les_latent_charges.py \
  --input data/response/liquid-64.xyz --models-dir models \
  --output runs/les_latent_charges.csv
python analysis/extract_les_bec.py \
  --input data/response/liquid-64.xyz --models-dir models \
  --output runs/les_bec.csv
```

The latent values are not calibrated physical partial charges. BECs are kept
in their raw model convention; Table 2 applies only one global sign flip for
comparison with the conventional water sign.

## Repository map

```text
analysis/       short public entry points and the audited transport estimator
configs/        one study manifest plus MD/training settings
data/           full molecular splits, compact paper data, and one liquid frame
environment/    pinned analysis and production dependencies
figures/        regenerated paper and supplementary figures
licenses/       upstream model/code notices
models/         checkpoint manifest; binary checkpoints are not distributed
production/     training, NPT, NPT analysis, and restart-safe NVT entry points
tools/          artifact validation
```

Exact provenance, internal-label mapping, data boundaries, and clean-room
checks are in [`PROVENANCE.md`](PROVENANCE.md), [`DATA_CARD.md`](DATA_CARD.md),
and [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).

## License boundary

Repository-authored code is covered by [`LICENSE`](LICENSE). MACE code, LES,
and MACE-OFF model artifacts retain their upstream terms; see
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). No model license is replaced
or broadened by this repository.
