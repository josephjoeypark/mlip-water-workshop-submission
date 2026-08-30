# Data card

## Molecular adaptation data

The source population contains 1,000 parent molecular configurations labeled
at $\omega$B97M-V/def2-TZVPD. Parents, rather than child clusters, are split
800/100/100 into train/validation/test sets to prevent related clusters from
crossing partitions.

| Lane | Train | Validation | Test |
|---|---:|---:|---:|
| G1-G2 | 5,600 records / 800 parents | 700 / 100 | 700 / 100 |
| G1-G6 | 8,800 records / 800 parents | 1,100 / 100 | 1,100 / 100 |

The larger record totals are deterministic cluster expansions, not additional
independent parents. File-level hashes and counts are stored beside each split
in `model_training/data/molecular_clusters_1000/*/manifest.json`. Figure 1 uses the validation parents; the
locked test parents were not used for plotted model selection.

## Liquid and paper data

- `md_simulations/data/initial_configuration/liquid-64.xyz`: the exact cited
  upstream 64-water configuration used to initialize MD; see its `SOURCE.md`.
- `model_training/data/bec_frame0/liquid-64.xyz`: the same frame used for
  one-frame response evaluation.
- `model_training/data/bec_frame0/les_latent_charges_64water_frame0.csv`: raw latent outputs;
  these are not calibrated partial charges.
- `model_training/data/bec_frame0/les_bec_64water_frame0.csv`: raw 3x3 atom-wise BEC tensors.
- `md_simulations/data/published_results`: compact state summaries and plotted curves.
- `model_training/data/model_parameters.csv` and
  `model_training/data/bec_frame0/bec_comparison.csv`: manuscript tables.

External experimental/reference curves retain their citations in `RESULTS.md`
and `FIGURES.md`. The compact tables report SD or SEM explicitly in column
names; those statistics must not be interchanged.

## Intended use and limits

The data support non-commercial academic reproduction of this molecular-to-
liquid deployment audit. They do not establish transferable charge labels,
thermodynamic-limit transport, or a controlled causal ablation between the two
different foundation checkpoints. Consult the upstream model licenses before
training or redistribution.
