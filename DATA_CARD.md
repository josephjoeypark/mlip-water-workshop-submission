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
in `data/training/*/manifest.json`. Figure 1 uses the validation parents; the
locked test parents were not used for plotted model selection.

## Liquid and paper data

- `data/response/liquid-64.xyz`: one periodic 64-water configuration used as
  the production starting structure and one-frame response example.
- `data/response/les_latent_charges_64water_frame0.csv`: raw latent outputs;
  these are not calibrated partial charges.
- `data/response/les_bec_64water_frame0.csv`: raw 3x3 atom-wise BEC tensors.
- `data/paper_figures`: compact state summaries and plotted curves.
- `data/paper_tables`: manuscript table values generated from source data.

External experimental/reference curves retain their citations in `RESULTS.md`
and `FIGURES.md`. The compact tables report SD or SEM explicitly in column
names; those statistics must not be interchanged.

## Intended use and limits

The data support non-commercial academic reproduction of this molecular-to-
liquid deployment audit. They do not establish transferable charge labels,
thermodynamic-limit transport, or a controlled causal ablation between the two
different foundation checkpoints. Consult the upstream model licenses before
training or redistribution.
