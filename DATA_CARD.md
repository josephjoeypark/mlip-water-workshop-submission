# Data card

## Molecular data

The dataset contains 1,000 parent configurations, split 800/100/100 into
training/validation/test sets. All child clusters stay in their parent's split.

| Coverage | Training records | Validation records | Test records |
|---|---:|---:|---:|
| G1–G2 | 5,600 | 700 | 700 |
| G1–G6 | 8,800 | 1,100 | 1,100 |

G2–G6 use raw total-energy and force labels documented as
omegaB97M-V/def2-TZVPD. G1 uses aligned fragment energies and
geometric-distortion forces; its source method metadata is incomplete.
[Label audits](reproducibility/PROVENANCE.md#reference-labels) document this
distinction and the monomer-force invariance residual.

Split files and hashes are in
[`model_training/data/molecular_clusters_1000/`](model_training/data/molecular_clusters_1000/).
Figure A1 uses validation configurations; the locked test split is not plotted.

## Liquid data

The [MD input](reproducibility/md/input/liquid-64.xyz) contains 64 water
molecules. The [compact results](reproducibility/results/2ns/data/) cover 168
lanes, each with 1 ns NPT and 2 ns NVT, plus per-seed summaries and plotting
arrays. Full trajectories and checkpoint binaries are not included.

Adapted uncertainty is sample SD across three training-seed means after
averaging MD replicas; baseline uncertainty is MD-replica sample SD. These
are descriptive sampling statistics, not confidence intervals or evidence
of thermodynamic-limit convergence.

Experimental reference curves and citations are recorded in
[source provenance](reproducibility/results/2ns/provenance/external_sources.json).
