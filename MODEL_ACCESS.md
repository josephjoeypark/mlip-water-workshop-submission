# Checkpoint access and submission decision

## Decision for this archive

The six model binaries are **not included**. Redistribution authorization for
the adapted and MACELES-derived checkpoints has not been confirmed. A
confidential reviewer upload is still a transfer to third parties, so review
confidentiality alone is not treated as permission to redistribute them.

This is a rights boundary, not a file-size workaround. The six uncompressed
checkpoints total more than 100 MB, but a compressed repository-plus-models
archive may fit below the NeurIPS 100 MB limit. They must still remain absent
unless the relevant rights holders explicitly authorize the proposed reviewer
distribution and all accompanying license obligations are satisfied.

## What reviewers can reproduce without checkpoints

Run:

```bash
bash reproduce.sh
```

This rebuilds every manuscript table and figure from the committed numerical
source data. `SHA256SUMS` authenticates the archive, and
`tools/validate_submission.py` checks its size, anonymity, dataset splits,
model names, response dimensions, and reported-data coverage.

The supplement also contains the complete 1,000-parent training dataset,
training and production programs, simulation settings, analysis estimators,
software revisions, and exact SHA-256 identifier for every omitted checkpoint.
Consequently, reviewers can audit the plotted values and methods, but cannot
re-run checkpoint-dependent inference or MD from this ZIP alone.

## What requires an authorized checkpoint bundle

- checkpoint-dependent validation for Figure 1;
- regeneration of NPT/NVT trajectories for Figures 2 and 3;
- LES latent-charge and Born-effective-charge inference;
- deployment of the six reported models.

The required filenames and hashes are in `models/manifest.json`. If written
authorization is obtained, place those exact files under `models/`, retain all
upstream licenses and notices, verify their hashes, and state the applicable
terms in the reviewer instructions. Do not silently substitute retrained or
different-version checkpoints.

The released MACE-OFF foundation may be obtained from its official project
under its Academic Software License. That public availability does not by
itself establish permission for redistribution of every adapted or
MACELES-derived binary used here.

Relevant upstream and submission documents:

- NeurIPS Code Submission Policy: <https://neurips.cc/public/guides/CodeSubmissionPolicy>
- official MACE-OFF repository: <https://github.com/ACEsuit/mace-off>
- official MACE repository: <https://github.com/ACEsuit/mace>
- copied license and attribution texts: `licenses/`
