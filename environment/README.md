# Environments

For figures/tables, create `environment.yml`; its pip section reads the pinned
analysis requirements. For training, MD, or LES response extraction, install
`requirements-production.txt` afterward. MACE and LES are pinned by immutable
Git commits in `source-revisions.json`.

CUDA wheels can be platform-specific. Install the PyTorch 2.7.1 build matching
the target GPU driver before the remaining production requirements if the
default pip resolution does not select a compatible build.
