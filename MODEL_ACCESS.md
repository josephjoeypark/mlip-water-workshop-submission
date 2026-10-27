# Checkpoint access

Checkpoint binaries are not bundled; redistribution authorization has not
been confirmed. The [14-model manifest](reproducibility/models/manifest.json)
records filenames, training seeds, optimizer updates, and checkpoint hashes.

Figures and tables can be reproduced from the included compact data without
these binaries. Training, inference, and MD require the appropriate foundation
or fitted checkpoint.

For MD, place authorized files in `reproducibility/models/` or pass
`--models-dir`. Training reads foundation paths from
[`configs/study.yaml`](configs/study.yaml). Verify each file against the
manifest before use and retain its [upstream license](THIRD_PARTY_NOTICES.md).
