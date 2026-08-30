# Model input boundary

Binary checkpoints are deliberately not committed. Put these files here:

```text
maceoff23m-baseline.model
maceoff23m-g1-g2.model
maceoff23m-g1-g6.model
maceles-off-baseline.model
maceles-off-g1-g2.model
maceles-off-g1-g6.model
```

`manifest.json` gives the exact SHA-256, parameter count, training seed, and
optimizer update for each file. Verify before production:

```bash
sha256sum models/*.model
```

The six uncompressed files total more than 100 MB, although ZIP compression can
bring a combined artifact below that limit. Size is therefore not the reason
for omission. The binaries are excluded because redistribution authorization
for the adapted and MACELES-derived checkpoints has not been confirmed. Do not
add them to a public or confidential submission merely because it is technically
possible; add them only after obtaining explicit permission consistent with all
upstream terms. Baseline files are the released foundations; G1-G2/G1-G6 files
are fused deployment exports with no remaining LoRA tensors. See
`../MODEL_ACCESS.md`.
