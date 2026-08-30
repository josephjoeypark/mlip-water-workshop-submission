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

The combined binaries are over 100 MB and carry upstream academic/non-
commercial terms. They therefore belong in a separately permissioned reviewer
bundle, not the public workshop code ZIP. Baseline files are the released
foundations; G1-G2/G1-G6 files are fused deployment exports with no remaining
LoRA tensors.
