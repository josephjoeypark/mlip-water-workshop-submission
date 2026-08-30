# Provenance and naming

The public names are fixed throughout the paper-facing code and data:

| Public condition | Archived campaign token |
|---|---|
| Baseline | ZS |
| G1-G2 | T12 |
| G1-G6 | D00 |

| Public foundation | Archived campaign token |
|---|---|
| MACEOFF23(M) | OFF |
| MACELES-OFF | LES |

Archived tokens remain only in `models/manifest.json` (`legacy_label`), in
private array keys inside the frozen Figure 3 NPZ, and in implementation-level
training arguments required to reconstruct historical checkpoints. They are
never presented as manuscript model names.

All adapted paper values use update 123,200. For G1-G2, 5,600 records with
batch size two give 2,800 updates per epoch, so the end of zero-indexed epoch
43 is exactly 44 x 2,800 = 123,200 updates. The exporter checks the optimizer
step stored in the checkpoint before fusing LoRA weights. G1-G6 training uses
an explicit maximum-step counter and its evaluator performs the same check.

The exact software revisions are recorded in
`environment/source-revisions.json`. Input/model hashes live in
`configs/study.yaml`, `models/manifest.json`, and the training manifests.
