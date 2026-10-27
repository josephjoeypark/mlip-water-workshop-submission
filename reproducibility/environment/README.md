# Pinned environments and version provenance

For figure reproduction, use Python 3.12.11 and install
`requirements-analysis.txt`. It includes the repository's pinned NumPy 1.26.4,
pandas 3.0.5, Matplotlib 3.11.1, SciPy 1.14.1, ASE 3.29.0 and PyYAML 6.0.3,
plus pypdf 6.1.1. This route needs no GPU or model binary.

| Component | Section 2.4 declaration | Corrected 2 ns campaign evidence |
|---|---|---|
| MACE | 0.3.16 | Frozen `mace/__version__.py` declares 0.3.17 |
| PyTorch | 2.7.1 | 2.7.1+cu128 |
| ASE | 3.29.0 | 3.29.0 |
| Execution precision | float64 | float64 |
| Python | Not specified in the quoted paragraph | 3.12.11 |

The manuscript declaration is recorded literally in
[`requirements-manuscript.txt`](requirements-manuscript.txt). It is not an
assertion that the corrected 2 ns run used the same MACE source. Reconcile
that sentence before making a single-version claim in the revised manuscript.
The version evidence is in [`software_versions.json`](software_versions.json).

For checkpoint-dependent work, install `requirements-production.txt`, which
retains the existing commit pins in
[`source-revisions.json`](../../environment/source-revisions.json). The corrected
MD launcher prepends the exact archived MACE, LES, and compatibility sources
under [`md/vendor`](../md/vendor/) in their original precedence order:
`compat`, `mace`, `les`. Those source bytes are checksum-identified in the
[release manifest](../manifest.json), including the native LES stress behavior.
The CUDA-specific PyTorch build depends on the machine; match the recorded
`2.7.1+cu128` build when replaying the original GPU environment.

The baseline MACELES checkpoint is stored in float32 and is converted to
float64 for simulation, as recorded in the model manifest. This does not change
the stored checkpoint hash.

The source snapshot includes third-party code. Retain the repository's
[notices](../../THIRD_PARTY_NOTICES.md), [MACE code license](../../licenses/MACE_CODE_MIT.md),
and [LES notice](../../licenses/LES_CC_BY_NC_4.0_NOTICE.md).
