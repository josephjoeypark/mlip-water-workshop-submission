# Release provenance

This page records technical lineage and changes between study versions.
The [release guide](README.md) is the entry point for reproduction.

## Source and validation

The 2 ns inputs come from the completed October 2026 analysis: 168 trajectories,
24 model/temperature states, and 14 model identities. The
[file manifest](manifest.json) records source and released hashes, including
the 42 individual NPT RDF inputs. Machine-specific locations were converted
to logical `archive/` references; numerical values were preserved.

The portable rebuild verifies identity and replica coverage, reconstructs the
RDF mean and uncertainty, and compares regenerated numerical tables with the
frozen results. It regenerates 14 figures and a 24-state appendix. Native-data
restart and trajectory audits are retained separately under
[`archived_validation/`](results/2ns/archived_validation/); the portable
validator does not claim to rerun those raw-trajectory checks.

Compact curves are thinned for display. Transport scalars were calculated
from full native 0.5 fs records. Lane-wise Yeh–Hummer finite-size adjustments
precede seed averaging, retaining covariance with viscosity. Sampling and
cell-size limitations remain part of the interpretation.

## Software versions

The manuscript excerpt specified MACE 0.3.16, PyTorch 2.7.1, and ASE 3.29.0.
The source snapshot used for the final 2 ns campaign declares MACE 0.3.17;
the recorded PyTorch build is 2.7.1+cu128. Both records are retained in
[`software_versions.json`](environment/software_versions.json).
[`requirements-manuscript.txt`](environment/requirements-manuscript.txt)
preserves the earlier declaration. The [environment guide](environment/README.md)
and hashed source snapshot identify the released simulation implementation.

## Reference labels

G2–G6 retain raw total quantum labels with omegaB97M-V/def2-TZVPD metadata.
G1 combines aligned fragment energies and geometric-distortion forces;
explicit source-method metadata is absent. Its validation references contain
a rotational-force component of approximately 0.897 meV/Å, incompatible with
an isolated invariant scalar-energy model. The release retains the original
labels and documents these limitations in the
[label-source audit](results/2ns/provenance/label_source_audit.json) and
[monomer invariance audit](results/2ns/provenance/monomer_reference_invariance.json).

## Version history

The earlier LES wrapper duplicated a cell-strain contribution already present
in the native implementation. All LES bulk trajectories in the final release
use native single-cell strain; affected historical LES outputs are excluded.
The earlier 2.4-fold viscosity split and persistent-mode interpretation are
withdrawn. The final 295 K G1–G6 LES/OFF ratio of mean viscosities is about
1.207. Molecular checkpoints and validation results were unchanged by this
MD correction.

The NVT implementation also includes the validated restoration of atomic
momenta and thermostat state at segment boundaries. The production sources
and their hashes are retained under [`md/internal/`](md/internal/).

The [700 ps workshop documentation](../archive/workshop_700ps/README.md)
preserves the earlier snapshot from commit `6729d5b`. Its bulk results are
superseded by the final release. Its original code and data remain available
for provenance; the main README and figure index point to the 2 ns release.
