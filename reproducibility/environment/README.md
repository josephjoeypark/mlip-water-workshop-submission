# Environment

For figure reproduction, use Python 3.12 and install
[`requirements-analysis.txt`](requirements-analysis.txt). It pins NumPy,
pandas, Matplotlib, SciPy, ASE, PyYAML, and pypdf.

The recorded simulation environment is:

| Component | Version |
|---|---|
| Python | 3.12.11 |
| MACE source | 0.3.17, as declared by the included snapshot |
| PyTorch | 2.7.1+cu128 |
| ASE | 3.29.0 |
| Runtime precision | float64 |

For training and MD, install
[`requirements-production.txt`](requirements-production.txt). The MD launcher
loads the included [MACE/LES source snapshot](../md/vendor/) in the recorded
order: `compat`, `mace`, `les`. Exact files are identified by the
[release manifest](../manifest.json).

The baseline MACELES checkpoint is stored in float32 and converted to float64
for simulation. See [version provenance](../PROVENANCE.md#software-versions)
for the manuscript declaration and [third-party notices](../../THIRD_PARTY_NOTICES.md)
for source licenses.
