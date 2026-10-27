# Reproducibility

The [software and reproducibility guide](reproducibility/README.md) contains
the complete release inventory and setup instructions.

```bash
python -m pip install -r reproducibility/environment/requirements-analysis.txt
bash reproducibility/run_reproduce.sh
```

This rebuilds the figures and tables from the included compact data. To check
the inputs without generating figures:

```bash
python reproducibility/scripts/verify.py
```

Training and MD require [separately supplied checkpoints](MODEL_ACCESS.md).
