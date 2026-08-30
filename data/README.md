# Data layout

- `training/g1-g2`: full train/validation/test extxyz splits.
- `training/g1-g6`: full JSONL splits plus validation/test extxyz exports.
- `paper_figures`: compact numerical inputs for Figures 1--3.
- `paper_tables`: generated Tables 1--2.
- `response`: one periodic frame plus raw LES latent/BEC outputs.
- `transport_sensitivity`: 0.5/1/2/5 fs correlation-grid summaries.

Run `python tools/validate_submission.py` to verify hashes, counts, and schema
coverage. See the root `DATA_CARD.md` for split policy and interpretation.
