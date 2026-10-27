# Molecular dynamics and analysis

The [protocol](../configs/protocol_2ns.json) defines 1 ns NPT followed by 2 ns
NVT for each of the [168 lanes](../configs/lanes.csv). The
[64-water starting configuration](input/liquid-64.xyz), production scripts,
and MACE/LES source snapshot are included here.

Install the [production environment](../environment/README.md) and supply the
models named in the [checkpoint manifest](../models/manifest.json).

## Run a lane

From the repository root:

```bash
python reproducibility/md/run_lane.py --lane 0 --stage gate
python reproducibility/md/run_lane.py --lane 0 --stage npt
python reproducibility/md/run_lane.py --lane 0 --stage analyze-npt
python reproducibility/md/run_lane.py --lane 0 --stage nvt --segment 1
```

These commands print the planned invocation. Add `--execute` inside a compute
allocation. Complete the stages in order, then run NVT segments **1–16**:
500 ps followed by fifteen 100 ps continuations. Each continuation restores
the preceding atomic and thermostat checkpoints.

Use `--models-dir` and `--runs-dir` to select input and output locations.

## Analyze trajectories

```bash
python reproducibility/md/analyze_transport.py --window all
```

Add `--execute` in a CPU allocation to run the analysis. Use `--runs-dir` to
locate trajectories and `--output` for a new result directory. Individual
windows are `1000`, `1500`, `2000`, and `1000:2000` ps; `--lane N` selects a lane.

The wrapper processes all 16 segments at native 0.5 fs cadence and retains
training-seed and MD-replica identities. Full-study outputs include lane,
seed, and model/temperature summaries. Use this wrapper as the analysis entry
point; the modules under `internal/` supply its estimators.
